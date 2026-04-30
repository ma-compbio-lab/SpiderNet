"""Module 4 — In silico spatial perturbation service.

Port of `spidernet_interactive_module4_spatial_perturbation_web.ipynb`.

The notebook supports two perturbation modes:
  1. Knockout — scale ligand genes on senders + receptor genes on receivers,
     restricted to chosen sender/receiver cell types.
  2. Cell replacement — sample donor cells of one type and overwrite cells of
     another type, preserving spatial coordinates.

Both modes re-run the trained SpiderNet model on cloned slices, then run a
paired t-test (BH-adjusted) on per-gene predicted expression for cells of
interest. Volcano + DE table + GO/KEGG enrichment follow.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from core.datasets import DatasetPaths
from core.loaders import get_core_bundle


# ============================================================================
# Defaults (mirrored from the notebook)
# ============================================================================

DEFAULT_TOP_N = 10
DEFAULT_KO_PERCENT = 50.0
DEFAULT_DE_LOGFC_THRESHOLD = 0.25
DEFAULT_DE_PADJ_THRESHOLD = 0.05
DEFAULT_ENRICH_DIRECTION = "up"
DEFAULT_ENRICH_SCOPE = "indirect_only"
DEFAULT_ENRICH_TOP_TERMS = 12
DEFAULT_MAX_CELLS_FOR_DE = 8000
DEFAULT_RANDOM_SEED = 0
DEFAULT_HIDDEN_CHANNELS = 256

CACHE_SCHEMA_VERSION = "v1_flask_m4"


# ============================================================================
# Cache layout
# ============================================================================

def _module_dir(ds: DatasetPaths) -> Path:
    p = ds.run_dir / "UI_Exports" / "module4_spatial_perturbation_web"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _runs_dir(ds: DatasetPaths) -> Path:
    p = _module_dir(ds) / "perturbation_runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


# In-memory caches: keyed by dataset name
_STATE_CACHE: dict[str, dict[str, Any]] = {}
_STATE_LOCK = Lock()
_RUN_CACHE: dict[str, dict[str, Any]] = {}     # cache_key -> {de_df, summary}
_RUN_LOCK = Lock()
_ENRICH_CACHE: dict[tuple, pd.DataFrame] = {}  # (kind, species, genes_tuple, top_n) -> df


# ============================================================================
# Generic helpers
# ============================================================================

def _to_numpy(x: Any) -> Optional[np.ndarray]:
    import scipy.sparse as sp
    import torch
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if sp.issparse(x):
        return x.toarray()
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def _ensure_str_array(x: Any) -> np.ndarray:
    arr = np.asarray(_to_list(x), dtype=object)
    return np.asarray([str(v) for v in arr], dtype=object)


def _get_data_attr(data_obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(data_obj, dict):
        return data_obj.get(key, default)
    return getattr(data_obj, key, default)


def _set_data_attr(data_obj: Any, key: str, value: Any) -> None:
    if isinstance(data_obj, dict):
        data_obj[key] = value
    else:
        setattr(data_obj, key, value)


def _standardize_edge_index(edge_index: Any) -> np.ndarray:
    edge_index = _to_numpy(edge_index)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape={edge_index.shape}")
    if edge_index.shape[1] == 2:
        return edge_index.astype(int)
    if edge_index.shape[0] == 2:
        return edge_index.T.astype(int)
    raise ValueError(f"Could not orient edge_index to [n_edges, 2]. Shape={edge_index.shape}")


def _format_lr_label(x: Any) -> str:
    if isinstance(x, (tuple, list, np.ndarray)):
        if len(x) >= 2:
            return f"{x[0]} -> {x[1]}"
        if len(x) == 1:
            return str(x[0])
    return str(x)


def _split_gene_token(x: Any) -> list[str]:
    if x is None:
        return []
    s = str(x).strip()
    if s == "":
        return []
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    s = s.replace(";", "+").replace("/", "+").replace(",", "+")
    return [p.strip() for p in s.split("+") if p.strip()]


def _parse_lr_entry(x: Any) -> dict:
    if isinstance(x, (tuple, list, np.ndarray)):
        if len(x) >= 2:
            ligand, receptor = str(x[0]), str(x[1])
        elif len(x) == 1:
            ligand = receptor = str(x[0])
        else:
            ligand = receptor = ""
    else:
        s = str(x)
        if "->" in s:
            ligand, receptor = [p.strip() for p in s.split("->", 1)]
        elif "|" in s:
            ligand, receptor = [p.strip() for p in s.split("|", 1)]
        else:
            ligand = receptor = s
    return {
        "label": _format_lr_label(x),
        "ligand": ligand,
        "receptor": receptor,
        "ligand_genes": _split_gene_token(ligand),
        "receptor_genes": _split_gene_token(receptor),
    }


def _ensure_mi_by_feature_matrix(arr: Any, n_mi: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    if arr.shape[0] == n_mi:
        return arr
    if arr.shape[1] == n_mi:
        return arr.T
    raise ValueError(
        f"Could not orient loading matrix to MI x feature format. "
        f"shape={arr.shape}, n_mi={n_mi}."
    )


def _column_sum_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    denom = np.sum(arr, axis=0, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return arr / denom


def _align_labels(labels: Any, n_features: int, prefix: str) -> list[str]:
    n_features = int(n_features)
    if labels is None:
        labels = []
    elif isinstance(labels, pd.Index):
        labels = labels.astype(str).tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif not isinstance(labels, (list, tuple)):
        try:
            labels = list(labels)
        except TypeError:
            labels = [labels]
    labels = [str(x) for x in labels]
    if len(labels) < n_features:
        labels = labels + [f"{prefix}-{i+1}" for i in range(len(labels), n_features)]
    if len(labels) > n_features:
        labels = labels[:n_features]
    return labels


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(pvals)
    if not np.any(finite_mask):
        return out
    p = pvals[finite_mask]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out_tmp = np.empty_like(q)
    out_tmp[order] = q
    out[finite_mask] = out_tmp
    return out


def _make_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _normalize_species_key(species: Any) -> str:
    s = str(species or "").strip().lower()
    alias = {
        "human": "human", "homo sapiens": "human", "h. sapiens": "human",
        "hs": "human", "hsapiens": "human", "enrichr": "human",
        "mouse": "mouse", "mus musculus": "mouse", "m. musculus": "mouse",
        "mm": "mouse", "mmusculus": "mouse",
    }
    return alias.get(s, "mouse" if s.startswith("mouse") else "human")


# ============================================================================
# Loading bundle (LR / sender / receiver loadings + labels)
# ============================================================================

def _load_gene_labels(ds: DatasetPaths, n_features: int) -> list[str]:
    candidate_files = []
    for base_dir in (ds.processed_dir, ds.run_dir):
        candidate_files.extend([
            base_dir / "genenames_train.pkl",
            base_dir / "genenames_train.npy",
            base_dir / "gene_names_train.pkl",
            base_dir / "gene_names_train.npy",
            base_dir / "genenames.pkl",
            base_dir / "genenames.npy",
        ])
    labels = None
    for path in candidate_files:
        if not path.exists():
            continue
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                labels = pickle.load(f)
        elif path.suffix == ".npy":
            labels = np.load(path, allow_pickle=True)
        if labels is not None:
            break
    return _align_labels(labels, n_features=n_features, prefix="Gene")


def _load_loading_bundle(ds: DatasetPaths) -> dict[str, Any]:
    lr = np.load(ds.run_dir / "loading_LR_use.npy")
    sender = np.load(ds.run_dir / "loading_sender_use.npy")
    receiver = np.load(ds.run_dir / "loading_receiver_use.npy")

    factor_path = ds.run_dir / "Factor_envir_use.npy"
    if factor_path.exists():
        factor = np.load(factor_path)
        n_mi = int(factor.shape[1])
    else:
        n_mi = int(min(
            lr.shape[0] if lr.ndim == 2 else 0,
            sender.shape[0],
            receiver.shape[0],
        ) or ds.dim_envir)

    lr = _ensure_mi_by_feature_matrix(lr, n_mi=n_mi)
    sender = _ensure_mi_by_feature_matrix(sender, n_mi=n_mi)
    receiver = _ensure_mi_by_feature_matrix(receiver, n_mi=n_mi)

    lr_list_path = ds.processed_dir / "LR_list.pkl"
    if not lr_list_path.exists():
        lr_list_path = ds.run_dir / "LR_list.pkl"
    lr_labels_raw = None
    if lr_list_path.exists():
        with open(lr_list_path, "rb") as f:
            lr_labels_raw = pickle.load(f)

    lr_labels = [_format_lr_label(x) for x in lr_labels_raw] if lr_labels_raw is not None else None
    lr_labels = _align_labels(lr_labels, n_features=lr.shape[1], prefix="LR")
    lr_meta_source = lr_labels_raw if lr_labels_raw is not None else lr_labels
    lr_meta = [_parse_lr_entry(x) for x in lr_meta_source]

    gene_labels = _load_gene_labels(ds, n_features=max(sender.shape[1], receiver.shape[1]))
    sender_gene_labels = _align_labels(gene_labels, n_features=sender.shape[1], prefix="Gene")
    receiver_gene_labels = _align_labels(gene_labels, n_features=receiver.shape[1], prefix="Gene")

    return {
        "mi_list": [f"MI-{i+1}" for i in range(n_mi)],
        "loading_lr_norm": _column_sum_normalize(lr),
        "loading_sender_norm": _column_sum_normalize(sender),
        "loading_receiver_norm": _column_sum_normalize(receiver),
        "lr_labels": lr_labels,
        "lr_meta": lr_meta,
        "sender_gene_labels": sender_gene_labels,
        "receiver_gene_labels": receiver_gene_labels,
    }


# ============================================================================
# Trained-model loading + baseline cache
# ============================================================================

def _select_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_training_cfg(ds: DatasetPaths) -> dict:
    for path in (
        ds.model_dir / "SpiderNet_model_config.json",
        ds.run_dir / "SpiderNet_model_config.json",
        ds.root / "config.json",
    ):
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return {}


def _make_training_config(cfg_dict: dict, n_mi: int, version: str) -> Any:
    try:
        from SpiderNet.config import TrainingConfig
    except Exception:
        return cfg_dict
    dim_envir = (
        cfg_dict.get("dim_envir")
        or cfg_dict.get("DIM_ENVIR")
        or cfg_dict.get("dim")
        or n_mi
    )
    n_jobs = cfg_dict.get("n_jobs", cfg_dict.get("N_JOBS", 1))
    max_epoch = cfg_dict.get("max_epoch", cfg_dict.get("MAX_EPOCH", 1))
    cfg_version = cfg_dict.get("version", cfg_dict.get("VERSION", version))
    try:
        return TrainingConfig(
            dim_envir=int(dim_envir),
            n_jobs=int(n_jobs),
            max_epoch=int(max_epoch),
            version=str(cfg_version),
        )
    except Exception:
        return cfg_dict


def _extract_epoch_from_name(path: Path) -> Optional[int]:
    name = path.stem.lower()
    for pat in (r"epoch[_-]?(\d+)", r"ep[_-]?(\d+)", r"checkpoint[_-]?(\d+)", r"model[_-]?epoch[_-]?(\d+)"):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def _find_checkpoint(model_dir: Path) -> Optional[Path]:
    candidates = []
    priority_terms = ["best", "final", "model", "checkpoint", "epoch"]
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        for path in model_dir.rglob(pattern):
            name = path.name.lower()
            priority = sum(len(priority_terms) - i for i, t in enumerate(priority_terms) if t in name)
            ep = _extract_epoch_from_name(path)
            candidates.append((ep is not None, ep or -1, priority, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-int(x[0]), -x[1], -x[2], -x[3], len(str(x[4]))))
    return candidates[0][4]


def _extract_state_dict(ckpt_obj: Any) -> Optional[dict]:
    import torch
    if ckpt_obj is None:
        return None
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    if hasattr(ckpt_obj, "state_dict"):
        try:
            return ckpt_obj.state_dict()
        except Exception:
            return None
    return None


def _infer_predictions_for_slices(model: Any, data_list: Any, slice_indices: list[int], device: str) -> dict[int, np.ndarray]:
    import torch
    pred = {}
    model.eval()
    with torch.inference_mode():
        for idx in slice_indices:
            data_obj = data_list[idx]
            data_for_model = data_obj.to(device) if hasattr(data_obj, "to") else data_obj
            outputs = model(data_for_model)
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
                exp_recon = outputs[0]
            else:
                raise RuntimeError(
                    "Unexpected SpiderNet forward output. Expected (exp_recon, ...)."
                )
            pred[idx] = _to_numpy(exp_recon).astype(np.float32)
            if device == "cuda":
                torch.cuda.empty_cache()
    return pred


def _build_state(ds: DatasetPaths, force_baseline_rebuild: bool = False) -> dict[str, Any]:
    """Heavy state: loads trained model, raw data list, baseline predictions."""
    try:
        from SpiderNet.api import build_model
        from SpiderNet.io import load_processed_data
    except Exception as e:
        raise ImportError(
            f"SpiderNet package import failed ({e}). M4 needs SpiderNet.api.build_model "
            "and SpiderNet.io.load_processed_data."
        )

    import torch

    bundle = get_core_bundle(ds)
    raw_data_list = bundle["pyg_list"]
    adata_list = bundle["adata_list"]
    if adata_list is not None and len(adata_list) != len(raw_data_list):
        adata_list = None

    processed = load_processed_data(ds.processed_dir)
    cfg_dict = _load_training_cfg(ds)
    loading_bundle = _load_loading_bundle(ds)
    n_mi = len(loading_bundle["mi_list"])
    train_cfg = _make_training_config(cfg_dict, n_mi=n_mi, version=ds.version)
    hidden_channels = int(cfg_dict.get(
        "hidden_channels", cfg_dict.get("HIDDEN_CHANNELS", DEFAULT_HIDDEN_CHANNELS)
    ))

    device = _select_device()

    model = build_model(
        processed=processed,
        train_cfg=train_cfg,
        device=device,
        hidden_channels=hidden_channels,
    )

    ckpt = _find_checkpoint(ds.model_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"Could not locate a model checkpoint under: {ds.model_dir}."
        )
    ckpt_obj = torch.load(ckpt, map_location=device)
    state_dict = _extract_state_dict(ckpt_obj)
    if state_dict is None:
        raise RuntimeError(f"Could not extract a state_dict from checkpoint: {ckpt}")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Gene labels — prefer the processed bundle's training gene names.
    gene_labels = getattr(processed, "genenames_train", None)
    if gene_labels is None:
        n_genes = int(_to_numpy(_get_data_attr(raw_data_list[0], "x")).shape[1])
        gene_labels = _load_gene_labels(ds, n_features=n_genes)
    gene_labels = np.asarray(gene_labels, dtype=object).astype(str)

    # Per-slice metadata + cell-type pools.
    celltype_col = ds.config.get("CELL_TYPE_COL", "cell.types")
    sample_key = ds.config.get("SAMPLE_ID_COL", "samples")

    slice_infos = []
    cell_pool_by_type: dict[str, list[tuple[int, int]]] = defaultdict(list)
    onehot_template_by_type: dict[str, np.ndarray] = {}
    global_celltypes: list = []
    global_slice_ids: list = []

    for slice_idx, data_obj in enumerate(raw_data_list):
        x = _to_numpy(_get_data_attr(data_obj, "x"))
        n_cells = int(x.shape[0])

        local_celltypes = _get_data_attr(data_obj, "cell_class", None)
        if local_celltypes is None and adata_list is not None:
            local_celltypes = adata_list[slice_idx].obs[celltype_col].astype(str).values
        local_celltypes = _ensure_str_array(local_celltypes)

        onehot = _to_numpy(_get_data_attr(data_obj, "cell_class_onehot", None))
        if onehot is not None:
            for i, ct in enumerate(local_celltypes):
                if ct not in onehot_template_by_type:
                    onehot_template_by_type[ct] = np.asarray(onehot[i], dtype=float)

        sample_name = None
        if adata_list is not None:
            obs = adata_list[slice_idx].obs
            if sample_key in obs.columns:
                unique_samples = pd.Index(obs[sample_key].astype(str)).unique().tolist()
                sample_name = unique_samples[0] if unique_samples else None

        slice_infos.append({
            "slice_index": slice_idx,
            "n_cells": n_cells,
            "sample_name": sample_name,
            "celltypes_present": sorted(np.unique(local_celltypes).tolist()),
        })
        global_celltypes.extend(local_celltypes.tolist())
        global_slice_ids.extend([slice_idx] * n_cells)
        for local_idx, ct in enumerate(local_celltypes):
            cell_pool_by_type[str(ct)].append((slice_idx, local_idx))

    global_celltypes = np.asarray(global_celltypes, dtype=object)
    global_slice_ids = np.asarray(global_slice_ids, dtype=int)
    gene_to_idx = {str(g): i for i, g in enumerate(gene_labels)}

    # Baseline predictions cache.
    baseline_npz = _module_dir(ds) / "baseline_predictions.npz"
    baseline_meta = _module_dir(ds) / "baseline_predictions_meta.json"
    if baseline_npz.exists() and baseline_meta.exists() and not force_baseline_rebuild:
        loaded = np.load(baseline_npz, allow_pickle=True)
        baseline_pred_by_slice = {int(k.replace("slice_", "")): loaded[k] for k in loaded.files}
    else:
        baseline_pred_by_slice = _infer_predictions_for_slices(
            model=model,
            data_list=raw_data_list,
            slice_indices=list(range(len(raw_data_list))),
            device=device,
        )
        np.savez_compressed(
            baseline_npz,
            **{f"slice_{k}": v for k, v in baseline_pred_by_slice.items()},
        )
        with open(baseline_meta, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_name": ds.name,
                "run_dir": str(ds.run_dir),
                "checkpoint": str(ckpt),
                "created_at": _make_timestamp(),
                "n_slices": len(raw_data_list),
                "n_genes": int(len(gene_labels)),
            }, f, indent=2)

    species = ds.setup.get("SPECIES") or ds.config.get("SPECIES") or (
        "mouse" if ds.name.lower().startswith("agingbrain") else "human"
    )

    return {
        "loading_bundle": loading_bundle,
        "model": model,
        "device": device,
        "checkpoint": ckpt,
        "raw_data_list": raw_data_list,
        "adata_list": adata_list,
        "celltype_col": celltype_col,
        "gene_labels": gene_labels,
        "gene_to_idx": gene_to_idx,
        "slice_infos": slice_infos,
        "global_celltypes": global_celltypes,
        "global_slice_ids": global_slice_ids,
        "cell_pool_by_type": dict(cell_pool_by_type),
        "onehot_template_by_type": onehot_template_by_type,
        "baseline_pred_by_slice": baseline_pred_by_slice,
        "available_celltypes": sorted(np.unique(global_celltypes).tolist()),
        "species": species,
    }


def get_state(ds: DatasetPaths, force_baseline_rebuild: bool = False) -> dict[str, Any]:
    """Return the cached M4 state for `ds`, building it on first call."""
    with _STATE_LOCK:
        cached = _STATE_CACHE.get(ds.name)
        if cached is not None and not force_baseline_rebuild:
            return cached
        state = _build_state(ds, force_baseline_rebuild=force_baseline_rebuild)
        _STATE_CACHE[ds.name] = state
        return state


def is_state_loaded(ds: DatasetPaths) -> bool:
    return ds.name in _STATE_CACHE


def get_loading_bundle(ds: DatasetPaths) -> dict[str, Any]:
    """Light-weight access to LR/sender/receiver loadings — does NOT load the model."""
    with _STATE_LOCK:
        cached = _STATE_CACHE.get(ds.name)
        if cached is not None:
            return cached["loading_bundle"]
    return _load_loading_bundle(ds)


# ============================================================================
# Feature tables (top rows per MI)
# ============================================================================

def top_rows_for_mi(loading_bundle: dict, mi_name: str, kind: str = "lr", top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    mi_list = loading_bundle["mi_list"]
    if mi_name not in mi_list:
        raise ValueError(f"{mi_name} not in MI list (size={len(mi_list)}).")
    mi_idx = mi_list.index(mi_name)
    top_n = max(1, min(int(top_n), 20))

    if kind == "lr":
        values = np.asarray(loading_bundle["loading_lr_norm"][mi_idx], dtype=float)
        labels = loading_bundle["lr_labels"]
        meta = loading_bundle["lr_meta"]
        order = np.argsort(values)[::-1][:top_n]
        rows = []
        for rank, i in enumerate(order, start=1):
            entry = meta[i] if i < len(meta) else _parse_lr_entry(labels[i])
            rows.append({
                "rank": int(rank),
                "label": str(entry["label"]),
                "ligand": str(entry["ligand"]),
                "receptor": str(entry["receptor"]),
                "normalized_loading": float(values[i]),
            })
        return pd.DataFrame(rows)

    if kind == "sender":
        values = np.asarray(loading_bundle["loading_sender_norm"][mi_idx], dtype=float)
        labels = loading_bundle["sender_gene_labels"]
    elif kind == "receiver":
        values = np.asarray(loading_bundle["loading_receiver_norm"][mi_idx], dtype=float)
        labels = loading_bundle["receiver_gene_labels"]
    else:
        raise ValueError("kind must be 'lr', 'sender', or 'receiver'.")

    order = np.argsort(values)[::-1][:top_n]
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1, dtype=int),
        "gene": [str(labels[i]) for i in order],
        "normalized_loading": values[order].astype(float),
    })


# ============================================================================
# Knockout / replacement perturbations
# ============================================================================

def _local_celltypes_for_slice(state: dict, slice_idx: int) -> np.ndarray:
    data_obj = state["raw_data_list"][slice_idx]
    local = _get_data_attr(data_obj, "cell_class", None)
    if local is None and state.get("adata_list") is not None:
        local = state["adata_list"][slice_idx].obs[state["celltype_col"]].astype(str).values
    return _ensure_str_array(local)


def _edge_mat_for_slice(data_obj: Any) -> np.ndarray:
    return _standardize_edge_index(_get_data_attr(data_obj, "edge_index"))


def _match_sender_receiver_cells(local_celltypes: np.ndarray, edge_mat: np.ndarray,
                                  sender_types: Optional[list], receiver_types: Optional[list]) -> tuple[np.ndarray, np.ndarray]:
    sender_types = set(sender_types or [])
    receiver_types = set(receiver_types or [])
    sender_mask_all = np.ones(len(local_celltypes), dtype=bool) if not sender_types else np.isin(local_celltypes, list(sender_types))
    receiver_mask_all = np.ones(len(local_celltypes), dtype=bool) if not receiver_types else np.isin(local_celltypes, list(receiver_types))

    src = edge_mat[:, 0]
    dst = edge_mat[:, 1]
    edge_mask = sender_mask_all[src] & receiver_mask_all[dst]
    if np.any(edge_mask):
        sender_cells = np.unique(src[edge_mask])
        receiver_cells = np.unique(dst[edge_mask])
    else:
        sender_cells = np.where(sender_mask_all)[0]
        receiver_cells = np.where(receiver_mask_all)[0]
    return sender_cells.astype(int), receiver_cells.astype(int)


def _filter_cells_by_type(local_celltypes: np.ndarray, cell_indices: np.ndarray,
                           allowed_types: Optional[list]) -> np.ndarray:
    cell_indices = np.asarray(cell_indices, dtype=int)
    if cell_indices.size == 0:
        return cell_indices
    allowed = [str(x) for x in (allowed_types or []) if str(x).strip()]
    if not allowed:
        return np.unique(cell_indices).astype(int)
    mask = np.isin(local_celltypes[cell_indices], allowed)
    return np.unique(cell_indices[mask]).astype(int)


def _neighbors_of(local_celltypes: np.ndarray, edge_mat: np.ndarray,
                   selected_cells: np.ndarray, allowed_types: Optional[list],
                   exclude_selected: bool = True) -> np.ndarray:
    selected_cells = np.asarray(selected_cells, dtype=int)
    if selected_cells.size == 0:
        return np.asarray([], dtype=int)
    src, dst = edge_mat[:, 0], edge_mat[:, 1]
    out_mask = np.isin(src, selected_cells)
    in_mask = np.isin(dst, selected_cells)
    neighbors = np.unique(np.concatenate([dst[out_mask], src[in_mask]])).astype(int)
    if exclude_selected and neighbors.size > 0:
        neighbors = neighbors[~np.isin(neighbors, selected_cells)]
    return _filter_cells_by_type(local_celltypes, neighbors, allowed_types).astype(int)


def _apply_knockout(data_obj: Any, local_celltypes: np.ndarray,
                     sender_gene_idx: list[int], receiver_gene_idx: list[int],
                     sender_types: Optional[list], receiver_types: Optional[list],
                     keep_pct: float) -> dict[str, Any]:
    edge_mat = _edge_mat_for_slice(data_obj)
    sender_cells, receiver_cells = _match_sender_receiver_cells(
        local_celltypes, edge_mat, sender_types, receiver_types
    )
    x = _get_data_attr(data_obj, "x")
    scale = float(keep_pct) / 100.0

    if len(sender_gene_idx) > 0 and len(sender_cells) > 0:
        gi = np.asarray(sender_gene_idx, dtype=int)
        x[sender_cells[:, None], gi] = x[sender_cells[:, None], gi] * scale
    if len(receiver_gene_idx) > 0 and len(receiver_cells) > 0:
        gi = np.asarray(receiver_gene_idx, dtype=int)
        x[receiver_cells[:, None], gi] = x[receiver_cells[:, None], gi] * scale
    _set_data_attr(data_obj, "x", x)

    modified = np.unique(np.concatenate([sender_cells, receiver_cells])).astype(int)
    return {
        "sender_cells_modified": int(len(sender_cells)),
        "receiver_cells_modified": int(len(receiver_cells)),
        "modified_cells_idx": modified,
    }


def _sample_donor_cells(state: dict, replacement_celltypes: list[str],
                          n_needed: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    pool = []
    for ct in replacement_celltypes:
        pool.extend(state["cell_pool_by_type"].get(str(ct), []))
    if not pool:
        raise ValueError(f"No cells available for replacement type(s): {replacement_celltypes}")
    replace = len(pool) < n_needed
    chosen_idx = rng.choice(len(pool), size=int(n_needed), replace=replace)
    return [pool[int(i)] for i in chosen_idx]


def _apply_replacement(data_obj: Any, state: dict, local_celltypes: np.ndarray,
                        replacement_celltypes: list[str], replaced_celltypes: list[str],
                        rng: np.random.Generator) -> dict[str, Any]:
    import torch
    replace_mask = np.isin(local_celltypes, list(replaced_celltypes))
    target_cells = np.where(replace_mask)[0].astype(int)
    if target_cells.size == 0:
        return {"cells_replaced": 0, "replaced_cells_idx": np.asarray([], dtype=int),
                "neighbor_cells_idx": np.asarray([], dtype=int)}

    edge_mat = _edge_mat_for_slice(data_obj)
    neighbor_cells = _neighbors_of(local_celltypes, edge_mat, target_cells,
                                    allowed_types=None, exclude_selected=True)
    sampled = _sample_donor_cells(state, replacement_celltypes, len(target_cells), rng)

    x = _get_data_attr(data_obj, "x")
    cell_class_onehot = _get_data_attr(data_obj, "cell_class_onehot", None)
    cell_class = _get_data_attr(data_obj, "cell_class", None)

    donor_x_rows: list = []
    donor_onehot_rows: list = []
    for slice_src, local_src in sampled:
        donor_slice = state["raw_data_list"][int(slice_src)]
        donor_x_rows.append(_get_data_attr(donor_slice, "x")[int(local_src)])
        donor_onehot = _get_data_attr(donor_slice, "cell_class_onehot", None)
        if donor_onehot is not None:
            donor_onehot_rows.append(_to_numpy(donor_onehot[int(local_src)]))

    if torch.is_tensor(x):
        donor_x_stack = torch.stack([
            v if torch.is_tensor(v) else torch.tensor(_to_numpy(v), dtype=x.dtype, device=x.device)
            for v in donor_x_rows
        ]).to(device=x.device, dtype=x.dtype)
    else:
        donor_x_stack = np.vstack([_to_numpy(v) for v in donor_x_rows]).astype(_to_numpy(x).dtype, copy=False)
    x[target_cells] = donor_x_stack
    _set_data_attr(data_obj, "x", x)

    if cell_class_onehot is not None and len(donor_onehot_rows) == len(target_cells):
        if torch.is_tensor(cell_class_onehot):
            donor_onehot_stack = torch.tensor(
                np.vstack(donor_onehot_rows),
                dtype=cell_class_onehot.dtype, device=cell_class_onehot.device,
            )
        else:
            donor_onehot_stack = np.vstack(donor_onehot_rows).astype(_to_numpy(cell_class_onehot).dtype, copy=False)
        cell_class_onehot[target_cells] = donor_onehot_stack
        _set_data_attr(data_obj, "cell_class_onehot", cell_class_onehot)

    if cell_class is not None:
        arr = np.asarray(cell_class, dtype=object).copy()
        label = (str(replacement_celltypes[0]) if len(replacement_celltypes) == 1
                 else "mixed_replacement_pool")
        arr[target_cells] = label
        if isinstance(cell_class, list):
            arr = arr.tolist()
        _set_data_attr(data_obj, "cell_class", arr)

    return {
        "cells_replaced": int(len(target_cells)),
        "replaced_cells_idx": target_cells,
        "neighbor_cells_idx": neighbor_cells,
    }


def _affected_slices_for_knockout(state: dict, sender_types: list, receiver_types: list) -> list[int]:
    sender_types = set(sender_types or [])
    receiver_types = set(receiver_types or [])
    out = []
    for info in state["slice_infos"]:
        present = set(info["celltypes_present"])
        sender_ok = True if not sender_types else bool(present & sender_types)
        receiver_ok = True if not receiver_types else bool(present & receiver_types)
        if sender_ok or receiver_ok:
            out.append(int(info["slice_index"]))
    return out


def _affected_slices_for_replacement(state: dict, replaced_celltypes: list) -> list[int]:
    replaced = set(replaced_celltypes or [])
    out = []
    for info in state["slice_infos"]:
        if set(info["celltypes_present"]) & replaced:
            out.append(int(info["slice_index"]))
    return out


def _collect_knockout_genes(lr_rows: list, lr_selected: list,
                              sender_rows: list, sender_selected: list,
                              receiver_rows: list, receiver_selected: list,
                              gene_to_idx: dict) -> dict[str, Any]:
    def _pick(rows, selected):
        if rows is None:
            return []
        rows = list(rows)
        if not selected:
            return rows
        return [rows[int(i)] for i in selected if 0 <= int(i) < len(rows)]

    sel_lr = _pick(lr_rows, lr_selected)
    sel_sender = _pick(sender_rows, sender_selected)
    sel_receiver = _pick(receiver_rows, receiver_selected)

    sender_genes: list[str] = []
    receiver_genes: list[str] = []
    for r in sel_lr:
        sender_genes.extend(_split_gene_token(r.get("ligand", "")))
        receiver_genes.extend(_split_gene_token(r.get("receptor", "")))
    for r in sel_sender:
        sender_genes.append(str(r.get("gene", "")))
    for r in sel_receiver:
        receiver_genes.append(str(r.get("gene", "")))

    sender_genes = [g for g in pd.unique(pd.Series(sender_genes, dtype=object)).tolist() if g in gene_to_idx]
    receiver_genes = [g for g in pd.unique(pd.Series(receiver_genes, dtype=object)).tolist() if g in gene_to_idx]

    return {
        "sender_gene_names": sender_genes,
        "receiver_gene_names": receiver_genes,
        "sender_gene_idx": [gene_to_idx[g] for g in sender_genes],
        "receiver_gene_idx": [gene_to_idx[g] for g in receiver_genes],
    }


def _compare_before_after(state: dict, before_pred: dict, after_pred: dict,
                           target_celltypes: list, max_cells_for_de: Optional[int],
                           selected_sender_genes: set, selected_receiver_genes: set,
                           selected_cells_by_slice: dict[int, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    target_celltypes = list(target_celltypes or [])
    before_blocks: list = []
    after_blocks: list = []

    for info in state["slice_infos"]:
        slice_idx = int(info["slice_index"])
        local_celltypes = _local_celltypes_for_slice(state, slice_idx)
        if selected_cells_by_slice:
            local_idx = np.asarray(selected_cells_by_slice.get(slice_idx, []), dtype=int)
            if local_idx.size == 0:
                continue
            if target_celltypes:
                local_idx = _filter_cells_by_type(local_celltypes, local_idx, target_celltypes)
            if local_idx.size == 0:
                continue
            before_blocks.append(before_pred[slice_idx][local_idx])
            after_blocks.append(after_pred[slice_idx][local_idx])
        else:
            mask = np.isin(local_celltypes, target_celltypes)
            if np.any(mask):
                before_blocks.append(before_pred[slice_idx][mask])
                after_blocks.append(after_pred[slice_idx][mask])

    if not before_blocks:
        raise ValueError("No cells satisfied the perturbation rule and target cell-type filter.")

    before = np.vstack(before_blocks)
    after = np.vstack(after_blocks)

    n_total = int(before.shape[0])
    if max_cells_for_de is not None and n_total > int(max_cells_for_de):
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
        keep_idx = np.sort(rng.choice(n_total, size=int(max_cells_for_de), replace=False))
        before = before[keep_idx]
        after = after[keep_idx]
        n_used = len(keep_idx)
        subsampled = True
    else:
        n_used = n_total
        subsampled = False

    eps = 1e-6
    mean_before = np.mean(before, axis=0)
    mean_after = np.mean(after, axis=0)
    log2fc = np.log2(mean_after + eps) - np.log2(mean_before + eps)

    ttest = stats.ttest_rel(after, before, axis=0, nan_policy="omit")
    pvals = np.asarray(ttest.pvalue, dtype=float)
    padj = _bh_adjust(pvals)

    gene_labels = np.asarray(state["gene_labels"]).astype(str)
    roles: list[str] = []
    for g in gene_labels:
        if g in selected_sender_genes and g in selected_receiver_genes:
            roles.append("sender+receiver selected")
        elif g in selected_sender_genes:
            roles.append("sender selected")
        elif g in selected_receiver_genes:
            roles.append("receiver selected")
        else:
            roles.append("not directly perturbed")

    df = pd.DataFrame({
        "gene": gene_labels,
        "mean_before": mean_before.astype(float),
        "mean_after": mean_after.astype(float),
        "log2FC": log2fc.astype(float),
        "pvalue": pvals.astype(float),
        "padj": padj.astype(float),
        "perturbation_role": roles,
    })
    df["neglog10_padj"] = -np.log10(np.clip(df["padj"].values.astype(float), 1e-300, 1.0))
    df = df.sort_values(["padj", "pvalue", "neglog10_padj"],
                        ascending=[True, True, False], na_position="last").reset_index(drop=True)

    meta = {
        "n_target_cells_total": n_total,
        "n_target_cells_used": n_used,
        "subsampled_target_cells": subsampled,
        "n_significant_padj_005": int(np.sum(df["padj"].fillna(1.0) < 0.05)),
        "n_abs_log2fc_ge_025": int(np.sum(df["log2FC"].abs().fillna(0.0) >= 0.25)),
    }
    return df, meta


def run_knockout_analysis(ds: DatasetPaths, state: dict, mi_name: str,
                           sender_gene_idx: list[int], receiver_gene_idx: list[int],
                           sender_gene_names: list[str], receiver_gene_names: list[str],
                           sender_types: list, receiver_types: list,
                           keep_pct: float, target_celltypes: list,
                           max_cells_for_de: Optional[int]) -> tuple[pd.DataFrame, dict]:
    sender_types = list(sender_types or [])
    receiver_types = list(receiver_types or [])
    target_celltypes = list(target_celltypes or [])

    if not sender_gene_idx and not receiver_gene_idx:
        raise ValueError("No genes were selected for knockout.")
    if not target_celltypes:
        raise ValueError("Please select at least one target cell type for downstream comparison.")

    affected = _affected_slices_for_knockout(state, sender_types, receiver_types)
    if not affected:
        raise ValueError("No slices contain the requested sender / receiver restriction.")

    clones = {idx: copy.deepcopy(state["raw_data_list"][idx]) for idx in affected}
    metrics = {"sender_cells_modified": 0, "receiver_cells_modified": 0}
    selected_cells_by_slice: dict[int, np.ndarray] = {}

    for slice_idx in affected:
        local_celltypes = _local_celltypes_for_slice(state, slice_idx)
        cur = _apply_knockout(
            clones[slice_idx], local_celltypes,
            sender_gene_idx, receiver_gene_idx,
            sender_types, receiver_types, keep_pct,
        )
        metrics["sender_cells_modified"] += cur["sender_cells_modified"]
        metrics["receiver_cells_modified"] += cur["receiver_cells_modified"]
        sel = _filter_cells_by_type(local_celltypes, cur["modified_cells_idx"], target_celltypes)
        if sel.size:
            selected_cells_by_slice[int(slice_idx)] = sel.astype(int)

    if not selected_cells_by_slice:
        raise ValueError("No perturbed cells remained after filtering by the target cell type(s).")

    perturbed_pred = dict(state["baseline_pred_by_slice"])
    rerun = _infer_predictions_for_slices(state["model"], clones, affected, state["device"])
    perturbed_pred.update(rerun)

    de_df, de_meta = _compare_before_after(
        state=state,
        before_pred=state["baseline_pred_by_slice"],
        after_pred=perturbed_pred,
        target_celltypes=target_celltypes,
        max_cells_for_de=max_cells_for_de,
        selected_sender_genes=set(sender_gene_names),
        selected_receiver_genes=set(receiver_gene_names),
        selected_cells_by_slice=selected_cells_by_slice,
    )
    summary = {
        "mode": "knockout",
        "mi_name": mi_name,
        "affected_slices": affected,
        "keep_pct": float(keep_pct),
        "sender_types": sender_types,
        "receiver_types": receiver_types,
        "target_celltypes": target_celltypes,
        "n_sender_genes": len(sender_gene_names),
        "n_receiver_genes": len(receiver_gene_names),
        "sender_gene_names": sender_gene_names,
        "receiver_gene_names": receiver_gene_names,
        **metrics,
        **de_meta,
    }
    return de_df, summary


def run_replacement_analysis(ds: DatasetPaths, state: dict,
                               replacement_celltypes: list, replaced_celltypes: list,
                               target_celltypes: list, max_cells_for_de: Optional[int],
                               random_seed: int = 0) -> tuple[pd.DataFrame, dict]:
    replacement_celltypes = [str(x) for x in (replacement_celltypes or []) if str(x).strip()]
    replaced_celltypes = [str(x) for x in (replaced_celltypes or []) if str(x).strip()]
    target_celltypes = list(target_celltypes or [])

    if not replacement_celltypes:
        raise ValueError("Please choose at least one replacement cell type.")
    if not replaced_celltypes:
        raise ValueError("Please choose at least one cell type to be replaced.")
    overlap = sorted(set(replacement_celltypes) & set(replaced_celltypes))
    if overlap:
        raise ValueError(f"Replacement and replaced cell types overlap: {overlap}")
    if not target_celltypes:
        raise ValueError("Please select at least one neighboring cell type of interest.")

    affected = _affected_slices_for_replacement(state, replaced_celltypes)
    if not affected:
        raise ValueError("No slices contain the requested replaced cell type(s).")

    rng = np.random.default_rng(int(random_seed))
    clones = {idx: copy.deepcopy(state["raw_data_list"][idx]) for idx in affected}
    cells_replaced_total = 0
    selected_cells_by_slice: dict[int, np.ndarray] = {}

    for slice_idx in affected:
        local_celltypes = _local_celltypes_for_slice(state, slice_idx)
        cur = _apply_replacement(
            clones[slice_idx], state, local_celltypes,
            replacement_celltypes, replaced_celltypes, rng,
        )
        cells_replaced_total += cur["cells_replaced"]
        sel = _filter_cells_by_type(local_celltypes, cur["neighbor_cells_idx"], target_celltypes)
        if sel.size:
            selected_cells_by_slice[int(slice_idx)] = sel.astype(int)

    if not selected_cells_by_slice:
        raise ValueError("No neighbors of replaced cells matched the target cell type(s).")

    perturbed_pred = dict(state["baseline_pred_by_slice"])
    rerun = _infer_predictions_for_slices(state["model"], clones, affected, state["device"])
    perturbed_pred.update(rerun)

    de_df, de_meta = _compare_before_after(
        state=state,
        before_pred=state["baseline_pred_by_slice"],
        after_pred=perturbed_pred,
        target_celltypes=target_celltypes,
        max_cells_for_de=max_cells_for_de,
        selected_sender_genes=set(),
        selected_receiver_genes=set(),
        selected_cells_by_slice=selected_cells_by_slice,
    )
    summary = {
        "mode": "cell_replacement",
        "replacement_celltypes": replacement_celltypes,
        "replaced_celltypes": replaced_celltypes,
        "target_celltypes": target_celltypes,
        "affected_slices": affected,
        "cells_replaced": int(cells_replaced_total),
        **de_meta,
    }
    return de_df, summary


# ============================================================================
# Enrichment (Enrichr via gseapy)
# ============================================================================

def _empty_enrichment_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Term", "Adjusted P-value", "P-value", "Genes", "Gene count", "Combined Score"])


def _format_enrichr_table(df: pd.DataFrame, top_terms: int = 12) -> pd.DataFrame:
    if df is None or getattr(df, "shape", (0, 0))[0] == 0:
        return _empty_enrichment_df()
    df = df.copy()
    if "Adjusted P-value" in df.columns:
        df["Adjusted P-value"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
        df = df.sort_values("Adjusted P-value", ascending=True)
    elif "P-value" in df.columns:
        df["P-value"] = pd.to_numeric(df["P-value"], errors="coerce")
        df = df.sort_values("P-value", ascending=True)
    if "Gene count" not in df.columns:
        if "Overlap" in df.columns:
            df["Gene count"] = df["Overlap"].astype(str).str.extract(r"^(\d+)").fillna(0).astype(int)
        elif "Genes" in df.columns:
            df["Gene count"] = df["Genes"].astype(str).apply(
                lambda x: len([g for g in re.split(r"[;,/]", x) if str(g).strip()])
            )
        else:
            df["Gene count"] = 0
    keep = [c for c in ("Term", "Adjusted P-value", "P-value", "Genes", "Gene count", "Combined Score") if c in df.columns]
    return df.loc[:, keep].head(int(top_terms)).reset_index(drop=True)


def _enrichr_libraries(species: str) -> dict:
    species_key = _normalize_species_key(species)
    if species_key == "mouse":
        organism_aliases = ["Mouse", "mouse", "mm", "mus musculus", "m. musculus"]
        kegg_candidates = ["KEGG_2021_Mouse", "KEGG_2019_Mouse"]
    else:
        organism_aliases = ["Human", "human", "hs", "homo sapiens", "h. sapiens", "enrichr"]
        kegg_candidates = ["KEGG_2021_Human", "KEGG_2019_Human"]
    return {
        "species_key": species_key,
        "organism_aliases": organism_aliases,
        "kegg_candidates": kegg_candidates,
        "go_bp_candidates": ["GO_Biological_Process_2023", "GO_Biological_Process_2021", "GO_Biological_Process_2018"],
    }


def _run_enrichr_safe(gene_list: list[str], gene_set_kind: str, species: str, top_terms: int = 12) -> pd.DataFrame:
    try:
        import gseapy as gp
    except Exception:
        return _empty_enrichment_df()

    genes = [str(g).strip() for g in gene_list if str(g).strip()]
    seen: set = set()
    deduped = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            deduped.append(g)
    if len(deduped) < 3:
        return _empty_enrichment_df()

    libs = _enrichr_libraries(species)
    library_candidates = libs["kegg_candidates"] if gene_set_kind == "kegg" else libs["go_bp_candidates"]

    for organism_name in libs["organism_aliases"]:
        gene_input = [g.upper() for g in deduped] if str(organism_name).lower().startswith("mouse") else list(deduped)
        try:
            valid_libs = set(gp.get_library_name(organism=organism_name))
        except Exception:
            valid_libs = None
        usable = [lib for lib in library_candidates if (valid_libs is None) or (lib in valid_libs)] or list(library_candidates)
        for lib in usable:
            try:
                enr = gp.enrichr(
                    gene_list=gene_input, gene_sets=lib,
                    organism=organism_name, outdir=None, cutoff=1.0,
                )
                if enr is not None and getattr(enr, "results", None) is not None:
                    out = _format_enrichr_table(enr.results.copy(), top_terms=top_terms)
                    if out.shape[0] > 0:
                        return out
            except Exception:
                continue
    return _empty_enrichment_df()


def cached_enrichment(gene_list: list[str], species: str, top_terms: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    seen: set = set()
    deduped = []
    for g in gene_list:
        s = str(g).strip()
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)
    if len(deduped) < 3:
        return _empty_enrichment_df(), _empty_enrichment_df()

    libs = _enrichr_libraries(species)
    key_go = ("go", libs["species_key"], tuple(deduped), int(top_terms))
    key_kegg = ("kegg", libs["species_key"], tuple(deduped), int(top_terms))
    if key_go not in _ENRICH_CACHE:
        _ENRICH_CACHE[key_go] = _run_enrichr_safe(deduped, "go", species, top_terms)
    if key_kegg not in _ENRICH_CACHE:
        _ENRICH_CACHE[key_kegg] = _run_enrichr_safe(deduped, "kegg", species, top_terms)
    return _ENRICH_CACHE[key_go].copy(), _ENRICH_CACHE[key_kegg].copy()


# ============================================================================
# DE-table filtering + enrichment gene picker
# ============================================================================

def filter_de_display_table(df: pd.DataFrame, lfc_threshold: float, padj_threshold: float) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["gene", "mean_before", "mean_after", "log2FC", "pvalue", "padj",
                                       "perturbation_role", "neglog10_padj"])
    out = pd.DataFrame(df).copy()
    for col in ("log2FC", "padj", "pvalue", "mean_before", "mean_after"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["neglog10_padj"] = pd.to_numeric(out.get("neglog10_padj", np.nan), errors="coerce")
    mask = out["padj"].lt(float(padj_threshold)) & out["log2FC"].abs().ge(float(lfc_threshold))
    return out.loc[mask].sort_values(
        ["padj", "pvalue", "neglog10_padj"], ascending=[True, True, False], na_position="last"
    ).reset_index(drop=True)


def genes_for_enrichment(df: pd.DataFrame, direction: str, scope: str,
                           lfc_threshold: float, padj_threshold: float) -> list[str]:
    out = filter_de_display_table(df, lfc_threshold, padj_threshold)
    if out.shape[0] == 0:
        return []
    direction = str(direction or "up").lower()
    if direction == "up":
        out = out.loc[out["log2FC"] > 0]
    elif direction == "down":
        out = out.loc[out["log2FC"] < 0]
    if str(scope or "indirect_only") == "indirect_only":
        out = out.loc[out["perturbation_role"].astype(str).eq("not directly perturbed")]
    seen: set = set()
    deduped = []
    for g in out["gene"].astype(str).tolist():
        if g and g not in seen:
            seen.add(g)
            deduped.append(g)
    return deduped


# ============================================================================
# Plotly figures (server-rendered)
# ============================================================================

def _theme_spec(theme_mode: str = "dark") -> dict:
    if str(theme_mode).lower() == "light":
        return {
            "paper": "#ffffff", "plot": "#ffffff", "text": "#111111",
            "subtext": "#4b5563", "grid": "rgba(17,17,17,0.12)",
            "border": "rgba(17,17,17,0.22)", "muted": "#6b7280",
            "accent": "#b45309", "accent2": "#2563eb",
            "volcano_other": "rgba(100,100,100,0.55)",
            "volcano_hit": "rgba(217,119,6,0.90)",
            "volcano_selected": "rgba(37,99,235,0.95)",
        }
    return {
        "paper": "#000000", "plot": "#000000", "text": "#ffffff",
        "subtext": "#d1d5db", "grid": "rgba(255,255,255,0.14)",
        "border": "rgba(255,255,255,0.72)", "muted": "#9ca3af",
        "accent": "#f59e0b", "accent2": "#60a5fa",
        "volcano_other": "rgba(180,180,180,0.45)",
        "volcano_hit": "rgba(245,158,11,0.90)",
        "volcano_selected": "rgba(96,165,250,0.95)",
    }


def _to_plotly_json(fig: go.Figure) -> dict:
    out = json.loads(fig.to_json(validate=False))
    out["layout"]["template"] = None
    return out


def _apply_plot_theme(fig: go.Figure, theme_mode: str = "dark") -> go.Figure:
    spec = _theme_spec(theme_mode)
    fig.update_layout(
        paper_bgcolor=spec["paper"],
        plot_bgcolor=spec["plot"],
        font=dict(color=spec["text"]),
        xaxis=dict(showgrid=True, gridcolor=spec["grid"], zeroline=False,
                   linecolor=spec["border"], mirror=True),
        yaxis=dict(showgrid=True, gridcolor=spec["grid"], zeroline=False,
                   linecolor=spec["border"], mirror=True),
        margin=dict(l=50, r=20, t=42, b=48),
    )
    return fig


def _message_figure(message: str, theme_mode: str = "dark", height: int = 240) -> dict:
    spec = _theme_spec(theme_mode)
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                        text=str(message), showarrow=False,
                        font=dict(size=16, color=spec["text"]), align="center")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=int(height),
        paper_bgcolor=spec["paper"], plot_bgcolor=spec["plot"],
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return _to_plotly_json(fig)


def stem_figure(df: pd.DataFrame, title_text: str, x_col: str, theme_mode: str = "dark", height: int = 300) -> dict:
    spec = _theme_spec(theme_mode)
    if df is None or len(df) == 0:
        return _message_figure("No rows available.", theme_mode=theme_mode, height=height)
    labels = df[x_col].astype(str).tolist()
    values = df["normalized_loading"].astype(float).tolist()
    fig = go.Figure()
    for lab, val in zip(labels, values):
        fig.add_trace(go.Scatter(
            x=[lab, lab], y=[0, val], mode="lines",
            line=dict(color=spec["accent2"], width=2),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="markers",
        marker=dict(size=9, color=spec["accent"]),
        showlegend=False,
        hovertemplate=f"{x_col}: %{{x}}<br>loading: %{{y:.4f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        height=int(height), xaxis_tickangle=-40,
    )
    _apply_plot_theme(fig, theme_mode=theme_mode)
    fig.update_yaxes(title="Normalized loading")
    fig.update_xaxes(title=None)
    return _to_plotly_json(fig)


def volcano_figure(df: pd.DataFrame, theme_mode: str = "dark",
                    lfc_threshold: float = 0.25, padj_threshold: float = 0.05,
                    annotate_top_n: int = 12) -> dict:
    spec = _theme_spec(theme_mode)
    if df is None or len(df) == 0:
        return _message_figure("No differential expression results yet.", theme_mode=theme_mode, height=520)

    p = df.copy()
    for col in ("log2FC", "padj", "neglog10_padj"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p["is_hit"] = p["padj"].lt(float(padj_threshold)) & p["log2FC"].abs().ge(float(lfc_threshold))
    p["is_selected_gene"] = p["perturbation_role"].astype(str).ne("not directly perturbed")

    fig = go.Figure()

    def _add(mask, name, color):
        sub = p.loc[mask]
        if len(sub) == 0:
            return
        fig.add_trace(go.Scattergl(
            x=sub["log2FC"].astype(float), y=sub["neglog10_padj"].astype(float),
            mode="markers", name=name, marker=dict(size=7, color=color),
            text=sub["gene"].astype(str),
            customdata=np.stack([sub["padj"].astype(float),
                                  sub["perturbation_role"].astype(str)], axis=1),
            hovertemplate=("Gene: %{text}<br>log2FC: %{x:.4f}<br>"
                           "-log10(adj P): %{y:.4f}<br>"
                           "adj P: %{customdata[0]:.3e}<br>"
                           "Role: %{customdata[1]}<extra></extra>"),
        ))

    other_mask = ~p["is_hit"] & ~p["is_selected_gene"]
    hit_mask = p["is_hit"] & ~p["is_selected_gene"]
    selected_mask = p["is_selected_gene"]
    _add(other_mask, "Other genes", spec["volcano_other"])
    _add(hit_mask, "Significant hits", spec["volcano_hit"])
    _add(selected_mask, "Directly perturbed genes", spec["volcano_selected"])

    annot = p.sort_values(
        ["is_selected_gene", "padj", "neglog10_padj"],
        ascending=[False, True, False],
    ).head(int(annotate_top_n))
    for _, row in annot.iterrows():
        fig.add_annotation(
            x=float(row["log2FC"]), y=float(row["neglog10_padj"]),
            text=str(row["gene"]), showarrow=False, yshift=10,
            font=dict(size=11, color=spec["text"]),
        )

    fig.add_vline(x=float(lfc_threshold), line_dash="dash", line_color=spec["muted"])
    fig.add_vline(x=-float(lfc_threshold), line_dash="dash", line_color=spec["muted"])
    fig.add_hline(y=-np.log10(max(float(padj_threshold), 1e-300)),
                   line_dash="dash", line_color=spec["muted"])

    fig.update_layout(
        title=dict(text="Predicted expression change after perturbation", x=0.5, xanchor="center"),
        height=560, legend=dict(orientation="h", x=0, y=1.08),
    )
    _apply_plot_theme(fig, theme_mode=theme_mode)
    fig.update_xaxes(title="log2 fold change")
    fig.update_yaxes(title="-log10 adjusted P")
    return _to_plotly_json(fig)


def bubble_figure(enrich_df: pd.DataFrame, theme_mode: str = "dark",
                   title: Optional[str] = None, height: int = 320) -> dict:
    if enrich_df is None or getattr(enrich_df, "shape", (0, 0))[0] == 0:
        return _message_figure("No enriched terms.", theme_mode=theme_mode, height=height)
    df = pd.DataFrame(enrich_df).copy()
    if df.shape[0] == 0 or "Term" not in df.columns:
        return _message_figure("No enriched terms.", theme_mode=theme_mode, height=height)
    spec = _theme_spec(theme_mode)
    if "Adjusted P-value" in df.columns:
        score = -np.log10(np.clip(pd.to_numeric(df["Adjusted P-value"], errors="coerce")
                                  .fillna(1.0).to_numpy(dtype=float), 1e-300, 1.0))
        x_label = "-log10 adjusted P"
    elif "P-value" in df.columns:
        score = -np.log10(np.clip(pd.to_numeric(df["P-value"], errors="coerce")
                                  .fillna(1.0).to_numpy(dtype=float), 1e-300, 1.0))
        x_label = "-log10 P-value"
    elif "Combined Score" in df.columns:
        score = pd.to_numeric(df["Combined Score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_label = "Combined score"
    else:
        score = np.arange(len(df), 0, -1, dtype=float)
        x_label = "Rank"

    sizes_raw = pd.to_numeric(df.get("Gene count", 1), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    if len(sizes_raw) == 0:
        sizes = np.array([])
    elif np.allclose(np.nanmax(sizes_raw), np.nanmin(sizes_raw)):
        sizes = np.full_like(sizes_raw, 12.0)
    else:
        sizes = 10.0 + 12.0 * (sizes_raw - np.nanmin(sizes_raw)) / max(np.nanmax(sizes_raw) - np.nanmin(sizes_raw), 1e-8)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=score, y=df["Term"].astype(str).tolist(),
        mode="markers",
        marker=dict(
            size=sizes, color=score, colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=x_label, thickness=10, len=0.70, x=1.01, xpad=2),
            opacity=0.88,
            line=dict(width=0.6,
                       color="rgba(0,0,0,0.20)" if str(theme_mode).lower() == "light" else "rgba(255,255,255,0.35)"),
        ),
        customdata=np.stack([sizes_raw], axis=1) if len(sizes_raw) > 0 else None,
        hovertemplate=("<b>%{y}</b><br>" + x_label + "=%{x:.2f}"
                       "<br>Gene count=%{customdata[0]:.0f}<extra></extra>"),
        showlegend=False,
    ))
    layout = dict(height=height, margin=dict(l=255, r=20, t=16 if not title else 38, b=42))
    if title:
        layout["title"] = dict(text=title, x=0.5, xanchor="center")
    fig.update_layout(**layout)
    _apply_plot_theme(fig, theme_mode=theme_mode)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="", autorange="reversed", automargin=True)
    return _to_plotly_json(fig)


# ============================================================================
# Response builders + run-cache
# ============================================================================

def _cache_key_for_run(ds_name: str, payload: dict) -> str:
    blob = json.dumps({"schema": CACHE_SCHEMA_VERSION, "dataset": ds_name, "payload": payload},
                       sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def store_run(cache_key: str, de_df: pd.DataFrame, summary: dict) -> None:
    with _RUN_LOCK:
        _RUN_CACHE[cache_key] = {"de_df": de_df, "summary": summary}


def get_run(cache_key: str) -> Optional[dict]:
    with _RUN_LOCK:
        return _RUN_CACHE.get(cache_key)


def export_run(ds: DatasetPaths, de_df: pd.DataFrame, summary: dict) -> tuple[Path, Path]:
    outdir = _runs_dir(ds)
    stamp = _make_timestamp()
    mode = str(summary.get("mode", "analysis"))
    csv_path = outdir / f"{stamp}_{mode}_gene_changes.csv"
    json_path = outdir / f"{stamp}_{mode}_summary.json"
    de_df.to_csv(csv_path, index=False)
    safe_summary = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in summary.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_summary, f, indent=2, default=str)
    return csv_path, json_path


def build_feature_panel_response(ds: DatasetPaths, mi_name: str, top_n: int, theme_mode: str = "dark") -> dict:
    bundle = get_loading_bundle(ds)
    if mi_name not in bundle["mi_list"]:
        mi_name = bundle["mi_list"][0] if bundle["mi_list"] else mi_name
    lr_df = top_rows_for_mi(bundle, mi_name, "lr", top_n)
    sender_df = top_rows_for_mi(bundle, mi_name, "sender", top_n)
    receiver_df = top_rows_for_mi(bundle, mi_name, "receiver", top_n)
    return {
        "mi_list": bundle["mi_list"],
        "mi_name": mi_name,
        "lr_rows": lr_df.to_dict("records"),
        "sender_rows": sender_df.to_dict("records"),
        "receiver_rows": receiver_df.to_dict("records"),
        "lr_figure": stem_figure(lr_df, f"{mi_name} | top LR pairs", "label", theme_mode),
        "sender_figure": stem_figure(sender_df, f"{mi_name} | top regulator genes", "gene", theme_mode),
        "receiver_figure": stem_figure(receiver_df, f"{mi_name} | top target genes", "gene", theme_mode),
    }


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", default_handler=str))


def build_run_response(ds: DatasetPaths, de_df: pd.DataFrame, summary: dict, cache_key: str,
                        lfc_threshold: float = DEFAULT_DE_LOGFC_THRESHOLD,
                        padj_threshold: float = DEFAULT_DE_PADJ_THRESHOLD,
                        enrich_direction: str = DEFAULT_ENRICH_DIRECTION,
                        enrich_scope: str = DEFAULT_ENRICH_SCOPE,
                        theme_mode: str = "dark",
                        top_terms: int = DEFAULT_ENRICH_TOP_TERMS) -> dict:
    display_df = filter_de_display_table(de_df, lfc_threshold, padj_threshold)
    enrich_genes = genes_for_enrichment(de_df, enrich_direction, enrich_scope, lfc_threshold, padj_threshold)
    state = _STATE_CACHE.get(ds.name)
    species = (state or {}).get("species") or ds.setup.get("SPECIES") or "human"
    go_df, kegg_df = cached_enrichment(enrich_genes, species=species, top_terms=top_terms)

    return {
        "cache_key": cache_key,
        "summary": {**{k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in summary.items()},
                     "lfc_threshold": float(lfc_threshold),
                     "padj_threshold": float(padj_threshold),
                     "n_enrichment_genes": len(enrich_genes),
                     "n_display_genes": int(display_df.shape[0])},
        "de_records": _df_to_records(display_df),
        "volcano_figure": volcano_figure(de_df, theme_mode, lfc_threshold, padj_threshold),
        "go_figure": bubble_figure(go_df, theme_mode, height=300),
        "kegg_figure": bubble_figure(kegg_df, theme_mode, height=300),
        "go_records": _df_to_records(go_df),
        "kegg_records": _df_to_records(kegg_df),
    }
