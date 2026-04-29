
import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
import scipy.sparse as sp
from torch_geometric.data import Data

try:
    from SpiderNet.utils import *
except ImportError:  # pragma: no cover
    from .utils import *  # type: ignore

ConfigDict = dict[str, Any]
HookFn = Callable[[sc.AnnData, dict[str, Any]], sc.AnnData]
PostConcatHookFn = Callable[[sc.AnnData, dict[str, Any]], sc.AnnData]

DEFAULT_CONFIG: ConfigDict = {
    "adata_folder_name": "adata",
    "file_suffix": ".h5ad",
    "file_filter_fn": None,
    "file_sort_key": None,
    "per_file_hook": None,
    "post_concat_hook": None,
    "output_dir": None,
    "n_hvg": 1000,
    "n_hvg_lr": 2000,
    "apply_hvg_selection": True,
    "expression_source": {
        "kind": "X",   # X | layer
        "name": None,  # layer name when kind == "layer"
    },
    "normalize_strategy": "auto",  # auto | always | never
    "normalize_target_sum": 1e4,
    "log1p": True,
    "remove_zero_count_cells": True,
    "obs_name_prefix_mode": "none",  # none | filename_stem | custom
    "obs_name_prefix_fn": None,
    "spatial_source": {
        "kind": "obsm",   # obsm | obs
        "key": "spatial",
        "cols": None,
    },
    "sample_col": None,
    "sample_group_col": "sample_name",
    "sample_group_builder": None,  # callable(adata, context) -> scalar/array-like
    "sample_sort": "auto",  # auto | lexicographic | none
    "cell_class_col": "celltype",
    "sample_name_obs_col": None,
    "sample_attr_obs_col": None,
    "pyg_obs_fields": {},  # {"patients": "patients", "age": "age"}
    "requested_genes": None,
    "gene_list_path": None,
    "fill_missing_requested_genes": False,
    "requested_gene_fill_value": 0.0,
    "lr_list_path": None,
    "num_neighbors": 8,
    "if_bothdirections": False,
    "cc_prop_threshold": None,
    "adaptive_lr_activation_thresholds": [0.01, 0.05, 0.2],
    "adaptive_lr_pair_count_caps": [150, 250],
    "skip_lr_filter_when_predefined": True,
    "apply_lr_corr_filter": True,
    "lr_corr_threshold": None,
    "cellchat_direct_use_min_pairs": 200,
    "if_subsetLRpair": False,
    "num_subsetLRpair_ratio": 0.8,
    "save_lr_heatmap": True,
    "neighbor_prop_celltype_col": None,
    "save_extra_tables": {},  # {"metadata_sample.csv": dataframe}
    "save_adata_file_list_pickle": False,
    "save_bundle_summary_json": True,
}

def _normalize_config(config: Mapping[str, Any]) -> ConfigDict:
    cfg = {**DEFAULT_CONFIG, **dict(config)}
    if cfg["output_dir"] is None:
        raise ValueError("config['output_dir'] must be provided.")
    if "data_path_main" not in cfg:
        raise ValueError("config['data_path_main'] must be provided.")
    if "ligand_receptor_filedir_cellchatdb" not in cfg:
        raise ValueError("config['ligand_receptor_filedir_cellchatdb'] must be provided.")
    if "ligand_receptor_filedir_scSeqComm" not in cfg:
        raise ValueError("config['ligand_receptor_filedir_scSeqComm'] must be provided.")

    sample_col = cfg.get("sample_col")
    if sample_col is not None:
        cfg["sample_group_col"] = sample_col
        if cfg.get("sample_name_obs_col") is None:
            cfg["sample_name_obs_col"] = sample_col
        if cfg.get("sample_attr_obs_col") is None:
            cfg["sample_attr_obs_col"] = sample_col

    if cfg["neighbor_prop_celltype_col"] is None:
        cfg["neighbor_prop_celltype_col"] = cfg["cell_class_col"]
    if cfg["apply_lr_corr_filter"] and cfg.get("lr_corr_threshold") is None:
        raise ValueError(
            "config['lr_corr_threshold'] must be provided when apply_lr_corr_filter=True. "
            "Please set it explicitly in the dataset-specific notebook or config."
        )
    return cfg

def normalize_optional_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path_str = str(path_str).strip()
    if path_str.lower() in {"", "none", "null", "nan"}:
        return None
    return path_str

def is_int_like(X: Any, tol: float = 1e-8, max_check: int = 1_000_000) -> bool:
    if sp.issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()

    if data.size == 0:
        return True

    if data.size > max_check:
        idx = np.random.choice(data.size, max_check, replace=False)
        data = data[idx]

    if not np.isfinite(data).all():
        return False

    return np.all(np.abs(data - np.round(data)) < tol)

def normalize_lr_list(LR_list: list[Any]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    normalized = []
    for pair in LR_list:
        lig_raw, rec_raw = pair
        lig = tuple(str(x) for x in lig_raw)
        rec = tuple(str(x) for x in rec_raw)
        normalized.append((lig, rec))

    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    unique_list = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def load_pickle_or_array(path: str | Path) -> Any:
    path = os.path.abspath(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif path.endswith(".npy"):
        obj = np.load(path, allow_pickle=True)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            obj = [line.strip() for line in f if line.strip()]
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            obj = df.iloc[:, 0].tolist()
        else:
            obj = df.values.tolist()
    else:
        raise ValueError(f"Unsupported file format for: {path}")
    return obj

def load_genenames(path: str | Path) -> list[str]:
    obj = load_pickle_or_array(path)

    if isinstance(obj, pd.Index):
        genes = obj.astype(str).tolist()
    elif isinstance(obj, np.ndarray):
        genes = obj.astype(str).tolist()
    elif isinstance(obj, (list, tuple)):
        genes = [str(x) for x in obj]
    else:
        raise ValueError(f"Unsupported gene-name object type: {type(obj)}")

    genes = [str(g) for g in genes]
    if len(genes) == 0:
        raise ValueError("Loaded gene-name list is empty.")
    return genes

def _resolve_adata_dir(data_path_main: str | Path, adata_folder_name: str) -> Path:
    data_path_main = Path(data_path_main)
    candidate = data_path_main / adata_folder_name
    if candidate.is_dir():
        return candidate
    if data_path_main.is_dir():
        return data_path_main
    raise FileNotFoundError(
        f"Cannot find data directory. Checked: {candidate} and {data_path_main}"
    )

def _auto_numeric_sort(values: list[str]) -> list[str]:
    def _key(x: str):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))
    return sorted(values, key=_key)

def _resolve_file_order(files: list[str], sort_key: Any) -> list[str]:
    if sort_key is None:
        return sorted(files)
    if sort_key == "auto":
        return sorted(files)
    if sort_key == "numeric_from_filename":
        return _auto_numeric_sort(files)
    if callable(sort_key):
        return sorted(files, key=sort_key)
    raise ValueError(f"Unsupported file_sort_key: {sort_key}")

def _ensure_obs_names_prefixed(
    adata: sc.AnnData,
    file_name: str,
    cfg: ConfigDict,
    context: dict[str, Any],
) -> sc.AnnData:
    mode = cfg["obs_name_prefix_mode"]
    if mode == "none":
        return adata

    if mode == "filename_stem":
        prefix = Path(file_name).stem
    elif mode == "custom":
        if cfg["obs_name_prefix_fn"] is None:
            raise ValueError("obs_name_prefix_mode='custom' requires obs_name_prefix_fn.")
        prefix = cfg["obs_name_prefix_fn"](adata, context)
    else:
        raise ValueError(f"Unsupported obs_name_prefix_mode: {mode}")

    adata.obs_names = [f"{prefix}_{x}" for x in adata.obs_names]
    return adata

def _resolve_expression_matrix(adata: sc.AnnData, cfg: ConfigDict) -> sc.AnnData:
    expr_cfg = dict(cfg["expression_source"])
    kind = expr_cfg.get("kind", "X")
    name = expr_cfg.get("name", None)

    if kind == "X":
        return adata

    if kind == "layer":
        if name is None:
            raise ValueError("expression_source.kind='layer' requires expression_source.name.")
        if name not in adata.layers:
            raise KeyError(
                f"Layer '{name}' was requested but is not available. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        adata.X = adata.layers[name].copy()
        return adata

    raise ValueError(f"Unsupported expression_source.kind: {kind}")

def _resolve_spatial(adata: sc.AnnData, cfg: ConfigDict) -> sc.AnnData:
    spatial_cfg = dict(cfg["spatial_source"])
    kind = spatial_cfg.get("kind", "obsm")

    if kind == "obsm":
        key = spatial_cfg.get("key", "spatial")
        if key not in adata.obsm:
            raise KeyError(
                f"Spatial key '{key}' not found in adata.obsm. "
                f"Available obsm keys: {list(adata.obsm.keys())}"
            )
        adata.obsm["spatial"] = np.asarray(adata.obsm[key])
        return adata

    if kind == "obs":
        cols = spatial_cfg.get("cols", None)
        if cols is None or len(cols) != 2:
            raise ValueError("spatial_source.kind='obs' requires exactly two column names in 'cols'.")
        missing_cols = [c for c in cols if c not in adata.obs.columns]
        if missing_cols:
            raise KeyError(f"Spatial obs columns not found: {missing_cols}")
        adata.obsm["spatial"] = np.asarray(adata.obs[cols], dtype=float)
        return adata

    raise ValueError(f"Unsupported spatial_source.kind: {kind}")

def _apply_sample_group_builder(
    adata: sc.AnnData,
    cfg: ConfigDict,
    context: dict[str, Any],
) -> sc.AnnData:
    sample_group_col = cfg["sample_group_col"]
    builder = cfg["sample_group_builder"]

    if callable(builder):
        values = builder(adata, context)
        if np.isscalar(values):
            adata.obs[sample_group_col] = values
        else:
            values = list(values)
            if len(values) != adata.n_obs:
                raise ValueError(
                    f"sample_group_builder returned {len(values)} values, but adata has {adata.n_obs} cells."
                )
            adata.obs[sample_group_col] = values
        return adata

    if sample_group_col not in adata.obs.columns:
        adata.obs[sample_group_col] = "Sample1"

    return adata

def _remove_zero_count_cells(adata: sc.AnnData, file_name: str) -> sc.AnnData:
    cell_total = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    keep_mask = cell_total > 0
    if keep_mask.sum() < adata.n_obs:
        print(f"{file_name}: remove {adata.n_obs - keep_mask.sum()} zero-count cells")
    return adata[keep_mask].copy()

def _normalize_if_needed(adata: sc.AnnData, cfg: ConfigDict) -> sc.AnnData:
    strategy = cfg["normalize_strategy"]
    do_normalize = False

    if strategy == "always":
        do_normalize = True
    elif strategy == "never":
        do_normalize = False
    elif strategy == "auto":
        do_normalize = is_int_like(adata.X)
    else:
        raise ValueError(f"Unsupported normalize_strategy: {strategy}")

    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=float(cfg["normalize_target_sum"]))
        if cfg["log1p"]:
            sc.pp.log1p(adata)
    return adata

def _subset_to_requested_genes_with_fill(
    adata: sc.AnnData,
    requested_genes: list[str],
    fill_value: float = 0.0,
) -> sc.AnnData:
    requested_genes = [str(g) for g in requested_genes]
    source_genes = [str(g) for g in adata.var_names]
    missing_genes = [g for g in requested_genes if g not in source_genes]

    def _align_matrix(matrix: Any) -> np.ndarray:
        arr = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        if arr.shape != (adata.n_obs, adata.n_vars):
            raise ValueError(
                f"Expected matrix with shape {(adata.n_obs, adata.n_vars)}, got {arr.shape}."
            )
        df = pd.DataFrame(arr, index=adata.obs_names, columns=source_genes)
        if missing_genes:
            fill_df = pd.DataFrame(
                fill_value,
                index=adata.obs_names,
                columns=missing_genes,
            )
            df = pd.concat([df, fill_df], axis=1)
        return df.loc[:, requested_genes].values

    expr_aligned = _align_matrix(adata.X)

    new_adata = sc.AnnData(
        X=expr_aligned,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=requested_genes),
    )
    if "spatial" in adata.obsm:
        new_adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"])
    for key, value in adata.obsm.items():
        if key != "spatial":
            new_adata.obsm[key] = value
    for key, value in adata.layers.items():
        try:
            new_adata.layers[key] = _align_matrix(value)
        except Exception as exc:
            print(f"Warning: skip layer '{key}' during requested-gene alignment because {exc}")
    return new_adata

def _build_lr_list(
    adata: sc.AnnData,
    cfg: ConfigDict,
) -> tuple[
    list[Any],
    list[Any],
    Any,
    Any,
]:
    """
    Build the LR list using the standard selection logic.

    Standard behavior:
      - If a predefined LR list is provided, use it directly after normalization.
      - Otherwise, extract CellChat LR pairs first.
      - If the number of CellChat pairs reaches the threshold, use CellChat only.
        and deduplicate within that direct-CellChat branch.
      - Otherwise, concatenate scSeqComm + CellChat without applying an extra
        global deduplication step.
    """
    lr_list_path = normalize_optional_path(cfg.get("lr_list_path"))
    use_predefined_lr_list = lr_list_path is not None

    LR_list_cellchatdb, LR_meta_cellchatdb = Ligand_Receptor_gene_extraction_CellChatdb(
        cfg["ligand_receptor_filedir_cellchatdb"],
        adata,
    )

    if use_predefined_lr_list:
        LR_list = normalize_lr_list(load_pickle_or_array(lr_list_path))
        gene_set = set(adata.var_names.tolist())
        missing_pairs = []
        for pair in LR_list:
            missing_genes = [g for g in list(pair[0]) + list(pair[1]) if g not in gene_set]
            if missing_genes:
                missing_pairs.append((pair, missing_genes))
        if missing_pairs:
            examples = [
                f"{pair[0]}->{pair[1]} missing {genes}"
                for pair, genes in missing_pairs[:5]
            ]
            raise ValueError(
                "The predefined LR list contains genes missing from processed AnnData. "
                f"Total problematic pairs: {len(missing_pairs)}. Examples: {examples}"
            )
        print(f"Use predefined LR_list from {lr_list_path}. Total LR pairs retained: {len(LR_list)}")
        LR_list_all = LR_list.copy()
        return LR_list, LR_list_all, LR_list_cellchatdb, LR_meta_cellchatdb

    if len(LR_list_cellchatdb) >= int(cfg["cellchat_direct_use_min_pairs"]):
        LR_list = LR_list_cellchatdb
        LR_pairs_str = ["+".join(pair[0]) + "->" + "+".join(pair[1]) for pair in LR_list]
        LR_unique_str = np.unique(LR_pairs_str)
        LR_unique_idx = [np.where(np.array(LR_pairs_str) == s)[0][0] for s in LR_unique_str]
        LR_list = [LR_list[i] for i in LR_unique_idx]
        LR_list_all = LR_list.copy()
        print(
            "Total number of Ligand–Receptor pairs after direct CellChat selection "
            f"and branch-local deduplication: {len(LR_list_all)}"
        )
        return LR_list, LR_list_all, LR_list_cellchatdb, LR_meta_cellchatdb

    LR_list_all_from_scseqcomm = Ligand_Receptor_gene_extraction_scSeqComm(
        cfg["ligand_receptor_filedir_scSeqComm"],
        adata,
    )
    LR_list = LR_list_all_from_scseqcomm + LR_list_cellchatdb
    LR_list_all = LR_list.copy()
    print(
        "Total number of Ligand–Receptor pairs after concatenating scSeqComm + CellChat "
        f"without global deduplication: {len(LR_list_all)}"
    )
    return LR_list, LR_list_all, LR_list_cellchatdb, LR_meta_cellchatdb

def _maybe_sort_sample_ids(batch_cell_unique: np.ndarray, cfg: ConfigDict) -> np.ndarray:
    values = batch_cell_unique.astype(str)
    mode = cfg["sample_sort"]
    if mode == "none":
        return values
    if mode == "lexicographic":
        return np.array(sorted(values), dtype=object)
    if mode == "auto":
        return np.array(_auto_numeric_sort(list(values)), dtype=object)
    raise ValueError(f"Unsupported sample_sort: {mode}")

def _save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _save_spidernet_pyg_list(obj: Any, output_dir: Path) -> Path:
    """Save SpiderNet PyG data as .pkl, falling back to .pt if needed.

    Older notebooks expect SpiderNet_data_pyg_list.pkl. For very large processed
    objects, pickle can fail or run out of memory on some systems; in that case
    we write SpiderNet_data_pyg_list.pt instead. The paired loader in
    SpiderNet.io resolves either extension automatically.
    """
    pkl_path = output_dir / "SpiderNet_data_pyg_list.pkl"
    pt_path = output_dir / "SpiderNet_data_pyg_list.pt"

    try:
        _save_pickle(obj, pkl_path)
        return pkl_path
    except (MemoryError, OverflowError, RuntimeError, OSError, pickle.PicklingError) as exc:
        if pkl_path.exists():
            try:
                pkl_path.unlink()
            except OSError:
                pass
        warnings.warn(
            "Saving SpiderNet_data_pyg_list.pkl failed; falling back to "
            f"SpiderNet_data_pyg_list.pt. Original error: {exc}",
            RuntimeWarning,
        )
        torch.save(obj, pt_path)
        return pt_path

def _json_safe(value: Any) -> Any:
    if callable(value):
        return getattr(value, "__name__", str(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

def _prepare_preview_state_until_lr_activation_filter(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Lightweight preview pipeline used only for LR-correlation inspection.

    It stops immediately after LR activation filtering, which is the earliest
    stage needed to visualize the non-diagonal LR-correlation distribution.
    It intentionally does not run the later saving/finalization steps.
    """
    cfg = _normalize_config(config)

    adata_dir = _resolve_adata_dir(cfg["data_path_main"], cfg["adata_folder_name"])
    print("Preview: load the data and run up to the LR-correlation stage")
    print(f"Reading AnnData files from: {adata_dir}")

    adata_files = [f for f in os.listdir(adata_dir) if f.endswith(cfg["file_suffix"])]
    if len(adata_files) == 0:
        raise FileNotFoundError(f"No {cfg['file_suffix']} files found in {adata_dir}")

    adata_files = _resolve_file_order(adata_files, cfg["file_sort_key"])

    adata_file_list: list[sc.AnnData] = []
    requested_genes = cfg["requested_genes"]
    if requested_genes is None and normalize_optional_path(cfg.get("gene_list_path")) is not None:
        requested_genes = load_genenames(cfg["gene_list_path"])

    genename_intersect: np.ndarray | None = None

    for adata_file in adata_files:
        adata_path = adata_dir / adata_file
        adata_cur = sc.read_h5ad(adata_path)
        context = {
            "file_name": adata_file,
            "file_path": adata_path,
            "config": cfg,
        }

        if callable(cfg["file_filter_fn"]) and not cfg["file_filter_fn"](adata_cur, context):
            continue

        adata_cur = _ensure_obs_names_prefixed(adata_cur, adata_file, cfg, context)

        if callable(cfg["per_file_hook"]):
            adata_cur = cfg["per_file_hook"](adata_cur, context)

        adata_cur = _resolve_expression_matrix(adata_cur, cfg)
        adata_cur = _resolve_spatial(adata_cur, cfg)
        adata_cur = _apply_sample_group_builder(adata_cur, cfg, context)

        if cfg["remove_zero_count_cells"]:
            adata_cur = _remove_zero_count_cells(adata_cur, adata_file)

        if requested_genes is not None and cfg["fill_missing_requested_genes"]:
            adata_cur = _subset_to_requested_genes_with_fill(
                adata_cur,
                requested_genes=requested_genes,
                fill_value=float(cfg["requested_gene_fill_value"]),
            )

        adata_cur = _normalize_if_needed(adata_cur, cfg)
        adata_file_list.append(adata_cur)

        if genename_intersect is None:
            genename_intersect = np.asarray(adata_cur.var_names)
        else:
            genename_intersect = np.intersect1d(genename_intersect, adata_cur.var_names)

    if len(adata_file_list) == 0:
        raise ValueError("No AnnData files were retained after file filtering.")

    if requested_genes is not None:
        requested_genes = [str(g) for g in requested_genes]
        if cfg["fill_missing_requested_genes"]:
            gene_order = np.asarray(requested_genes)
        else:
            missing_genes = [g for g in requested_genes if g not in set(genename_intersect.tolist())]
            if missing_genes:
                raise ValueError(
                    "The requested gene list contains genes that are not available after "
                    f"intersecting all files. Missing {len(missing_genes)} genes, examples: {missing_genes[:10]}"
                )
            gene_order = np.asarray(requested_genes)
    else:
        gene_order = genename_intersect

    for i in range(len(adata_file_list)):
        adata_file_list[i] = adata_file_list[i][:, gene_order].copy()

    adata = sc.AnnData.concatenate(*adata_file_list, batch_key="_concat_batch", index_unique=None)
    adata_copy = adata.copy()

    if callable(cfg["post_concat_hook"]):
        adata = cfg["post_concat_hook"](adata, {"config": cfg, "stage": "post_concat"})
        adata_copy = adata.copy()

    if cfg["apply_hvg_selection"]:
        if max(int(cfg["n_hvg"]), int(cfg["n_hvg_lr"])) < adata.n_vars:
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=max(int(cfg["n_hvg"]), int(cfg["n_hvg_lr"])),
                subset=True,
            )
        if int(cfg["n_hvg"]) < adata_copy.n_vars:
            sc.pp.highly_variable_genes(
                adata_copy,
                flavor="seurat_v3",
                n_top_genes=int(cfg["n_hvg"]),
                subset=True,
            )
        genenames = adata.var_names
        genenames_train = adata_copy.var_names
        genenames_train_index = np.where(np.isin(genenames, genenames_train))[0]
        genenames_train = genenames[genenames_train_index]
    else:
        genenames = adata.var_names
        genenames_train = adata_copy.var_names
        genenames_train_index = np.arange(adata.n_vars)

    sample_group_col = cfg["sample_group_col"]
    batch_cell = adata.obs[sample_group_col].astype(str).values
    batch_cell_unique = np.unique(batch_cell.astype(str))
    batch_cell_unique = _maybe_sort_sample_ids(batch_cell_unique, cfg)

    LR_list, LR_list_all, LR_list_cellchatdb, LR_meta_cellchatdb = _build_lr_list(adata, cfg)

    SpiderNet_data_list = []
    for sample in batch_cell_unique:
        adata_curbatch = adata[adata.obs[sample_group_col].astype(str) == str(sample), :].copy()

        expr = adata_curbatch.X.toarray() if sp.issparse(adata_curbatch.X) else np.asarray(adata_curbatch.X)
        data_dict: dict[str, Any] = {
            "num_cells": adata_curbatch.n_obs,
            "cellnames": adata_curbatch.obs_names,
            "num_genes": adata_curbatch.n_vars,
            "genenames": adata_curbatch.var_names,
            "spatial_location": np.asarray(adata_curbatch.obsm["spatial"]),
            "expression_normalized": np.asarray(expr, dtype=np.float32),
            "cell_class": np.asarray(adata_curbatch.obs[cfg["cell_class_col"]]),
            "sample_name": np.asarray(
                adata_curbatch.obs[
                    cfg["sample_name_obs_col"] if cfg["sample_name_obs_col"] is not None else sample_group_col
                ]
            ),
            "sample": np.asarray(
                adata_curbatch.obs[
                    cfg["sample_attr_obs_col"] if cfg["sample_attr_obs_col"] is not None else sample_group_col
                ]
            ),
        }

        for attr_name, obs_col in dict(cfg["pyg_obs_fields"]).items():
            if obs_col not in adata_curbatch.obs.columns:
                raise KeyError(
                    f"Requested pyg_obs_fields column '{obs_col}' not found in AnnData.obs."
                )
            data_dict[attr_name] = np.asarray(adata_curbatch.obs[obs_col])

        SpiderNet_data_list.append(data_dict)

    for sample in batch_cell_unique:
        idx = np.where(batch_cell_unique == sample)[0][0]
        _, edge_index = spatial_neighborindex_generation(
            cell_spatial=SpiderNet_data_list[idx]["spatial_location"],
            num_neighbor_available=int(cfg["num_neighbors"]),
        )
        if cfg["if_bothdirections"]:
            edge_index_reverse = edge_index[:, [1, 0]]
            edge_index_combined = np.concatenate([edge_index, edge_index_reverse], axis=0)
            edge_index = np.unique(edge_index_combined, axis=0)
        SpiderNet_data_list[idx]["edge_index"] = edge_index

    cellclass = np.hstack([SpiderNet_data_list[i]["cell_class"] for i in range(len(SpiderNet_data_list))])
    cellclass_unique = np.unique(cellclass)

    for idx in range(len(batch_cell_unique)):
        cellclass_curbatch = SpiderNet_data_list[idx]["cell_class"]
        num_cells = cellclass_curbatch.shape[0]
        cellclass_onehot = np.zeros((num_cells, cellclass_unique.shape[0]), dtype=float)
        for i, ct in enumerate(cellclass_unique):
            cellclass_onehot[cellclass_curbatch == ct, i] = 1

        cellclass_onehot_pd = pd.DataFrame(cellclass_onehot, columns=cellclass_unique)
        cellclass_onehot_pd.index = SpiderNet_data_list[idx]["cellnames"]
        SpiderNet_data_list[idx]["cell_class_onehot"] = cellclass_onehot_pd

    SpiderNet_data_pyg_list = []
    for idx in range(len(batch_cell_unique)):
        pyg_kwargs = {
            "x": torch.tensor(SpiderNet_data_list[idx]["expression_normalized"], dtype=torch.float),
            "edge_index": torch.tensor(SpiderNet_data_list[idx]["edge_index"], dtype=torch.long),
            "pos": torch.tensor(SpiderNet_data_list[idx]["spatial_location"], dtype=torch.float),
            "cell_class_onehot": torch.tensor(
                np.asarray(SpiderNet_data_list[idx]["cell_class_onehot"]),
                dtype=torch.float,
            ),
            "cell_class_unique": cellclass_unique.tolist(),
            "cellnames": SpiderNet_data_list[idx]["cellnames"],
            "genenames": SpiderNet_data_list[idx]["genenames"],
            "num_cells": SpiderNet_data_list[idx]["num_cells"],
            "num_genes": SpiderNet_data_list[idx]["num_genes"],
            "sample_name": SpiderNet_data_list[idx]["sample_name"],
            "sample": SpiderNet_data_list[idx]["sample"],
        }

        if "cell_subclass" in SpiderNet_data_list[idx]:
            pyg_kwargs["cell_subclass"] = SpiderNet_data_list[idx]["cell_subclass"]

        for attr_name in dict(cfg["pyg_obs_fields"]).keys():
            pyg_kwargs[attr_name] = SpiderNet_data_list[idx][attr_name]

        SpiderNet_data_pyg_list.append(Data(**pyg_kwargs))

    for idx in range(len(batch_cell_unique)):
        edge_index = SpiderNet_data_pyg_list[idx].edge_index.cpu().numpy()
        cellpair_LRpair = np.zeros((edge_index.shape[0], len(LR_list)), dtype=np.float32)

        for LR_idx, LR_pair in enumerate(LR_list):
            ligand_idx = np.where(np.isin(SpiderNet_data_list[idx]["genenames"], LR_pair[0]))[0]
            receptor_idx = np.where(np.isin(SpiderNet_data_list[idx]["genenames"], LR_pair[1]))[0]

            exp_ligand = np.asarray(SpiderNet_data_list[idx]["expression_normalized"][:, ligand_idx])
            exp_receptor = np.asarray(SpiderNet_data_list[idx]["expression_normalized"][:, receptor_idx])

            exp_ligand = (
                np.power(np.prod(exp_ligand, axis=1), 1 / exp_ligand.shape[1])
                if exp_ligand.shape[1] > 1 else exp_ligand[:, 0]
            )
            exp_receptor = (
                np.power(np.prod(exp_receptor, axis=1), 1 / exp_receptor.shape[1])
                if exp_receptor.shape[1] > 1 else exp_receptor[:, 0]
            )

            cellpair_LRpair[:, LR_idx] = np.sqrt(
                exp_ligand[edge_index[:, 0]] * exp_receptor[edge_index[:, 1]]
            )

        SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"] = torch.from_numpy(cellpair_LRpair)

    if not (normalize_optional_path(cfg.get("lr_list_path")) is not None and cfg["skip_lr_filter_when_predefined"]):
        cellpair_LRpair_prop = np.array([
            (torch.sum(d["cellpair_LRpair_neigh"] > 0, dim=0) / d["cellpair_LRpair_neigh"].shape[0]).numpy()
            for d in SpiderNet_data_pyg_list
        ])
        cellpair_LRpair_prop_max = np.max(cellpair_LRpair_prop, axis=0)

        if cfg["cc_prop_threshold"] is not None:
            selected_LR_idx = np.where(cellpair_LRpair_prop_max > float(cfg["cc_prop_threshold"]))[0]
        else:
            thresholds = list(cfg["adaptive_lr_activation_thresholds"])
            caps = list(cfg["adaptive_lr_pair_count_caps"])
            selected_LR_idx = np.where(cellpair_LRpair_prop_max > thresholds[0])[0]
            if len(selected_LR_idx) > caps[0]:
                selected_LR_idx = np.where(cellpair_LRpair_prop_max > thresholds[1])[0]
                if len(selected_LR_idx) > caps[1]:
                    selected_LR_idx = np.where(cellpair_LRpair_prop_max > thresholds[2])[0]

        selected_LR_idx = torch.tensor(selected_LR_idx, dtype=torch.long)
        for idx in range(len(SpiderNet_data_pyg_list)):
            SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"] = (
                SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"][:, selected_LR_idx]
            )
        LR_list = [LR_list[i] for i in selected_LR_idx.numpy()]

    return {
        "config": cfg,
        "adata": adata_copy,
        "adata_file_list": adata_file_list,
        "SpiderNet_data_pyg_list": SpiderNet_data_pyg_list,
        "LR_list": LR_list,
        "LR_list_all": LR_list_all,
        "LR_list_cellchatdb": LR_list_cellchatdb,
        "LR_meta_cellchatdb": LR_meta_cellchatdb,
        "batch_cell_unique": batch_cell_unique,
        "batch_cell": batch_cell,
        "genenames": np.asarray(genenames),
        "genenames_train": np.asarray(genenames_train),
        "genenames_train_index": np.asarray(genenames_train_index),
        "cellclass_unique": cellclass_unique,
        "output_dir": Path(cfg["output_dir"]),
        "sample_group_col": sample_group_col,
    }

def _compute_lr_corr_matrix_from_bundle(bundle: Mapping[str, Any]) -> pd.DataFrame:
    spider_data = bundle["SpiderNet_data_pyg_list"]
    lr_list = bundle["LR_list"]

    if len(lr_list) <= 1:
        raise ValueError("At least two LR pairs are required to compute an LR-correlation matrix.")

    cellpair_LRpair_corr = np.zeros((len(lr_list), len(lr_list)))
    for idx in range(len(spider_data)):
        arr = spider_data[idx]["cellpair_LRpair_neigh"].cpu().numpy()
        corr_cur = np.corrcoef(arr, rowvar=False)
        corr_cur[np.isnan(corr_cur)] = 0
        cellpair_LRpair_corr = np.maximum(cellpair_LRpair_corr, corr_cur)

    return pd.DataFrame(
        cellpair_LRpair_corr,
        index=[f"LR{i}" for i in range(len(lr_list))],
        columns=[f"LR{i}" for i in range(len(lr_list))],
    )

def summarize_lr_corr_distribution_from_bundle(
    bundle: Mapping[str, Any],
    output_dir: str | Path | None = None,
    show_plot: bool = True,
) -> dict[str, Any]:
    """
    Summarize the non-diagonal LR-correlation distribution before thresholding.

    Use this summary to choose lr_corr_threshold.
    """
    corr = _compute_lr_corr_matrix_from_bundle(bundle)
    corr_no_diag = corr.copy()
    np.fill_diagonal(corr_no_diag.values, np.nan)

    values = corr_no_diag.values[~np.isnan(corr_no_diag.values)]
    if values.size == 0:
        raise ValueError("No non-diagonal LR-correlation values are available.")

    quantile_points = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    quantile_summary = pd.Series(
        {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in quantile_points},
        name="non_diagonal_lr_correlation",
    )
    summary = pd.Series(
        {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        },
        name="non_diagonal_lr_correlation",
    )
    summary_df = pd.concat([summary, quantile_summary])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(values, bins=50, stat="density", kde=True, ax=ax)
    ax.set_title("Non-diagonal LR-correlation distribution before thresholding")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density")
    fig.tight_layout()

    out_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "LR_corr_distribution_before_threshold.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "corr": corr,
        "corr_no_diag": corr_no_diag,
        "non_diagonal_values": values,
        "summary": summary_df,
        "plot_path": out_path,
    }

def preview_lr_corr_distribution(
    config: Mapping[str, Any],
    preview_output_dir: str | Path | None = None,
    show_plot: bool = True,
) -> dict[str, Any]:
    """
    Run only the minimal preprocessing needed to visualize the non-diagonal
    LR-correlation distribution before thresholding.

    Unlike the full unified loader, this preview stops right after LR activation
    filtering and does not run the later finalization or saving steps.
    """
    config_preview = dict(config)
    base_output_dir = Path(config_preview["output_dir"])
    if preview_output_dir is None:
        preview_output_dir = base_output_dir / "_lr_corr_preview"

    config_preview["output_dir"] = Path(preview_output_dir)
    config_preview["apply_lr_corr_filter"] = False
    config_preview["save_lr_heatmap"] = False

    preview_state = _prepare_preview_state_until_lr_activation_filter(config_preview)
    diagnostics = summarize_lr_corr_distribution_from_bundle(
        preview_state,
        output_dir=preview_output_dir,
        show_plot=show_plot,
    )
    diagnostics["preview_output_dir"] = Path(preview_output_dir)
    return diagnostics

def prepare_processed_bundle_unified(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Unified SpiderNet data-loading pipeline.

    Expected usage:
        bundle = prepare_processed_bundle_unified(config)

    The most important dataset-specific knobs are:
      - per_file_hook
      - post_concat_hook
      - expression_source
      - spatial_source
      - sample_group_builder
      - cell_class_col
      - requested_genes / gene_list_path
      - lr_list_path
      - pyg_obs_fields
    """
    cfg = _normalize_config(config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_dir = _resolve_adata_dir(cfg["data_path_main"], cfg["adata_folder_name"])
    print("Step 1: Load the data and perform preprocessing")
    print(f"Reading AnnData files from: {adata_dir}")

    adata_files = [f for f in os.listdir(adata_dir) if f.endswith(cfg["file_suffix"])]
    if len(adata_files) == 0:
        raise FileNotFoundError(f"No {cfg['file_suffix']} files found in {adata_dir}")

    adata_files = _resolve_file_order(adata_files, cfg["file_sort_key"])

    adata_file_list: list[sc.AnnData] = []
    requested_genes = cfg["requested_genes"]
    if requested_genes is None and normalize_optional_path(cfg.get("gene_list_path")) is not None:
        requested_genes = load_genenames(cfg["gene_list_path"])

    genename_intersect: np.ndarray | None = None

    for adata_file in adata_files:
        adata_path = adata_dir / adata_file
        adata_cur = sc.read_h5ad(adata_path)
        context = {
            "file_name": adata_file,
            "file_path": adata_path,
            "config": cfg,
        }

        if callable(cfg["file_filter_fn"]) and not cfg["file_filter_fn"](adata_cur, context):
            print(f"Skip file: {adata_file}")
            continue

        adata_cur = _ensure_obs_names_prefixed(adata_cur, adata_file, cfg, context)

        if callable(cfg["per_file_hook"]):
            adata_cur = cfg["per_file_hook"](adata_cur, context)

        adata_cur = _resolve_expression_matrix(adata_cur, cfg)
        adata_cur = _resolve_spatial(adata_cur, cfg)
        adata_cur = _apply_sample_group_builder(adata_cur, cfg, context)

        if cfg["remove_zero_count_cells"]:
            adata_cur = _remove_zero_count_cells(adata_cur, adata_file)

        if requested_genes is not None and cfg["fill_missing_requested_genes"]:
            adata_cur = _subset_to_requested_genes_with_fill(
                adata_cur,
                requested_genes=requested_genes,
                fill_value=float(cfg["requested_gene_fill_value"]),
            )

        adata_cur = _normalize_if_needed(adata_cur, cfg)
        adata_file_list.append(adata_cur)

        if genename_intersect is None:
            genename_intersect = np.asarray(adata_cur.var_names)
        else:
            genename_intersect = np.intersect1d(genename_intersect, adata_cur.var_names)

        if len(adata_file_list) == 1:
            print(adata_cur)

    if len(adata_file_list) == 0:
        raise ValueError("No AnnData files were retained after file filtering.")

    if requested_genes is not None:
        requested_genes = [str(g) for g in requested_genes]
        if cfg["fill_missing_requested_genes"]:
            gene_order = np.asarray(requested_genes)
        else:
            missing_genes = [g for g in requested_genes if g not in set(genename_intersect.tolist())]
            if missing_genes:
                raise ValueError(
                    "The requested gene list contains genes that are not available after "
                    f"intersecting all files. Missing {len(missing_genes)} genes, examples: {missing_genes[:10]}"
                )
            gene_order = np.asarray(requested_genes)
    else:
        gene_order = genename_intersect

    for i in range(len(adata_file_list)):
        adata_file_list[i] = adata_file_list[i][:, gene_order].copy()

    # adata = sc.AnnData.concatenate(*adata_file_list, batch_key="sample_name", index_unique=None)
    adata = sc.AnnData.concatenate(*adata_file_list, batch_key="_concat_batch", index_unique=None)
    adata_copy = adata.copy()

    if callable(cfg["post_concat_hook"]):
        adata = cfg["post_concat_hook"](adata, {"config": cfg, "stage": "post_concat"})
        adata_copy = adata.copy()

    if cfg["apply_hvg_selection"]:
        if max(int(cfg["n_hvg"]), int(cfg["n_hvg_lr"])) < adata.n_vars:
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=max(int(cfg["n_hvg"]), int(cfg["n_hvg_lr"])),
                subset=True,
            )
        if int(cfg["n_hvg"]) < adata_copy.n_vars:
            sc.pp.highly_variable_genes(
                adata_copy,
                flavor="seurat_v3",
                n_top_genes=int(cfg["n_hvg"]),
                subset=True,
            )
        genenames = adata.var_names
        genenames_train = adata_copy.var_names
        genenames_train_index = np.where(np.isin(genenames, genenames_train))[0]
        genenames_train = genenames[genenames_train_index]
    else:
        genenames = adata.var_names
        genenames_train = adata_copy.var_names
        genenames_train_index = np.arange(adata.n_vars)

    sample_group_col = cfg["sample_group_col"]
    batch_cell = adata.obs[sample_group_col].astype(str).values
    batch_cell_unique = np.unique(batch_cell.astype(str))
    batch_cell_unique = _maybe_sort_sample_ids(batch_cell_unique, cfg)

    LR_list, LR_list_all, LR_list_cellchatdb, LR_meta_cellchatdb = _build_lr_list(adata, cfg)
    _save_pickle(LR_list_all, output_dir / "LR_list_all.pkl")

    print("Step 2: Build sample-level SpiderNet data dictionaries")
    SpiderNet_data_list = []
    for sample in batch_cell_unique:
        adata_curbatch = adata[adata.obs[sample_group_col].astype(str) == str(sample), :].copy()

        expr = adata_curbatch.X.toarray() if sp.issparse(adata_curbatch.X) else np.asarray(adata_curbatch.X)
        data_dict: dict[str, Any] = {
            "num_cells": adata_curbatch.n_obs,
            "cellnames": adata_curbatch.obs_names,
            "num_genes": adata_curbatch.n_vars,
            "genenames": adata_curbatch.var_names,
            "spatial_location": np.asarray(adata_curbatch.obsm["spatial"]),
            "expression_normalized": np.asarray(expr, dtype=np.float32),
            "cell_class": np.asarray(adata_curbatch.obs[cfg["cell_class_col"]]),
            "sample_name": np.asarray(
                adata_curbatch.obs[
                    cfg["sample_name_obs_col"] if cfg["sample_name_obs_col"] is not None else sample_group_col
                ]
            ),
            "sample": np.asarray(
                adata_curbatch.obs[
                    cfg["sample_attr_obs_col"] if cfg["sample_attr_obs_col"] is not None else sample_group_col
                ]
            ),
        }

        for attr_name, obs_col in dict(cfg["pyg_obs_fields"]).items():
            if obs_col not in adata_curbatch.obs.columns:
                raise KeyError(
                    f"Requested pyg_obs_fields column '{obs_col}' not found in AnnData.obs."
                )
            data_dict[attr_name] = np.asarray(adata_curbatch.obs[obs_col])

        SpiderNet_data_list.append(data_dict)

    print("Step 3: Build spatial neighbor graph")
    for sample in batch_cell_unique:
        idx = np.where(batch_cell_unique == sample)[0][0]
        _, edge_index = spatial_neighborindex_generation(
            cell_spatial=SpiderNet_data_list[idx]["spatial_location"],
            num_neighbor_available=int(cfg["num_neighbors"]),
        )

        if cfg["if_bothdirections"]:
            edge_index_reverse = edge_index[:, [1, 0]]
            edge_index_combined = np.concatenate([edge_index, edge_index_reverse], axis=0)
            edge_index = np.unique(edge_index_combined, axis=0)

        SpiderNet_data_list[idx]["edge_index"] = edge_index

    num_edges = np.sum(
        [SpiderNet_data_list[i]["edge_index"].shape[0] for i in range(len(SpiderNet_data_list))]
    )
    print(f"Total number of edges across all samples: {num_edges}")

    print("Step 4: One-hot encoding of cell types and PyG object creation")
    cellclass = np.hstack([SpiderNet_data_list[i]["cell_class"] for i in range(len(SpiderNet_data_list))])
    cellclass_unique = np.unique(cellclass)

    for idx in range(len(batch_cell_unique)):
        cellclass_curbatch = SpiderNet_data_list[idx]["cell_class"]
        num_cells = cellclass_curbatch.shape[0]

        cellclass_onehot = np.zeros((num_cells, cellclass_unique.shape[0]), dtype=float)
        for i, ct in enumerate(cellclass_unique):
            cellclass_onehot[cellclass_curbatch == ct, i] = 1

        cellclass_onehot_pd = pd.DataFrame(cellclass_onehot, columns=cellclass_unique)
        cellclass_onehot_pd.index = SpiderNet_data_list[idx]["cellnames"]
        SpiderNet_data_list[idx]["cell_class_onehot"] = cellclass_onehot_pd

    SpiderNet_data_pyg_list = []
    for idx in range(len(batch_cell_unique)):
        pyg_kwargs = {
            "x": torch.tensor(SpiderNet_data_list[idx]["expression_normalized"], dtype=torch.float),
            "edge_index": torch.tensor(SpiderNet_data_list[idx]["edge_index"], dtype=torch.long),
            "pos": torch.tensor(SpiderNet_data_list[idx]["spatial_location"], dtype=torch.float),
            "cell_class_onehot": torch.tensor(
                np.asarray(SpiderNet_data_list[idx]["cell_class_onehot"]),
                dtype=torch.float,
            ),
            "cell_class_unique": cellclass_unique.tolist(),
            "cellnames": SpiderNet_data_list[idx]["cellnames"],
            "genenames": SpiderNet_data_list[idx]["genenames"],
            "num_cells": SpiderNet_data_list[idx]["num_cells"],
            "num_genes": SpiderNet_data_list[idx]["num_genes"],
            "sample_name": SpiderNet_data_list[idx]["sample_name"],
            "sample": SpiderNet_data_list[idx]["sample"],
        }

        if "cell_subclass" in SpiderNet_data_list[idx]:
            pyg_kwargs["cell_subclass"] = SpiderNet_data_list[idx]["cell_subclass"]

        for attr_name in dict(cfg["pyg_obs_fields"]).keys():
            pyg_kwargs[attr_name] = SpiderNet_data_list[idx][attr_name]

        data_pyg = Data(**pyg_kwargs)
        SpiderNet_data_pyg_list.append(data_pyg)

    print("Step 5: Compute cell–cell–LRpair tensor")
    for idx in range(len(batch_cell_unique)):
        edge_index = SpiderNet_data_pyg_list[idx].edge_index.cpu().numpy()
        cellpair_LRpair = np.zeros((edge_index.shape[0], len(LR_list)), dtype=np.float32)

        for LR_idx, LR_pair in enumerate(LR_list):
            ligand_idx = np.where(np.isin(SpiderNet_data_list[idx]["genenames"], LR_pair[0]))[0]
            receptor_idx = np.where(np.isin(SpiderNet_data_list[idx]["genenames"], LR_pair[1]))[0]

            exp_ligand = np.asarray(SpiderNet_data_list[idx]["expression_normalized"][:, ligand_idx])
            exp_receptor = np.asarray(SpiderNet_data_list[idx]["expression_normalized"][:, receptor_idx])

            exp_ligand = (
                np.power(np.prod(exp_ligand, axis=1), 1 / exp_ligand.shape[1])
                if exp_ligand.shape[1] > 1 else exp_ligand[:, 0]
            )
            exp_receptor = (
                np.power(np.prod(exp_receptor, axis=1), 1 / exp_receptor.shape[1])
                if exp_receptor.shape[1] > 1 else exp_receptor[:, 0]
            )

            cellpair_LRpair[:, LR_idx] = np.sqrt(
                exp_ligand[edge_index[:, 0]] * exp_receptor[edge_index[:, 1]]
            )

        SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"] = torch.from_numpy(cellpair_LRpair)

    if not (normalize_optional_path(cfg.get("lr_list_path")) is not None and cfg["skip_lr_filter_when_predefined"]):
        print("Step 6: Filter LR pairs based on coverage across edges")
        cellpair_LRpair_prop = np.array([
            (torch.sum(d["cellpair_LRpair_neigh"] > 0, dim=0) / d["cellpair_LRpair_neigh"].shape[0]).numpy()
            for d in SpiderNet_data_pyg_list
        ])
        cellpair_LRpair_prop_max = np.max(cellpair_LRpair_prop, axis=0)
        print(
            "Quantiles of LR pair activation proportion:",
            np.quantile(cellpair_LRpair_prop_max, (0, 0.2, 0.5, 0.8, 1.0)),
        )

        if cfg["cc_prop_threshold"] is not None:
            selected_LR_idx = np.where(cellpair_LRpair_prop_max > float(cfg["cc_prop_threshold"]))[0]
        else:
            thresholds = list(cfg["adaptive_lr_activation_thresholds"])
            caps = list(cfg["adaptive_lr_pair_count_caps"])
            selected_LR_idx = np.where(cellpair_LRpair_prop_max > thresholds[0])[0]
            if len(selected_LR_idx) > caps[0]:
                selected_LR_idx = np.where(cellpair_LRpair_prop_max > thresholds[1])[0]
                if len(selected_LR_idx) > caps[1]:
                    selected_LR_idx = np.where(cellpair_LRpair_prop_max > thresholds[2])[0]

        selected_LR_idx = torch.tensor(selected_LR_idx, dtype=torch.long)
        for idx in range(len(SpiderNet_data_pyg_list)):
            SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"] = (
                SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"][:, selected_LR_idx]
            )
        LR_list = [LR_list[i] for i in selected_LR_idx.numpy()]
    else:
        print("Step 6: Skip LR activation filtering because a predefined LR_list was provided")

    corr = None
    corr_binary = None
    corr_clustered = None

    if cfg["apply_lr_corr_filter"] and len(LR_list) > 1:
        print("Step 7: Compute pairwise LR correlation across samples")
        cellpair_LRpair_corr = np.zeros((len(LR_list), len(LR_list)))
        for idx in range(len(SpiderNet_data_pyg_list)):
            arr = SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"].cpu().numpy()
            corr_cur = np.corrcoef(arr, rowvar=False)
            corr_cur[np.isnan(corr_cur)] = 0
            cellpair_LRpair_corr = np.maximum(cellpair_LRpair_corr, corr_cur)

        corr = pd.DataFrame(
            cellpair_LRpair_corr,
            index=[f"LR{i}" for i in range(len(LR_list))],
            columns=[f"LR{i}" for i in range(len(LR_list))],
        )

        choose = np.ones(len(LR_list), dtype=bool)
        if len(LR_list) > 2:
            corr_no_diag = corr.copy()
            np.fill_diagonal(corr_no_diag.values, np.nan)
            choose = np.nanmax(corr_no_diag.values, axis=0) > float(cfg["lr_corr_threshold"])
            if choose.sum() == 0:
                choose = np.ones(len(LR_list), dtype=bool)

        choose_tensor = torch.tensor(choose, dtype=torch.bool)
        for idx in range(len(SpiderNet_data_pyg_list)):
            SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"] = (
                SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"][:, choose_tensor]
            )
        LR_list = [LR_list[i] for i in np.where(choose)[0]]
        corr = corr.loc[choose, choose]
        corr_binary = (corr > float(cfg["lr_corr_threshold"])).astype(int)
        np.fill_diagonal(corr_binary.values, 1)

        if len(corr) > 1:
            dist_matrix = 1 - corr.fillna(1).values
            row_linkage = linkage(squareform(dist_matrix, checks=False), method="average")
            col_linkage = linkage(squareform(dist_matrix.T, checks=False), method="average")
            row_order = leaves_list(row_linkage)
            col_order = leaves_list(col_linkage)
            corr_clustered = corr.iloc[row_order, col_order]

            if cfg["save_lr_heatmap"]:
                plt.figure(figsize=(10, 10))
                sns.heatmap(
                    corr_clustered,
                    cmap="viridis",
                    annot=False,
                    fmt=".2f",
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    mask=np.eye(len(corr_clustered)),
                )
                plt.title("Correlation of LR pairs (Clustered)")
                plt.savefig(output_dir / "LR_pairs_correlation_heatmap_clustered.png", dpi=300, bbox_inches="tight")
                plt.close()

    print("Step 8: Finalize PyG features and cell labels")
    for idx in range(len(batch_cell_unique)):
        SpiderNet_data_pyg_list[idx]["x"] = SpiderNet_data_pyg_list[idx]["x"][
            :, torch.tensor(genenames_train_index, dtype=torch.long)
        ]
        SpiderNet_data_pyg_list[idx]["cell_class"] = np.asarray(
            SpiderNet_data_pyg_list[idx]["cell_class_unique"]
        )[np.where(SpiderNet_data_pyg_list[idx]["cell_class_onehot"] == 1)[1]]

    print("Step 9: Compute neighboring cell-type proportions")
    cell_types = np.unique(adata_copy.obs[cfg["neighbor_prop_celltype_col"]])
    neighbor_props = {ct: [] for ct in cell_types}

    for idx in range(len(batch_cell_unique)):
        SpiderNet_data_pyg_cur = SpiderNet_data_pyg_list[idx]
        cellclass_cur = SpiderNet_data_pyg_cur["cell_class"]
        edge_index_cur = SpiderNet_data_pyg_cur["edge_index"].cpu().numpy()
        src, dst = edge_index_cur[:, 0], edge_index_cur[:, 1]
        n_nodes = len(cellclass_cur)

        neigh_cnt = np.bincount(src, minlength=n_nodes).astype(float)
        safe_div = lambda num, den: np.divide(num, np.where(den == 0, 1.0, den))

        for ct in neighbor_props.keys():
            ct_cnt = np.bincount(
                src,
                weights=(cellclass_cur[dst] == ct).astype(float),
                minlength=n_nodes,
            )
            ct_prop = safe_div(ct_cnt, neigh_cnt)
            neighbor_props[ct].extend(ct_prop.tolist())

    for ct, values in neighbor_props.items():
        adata_copy.obs[f"{ct}_prop"] = values

    LR_list_val = []
    if cfg["if_subsetLRpair"]:
        print("Step 10: Randomly subset LR pairs for training/validation")
        num_subsetLRpair = int(float(cfg["num_subsetLRpair_ratio"]) * len(LR_list))
        if num_subsetLRpair <= 0 or num_subsetLRpair >= len(LR_list):
            raise ValueError(
                "num_subsetLRpair_ratio must retain at least 1 LR pair and leave at least 1 for validation."
            )
        np.random.seed(0)
        sample_index = np.random.choice(range(len(LR_list)), num_subsetLRpair, replace=False)
        unsample_index = np.setdiff1d(range(len(LR_list)), sample_index)
        LR_list_val = [LR_list[i] for i in unsample_index]
        LR_list = [LR_list[i] for i in sample_index]
        _save_pickle(LR_list_val, output_dir / "LR_list_val.pkl")
        for idx in range(len(batch_cell_unique)):
            SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"] = (
                SpiderNet_data_pyg_list[idx]["cellpair_LRpair_neigh"][
                    :, torch.tensor(sample_index, dtype=torch.long)
                ]
            )

    print("Step 11: Build batch-aligned adata_list")
    adata_list = []
    for sample in batch_cell_unique:
        adata_batch = adata_copy[adata_copy.obs[sample_group_col].astype(str) == str(sample), :].copy()
        adata_list.append(adata_batch)

    print("Step 12: Save processed objects")
    adata_copy.write_h5ad(output_dir / "adata_all.h5ad", compression="gzip")
    _save_spidernet_pyg_list(SpiderNet_data_pyg_list, output_dir)
    _save_pickle(LR_list, output_dir / "LR_list.pkl")
    _save_pickle(LR_list_cellchatdb, output_dir / "LR_list_cellchatdb.pkl")
    _save_pickle(LR_meta_cellchatdb, output_dir / "LR_meta_cellchatdb.pkl")
    _save_pickle(batch_cell_unique, output_dir / "batch_cell_unique.pkl")
    _save_pickle(batch_cell, output_dir / "batch_cell.pkl")
    _save_pickle(np.asarray(genenames), output_dir / "genenames.pkl")
    _save_pickle(np.asarray(genenames_train), output_dir / "genenames_train.pkl")
    _save_pickle(adata_list, output_dir / "adata_list.pkl")
    _save_pickle(cellclass_unique, output_dir / "cellclass_unique.pkl")

    if cfg["save_adata_file_list_pickle"]:
        _save_pickle(adata_file_list, output_dir / "adata_file_list.pkl")

    for file_name, table in dict(cfg["save_extra_tables"]).items():
        out_path = output_dir / file_name
        if isinstance(table, pd.DataFrame):
            table.to_csv(out_path, index=False)
        else:
            raise ValueError(f"save_extra_tables only supports pandas DataFrames. Got: {type(table)}")

    if cfg["save_bundle_summary_json"]:
        summary = {
            "data_path_main": str(cfg["data_path_main"]),
            "adata_dir": str(adata_dir),
            "num_input_files": len(adata_files),
            "num_used_files": len(adata_file_list),
            "num_batches": int(len(batch_cell_unique)),
            "num_cells_total": int(adata_copy.n_obs),
            "num_genes_all": int(len(genenames)),
            "num_genes_train": int(len(genenames_train)),
            "num_lr_pairs_train": int(len(LR_list)),
            "num_lr_pairs_val": int(len(LR_list_val)),
            "config_snapshot": _json_safe(cfg),
        }
        with open(output_dir / "bundle_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    return {
        "adata": adata_copy,
        "adata_list": adata_list,
        "adata_file_list": adata_file_list,
        "SpiderNet_data_pyg_list": SpiderNet_data_pyg_list,
        "LR_list": LR_list,
        "LR_list_val": LR_list_val,
        "LR_list_all": LR_list_all,
        "LR_list_cellchatdb": LR_list_cellchatdb,
        "LR_meta_cellchatdb": LR_meta_cellchatdb,
        "batch_cell_unique": batch_cell_unique,
        "batch_cell": batch_cell,
        "genenames": np.asarray(genenames),
        "genenames_train": np.asarray(genenames_train),
        "cellclass_unique": cellclass_unique,
        "corr": corr,
        "corr_binary": corr_binary,
        "corr_clustered": corr_clustered,
        "output_dir": output_dir,
        "config": cfg,
    }

def _load_json_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified SpiderNet data loading.")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file.")
    args = parser.parse_args()
    config = _load_json_config(args.config)
    prepare_processed_bundle_unified(config)

if __name__ == "__main__":
    main()
