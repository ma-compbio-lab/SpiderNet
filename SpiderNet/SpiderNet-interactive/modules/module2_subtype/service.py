"""Module 2 — Subtype Discovery service.

Pragmatic port of the v13 pipeline from
`spidernet_interactive_module2_subtype_discovery_web.ipynb` (cells 3 + 9 + 11 + 12).

This first cut keeps the analytic pipeline correct but drops the multi-layer
disk-cache infrastructure (embedding/profile/PCA/neighbors/louvain/umap caches).
Only the per-dataset embedding store is cached on disk; everything else is
recomputed per request. DEG + GO/KEGG are deferred to milestone 4b.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sp

from core.datasets import DatasetPaths
from core.loaders import get_core_bundle


# ============================================================================
# Constants (cell 1 + later patches)
# ============================================================================

EMBEDDING_STORE_VERSION = "v2_full_obs_columns"
DEFAULT_AGG_MODE = "max"
DEFAULT_MI_THRESHOLD = 0.6
DEFAULT_N_NEIGHBORS = 25
DEFAULT_N_PCS = 50
DEFAULT_LOUVAIN_RES = 0.17
DEFAULT_UMAP_MIN_DIST = 0.10
DEFAULT_MAX_CELLS_FOR_PLOT = 15000
PCA_VAR_TARGET = 0.85
MAX_PCA_FIT_COMPONENTS = 50
UMAP_N_NEIGHBORS = 30
UMAP_RANDOM_STATE = 42


# ============================================================================
# Edge → per-cell aggregator (cell 3, pure NumPy)
# ============================================================================

def _edge_index_e_by_2(edge_index: Any) -> np.ndarray:
    import torch
    if torch.is_tensor(edge_index):
        arr = edge_index.detach().cpu().numpy()
    else:
        arr = np.asarray(edge_index)
    if arr.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {arr.shape}")
    if arr.shape[1] == 2:
        return arr.astype(np.int64, copy=False)
    if arr.shape[0] == 2:
        return arr.T.astype(np.int64, copy=False)
    raise ValueError(f"unrecognized edge_index shape {arr.shape}")


def _aggregate_per_cell(edge_index: Any, values: np.ndarray, num_cells: int, agg: str = "max"):
    edge_index = _edge_index_e_by_2(edge_index)
    values = np.asarray(values, dtype=np.float32)
    if edge_index.shape[0] != values.shape[0]:
        raise ValueError(f"edges {edge_index.shape[0]} != values rows {values.shape[0]}")
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    dim = values.shape[1]
    if agg == "max":
        send = np.full((num_cells, dim), -np.inf, dtype=np.float32)
        recv = np.full((num_cells, dim), -np.inf, dtype=np.float32)
        np.maximum.at(send, src, values)
        np.maximum.at(recv, dst, values)
        send[~np.isfinite(send)] = 0.0
        recv[~np.isfinite(recv)] = 0.0
    elif agg == "mean":
        send = np.zeros((num_cells, dim), dtype=np.float32)
        recv = np.zeros((num_cells, dim), dtype=np.float32)
        send_count = np.zeros(num_cells, dtype=np.int64)
        recv_count = np.zeros(num_cells, dtype=np.int64)
        np.add.at(send, src, values)
        np.add.at(recv, dst, values)
        np.add.at(send_count, src, 1)
        np.add.at(recv_count, dst, 1)
        send = send / np.maximum(send_count[:, None], 1)
        recv = recv / np.maximum(recv_count[:, None], 1)
    else:
        raise ValueError(f"agg must be 'max' or 'mean', got {agg!r}")
    return send.astype(np.float32, copy=False), recv.astype(np.float32, copy=False)


# ============================================================================
# Embedding store: per-cell aggregated MI vectors across all slices
# ============================================================================

@dataclass
class EmbeddingStore:
    meta: pd.DataFrame              # one row per cell, with cell_type / sample_id / barcode / global_cell_id
    X: np.ndarray                   # (n_cells, 2 * dim_envir): [send_MI1..send_MIK | recv_MI1..recv_MIK]
    mi_columns: list[str]
    summary: dict


# (dataset_name, agg_mode) -> EmbeddingStore
_EMBED_CACHE: dict[tuple[str, str], EmbeddingStore] = {}
_EMBED_LOCK = Lock()


def _embed_cache_dir(ds: DatasetPaths) -> Path:
    p = ds.run_dir / "UI_Exports" / "module2_cell_subtype_discovery_web" / "embedding_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_embedding_store(ds: DatasetPaths, agg_mode: str = DEFAULT_AGG_MODE) -> EmbeddingStore:
    key = (ds.name, agg_mode)
    with _EMBED_LOCK:
        cached = _EMBED_CACHE.get(key)
        if cached is not None:
            return cached

    cache_dir = _embed_cache_dir(ds)
    meta_path = cache_dir / f"cell_metadata_{agg_mode}.csv.gz"
    x_path = cache_dir / f"cell_embedding_{agg_mode}.npy"
    cols_path = cache_dir / f"mi_columns_{agg_mode}.json"
    summary_path = cache_dir / f"embedding_summary_{agg_mode}.json"

    # Disk cache?
    if meta_path.exists() and x_path.exists() and cols_path.exists() and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("embedding_store_version") == EMBEDDING_STORE_VERSION:
                meta_df = pd.read_csv(meta_path)
                X = np.load(x_path, mmap_mode="r")
                mi_columns = json.loads(cols_path.read_text(encoding="utf-8"))
                store = EmbeddingStore(meta=meta_df, X=X, mi_columns=mi_columns, summary=summary)
                with _EMBED_LOCK:
                    _EMBED_CACHE[key] = store
                return store
        except Exception:
            pass

    print(f"[m2] Building embedding store for {ds.name} (agg={agg_mode})...")
    bundle = get_core_bundle(ds)
    adata_list = bundle["adata_list"]
    pyg_list = bundle["pyg_list"]
    fe_list = bundle["factor_envir_list"]

    celltype_col = ds.config.get("CELL_TYPE_COL")
    sample_key = ds.config.get("SAMPLE_ID_COL")
    if not celltype_col:
        raise KeyError("config.json is missing CELL_TYPE_COL")

    meta_parts = []
    embedding_parts = []
    dim_envir = None

    for slice_idx, (adata, pyg_obj, factor_envir) in enumerate(zip(adata_list, pyg_list, fe_list)):
        n_cells = int(adata.n_obs)
        edge_index = pyg_obj["edge_index"] if isinstance(pyg_obj, dict) else getattr(pyg_obj, "edge_index")
        factor = np.asarray(factor_envir, dtype=np.float32)
        if factor.ndim != 2:
            raise ValueError(f"slice {slice_idx} factor_envir must be 2D, got {factor.shape}")
        if dim_envir is None:
            dim_envir = int(factor.shape[1])

        send, recv = _aggregate_per_cell(edge_index, factor, n_cells, agg=agg_mode)
        X_slice = np.concatenate([send, recv], axis=1).astype(np.float32, copy=False)

        obs = adata.obs.copy().reset_index(drop=True)
        if celltype_col not in obs.columns:
            raise KeyError(f"slice {slice_idx} missing column {celltype_col!r}")
        sample_values = (
            obs[sample_key].astype(str).to_numpy()
            if sample_key in obs.columns
            else np.repeat(f"slice_{slice_idx}", n_cells)
        )
        barcodes = pd.Index(adata.obs_names).astype(str)
        meta_slice = pd.DataFrame({
            "barcode": barcodes,
            "cell_type": obs[celltype_col].astype(str).to_numpy(),
            "sample_id": sample_values,
            "slice_index": int(slice_idx),
        })
        meta_slice["global_cell_id"] = "s" + str(slice_idx) + "::" + meta_slice["barcode"].astype(str)

        meta_parts.append(meta_slice)
        embedding_parts.append(X_slice)

    meta_df = pd.concat(meta_parts, axis=0, ignore_index=True)
    X = np.vstack(embedding_parts).astype(np.float32, copy=False)
    mi_columns = (
        [f"MI-{i + 1}_Sender" for i in range(dim_envir)]
        + [f"MI-{i + 1}_Receiver" for i in range(dim_envir)]
    )

    meta_df.to_csv(meta_path, index=False, compression="gzip")
    np.save(x_path, X)
    cols_path.write_text(json.dumps(mi_columns, indent=2), encoding="utf-8")
    summary = {
        "embedding_store_version": EMBEDDING_STORE_VERSION,
        "n_cells": int(meta_df.shape[0]),
        "n_features": int(X.shape[1]),
        "dim_envir": int(dim_envir),
        "agg_mode": str(agg_mode),
        "cell_types": sorted(pd.unique(meta_df["cell_type"]).tolist()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    store = EmbeddingStore(meta=meta_df, X=X, mi_columns=mi_columns, summary=summary)
    with _EMBED_LOCK:
        _EMBED_CACHE[key] = store
    return store


# ============================================================================
# MI feature selection from Avg_MI_cellclass_pair_merge_use (v13)
# ============================================================================

def _load_avg_mi_pair(ds: DatasetPaths) -> Optional[pd.DataFrame]:
    """Find the cached Avg_MI_cellclass_pair_merge_use table from M1's enrichment."""
    candidates = [
        # M1 enrichment cache
        ds.run_dir / "UI_Exports" / "module1_in_situ_professional" / "enrichment" / "Avg_MI_cellclass_pair_merge_use.csv",
        # legacy notebook locations
        ds.run_dir / "Avg_MI_cellclass_pair_merge_use.pkl",
        ds.run_dir / "Avg_MI_cellclass_pair_merge_use.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".pkl":
                with open(p, "rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame):
                    return df.copy()
            else:
                return pd.read_csv(p)
    return None


def _select_mi_features(ds: DatasetPaths, cell_type: str, mi_columns: list[str], mi_threshold: float) -> tuple[list[str], dict]:
    """v13: pick MI columns where this cell type appears as Sender (z>thr) or Receiver (z>thr)."""
    pair_df = _load_avg_mi_pair(ds)
    if pair_df is None or "Sender" not in pair_df.columns or "Receiver" not in pair_df.columns:
        # No enrichment table — keep all features.
        return list(mi_columns), {"sender_mis": [], "receiver_mis": [], "fallback": True}

    mi_pair_cols = [c for c in pair_df.columns if isinstance(c, str) and c.startswith("MI-")]
    chosen_sender = pair_df.loc[pair_df["Sender"].astype(str) == str(cell_type), mi_pair_cols]
    chosen_receiver = pair_df.loc[pair_df["Receiver"].astype(str) == str(cell_type), mi_pair_cols]

    def _select(df: pd.DataFrame) -> list[str]:
        if df is None or df.empty:
            return []
        v = df.to_numpy(dtype=np.float32, copy=False)
        keep = np.nanmax(v, axis=0) > float(mi_threshold)
        return [mi_pair_cols[i] for i in np.where(keep)[0]]

    sender_mis = _select(chosen_sender)
    receiver_mis = _select(chosen_receiver)
    requested = set([f"{m}_Sender" for m in sender_mis] + [f"{m}_Receiver" for m in receiver_mis])
    selected = [c for c in mi_columns if c in requested]
    if not selected:
        return list(mi_columns), {"sender_mis": sender_mis, "receiver_mis": receiver_mis, "fallback": True}
    return selected, {"sender_mis": sender_mis, "receiver_mis": receiver_mis, "fallback": False}


# ============================================================================
# Cluster palette + natural sort
# ============================================================================

def _natural_cluster_sort(labels: list[str]) -> list[str]:
    def _key(s: str):
        try:
            return (0, int(s))
        except (TypeError, ValueError):
            return (1, str(s))
    return sorted(set(labels), key=_key)


_CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _build_cluster_palette(cluster_order: list[str]) -> dict[str, str]:
    return {c: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i, c in enumerate(cluster_order)}


# ============================================================================
# Subtype discovery pipeline (single-pass, no disk cache)
# ============================================================================

@dataclass
class SubtypeResult:
    cell_type: str
    n_cells_total: int
    n_cells_used: int
    n_clusters: int
    cluster_order: list[str]
    cluster_palette: dict[str, str]
    umap_df: pd.DataFrame                # rows = cells used, cols: UMAP_1, UMAP_2, cluster, sample_id, ...
    summary_df: pd.DataFrame             # cluster summary: n_cells, fraction, dominant_sample
    sender_cluster_mean: pd.DataFrame    # n_clusters × n_MI
    receiver_cluster_mean: pd.DataFrame  # n_clusters × n_MI
    selected_features: list[str]
    feature_selection_summary: dict
    runtime_seconds: float
    cluster_assignment: pd.DataFrame     # global_cell_id -> cluster (for downstream DEG)
    cache_key: str = ""                  # hash of (cell_type + run params) — keys downstream caches


# In-memory map: cache_key -> SubtypeResult so DEG calls can reuse the run.
_RUN_CACHE: dict[str, SubtypeResult] = {}
_RUN_LOCK = Lock()


def _hash_params(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def get_cached_run(cache_key: str) -> Optional[SubtypeResult]:
    with _RUN_LOCK:
        return _RUN_CACHE.get(cache_key)


def run_subtype_analysis(
    ds: DatasetPaths,
    cell_type: str,
    *,
    agg_mode: str = DEFAULT_AGG_MODE,
    mi_threshold: float = DEFAULT_MI_THRESHOLD,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    n_pcs: int = DEFAULT_N_PCS,
    louvain_resolution: float = DEFAULT_LOUVAIN_RES,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    random_state: int = 0,
    max_cells_for_plot: int = DEFAULT_MAX_CELLS_FOR_PLOT,
) -> SubtypeResult:
    import scanpy as sc
    import anndata as ad
    import umap as umap_lib
    from sklearn.decomposition import PCA

    # Quiet down scanpy's chatty warnings.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    sc.settings.verbosity = 0

    t0 = time.time()
    store = get_embedding_store(ds, agg_mode=agg_mode)
    meta_all = store.meta
    X_all = np.asarray(store.X)
    mi_columns = list(store.mi_columns)

    sel = (meta_all["cell_type"].astype(str) == str(cell_type)).to_numpy()
    if sel.sum() < 3:
        raise ValueError(f"Cell type {cell_type!r} has only {int(sel.sum())} cells; need ≥3.")
    meta_cell = meta_all.loc[sel].reset_index(drop=True)
    X_cell = np.asarray(X_all[sel], dtype=np.float32)

    selected_cols, fs_summary = _select_mi_features(
        ds, cell_type=cell_type, mi_columns=mi_columns, mi_threshold=mi_threshold,
    )
    feature_idx = np.array([mi_columns.index(c) for c in selected_cols], dtype=np.int64)
    if feature_idx.size < 2:
        raise ValueError(
            f"Selected only {feature_idx.size} MI features for {cell_type!r}; "
            "lower the MI threshold."
        )

    X_sel = X_cell[:, feature_idx].astype(np.float32, copy=True)
    col_mean = X_sel.mean(axis=0, keepdims=True)
    col_std = X_sel.std(axis=0, keepdims=True)
    col_std = np.where(col_std > 1e-8, col_std, 1.0)
    X_z = ((X_sel - col_mean) / col_std).astype(np.float32, copy=False)
    X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA
    n_obs, n_vars = X_z.shape
    max_fit = int(min(max(int(n_pcs), 2), MAX_PCA_FIT_COMPONENTS, n_vars, n_obs))
    pca = PCA(n_components=max_fit, svd_solver="auto")
    X_pca_full = pca.fit_transform(X_z).astype(np.float32, copy=False)
    explained = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
    cum = np.cumsum(explained)
    hit = np.where(cum >= PCA_VAR_TARGET)[0]
    cutoff = int(hit[0]) if hit.size > 0 else int(X_pca_full.shape[1])
    eff_pcs = max(1, min(int(cutoff), int(X_pca_full.shape[1])))
    X_pca = np.ascontiguousarray(X_pca_full[:, :eff_pcs], dtype=np.float32)

    # kNN — try umap method first, fall back to sklearn (cell-11 stability shim)
    eff_neighbors = int(min(max(int(n_neighbors), 2), max(2, n_obs - 1)))
    adata_pca = ad.AnnData(X=X_pca.copy(order="C"))
    try:
        sc.pp.neighbors(adata_pca, n_neighbors=eff_neighbors, use_rep="X",
                        method="umap", metric="euclidean")
        backend = "umap"
    except Exception:
        adata_pca = ad.AnnData(X=X_pca.copy(order="C"))
        sc.pp.neighbors(adata_pca, n_neighbors=eff_neighbors, use_rep="X",
                        method="umap", metric="euclidean", transformer="sklearn")
        backend = "sklearn_fallback"

    # Louvain on the kNN graph
    sc.tl.louvain(adata_pca, resolution=float(louvain_resolution),
                  key_added="subcluster", random_state=int(random_state))
    raw_labels = adata_pca.obs["subcluster"].astype(str).to_numpy()
    ordered_raw = _natural_cluster_sort(pd.unique(raw_labels).tolist())
    relabel = {lab: str(i + 1) for i, lab in enumerate(ordered_raw)}
    cluster_labels = np.array([relabel[r] for r in raw_labels], dtype=object)
    cluster_order = _natural_cluster_sort(pd.unique(cluster_labels).tolist())
    cluster_palette = _build_cluster_palette(cluster_order)

    # UMAP directly on PCA (notebook v13 mode)
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=float(umap_min_dist), metric="euclidean",
        random_state=UMAP_RANDOM_STATE,
    )
    coords = np.asarray(reducer.fit_transform(X_pca), dtype=np.float32)

    # Assemble result dataframes
    umap_df = meta_cell.copy()
    umap_df["UMAP_1"] = coords[:, 0]
    umap_df["UMAP_2"] = coords[:, 1]
    umap_df["cluster"] = cluster_labels.astype(str)

    # Per-cluster mean MI heatmaps from FULL joint (sender|receiver) profile
    cluster_mean = (
        pd.DataFrame(X_cell, columns=mi_columns)
        .assign(cluster=cluster_labels)
        .groupby("cluster", sort=False)
        .mean(numeric_only=True)
        .loc[cluster_order, :]
    )
    sender_cols = [c for c in cluster_mean.columns if c.endswith("_Sender")]
    receiver_cols = [c for c in cluster_mean.columns if c.endswith("_Receiver")]

    summary_df = (
        umap_df.groupby("cluster", sort=False)
        .agg(
            n_cells=("barcode", "size"),
            n_samples=("sample_id", "nunique"),
            dominant_sample=("sample_id", lambda x: x.value_counts().index[0] if len(x) > 0 else ""),
        )
        .reset_index()
    )
    total = summary_df["n_cells"].sum()
    summary_df["fraction"] = summary_df["n_cells"] / max(total, 1)

    cluster_assignment = umap_df[["global_cell_id", "cluster"]].drop_duplicates().reset_index(drop=True)

    cache_key = _hash_params({
        "dataset": ds.name,
        "cell_type": cell_type,
        "agg_mode": agg_mode,
        "mi_threshold": mi_threshold,
        "n_neighbors": n_neighbors,
        "n_pcs": n_pcs,
        "louvain_resolution": louvain_resolution,
        "umap_min_dist": umap_min_dist,
        "random_state": random_state,
    })

    result = SubtypeResult(
        cell_type=str(cell_type),
        n_cells_total=int(meta_cell.shape[0]),
        n_cells_used=int(X_pca.shape[0]),
        n_clusters=int(len(cluster_order)),
        cluster_order=cluster_order,
        cluster_palette=cluster_palette,
        umap_df=umap_df,
        summary_df=summary_df,
        sender_cluster_mean=cluster_mean.loc[:, sender_cols],
        receiver_cluster_mean=cluster_mean.loc[:, receiver_cols],
        selected_features=selected_cols,
        feature_selection_summary=fs_summary,
        runtime_seconds=float(time.time() - t0),
        cluster_assignment=cluster_assignment,
        cache_key=cache_key,
    )
    with _RUN_LOCK:
        _RUN_CACHE[cache_key] = result
    return result


# ============================================================================
# Plotly figure builders
# ============================================================================

def _theme_spec(theme_mode: str = "dark") -> dict:
    if str(theme_mode).lower() == "light":
        return {"bg": "#ffffff", "text": "#111111", "grid": "rgba(17,17,17,0.10)"}
    return {"bg": "#000000", "text": "#ffffff", "grid": "rgba(255,255,255,0.10)"}


def _to_plotly_json(fig: go.Figure) -> dict:
    out = json.loads(fig.to_json(validate=False))
    out["layout"]["template"] = None
    return out


def build_umap_figure(result: SubtypeResult, theme_mode: str = "dark", marker_size: float = 4.0) -> dict:
    theme = _theme_spec(theme_mode)
    fig = go.Figure()
    df = result.umap_df
    for cluster in result.cluster_order:
        sub = df.loc[df["cluster"] == cluster]
        fig.add_trace(go.Scatter(
            x=sub["UMAP_1"].astype(float).tolist(),
            y=sub["UMAP_2"].astype(float).tolist(),
            mode="markers",
            name=f"Cluster {cluster} (n={len(sub)})",
            customdata=sub[["barcode", "sample_id"]].astype(str).values.tolist(),
            marker=dict(
                size=float(marker_size),
                color=result.cluster_palette[cluster],
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>Cluster {cluster}</b><br>"
                "%{customdata[0]} · %{customdata[1]}<br>"
                "UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        autosize=True, height=560,
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=12),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(font=dict(color=theme["text"], size=11), bgcolor="rgba(0,0,0,0)"),
        title=None,
    )
    fig.update_xaxes(title="UMAP 1", color=theme["text"], gridcolor=theme["grid"], zeroline=False)
    fig.update_yaxes(title="UMAP 2", color=theme["text"], gridcolor=theme["grid"], zeroline=False, scaleanchor="x", scaleratio=1)
    return _to_plotly_json(fig)


def build_cluster_size_figure(result: SubtypeResult, theme_mode: str = "dark") -> dict:
    theme = _theme_spec(theme_mode)
    df = result.summary_df.copy()
    fig = go.Figure(go.Bar(
        x=df["cluster"].astype(str).tolist(),
        y=df["n_cells"].astype(int).tolist(),
        marker=dict(color=[result.cluster_palette[c] for c in df["cluster"].astype(str)],
                    line=dict(color=theme["text"], width=0.5)),
        text=df["fraction"].map(lambda f: f"{f * 100:.1f}%").tolist(),
        textposition="outside",
        textfont=dict(color=theme["text"]),
        hovertemplate="Cluster %{x}<br>n=%{y:,}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        height=280, autosize=True,
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=12),
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=False,
    )
    fig.update_xaxes(title="Cluster", color=theme["text"], showgrid=False)
    fig.update_yaxes(title="Cells", color=theme["text"], gridcolor=theme["grid"])
    return _to_plotly_json(fig)


def _heatmap_figure(df: pd.DataFrame, title: str, theme: dict) -> dict:
    if df.shape[0] == 0 or df.shape[1] == 0:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="No data.", showarrow=False,
                           font=dict(color=theme["text"], size=14))
        fig.update_layout(paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=320)
        return _to_plotly_json(fig)
    y_labels = [f"Cluster-{c}" for c in df.index.astype(str).tolist()]
    fig = go.Figure(go.Heatmap(
        z=df.values.astype(float).tolist(),
        x=df.columns.astype(str).tolist(),
        y=y_labels,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="Mean", font=dict(color=theme["text"], size=11)),
                      tickfont=dict(color=theme["text"], size=10),
                      outlinecolor=theme["text"], outlinewidth=1.0),
        xgap=0.5, ygap=0.5,
        hovertemplate="%{y}<br>%{x}<br>Mean MI: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["text"], size=12)),
        height=max(280, 60 + df.shape[0] * 28), autosize=True,
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=11),
        margin=dict(l=70, r=20, t=40, b=110),
    )
    fig.update_xaxes(tickangle=55, color=theme["text"], showgrid=False, automargin=True)
    fig.update_yaxes(autorange="reversed", color=theme["text"], showgrid=False, automargin=True)
    return _to_plotly_json(fig)


def build_cluster_heatmaps(result: SubtypeResult, theme_mode: str = "dark") -> tuple[dict, dict]:
    theme = _theme_spec(theme_mode)
    sender = _heatmap_figure(result.sender_cluster_mean, "Sending MI · cluster mean", theme)
    receiver = _heatmap_figure(result.receiver_cluster_mean, "Receiving MI · cluster mean", theme)
    return sender, receiver


def build_full_response(result: SubtypeResult, theme_mode: str = "dark", marker_size: float = 4.0) -> dict:
    sender_fig, receiver_fig = build_cluster_heatmaps(result, theme_mode=theme_mode)
    return {
        "cache_key": result.cache_key,
        "umap_figure": build_umap_figure(result, theme_mode=theme_mode, marker_size=marker_size),
        "cluster_size_figure": build_cluster_size_figure(result, theme_mode=theme_mode),
        "sender_heatmap_figure": sender_fig,
        "receiver_heatmap_figure": receiver_fig,
        "summary": {
            "cell_type": result.cell_type,
            "n_cells_total": result.n_cells_total,
            "n_cells_used": result.n_cells_used,
            "n_clusters": result.n_clusters,
            "cluster_order": result.cluster_order,
            "cluster_palette": result.cluster_palette,
            "n_selected_features": len(result.selected_features),
            "selected_features": result.selected_features,
            "feature_selection": result.feature_selection_summary,
            "runtime_seconds": result.runtime_seconds,
        },
        "summary_table": result.summary_df.to_dict(orient="records"),
    }


# ============================================================================
# DEG + Enrichment (sub-milestone 4b)
# ============================================================================

def _sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))


def _expression_cache_dir(ds: DatasetPaths) -> Path:
    p = ds.run_dir / "UI_Exports" / "module2_cell_subtype_discovery_web" / "expression_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _deg_cache_dir(ds: DatasetPaths) -> Path:
    p = ds.run_dir / "UI_Exports" / "module2_cell_subtype_discovery_web" / "deg_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_expression_cache(ds: DatasetPaths, cell_type: str) -> Path:
    """Concatenate per-slice AnnData rows for one cell type and write a single
    h5ad. Lazy: only builds if missing.
    """
    import anndata as ad
    cache_dir = _expression_cache_dir(ds)
    safe = _sanitize_name(cell_type)
    cache_path = cache_dir / f"expression_{safe}.h5ad"
    if cache_path.exists():
        return cache_path

    print(f"[m2] Building expression cache for {ds.name}/{cell_type}...")
    bundle = get_core_bundle(ds)
    adata_list = bundle["adata_list"]
    celltype_col = ds.config.get("CELL_TYPE_COL")
    sample_key = ds.config.get("SAMPLE_ID_COL")

    subset_list = []
    gene_intersection: Optional[set] = None

    for slice_idx, adata_cur in enumerate(adata_list):
        obs = adata_cur.obs
        if celltype_col not in obs.columns:
            raise KeyError(f"slice {slice_idx} missing column {celltype_col!r}")
        mask = obs[celltype_col].astype(str).to_numpy() == str(cell_type)
        if not np.any(mask):
            continue
        sub = adata_cur[mask].copy()
        sub.obs = pd.DataFrame(index=sub.obs_names)
        sub.obs["barcode"] = pd.Index(sub.obs_names).astype(str)
        sub.obs["cell_type"] = str(cell_type)
        if sample_key in obs.columns:
            sub.obs["sample_id"] = adata_cur.obs.loc[sub.obs_names, sample_key].astype(str).to_numpy()
        else:
            sub.obs["sample_id"] = np.repeat(f"slice_{slice_idx}", sub.n_obs)
        sub.obs["slice_index"] = int(slice_idx)
        sub.obs["global_cell_id"] = "s" + str(slice_idx) + "::" + sub.obs["barcode"].astype(str)

        subset_list.append(sub)
        gene_set = set(pd.Index(sub.var_names).astype(str))
        gene_intersection = gene_set if gene_intersection is None else (gene_intersection & gene_set)

    if not subset_list:
        raise ValueError(f"No cells of type {cell_type!r} in expression data.")

    common_genes = sorted(gene_intersection) if gene_intersection else list(pd.Index(subset_list[0].var_names).astype(str))
    subset_list = [x[:, common_genes].copy() for x in subset_list]
    expr = ad.concat(subset_list, axis=0, join="inner", merge="same")
    if sp.issparse(expr.X):
        expr.X = expr.X.tocsr().astype(np.float32)
    else:
        expr.X = np.asarray(expr.X, dtype=np.float32)
    expr.write_h5ad(cache_path, compression="lzf")
    return cache_path


# ----------------------------------------------------------------------------
# Enrichr (cell 13 v12)
# ----------------------------------------------------------------------------

def _normalize_species_key(species: str) -> str:
    s = str(species or "").strip().lower()
    if s.startswith("mouse") or s in ("mm", "m. musculus", "mus musculus", "mmusculus"):
        return "mouse"
    if s in ("human", "homo sapiens", "h. sapiens", "hs", "hsapiens", "enrichr"):
        return "human"
    if s in ("yeast", "saccharomyces", "saccharomyces cerevisiae", "s. cerevisiae"):
        return "yeast"
    if s in ("worm", "caenorhabditis elegans", "c. elegans"):
        return "worm"
    if s in ("fish", "zebrafish", "danio rerio", "d. rerio"):
        return "fish"
    if s in ("fly", "drosophila", "drosophila melanogaster", "d. melanogaster"):
        return "fly"
    return "human"


def _enrichment_lib_info(species: str) -> dict:
    sk = _normalize_species_key(species)
    organism_primary = {"human": "Human", "mouse": "Mouse", "yeast": "Yeast",
                        "worm": "Worm", "fish": "Fish", "fly": "Fly"}[sk]
    kegg_candidates = {
        "human": ["KEGG_2021_Human", "KEGG_2019_Human"],
        "mouse": ["KEGG_2021_Mouse", "KEGG_2019_Mouse"],
        "yeast": ["KEGG_2021_Human", "KEGG_2019_Human"],
        "worm": ["KEGG_2021_Human", "KEGG_2019_Human"],
        "fish": ["KEGG_2021_Human", "KEGG_2019_Human"],
        "fly": ["KEGG_2021_Human", "KEGG_2019_Human"],
    }[sk]
    return {
        "species_key": sk,
        "organism": organism_primary,
        "kegg": kegg_candidates[0],
        "kegg_candidates": kegg_candidates,
        "go_bp": "GO_Biological_Process_2023",
        "go_bp_candidates": ["GO_Biological_Process_2023", "GO_Biological_Process_2021", "GO_Biological_Process_2018"],
    }


def _empty_enrichment_records() -> list[dict]:
    return []


def _format_enrichr_table(df: pd.DataFrame, top_terms: int) -> pd.DataFrame:
    if df is None or df.shape[0] == 0:
        return pd.DataFrame(columns=["Term", "Adjusted P-value", "Genes", "Gene count", "Combined Score"])
    df = df.copy()
    if "Adjusted P-value" in df.columns:
        df["Adjusted P-value"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
        df = df.sort_values("Adjusted P-value", ascending=True)
    if "Gene count" not in df.columns:
        if "Overlap" in df.columns:
            df["Gene count"] = df["Overlap"].astype(str).str.extract(r"^(\d+)").fillna(0).astype(int)
        elif "Genes" in df.columns:
            df["Gene count"] = df["Genes"].astype(str).apply(
                lambda x: len([g for g in re.split(r"[;,/]", x) if g.strip()])
            )
        else:
            df["Gene count"] = 0
    keep = [c for c in ["Term", "Adjusted P-value", "P-value", "Genes", "Gene count", "Combined Score"] if c in df.columns]
    return df.loc[:, keep].head(int(top_terms)).reset_index(drop=True)


def run_enrichr_safe(gene_list: list[str], gene_set_library: str, organism: str, top_terms: int = 12) -> pd.DataFrame:
    """Robust Enrichr call with multi-organism alias retry. Returns either a
    formatted result table or a one-row DataFrame describing the failure."""
    import gseapy as gp

    genes = [str(g).strip() for g in gene_list if str(g).strip()]
    # de-dup keep order
    seen, deduped = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g); deduped.append(g)
    genes = deduped
    if len(genes) < 3:
        return _format_enrichr_table(pd.DataFrame(), top_terms)

    info = _enrichment_lib_info(organism)
    if info["species_key"] == "mouse":
        genes = [g.upper() for g in genes]

    if str(gene_set_library).startswith("KEGG"):
        candidates = list(dict.fromkeys([gene_set_library] + info["kegg_candidates"]))
    elif str(gene_set_library).startswith("GO_Biological_Process"):
        candidates = list(dict.fromkeys([gene_set_library] + info["go_bp_candidates"]))
    else:
        candidates = [gene_set_library]

    last_err = None
    for org in [info["organism"], info["species_key"]]:
        for lib in candidates:
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism=org, outdir=None, cutoff=1.0)
                if enr is not None and getattr(enr, "results", None) is not None:
                    df = _format_enrichr_table(enr.results, top_terms=top_terms)
                    if df.shape[0] > 0:
                        return df
            except Exception as e:
                last_err = e

    err_text = str(last_err) if last_err else "no enrichment"
    err_text = err_text.replace("\n", " ")[:200]
    return pd.DataFrame({
        "Term": [f"Enrichment failed ({info['species_key']}): {err_text}"],
        "Adjusted P-value": [1.0], "Genes": [""], "Gene count": [0],
    })


# ----------------------------------------------------------------------------
# DEG (Wilcoxon, scanpy.rank_genes_groups)
# ----------------------------------------------------------------------------

def _deg_cache_path(ds: DatasetPaths, run_cache_key: str, lfc: float, padj: float) -> Path:
    sub = _hash_params({"run": run_cache_key, "lfc": lfc, "padj": padj})
    return _deg_cache_dir(ds) / f"deg_{sub}.pkl"


def compute_deg_for_run(
    ds: DatasetPaths,
    result: SubtypeResult,
    *,
    lfc_thresh: float = 0.5,
    padj_thresh: float = 0.05,
    max_genes_for_enrichment: int = 100,
    top_enrich_terms: int = 12,
) -> dict:
    """Wilcoxon DEG per cluster vs. rest, plus Enrichr GO_BP + KEGG."""
    import scanpy as sc
    sc.settings.verbosity = 0
    warnings.filterwarnings("ignore")

    cache_path = _deg_cache_path(ds, result.cache_key, lfc_thresh, padj_thresh)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    expr_path = build_expression_cache(ds, result.cell_type)
    expr = sc.read_h5ad(expr_path)

    expr.obs = expr.obs.copy()
    expr.obs["global_cell_id"] = expr.obs["global_cell_id"].astype(str)
    gid_to_cluster = dict(zip(
        result.cluster_assignment["global_cell_id"].astype(str).tolist(),
        result.cluster_assignment["cluster"].astype(str).tolist(),
    ))
    mask = expr.obs["global_cell_id"].isin(gid_to_cluster.keys()).to_numpy()
    expr = expr[mask].copy()
    if expr.n_obs == 0:
        raise ValueError("no overlapping cells between expression cache and clustering")

    expr.obs["subcluster"] = expr.obs["global_cell_id"].map(gid_to_cluster).astype(str)
    expr.obs["subcluster"] = pd.Categorical(expr.obs["subcluster"], categories=result.cluster_order, ordered=True)

    out: dict[str, Any] = {
        "deg_full": {},
        "marker_tables": {},
        "enrichment": {},
        "deg_overview": [],
        "expression_summary": {"n_cells": int(expr.n_obs), "n_genes": int(expr.n_vars)},
        "params": {"lfc": lfc_thresh, "padj": padj_thresh,
                   "max_genes_for_enrichment": max_genes_for_enrichment,
                   "top_enrich_terms": top_enrich_terms},
    }

    if len(pd.unique(expr.obs["subcluster"].astype(str))) < 2:
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        return out

    sc.tl.rank_genes_groups(expr, groupby="subcluster", method="wilcoxon",
                             use_raw=False, pts=False)

    species = ds.config.get("SPECIES", "human")
    info = _enrichment_lib_info(species)
    print(f"[m2] DEG/Enrichr: {expr.n_obs} cells × {expr.n_vars} genes, {len(result.cluster_order)} clusters, species={species}")

    for cluster in result.cluster_order:
        df = sc.get.rank_genes_groups_df(expr, group=cluster).copy()
        df = df.rename(columns={
            "names": "gene", "pvals_adj": "padj", "pvals": "pval",
            "logfoldchanges": "logfoldchange", "scores": "score",
        })
        for col in ["padj", "pval", "logfoldchange", "score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["gene"] = df["gene"].astype(str)
        df["neglog10_padj"] = np.minimum(
            -np.log10(np.clip(df["padj"].fillna(1.0).to_numpy(dtype=float), 1e-300, 1.0)),
            10.0,
        )

        df_up = df[(df["logfoldchange"] >= float(lfc_thresh)) & (df["padj"] <= float(padj_thresh))]
        df_up = df_up.sort_values(["padj", "logfoldchange"], ascending=[True, False]).head(200).reset_index(drop=True)

        df_down = df[(df["logfoldchange"] <= -float(lfc_thresh)) & (df["padj"] <= float(padj_thresh))]
        df_down = df_down.sort_values(["padj", "logfoldchange"], ascending=[True, True]).head(200).reset_index(drop=True)

        up_genes = df_up["gene"].tolist()[:int(max_genes_for_enrichment)]
        down_genes = df_down["gene"].tolist()[:int(max_genes_for_enrichment)]

        enr = {"up": {}, "down": {}}
        for direction, genes in (("up", up_genes), ("down", down_genes)):
            enr[direction]["go"] = run_enrichr_safe(
                genes, gene_set_library=info["go_bp"], organism=info["organism"],
                top_terms=top_enrich_terms,
            ).to_dict("records")
            enr[direction]["kegg"] = run_enrichr_safe(
                genes, gene_set_library=info["kegg"], organism=info["organism"],
                top_terms=top_enrich_terms,
            ).to_dict("records")

        out["deg_full"][str(cluster)] = df.to_dict("records")
        out["marker_tables"][str(cluster)] = {
            "up": df_up.to_dict("records"),
            "down": df_down.to_dict("records"),
        }
        out["enrichment"][str(cluster)] = enr
        out["deg_overview"].append({
            "cluster": str(cluster),
            "n_up": int(df_up.shape[0]),
            "n_down": int(df_down.shape[0]),
        })

    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


# ----------------------------------------------------------------------------
# DEG plot builders
# ----------------------------------------------------------------------------

def build_volcano_figure(deg_records: list[dict], cluster: str, lfc_thresh: float, padj_thresh: float, theme_mode: str = "dark") -> dict:
    theme = _theme_spec(theme_mode)
    if not deg_records:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="No DEG data.", showarrow=False,
                           font=dict(color=theme["text"]))
        fig.update_layout(paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=320)
        return _to_plotly_json(fig)

    df = pd.DataFrame(deg_records)
    df["lfc"] = pd.to_numeric(df["logfoldchange"], errors="coerce")
    df["nlp"] = pd.to_numeric(df["neglog10_padj"], errors="coerce")
    df["padj_v"] = pd.to_numeric(df["padj"], errors="coerce")
    df = df.dropna(subset=["lfc", "nlp"])

    up_mask = (df["lfc"] >= float(lfc_thresh)) & (df["padj_v"] <= float(padj_thresh))
    down_mask = (df["lfc"] <= -float(lfc_thresh)) & (df["padj_v"] <= float(padj_thresh))
    other_mask = ~(up_mask | down_mask)

    fig = go.Figure()
    for sub_df, name, color in [
        (df.loc[other_mask], "n.s.", "rgba(150,150,150,0.5)"),
        (df.loc[down_mask], "down", "rgba(60,130,200,0.85)"),
        (df.loc[up_mask], "up", "rgba(220,80,80,0.85)"),
    ]:
        if sub_df.shape[0] == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=sub_df["lfc"].astype(float).tolist(),
            y=sub_df["nlp"].astype(float).tolist(),
            mode="markers",
            name=f"{name} (n={len(sub_df)})",
            customdata=sub_df["gene"].astype(str).tolist(),
            marker=dict(size=4.5, color=color, line=dict(width=0)),
            hovertemplate="<b>%{customdata}</b><br>log2FC: %{x:.2f}<br>-log10 padj: %{y:.2f}<extra></extra>",
        ))

    fig.add_vline(x=float(lfc_thresh), line_dash="dot", line_color=theme["text"], opacity=0.4)
    fig.add_vline(x=-float(lfc_thresh), line_dash="dot", line_color=theme["text"], opacity=0.4)
    if padj_thresh > 0:
        fig.add_hline(y=-np.log10(float(padj_thresh)), line_dash="dot", line_color=theme["text"], opacity=0.4)

    fig.update_layout(
        title=dict(text=f"Volcano · cluster {cluster}", font=dict(color=theme["text"], size=12)),
        autosize=True, height=420,
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=11),
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(font=dict(color=theme["text"], size=10), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title="log2 fold change", color=theme["text"], gridcolor=theme["grid"])
    fig.update_yaxes(title="-log10 adj p", color=theme["text"], gridcolor=theme["grid"])
    return _to_plotly_json(fig)


def build_enrichment_bubble(records: list[dict], title: str, theme_mode: str = "dark") -> dict:
    theme = _theme_spec(theme_mode)
    if not records:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="No terms.", showarrow=False,
                           font=dict(color=theme["text"]))
        fig.update_layout(paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
                          title=dict(text=title, font=dict(color=theme["text"], size=12)),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=320, margin=dict(l=10, r=10, t=40, b=10))
        return _to_plotly_json(fig)

    df = pd.DataFrame(records)
    df["padj"] = pd.to_numeric(df.get("Adjusted P-value", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    df["neglog10"] = -np.log10(np.clip(df["padj"].to_numpy(dtype=float), 1e-300, 1.0))
    df["count"] = pd.to_numeric(df.get("Gene count", pd.Series(1, index=df.index)), errors="coerce").fillna(1).astype(int)
    df = df.iloc[::-1].reset_index(drop=True)  # plot best-padj at top

    terms = df.get("Term", pd.Series([""] * len(df))).astype(str).tolist()
    short_terms = [t if len(t) <= 60 else t[:57] + "..." for t in terms]

    sizes = np.clip(df["count"].to_numpy(dtype=float) * 1.4, 6, 28)
    fig = go.Figure(go.Scatter(
        x=df["neglog10"].astype(float).tolist(),
        y=short_terms,
        mode="markers",
        marker=dict(
            size=sizes.tolist(),
            color=df["neglog10"].astype(float).tolist(),
            colorscale="Viridis",
            colorbar=dict(title=dict(text="-log10 padj", font=dict(color=theme["text"], size=10)),
                          tickfont=dict(color=theme["text"], size=9),
                          outlinecolor=theme["text"], outlinewidth=1.0),
            line=dict(width=0),
        ),
        customdata=np.column_stack([df["count"].astype(int).to_numpy(),
                                     df["padj"].astype(float).to_numpy(),
                                     terms]).tolist(),
        hovertemplate="<b>%{customdata[2]}</b><br>genes: %{customdata[0]}<br>padj: %{customdata[1]:.2e}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["text"], size=12)),
        autosize=True, height=max(320, 60 + len(df) * 22),
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=10),
        margin=dict(l=10, r=20, t=40, b=40),
    )
    fig.update_xaxes(title="-log10 adj p", color=theme["text"], gridcolor=theme["grid"])
    fig.update_yaxes(color=theme["text"], showgrid=False, automargin=True)
    return _to_plotly_json(fig)


def build_deg_response(
    ds: DatasetPaths,
    result: SubtypeResult,
    cluster: str,
    direction: str,
    *,
    lfc_thresh: float = 0.5,
    padj_thresh: float = 0.05,
    max_genes_for_enrichment: int = 100,
    top_enrich_terms: int = 12,
    theme_mode: str = "dark",
) -> dict:
    """Compute (or load cached) DEG + Enrichment, then return figures + tables
    for the chosen cluster + direction (up/down)."""
    deg = compute_deg_for_run(
        ds, result, lfc_thresh=lfc_thresh, padj_thresh=padj_thresh,
        max_genes_for_enrichment=max_genes_for_enrichment,
        top_enrich_terms=top_enrich_terms,
    )

    cluster_str = str(cluster)
    direction_str = str(direction).lower()
    if cluster_str not in deg["deg_full"]:
        raise KeyError(f"cluster {cluster_str!r} not found")
    if direction_str not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")

    full_records = deg["deg_full"][cluster_str]
    marker_records = deg["marker_tables"][cluster_str].get(direction_str, [])
    enr = deg["enrichment"][cluster_str].get(direction_str, {"go": [], "kegg": []})

    return {
        "cluster": cluster_str,
        "direction": direction_str,
        "thresholds": {"lfc": float(lfc_thresh), "padj": float(padj_thresh)},
        "expression_summary": deg["expression_summary"],
        "deg_overview": deg["deg_overview"],
        "volcano_figure": build_volcano_figure(full_records, cluster_str, lfc_thresh, padj_thresh, theme_mode),
        "go_figure": build_enrichment_bubble(enr.get("go", []),
                                              f"GO_BP · cluster {cluster_str} ({direction_str})",
                                              theme_mode),
        "kegg_figure": build_enrichment_bubble(enr.get("kegg", []),
                                                f"KEGG · cluster {cluster_str} ({direction_str})",
                                                theme_mode),
        "marker_table": marker_records[:50],
        "go_table": enr.get("go", [])[:25],
        "kegg_table": enr.get("kegg", [])[:25],
    }
