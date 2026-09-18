"""Module 3 — MI Cascade service.

Sub-milestone 5a: heavy permutation test + clustered MI×MI heatmap +
significant-pair table + per-pair celltype-triple stem plot.

Sub-milestones 5b/5c (in-situ rendering + DEG/GO) are appended to the same
module by later milestones.
"""
from __future__ import annotations

import hashlib
import html
import textwrap
import json
import pickle
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist

from core.datasets import DatasetPaths
from core.loaders import get_core_bundle


# ============================================================================
# Constants (cell 1 + cell 3 defaults)
# ============================================================================

DEFAULT_MI_THRESHOLD = 0.60
DEFAULT_ZSCORE_THRESHOLD = 1.40
DEFAULT_PADJ_THRESHOLD = 0.001
DEFAULT_NPERM = 100
DEFAULT_FDR_ALPHA = 0.05
DEFAULT_PROGRESS_EVERY = 5
DEFAULT_TRIPLE_PROP_THRESHOLD = 0.0
STEM_PROP_MIN = 0.01
STEM_TOPN = 20
CACHE_SCHEMA_VERSION = "v1_flask"


# ============================================================================
# Cache layout
# ============================================================================

def _module_dir(ds: DatasetPaths) -> Path:
    p = ds.run_dir / "UI_Exports" / "module3_mi_cascade_web"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_dir(ds: DatasetPaths) -> Path:
    p = _module_dir(ds) / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# In-memory caches
_RUN_CACHE: dict[str, dict[str, Any]] = {}     # cache_key -> analysis payload
_RUN_LOCK = Lock()


def _normalize_params(params: dict) -> dict:
    return {
        "MI_threshold": round(float(params.get("MI_threshold", DEFAULT_MI_THRESHOLD)), 6),
        "zscore_countcolocal_threshold": round(float(params.get("zscore_countcolocal_threshold", DEFAULT_ZSCORE_THRESHOLD)), 6),
        "pvalue_adjusted_threshold": round(float(params.get("pvalue_adjusted_threshold", DEFAULT_PADJ_THRESHOLD)), 8),
        "nperm": int(params.get("nperm", DEFAULT_NPERM)),
        "fdr_alpha": round(float(params.get("fdr_alpha", DEFAULT_FDR_ALPHA)), 6),
    }


def _params_cache_key(ds_name: str, params: dict) -> str:
    blob = json.dumps(
        {"schema": CACHE_SCHEMA_VERSION, "dataset": ds_name, "params": _normalize_params(params)},
        sort_keys=True,
    )
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


# ============================================================================
# Permutation test wrapper
# ============================================================================

def _ensure_dataframe(arr: Any, dim_envir: int) -> pd.DataFrame:
    """Coerce SpiderNet output (np.ndarray or DataFrame) to a labeled MI×MI DF."""
    if isinstance(arr, pd.DataFrame):
        return arr
    arr = np.asarray(arr)
    labels = [f"MI-{i + 1}" for i in range(dim_envir)]
    return pd.DataFrame(arr, index=labels, columns=labels)


def _build_significant_summary(z_df: pd.DataFrame, p_df: pd.DataFrame,
                                z_threshold: float, p_threshold: float) -> pd.DataFrame:
    """Filter MI×MI pairs by z and adj-P. Cutoff is `z > log2(z_threshold)`."""
    z_cutoff = float(np.log2(max(z_threshold, 1e-8)))
    rows = []
    for i, mi_first in enumerate(z_df.index):
        for j, mi_second in enumerate(z_df.columns):
            z_val = float(z_df.iloc[i, j])
            p_val = float(p_df.iloc[i, j])
            if (np.isfinite(z_val) and np.isfinite(p_val)
                    and z_val > z_cutoff and p_val < p_threshold):
                rows.append({
                    "MI_first": mi_first, "MI_second": mi_second,
                    "zscore_countcolocal": z_val, "pvalue_adjusted": p_val,
                })
    if not rows:
        return pd.DataFrame(columns=["MI_first", "MI_second", "zscore_countcolocal", "pvalue_adjusted"])
    out = pd.DataFrame(rows).sort_values(
        ["zscore_countcolocal", "pvalue_adjusted"], ascending=[False, True]
    ).reset_index(drop=True)
    return out


def _mi_id_from_label(label: str) -> int:
    """`'MI-7'` -> 7."""
    s = str(label).strip()
    if s.startswith("MI-"):
        s = s[3:]
    return int(s)


def _canonical_pair_key(mi_first: int | str, mi_second: int | str) -> str:
    a = _mi_id_from_label(mi_first)
    b = _mi_id_from_label(mi_second)
    return f"MI-{a} -> MI-{b}"


def _build_sig_df_numeric(sig_df: pd.DataFrame) -> pd.DataFrame:
    """Add `pair_key` + numeric `MI_first`/`MI_second` indices for downstream
    SpiderNet calls that expect 1-based ints."""
    out = sig_df.copy()
    if out.shape[0] == 0:
        out["pair_key"] = []
        out["MI_first_int"] = []
        out["MI_second_int"] = []
        return out
    out["pair_key"] = [_canonical_pair_key(a, b) for a, b in zip(out["MI_first"], out["MI_second"])]
    out["MI_first_int"] = out["MI_first"].apply(_mi_id_from_label).astype(int)
    out["MI_second_int"] = out["MI_second"].apply(_mi_id_from_label).astype(int)
    return out


# ----------------------------------------------------------------------------
# Triple proportions
# ----------------------------------------------------------------------------

def _extract_pair_series_from_pivot(
    prop_pivot: pd.DataFrame, sig_df_numeric: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """For each significant pair, extract the (celltype_triple, proportion) series
    from `prop_avg_allsample_filter_merge_pivot_select`. The pivot may be oriented
    as either pairs × triples (columns) or triples × pairs — try both.
    """
    if prop_pivot is None or prop_pivot.shape[0] == 0:
        return {}

    pair_keys = sig_df_numeric["pair_key"].astype(str).tolist()
    rows_str = [str(x) for x in prop_pivot.index]
    cols_str = [str(x) for x in prop_pivot.columns]

    def _try_extract(orientation: str) -> Optional[dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, Any]] = {}
        if orientation == "rows_pairs_cols_triples":
            for pk in pair_keys:
                if pk not in rows_str:
                    return None
                series = prop_pivot.loc[pk] if pk in prop_pivot.index else None
                if series is None:
                    return None
                pairs = list(zip(prop_pivot.columns.astype(str), pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy()))
                pairs.sort(key=lambda t: -float(t[1]))
                out[pk] = {
                    "celltype_triples": [t[0] for t in pairs],
                    "proportions": [float(t[1]) for t in pairs],
                    "n_celltype_triples": int(sum(1 for t in pairs if t[1] > 0)),
                }
            return out
        # cols are pairs, rows are triples
        for pk in pair_keys:
            if pk not in cols_str:
                return None
            series = prop_pivot[pk] if pk in prop_pivot.columns else None
            if series is None:
                return None
            pairs = list(zip(prop_pivot.index.astype(str), pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy()))
            pairs.sort(key=lambda t: -float(t[1]))
            out[pk] = {
                "celltype_triples": [t[0] for t in pairs],
                "proportions": [float(t[1]) for t in pairs],
                "n_celltype_triples": int(sum(1 for t in pairs if t[1] > 0)),
            }
        return out

    payload = _try_extract("rows_pairs_cols_triples")
    if payload is None:
        payload = _try_extract("cols_pairs_rows_triples")
    if payload is None:
        # Try with pair labels normalized through _canonical_pair_key (in case the
        # pivot uses a different MI label spelling)
        payload = {}
    return payload or {}


def _compute_celltype_triple_proportions(
    ds: DatasetPaths,
    bundle: dict[str, Any],
    coloc_result: dict[str, Any],
    sig_df: pd.DataFrame,
    mi_threshold: float,
) -> tuple[dict[str, dict[str, Any]], Optional[pd.DataFrame]]:
    from SpiderNet.analysis import summarize_mi_cascade_celltype_triples

    sig_df_numeric = _build_sig_df_numeric(sig_df)
    if len(sig_df_numeric) == 0:
        return {}, None

    sig_for_call = sig_df_numeric.drop(columns=["MI_first", "MI_second"]).rename(
        columns={"MI_first_int": "MI_first", "MI_second_int": "MI_second"}
    )
    cascade = summarize_mi_cascade_celltype_triples(
        MI_colocal_summary_significant=sig_for_call[
            ["zscore_countcolocal", "pvalue_adjusted", "MI_first", "MI_second"]
        ],
        Factor_envir_norm_list=coloc_result["Factor_envir_norm_list"],
        hyper_edge_adj_list=coloc_result["hyper_edge_adj_list"],
        SpiderNet_data_pyg_list=bundle["pyg_list"],
        adata_list=bundle["adata_list"],
        MI_threshold=float(mi_threshold),
        celltype_col=ds.config.get("CELL_TYPE_COL"),
        sample_col=ds.config.get("SAMPLE_ID_COL"),
        select_triple_prop_threshold=DEFAULT_TRIPLE_PROP_THRESHOLD,
        progress_every=max(1, DEFAULT_PROGRESS_EVERY),
    )

    prop_pivot = cascade.get("prop_avg_allsample_filter_merge_pivot_select")
    if not isinstance(prop_pivot, pd.DataFrame):
        # SpiderNet returned no usable pivot
        return {}, None
    payload = _extract_pair_series_from_pivot(prop_pivot, sig_df_numeric)
    return payload, prop_pivot


def run_cascade_analysis(
    ds: DatasetPaths,
    *,
    mi_threshold: float = DEFAULT_MI_THRESHOLD,
    zscore_countcolocal_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    pvalue_adjusted_threshold: float = DEFAULT_PADJ_THRESHOLD,
    nperm: int = DEFAULT_NPERM,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Heavy permutation analysis. Cached by (params, dataset)."""
    from SpiderNet.analysis import MI_colocalization_analysis

    warnings.filterwarnings("ignore")
    params = _normalize_params({
        "MI_threshold": mi_threshold,
        "zscore_countcolocal_threshold": zscore_countcolocal_threshold,
        "pvalue_adjusted_threshold": pvalue_adjusted_threshold,
        "nperm": nperm, "fdr_alpha": fdr_alpha,
    })
    cache_key = _params_cache_key(ds.name, params)
    cache_path = _cache_dir(ds) / f"mi_colocalization_{cache_key}.pkl"

    with _RUN_LOCK:
        if not force_recompute and cache_key in _RUN_CACHE:
            return _RUN_CACHE[cache_key]
    if not force_recompute and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            with _RUN_LOCK:
                _RUN_CACHE[cache_key] = payload
            return payload
        except Exception:
            pass

    print(f"[m3] Cascade analysis for {ds.name}: nperm={params['nperm']}, "
          f"MI_threshold={params['MI_threshold']}, z_thr={params['zscore_countcolocal_threshold']}, "
          f"padj<{params['pvalue_adjusted_threshold']}")
    bundle = get_core_bundle(ds)
    t0 = time.time()
    coloc = MI_colocalization_analysis(
        Factor_envir_list=bundle["factor_envir_list"],
        SpiderNet_data_pyg_list=bundle["pyg_list"],
        adata_list=bundle["adata_list"],
        MI_threshold=params["MI_threshold"],
        nperm=params["nperm"],
        fdr_alpha=params["fdr_alpha"],
        celltype_col=ds.config.get("CELL_TYPE_COL"),
        progress_every=DEFAULT_PROGRESS_EVERY,
        zscore_countcolocal_threshold=params["zscore_countcolocal_threshold"],
        pvalue_adjusted_threshold=params["pvalue_adjusted_threshold"],
    )
    t_coloc = time.time()

    z_df = _ensure_dataframe(coloc["colocal_count_merge_sum_zscore"], ds.dim_envir)
    p_df = _ensure_dataframe(coloc["colocal_count_merge_sum_pvalue_fdr"], ds.dim_envir)
    sig_df = _build_significant_summary(
        z_df, p_df,
        z_threshold=params["zscore_countcolocal_threshold"],
        p_threshold=params["pvalue_adjusted_threshold"],
    )

    print(f"[m3] {len(sig_df)} significant pairs after filtering")
    pair_payload, _ = _compute_celltype_triple_proportions(
        ds=ds, bundle=bundle, coloc_result=coloc, sig_df=sig_df,
        mi_threshold=params["MI_threshold"],
    )
    t_triples = time.time()

    # Build pair option list for the UI radio
    sig_df_numeric = _build_sig_df_numeric(sig_df)
    if not sig_df_numeric.empty:
        sig_df_numeric["n_celltype_triples"] = [
            int(pair_payload.get(pk, {}).get("n_celltype_triples", 0))
            for pk in sig_df_numeric["pair_key"]
        ]

    pair_options = [
        {
            "pair_key": str(row["pair_key"]),
            "mi_first": int(row["MI_first_int"]),
            "mi_second": int(row["MI_second_int"]),
            "zscore": float(row["zscore_countcolocal"]),
            "pvalue_adjusted": float(row["pvalue_adjusted"]),
            "n_celltype_triples": int(row.get("n_celltype_triples", 0)),
        }
        for _, row in sig_df_numeric.iterrows()
    ]

    payload = {
        "cache_key": cache_key,
        "params": params,
        "z_df": z_df,
        "p_df": p_df,
        "sig_df_numeric": sig_df_numeric,
        "pair_payload": pair_payload,
        "pair_options": pair_options,
        "default_pair_key": pair_options[0]["pair_key"] if pair_options else None,
        "elapsed_seconds": float(time.time() - t0),
        "coloc_seconds": float(t_coloc - t0),
        "triples_seconds": float(t_triples - t_coloc),
        "n_significant_pairs": int(len(sig_df_numeric)),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with _RUN_LOCK:
        _RUN_CACHE[cache_key] = payload
    return payload


def get_cached_analysis(cache_key: str) -> Optional[dict[str, Any]]:
    with _RUN_LOCK:
        return _RUN_CACHE.get(cache_key)


# ============================================================================
# Plotly figures
# ============================================================================

def _theme_spec(theme_mode: str = "dark") -> dict:
    if str(theme_mode).lower() == "light":
        return {"bg": "#ffffff", "text": "#111111", "grid": "rgba(17,17,17,0.10)"}
    return {"bg": "#000000", "text": "#ffffff", "grid": "rgba(255,255,255,0.10)"}


def _to_plotly_json(fig: go.Figure) -> dict:
    out = json.loads(fig.to_json(validate=False))
    out["layout"]["template"] = None
    return out


def _cluster_linkage(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Hierarchical clustering for the heatmap row/col ordering."""
    if matrix.shape[0] < 2:
        return None, list(range(matrix.shape[0]))
    finite = np.where(np.isfinite(matrix), matrix, 0.0)
    distances = pdist(finite, metric="euclidean")
    if not np.any(np.isfinite(distances)):
        return None, list(range(matrix.shape[0]))
    Z = linkage(distances, method="average", optimal_ordering=True)
    return Z, list(leaves_list(Z))


def _dendro_traces(Z: Optional[np.ndarray], orientation: str, theme: dict) -> list[go.Scatter]:
    """Build dendrogram traces aligned to the heatmap's integer leaf indices.

    scipy's `dendrogram` returns icoord = leaf-axis coordinates (leaves at
    5, 15, 25, … in conventional units) and dcoord = depth-axis coordinates.
    For top/bottom orientations the leaves run along x; for left/right
    orientations the leaves run along y. Two adjustments matter:

      1. Map icoord/dcoord onto x/y based on orientation, so the tree's
         branches grow perpendicular to the heatmap's matching axis.
      2. Rescale icoord from scipy's `5 + 10*i` convention to the integer
         leaf indices (0, 1, 2, …) that Plotly uses for category axes,
         so the dendrogram leaves line up with the heatmap rows/columns.
    """
    if Z is None:
        return []
    dendro = dendrogram(Z, orientation=orientation, no_plot=True, color_threshold=-1)
    is_horizontal = orientation in ("left", "right")
    traces = []
    for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"]):
        leaf_pos = [(c - 5.0) / 10.0 for c in icoord]    # 5,15,25 -> 0,1,2
        if is_horizontal:
            xs, ys = dcoord, leaf_pos
        else:
            xs, ys = leaf_pos, dcoord
        traces.append(go.Scatter(
            x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=theme["text"], width=1.0),
        ))
    return traces


def build_heatmap_figure(payload: dict[str, Any], theme_mode: str = "dark") -> dict:
    """Clustered MI×MI z-score heatmap with significant-pair markers."""
    theme = _theme_spec(theme_mode)
    z_df: pd.DataFrame = payload["z_df"]
    p_df: pd.DataFrame = payload["p_df"]
    sig_df = payload["sig_df_numeric"]

    z = z_df.values.astype(float)
    z = np.where(np.isfinite(z), z, 0.0)
    Z_row, row_order = _cluster_linkage(z)
    Z_col, col_order = _cluster_linkage(z.T)

    z_ord = z[np.ix_(row_order, col_order)]
    row_labels = [str(x) for x in z_df.index[row_order]]
    col_labels = [str(x) for x in z_df.columns[col_order]]

    # Build hover text including p-value
    p = p_df.values.astype(float)
    p_ord = p[np.ix_(row_order, col_order)]
    text = np.empty(z_ord.shape, dtype=object)
    for i in range(z_ord.shape[0]):
        for j in range(z_ord.shape[1]):
            text[i, j] = f"upstream MI: {row_labels[i]}<br>downstream MI: {col_labels[j]}<br>z: {z_ord[i, j]:.3f}<br>adj P: {p_ord[i, j]:.2e}"

    z_abs = float(np.nanmax(np.abs(z_ord))) if np.isfinite(z_ord).any() else 1.0
    z_abs = max(z_abs, 1e-3)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.13, 0.87],
        row_heights=[0.13, 0.87],
        horizontal_spacing=0.08,           # gap so dendrogram doesn't kiss labels
        vertical_spacing=0.015,
        shared_xaxes=False,
        shared_yaxes=False,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "heatmap"}]],
    )
    # column dendrogram (top right) — branches grow downward toward heatmap
    for tr in _dendro_traces(Z_col, "top", theme):
        fig.add_trace(tr, row=1, col=2)
    # row dendrogram (bottom left) — branches grow rightward toward heatmap.
    # We use `orientation="right"` so the trunk sits on the left of its
    # subplot and the leaves on the right (next to the heatmap labels).
    for tr in _dendro_traces(Z_row, "right", theme):
        fig.add_trace(tr, row=2, col=1)

    # heatmap
    fig.add_trace(go.Heatmap(
        z=z_ord.tolist(), x=col_labels, y=row_labels,
        text=text.tolist(), hoverinfo="text",
        colorscale="RdBu_r", zmin=-z_abs, zmax=z_abs,
        colorbar=dict(
            title=dict(text="z-score", font=dict(color=theme["text"], size=11)),
            tickfont=dict(color=theme["text"], size=10),
            outlinecolor=theme["text"], outlinewidth=1.0,
            x=1.02,
        ),
        xgap=0.5, ygap=0.5,
    ), row=2, col=2)

    # significant markers as `*`
    if not sig_df.empty:
        x_idx = {lab: i for i, lab in enumerate(col_labels)}
        y_idx = {lab: i for i, lab in enumerate(row_labels)}
        xs, ys = [], []
        for _, row in sig_df.iterrows():
            f, s = str(row["MI_first"]), str(row["MI_second"])
            if f in y_idx and s in x_idx:
                xs.append(col_labels[x_idx[s]])
                ys.append(row_labels[y_idx[f]])
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="text", text=["*"] * len(xs),
                showlegend=False, hoverinfo="skip",
                textfont=dict(color="#FDE047", size=18),
            ), row=2, col=2)

    n_mi = len(row_labels)
    h = max(540, 60 + n_mi * 28)
    fig.update_layout(
        autosize=True, height=h,
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=11),
        margin=dict(l=10, r=20, t=10, b=80),
        showlegend=False,
    )
    # Hide all dendrogram-subplot axes (ticks, gridlines, labels).
    for ax in ("xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"):
        if ax in fig.layout:
            try:
                fig.layout[ax].update(showgrid=False, zeroline=False, showticklabels=False, ticks="")
            except Exception:
                pass
    # Heatmap axes (xaxis4, yaxis4 in the 2x2 subplot grid).
    fig.update_xaxes(showticklabels=True, tickangle=55, color=theme["text"],
                     showgrid=False, automargin=True, row=2, col=2)
    fig.update_yaxes(showticklabels=True, color=theme["text"],
                     autorange="reversed", showgrid=False, automargin=True, row=2, col=2)

    # Match the row-dendrogram's y-range to the heatmap's leaf positions
    # (0..n-1, reversed because heatmap autorange is reversed). This is what
    # actually makes the branches line up with the rows.
    fig.update_yaxes(range=[n_mi - 0.5, -0.5], row=2, col=1)
    # scipy returns leaf depths at dcoord=0 and the trunk at max depth. By
    # default Plotly puts 0 on the left → leaves end up on the wrong side.
    # Reverse the row-dendrogram x-axis so dcoord=0 (leaves) sits on the
    # right edge, adjacent to the heatmap's y-axis labels.
    fig.update_xaxes(autorange="reversed", row=2, col=1)
    # Match the column-dendrogram's x-range to the heatmap's column positions.
    n_col = len(col_labels)
    fig.update_xaxes(range=[-0.5, n_col - 0.5], row=1, col=2)
    return _to_plotly_json(fig)


def build_pair_stem_figure(payload: dict[str, Any], pair_key: str, theme_mode: str = "dark", top_n: int = STEM_TOPN) -> dict:
    """For one significant pair, plot top-N celltype-triple proportions as a stem plot."""
    theme = _theme_spec(theme_mode)
    pair_payload = payload.get("pair_payload", {})
    triple_data = pair_payload.get(pair_key)
    if not triple_data:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text=f"No celltype triples for {pair_key}.",
                           showarrow=False, font=dict(color=theme["text"], size=14))
        fig.update_layout(paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          margin=dict(l=10, r=10, t=10, b=10), height=320)
        return _to_plotly_json(fig)

    triples = list(triple_data["celltype_triples"])
    proportions = list(triple_data["proportions"])
    pairs = [(t, p) for t, p in zip(triples, proportions) if p > STEM_PROP_MIN]
    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[:max(1, min(100, int(top_n)))]

    if not pairs:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text=f"No triples > {STEM_PROP_MIN:.2f} for {pair_key}.",
                           showarrow=False, font=dict(color=theme["text"], size=13))
        fig.update_layout(paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=320, margin=dict(l=10, r=10, t=10, b=10))
        return _to_plotly_json(fig)

    labels = [t for t, _ in pairs]
    values = [p for _, p in pairs]
    short_labels = [lab if len(lab) <= 36 else lab[:33] + "..." for lab in labels]
    x = np.arange(len(values), dtype=float)
    line_x = np.column_stack([x, x, np.full_like(x, np.nan)]).ravel().tolist()
    line_y = np.column_stack([np.zeros_like(values), values, np.full_like(values, np.nan)]).ravel().tolist()

    fig = go.Figure()
    fig.add_hline(y=0.0, line_width=1.0, line_color=theme["grid"])
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=theme["text"], width=2.0),
    ))
    fig.add_trace(go.Scatter(
        x=x.tolist(), y=values, mode="markers", showlegend=False,
        customdata=labels,
        marker=dict(size=11, color="#00F7FF", line=dict(color=theme["text"], width=1.0)),
        hovertemplate="<b>%{customdata}</b><br>proportion: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Cell-type triple proportions · {pair_key}", font=dict(color=theme["text"], size=12)),
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=11),
        margin=dict(l=60, r=20, t=40, b=160),
        height=420, autosize=True,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=x.tolist(), ticktext=short_labels,
        tickangle=42, color=theme["text"], showgrid=False, zeroline=False,
    )
    fig.update_yaxes(
        title="Avg proportion across slices", color=theme["text"],
        showgrid=True, gridcolor=theme["grid"], zeroline=False,
    )
    return _to_plotly_json(fig)


# ============================================================================
# Public API for routes
# ============================================================================

def build_run_response(payload: dict[str, Any], theme_mode: str = "dark") -> dict:
    sig_df = payload["sig_df_numeric"]
    sig_records = []
    if not sig_df.empty:
        for _, row in sig_df.iterrows():
            sig_records.append({
                "pair_key": str(row["pair_key"]),
                "mi_first": int(row["MI_first_int"]),
                "mi_second": int(row["MI_second_int"]),
                "zscore": round(float(row["zscore_countcolocal"]), 4),
                "pvalue_adjusted": float(row["pvalue_adjusted"]),
                "n_celltype_triples": int(row.get("n_celltype_triples", 0)),
            })
    return {
        "cache_key": payload["cache_key"],
        "params": payload["params"],
        "n_significant_pairs": int(payload["n_significant_pairs"]),
        "default_pair_key": payload.get("default_pair_key"),
        "pair_options": sig_records,
        "heatmap_figure": build_heatmap_figure(payload, theme_mode=theme_mode),
        "stem_figure": (
            build_pair_stem_figure(payload, payload["default_pair_key"], theme_mode=theme_mode)
            if payload.get("default_pair_key") else None
        ),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
        "coloc_seconds": float(payload.get("coloc_seconds", 0.0)),
        "triples_seconds": float(payload.get("triples_seconds", 0.0)),
    }


def build_stem_response(payload: dict[str, Any], pair_key: str, theme_mode: str = "dark", top_n: int = STEM_TOPN) -> dict:
    return {
        "pair_key": pair_key,
        "stem_figure": build_pair_stem_figure(payload, pair_key, theme_mode=theme_mode, top_n=top_n),
    }


# ============================================================================
# Sub-milestone 5b: in-situ rendering + DEG/GO
# ============================================================================

from matplotlib import colormaps as _colormaps
import matplotlib.colors as _mcolors
from scipy import stats as _stats

_SLICE_BUNDLE_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_INSITU_OPTIONS_CACHE: dict[str, dict[str, Any]] = {}
_PALETTE_CACHE: dict[str, dict[str, str]] = {}
_DEGGO_CACHE: dict[str, dict[str, Any]] = {}
_5B_LOCK = Lock()


# ---------------------------- Slice bundle ----------------------------------

def _edge_index_to_2_by_n(sp_data: Any) -> np.ndarray:
    import torch
    if hasattr(sp_data, "edge_index"):
        ei = sp_data.edge_index
    elif isinstance(sp_data, dict):
        ei = sp_data["edge_index"]
    else:
        raise AttributeError("no edge_index on slice")
    if torch.is_tensor(ei):
        ei = ei.detach().cpu().numpy()
    arr = np.asarray(ei)
    if arr.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {arr.shape}")
    if arr.shape[0] == 2:
        out = arr
    elif arr.shape[1] == 2:
        out = arr.T
    else:
        raise ValueError(f"edge_index shape {arr.shape}")
    return np.asarray(out, dtype=np.int64)


def _ensure_edge_factor_matrix(factor: Any, n_edges: int) -> np.ndarray:
    arr = np.asarray(factor, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"factor_envir must be 2D, got {arr.shape}")
    if arr.shape[0] == n_edges:
        out = arr
    elif arr.shape[1] == n_edges:
        out = arr.T
    else:
        raise ValueError(f"factor_envir shape {arr.shape} vs n_edges={n_edges}")
    return np.asarray(out, dtype=np.float32)


def _get_obs_vector(adata: Any, key: str, fallback: str) -> np.ndarray:
    try:
        if key and key in adata.obs.columns:
            return adata.obs[key].astype(str).to_numpy()
    except Exception:
        pass
    return np.asarray([str(fallback)] * int(adata.n_obs), dtype=object)


def _load_slice_bundle(ds: DatasetPaths, slice_index: int) -> dict[str, Any]:
    key = (ds.name, int(slice_index))
    with _5B_LOCK:
        if key in _SLICE_BUNDLE_CACHE:
            return _SLICE_BUNDLE_CACHE[key]

    bundle = get_core_bundle(ds)
    adata = bundle["adata_list"][slice_index]
    pyg_obj = bundle["pyg_list"][slice_index]
    factor = bundle["factor_envir_list"][slice_index]

    edge_index = _edge_index_to_2_by_n(pyg_obj)        # (2, n_edges)
    n_edges = int(edge_index.shape[1])
    factor_mat = _ensure_edge_factor_matrix(factor, n_edges)

    spatial_key = ds.config.get("SPATIAL_KEY", "spatial")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{spatial_key!r}] missing on slice {slice_index}")
    spatial = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]

    celltype_col = ds.config.get("CELL_TYPE_COL")
    sample_col = ds.config.get("SAMPLE_ID_COL")
    barcodes = pd.Index(adata.obs_names).astype(str).to_numpy()
    celltypes = _get_obs_vector(adata, celltype_col, fallback="Unknown")
    sample_vec = _get_obs_vector(adata, sample_col, fallback=f"slice_{slice_index}")
    sample_name = str(sample_vec[0]) if len(sample_vec) > 0 else f"slice_{slice_index}"

    sender_idx = edge_index[0].astype(np.int64)
    receiver_idx = edge_index[1].astype(np.int64)

    cells_df = pd.DataFrame({
        "cell_index": np.arange(int(adata.n_obs), dtype=int),
        "barcode": barcodes,
        "celltype": np.asarray(celltypes, dtype=object),
        "sample_name": np.asarray(sample_vec, dtype=object),
        "x": spatial[:, 0],
        "y": spatial[:, 1],
    })

    edge_df = pd.DataFrame({
        "edge_index": np.arange(n_edges, dtype=int),
        "sender_idx": sender_idx,
        "receiver_idx": receiver_idx,
        "sender_celltype": cells_df["celltype"].to_numpy()[sender_idx],
        "receiver_celltype": cells_df["celltype"].to_numpy()[receiver_idx],
        "sender_x": cells_df["x"].to_numpy(dtype=float)[sender_idx],
        "sender_y": cells_df["y"].to_numpy(dtype=float)[sender_idx],
        "receiver_x": cells_df["x"].to_numpy(dtype=float)[receiver_idx],
        "receiver_y": cells_df["y"].to_numpy(dtype=float)[receiver_idx],
    })

    out = {
        "slice_index": int(slice_index),
        "sample_name": sample_name,
        "cells_df": cells_df,
        "edge_df": edge_df,
        "factor_mat": factor_mat,
    }
    with _5B_LOCK:
        _SLICE_BUNDLE_CACHE[key] = out
    return out


def _get_global_palette(ds: DatasetPaths) -> dict[str, str]:
    with _5B_LOCK:
        if ds.name in _PALETTE_CACHE:
            return _PALETTE_CACHE[ds.name]
    bundle = get_core_bundle(ds)
    celltype_col = ds.config.get("CELL_TYPE_COL")
    seen: list[str] = []
    seen_set: set[str] = set()
    for adata in bundle["adata_list"]:
        for ct in _get_obs_vector(adata, celltype_col, "Unknown"):
            s = str(ct)
            if s not in seen_set:
                seen.append(s); seen_set.add(s)
    n = max(len(seen), 1)
    cmap = _colormaps["tab20"].resampled(n)
    colors = [_mcolors.to_hex(cmap(i)) for i in range(n)]
    palette = {ct: colors[i % len(colors)] for i, ct in enumerate(seen)}
    with _5B_LOCK:
        _PALETTE_CACHE[ds.name] = palette
    return palette


# ---------------------------- Triple counting ----------------------------------

def _count_pair_triples_in_slice(bundle: dict[str, Any], mi_first: int, mi_second: int, mi_threshold: float) -> Counter:
    factor_mat = bundle["factor_mat"]
    edge_df = bundle["edge_df"]
    cells_df = bundle["cells_df"]
    if mi_first < 1 or mi_second < 1 or mi_first > factor_mat.shape[1] or mi_second > factor_mat.shape[1]:
        raise IndexError("MI index out of bounds")

    mask_first = factor_mat[:, mi_first - 1] >= float(mi_threshold)
    mask_second = factor_mat[:, mi_second - 1] >= float(mi_threshold)
    idx_first = np.flatnonzero(mask_first)
    idx_second = np.flatnonzero(mask_second)
    if idx_first.size == 0 or idx_second.size == 0:
        return Counter()

    sender = edge_df["sender_idx"].to_numpy(dtype=int)
    receiver = edge_df["receiver_idx"].to_numpy(dtype=int)
    celltypes = cells_df["celltype"].astype(str).to_numpy()

    first_by_mid: dict[int, list[int]] = defaultdict(list)
    for e in idx_first.tolist():
        first_by_mid[int(receiver[e])].append(int(e))
    second_by_mid: dict[int, list[int]] = defaultdict(list)
    for e in idx_second.tolist():
        second_by_mid[int(sender[e])].append(int(e))

    counter: Counter = Counter()
    shared = set(first_by_mid.keys()).intersection(second_by_mid.keys())
    for mid in shared:
        ct2 = str(celltypes[mid])
        ct3_counter = Counter(str(celltypes[int(receiver[e2])]) for e2 in second_by_mid[mid])
        for e1 in first_by_mid[mid]:
            ct1 = str(celltypes[int(sender[e1])])
            for ct3, n2 in ct3_counter.items():
                counter[(ct1, ct2, ct3)] += int(n2)
    return counter


def summarize_pair_triples_for_insitu(
    ds: DatasetPaths, analysis_payload: dict[str, Any], pair_key: str
) -> dict[str, Any]:
    cache_token = f"{analysis_payload.get('cache_key', 'nocache')}::{pair_key}::insitu-summary"
    with _5B_LOCK:
        if cache_token in _INSITU_OPTIONS_CACHE:
            return _INSITU_OPTIONS_CACHE[cache_token]

    sig_df = analysis_payload["sig_df_numeric"]
    row = sig_df.loc[sig_df["pair_key"] == pair_key]
    if row.empty:
        raise KeyError(f"pair_key {pair_key!r} not in significant pairs")
    mi_first = int(row.iloc[0]["MI_first_int"])
    mi_second = int(row.iloc[0]["MI_second_int"])
    base_threshold = float(analysis_payload["params"]["MI_threshold"])

    bundle = get_core_bundle(ds)
    rows = []
    for slice_index in range(len(bundle["adata_list"])):
        sb = _load_slice_bundle(ds, slice_index)
        counter = _count_pair_triples_in_slice(sb, mi_first, mi_second, base_threshold)
        if not counter:
            continue
        sample_name = sb["sample_name"]
        for (ct1, ct2, ct3), count in counter.items():
            rows.append({
                "slice_index": int(slice_index),
                "sample_name": str(sample_name),
                "cell1_type": str(ct1), "cell2_type": str(ct2), "cell3_type": str(ct3),
                "triple_label": f"{ct1} → {ct2} → {ct3}",
                "count": int(count),
            })

    if rows:
        df = pd.DataFrame(rows).sort_values(
            ["count", "cell1_type", "cell2_type", "cell3_type", "slice_index"],
            ascending=[False, True, True, True, True],
        ).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["slice_index", "sample_name", "cell1_type", "cell2_type", "cell3_type", "triple_label", "count"])

    if df.shape[0] > 0:
        d = df.iloc[0]
        default_cell1, default_cell2, default_cell3 = str(d["cell1_type"]), str(d["cell2_type"]), str(d["cell3_type"])
        cell1_options = sorted(df["cell1_type"].unique().tolist())
        cell2_options = sorted(df["cell2_type"].unique().tolist())
        cell3_options = sorted(df["cell3_type"].unique().tolist())
    else:
        default_cell1 = default_cell2 = default_cell3 = ""
        cell1_options = cell2_options = cell3_options = []

    payload = {
        "pair_key": str(pair_key),
        "mi_first": mi_first, "mi_second": mi_second,
        "base_threshold": base_threshold,
        "cell1_options": cell1_options,
        "cell2_options": cell2_options,
        "cell3_options": cell3_options,
        "default_cell1": default_cell1,
        "default_cell2": default_cell2,
        "default_cell3": default_cell3,
        "summary_rows": df.to_dict(orient="records"),
        "n_summary_rows": int(df.shape[0]),
        "total_cascade_count": int(df["count"].sum()) if df.shape[0] > 0 else 0,
    }
    with _5B_LOCK:
        _INSITU_OPTIONS_CACHE[cache_token] = payload
    return payload


def build_insitu_options_payload(ds: DatasetPaths, analysis_payload: dict[str, Any], pair_key: str) -> dict[str, Any]:
    payload = summarize_pair_triples_for_insitu(ds, analysis_payload, pair_key)
    summary_rows = payload.get("summary_rows", [])
    d1, d2, d3 = payload["default_cell1"], payload["default_cell2"], payload["default_cell3"]
    matching = [r for r in summary_rows if r["cell1_type"] == d1 and r["cell2_type"] == d2 and r["cell3_type"] == d3]
    matching = sorted(matching, key=lambda r: (-int(r["count"]), int(r["slice_index"])))
    default_slice = int(matching[0]["slice_index"]) if matching else (
        int(summary_rows[0]["slice_index"]) if summary_rows else 0
    )
    out = dict(payload)
    out["default_slice"] = default_slice
    out["palette"] = _get_global_palette(ds)
    return out


# ---------------------------- Slice cascade enumerator ----------------------------------

def _extract_slice_cascade_instances(
    bundle: dict[str, Any], mi_first: int, mi_second: int, mi_threshold: float,
    cell1_type: str, cell2_type: str, cell3_type: str,
) -> pd.DataFrame:
    factor_mat = bundle["factor_mat"]
    edge_df = bundle["edge_df"]
    cells_df = bundle["cells_df"]

    sender = edge_df["sender_idx"].to_numpy(dtype=int)
    receiver = edge_df["receiver_idx"].to_numpy(dtype=int)
    celltypes = cells_df["celltype"].astype(str).to_numpy()

    mask_first = factor_mat[:, mi_first - 1] >= float(mi_threshold)
    mask_second = factor_mat[:, mi_second - 1] >= float(mi_threshold)
    idx_first = np.flatnonzero(mask_first)
    idx_second = np.flatnonzero(mask_second)
    if idx_first.size == 0 or idx_second.size == 0:
        return pd.DataFrame()

    first_by_mid: dict[int, list[int]] = defaultdict(list)
    for e in idx_first.tolist():
        mid = int(receiver[e])
        if str(celltypes[mid]) != str(cell2_type): continue
        if str(celltypes[int(sender[e])]) != str(cell1_type): continue
        first_by_mid[mid].append(int(e))
    second_by_mid: dict[int, list[int]] = defaultdict(list)
    for e in idx_second.tolist():
        mid = int(sender[e])
        if str(celltypes[mid]) != str(cell2_type): continue
        if str(celltypes[int(receiver[e])]) != str(cell3_type): continue
        second_by_mid[mid].append(int(e))

    records = []
    for mid in set(first_by_mid).intersection(second_by_mid):
        for e1 in first_by_mid[mid]:
            c1, c2 = int(sender[e1]), int(receiver[e1])
            v1 = float(factor_mat[e1, mi_first - 1])
            for e2 in second_by_mid[mid]:
                c3 = int(receiver[e2])
                v2 = float(factor_mat[e2, mi_second - 1])
                records.append({
                    "edge1_index": int(e1), "edge2_index": int(e2),
                    "cell1_index": c1, "cell2_index": c2, "cell3_index": c3,
                    "mi_first_value": v1, "mi_second_value": v2,
                })
    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["mi_first_value", "mi_second_value"], ascending=[False, False])
        .reset_index(drop=True)
    )


# ---------------------------- In-situ figure ----------------------------------

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _mcolors.to_rgb(str(hex_color))
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))},{a:.4f})"


def _arrow_segments(sx: np.ndarray, sy: np.ndarray, rx: np.ndarray, ry: np.ndarray, arrow_size: float):
    if sx.size == 0:
        return [], []
    dx = rx - sx; dy = ry - sy
    length = np.sqrt(dx ** 2 + dy ** 2)
    length = np.where(length <= 1e-12, 1e-12, length)
    ux = dx / length; uy = dy / length
    px = -uy; py = ux
    head_len = np.maximum(length * (0.06 + 0.03 * float(arrow_size)), 1.5)
    head_w = head_len * (0.55 + 0.45 * float(arrow_size))
    bx = rx - ux * head_len; by = ry - uy * head_len
    lx = bx + px * head_w * 0.5; ly = by + py * head_w * 0.5
    rx2 = bx - px * head_w * 0.5; ry2 = by - py * head_w * 0.5
    n = sx.size
    arrow_x = np.column_stack([lx, rx, rx2, np.full(n, np.nan)]).ravel().tolist()
    arrow_y = np.column_stack([ly, ry, ry2, np.full(n, np.nan)]).ravel().tolist()
    return arrow_x, arrow_y


def build_insitu_figure(
    ds: DatasetPaths,
    pair_payload: dict[str, Any],
    *,
    pair_key: str,
    cell1_type: str,
    cell2_type: str,
    cell3_type: str,
    slice_index: int,
    mi_threshold: float = DEFAULT_MI_THRESHOLD,
    cell_alpha: float = 0.82,
    cell_size: float = 6.0,
    edge_width: float = 1.2,
    arrow_size: float = 0.55,
    theme_mode: str = "dark",
) -> dict[str, Any]:
    theme = _theme_spec(theme_mode)
    sig_df = pair_payload["sig_df_numeric"]
    row = sig_df.loc[sig_df["pair_key"] == pair_key]
    if row.empty:
        raise KeyError(f"pair {pair_key!r} not significant")
    mi_first = int(row.iloc[0]["MI_first_int"])
    mi_second = int(row.iloc[0]["MI_second_int"])

    sb = _load_slice_bundle(ds, int(slice_index))
    cells_df = sb["cells_df"]
    edge_df = sb["edge_df"]
    palette = _get_global_palette(ds)

    instances = _extract_slice_cascade_instances(
        sb, mi_first, mi_second, mi_threshold, cell1_type, cell2_type, cell3_type,
    )

    fig = go.Figure()

    # All cells (background) — colored by celltype
    base_colors = cells_df["celltype"].astype(str).map(palette).fillna("#999999").tolist()
    fig.add_trace(go.Scattergl(
        x=cells_df["x"].astype(float).tolist(),
        y=cells_df["y"].astype(float).tolist(),
        mode="markers",
        marker=dict(size=cell_size * 0.6,
                    color=[_hex_to_rgba(c, cell_alpha * 0.6) for c in base_colors],
                    line=dict(width=0)),
        customdata=cells_df[["barcode", "celltype"]].astype(str).values.tolist(),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
        name="All cells", showlegend=False,
    ))

    if not instances.empty:
        # Highlighted cascade cells per position
        c1_idx = sorted(set(instances["cell1_index"].astype(int).tolist()))
        c2_idx = sorted(set(instances["cell2_index"].astype(int).tolist()))
        c3_idx = sorted(set(instances["cell3_index"].astype(int).tolist()))
        for label, idx, color in [
            (f"Cell1 ({cell1_type})", c1_idx, "#FF0087"),
            (f"Cell2 ({cell2_type})", c2_idx, "#FFCC00"),
            (f"Cell3 ({cell3_type})", c3_idx, "#00F7FF"),
        ]:
            if not idx:
                continue
            sub = cells_df.iloc[idx]
            fig.add_trace(go.Scattergl(
                x=sub["x"].astype(float).tolist(),
                y=sub["y"].astype(float).tolist(),
                mode="markers",
                name=label,
                marker=dict(size=cell_size * 1.45, color=color,
                            line=dict(color=theme["text"], width=1.0)),
                hovertemplate=f"<b>{label}</b><br>%{{customdata[0]}}<extra></extra>",
                customdata=sub[["barcode"]].astype(str).values.tolist(),
            ))

        # Edges: upstream (e1) and downstream (e2)
        for role, e_col, color in [("upstream", "edge1_index", "#FF0087"),
                                    ("downstream", "edge2_index", "#00F7FF")]:
            e_indices = instances[e_col].astype(int).unique().tolist()
            if not e_indices:
                continue
            es = edge_df.iloc[e_indices]
            sx = es["sender_x"].to_numpy(dtype=float); sy = es["sender_y"].to_numpy(dtype=float)
            rx = es["receiver_x"].to_numpy(dtype=float); ry = es["receiver_y"].to_numpy(dtype=float)
            line_x = np.column_stack([sx, rx, np.full_like(sx, np.nan)]).ravel().tolist()
            line_y = np.column_stack([sy, ry, np.full_like(sy, np.nan)]).ravel().tolist()
            fig.add_trace(go.Scattergl(
                x=line_x, y=line_y, mode="lines",
                line=dict(color=_hex_to_rgba(color, 0.55), width=edge_width),
                name=f"{role} (MI-{mi_first if role=='upstream' else mi_second})",
                hoverinfo="skip", showlegend=True,
            ))
            ax, ay = _arrow_segments(sx, sy, rx, ry, arrow_size)
            fig.add_trace(go.Scattergl(
                x=ax, y=ay, mode="lines",
                line=dict(color=_hex_to_rgba(color, 0.85), width=max(edge_width * 0.95, 0.5)),
                name=f"{role} arrows", hoverinfo="skip", showlegend=False,
            ))

    # Geometry
    x_arr = cells_df["x"].to_numpy(dtype=float)
    y_arr = cells_df["y"].to_numpy(dtype=float)
    pad_frac = 0.05
    x_pad = max((x_arr.max() - x_arr.min()) * pad_frac, 40.0)
    y_pad = max((y_arr.max() - y_arr.min()) * pad_frac, 40.0)
    x_range = [x_arr.min() - x_pad, x_arr.max() + x_pad]
    y_range = [y_arr.max() + y_pad, y_arr.min() - y_pad]

    fig.update_layout(
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=11),
        autosize=True, height=720,
        margin=dict(l=10, r=10, t=10, b=80),
        legend=dict(orientation="h", x=0, y=-0.05, font=dict(color=theme["text"], size=10),
                    bgcolor="rgba(0,0,0,0)"),
        dragmode="pan",
        hovermode="closest",
    )
    fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False,
                     showline=False, range=x_range, constrain="domain")
    fig.update_yaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False,
                     showline=False, range=y_range, scaleanchor="x", scaleratio=1, constrain="domain")

    return {
        "figure": _to_plotly_json(fig),
        "meta": {
            "pair_key": str(pair_key),
            "mi_first": mi_first, "mi_second": mi_second,
            "slice_index": int(slice_index),
            "sample_name": sb["sample_name"],
            "cell1_type": str(cell1_type),
            "cell2_type": str(cell2_type),
            "cell3_type": str(cell3_type),
            "n_cascade_instances": int(instances.shape[0]),
            "n_cell1": int(len(set(instances["cell1_index"].tolist())) if not instances.empty else 0),
            "n_cell2": int(len(set(instances["cell2_index"].tolist())) if not instances.empty else 0),
            "n_cell3": int(len(set(instances["cell3_index"].tolist())) if not instances.empty else 0),
        },
    }


# ---------------------------- DEG + GO ----------------------------------

DEG_MIN_CELLS_PER_GROUP = 3
DEFAULT_DEG_LFC = 0.5
DEFAULT_DEG_PADJ = 0.05
DEFAULT_GO_TOPN = 20
DEFAULT_GO_GENESETS = ("GO_Biological_Process_2021",)


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.ones_like(p, dtype=float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return out
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    n = ranked.size
    adj = ranked * n / np.arange(1, n + 1, dtype=float)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[finite] = tmp
    return out


def _normalize_total_log1p(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X.copy(), X.copy()
    totals = X.sum(axis=1, keepdims=True)
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    Xn = X * scale
    Xlog = np.log1p(Xn)
    return Xn, Xlog


def _extract_matrix(adata, indices: list[int]) -> np.ndarray:
    import scipy.sparse as _sp
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return np.zeros((0, int(adata.n_vars)), dtype=float)
    src = None
    if hasattr(adata, "layers") and "counts" in getattr(adata, "layers", {}):
        try:
            src = adata.layers["counts"]
        except Exception:
            src = None
    if src is None:
        src = adata.X
    X = src[idx]
    if _sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _collect_group_expression(ds: DatasetPaths, cell_df: pd.DataFrame):
    if cell_df is None or cell_df.empty:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.asarray([], dtype=object)
    bundle = get_core_bundle(ds)
    parts_norm, parts_log = [], []
    gene_ref = None
    for slice_index, sub in cell_df.groupby("slice_index", sort=True):
        adata = bundle["adata_list"][int(slice_index)]
        X = _extract_matrix(adata, sub["cell_index"].astype(int).tolist())
        gene_cur = adata.var_names.astype(str).to_numpy()
        if gene_ref is None:
            gene_ref = gene_cur
        elif len(gene_cur) != len(gene_ref) or np.any(gene_cur != gene_ref):
            raise ValueError("Gene order differs across slices")
        Xn, Xlog = _normalize_total_log1p(X)
        parts_norm.append(Xn); parts_log.append(Xlog)
    if gene_ref is None:
        gene_ref = np.asarray([], dtype=object)
    Xn = np.vstack(parts_norm) if parts_norm else np.zeros((0, len(gene_ref)))
    Xlog = np.vstack(parts_log) if parts_log else np.zeros((0, len(gene_ref)))
    return Xn, Xlog, gene_ref


def _collect_triplet_cells_across_slices(
    ds: DatasetPaths, mi_first: int, mi_second: int, mi_threshold: float,
    cell1_type: str, cell2_type: str, cell3_type: str,
    upstream_active: bool, downstream_active: bool,
) -> dict[str, Any]:
    bundle = get_core_bundle(ds)
    pos_frames: dict[int, list[pd.DataFrame]] = {1: [], 2: [], 3: []}
    total_triplets = 0
    n_slices_with_hits = 0

    for slice_index in range(len(bundle["adata_list"])):
        sb = _load_slice_bundle(ds, int(slice_index))
        factor_mat = sb["factor_mat"]
        edge_df = sb["edge_df"]
        cells_df = sb["cells_df"]

        sender_arr = edge_df["sender_idx"].to_numpy(dtype=int)
        receiver_arr = edge_df["receiver_idx"].to_numpy(dtype=int)
        sender_ct = edge_df["sender_celltype"].astype(str).to_numpy()
        receiver_ct = edge_df["receiver_celltype"].astype(str).to_numpy()

        v1 = factor_mat[:, mi_first - 1]
        v2 = factor_mat[:, mi_second - 1]
        m1 = (v1 >= float(mi_threshold)) if upstream_active else (v1 < float(mi_threshold))
        m2 = (v2 >= float(mi_threshold)) if downstream_active else (v2 < float(mi_threshold))
        m1 &= (sender_ct == str(cell1_type)) & (receiver_ct == str(cell2_type))
        m2 &= (sender_ct == str(cell2_type)) & (receiver_ct == str(cell3_type))

        idx1 = np.flatnonzero(m1); idx2 = np.flatnonzero(m2)
        if idx1.size == 0 or idx2.size == 0:
            continue

        first_by_mid: dict[int, list[int]] = defaultdict(list)
        for e in idx1.tolist():
            first_by_mid[int(receiver_arr[e])].append(int(e))
        second_by_mid: dict[int, list[int]] = defaultdict(list)
        for e in idx2.tolist():
            second_by_mid[int(sender_arr[e])].append(int(e))
        shared = set(first_by_mid).intersection(second_by_mid)
        if not shared:
            continue

        c1_set, c2_set, c3_set = set(), set(), set()
        for mid in shared:
            e1s, e2s = first_by_mid[mid], second_by_mid[mid]
            if not e1s or not e2s: continue
            c2_set.add(int(mid))
            for e1 in e1s: c1_set.add(int(sender_arr[e1]))
            for e2 in e2s: c3_set.add(int(receiver_arr[e2]))
            total_triplets += int(len(e1s) * len(e2s))

        if not c1_set and not c2_set and not c3_set:
            continue
        n_slices_with_hits += 1
        for pos, indices in [(1, c1_set), (2, c2_set), (3, c3_set)]:
            if not indices: continue
            sub = cells_df.iloc[sorted(indices)].copy()
            sub.insert(0, "slice_index", int(slice_index))
            sub["cell_key"] = sub["slice_index"].astype(str) + "::" + sub["cell_index"].astype(int).astype(str)
            sub["position"] = int(pos)
            pos_frames[pos].append(sub[["slice_index", "cell_index", "barcode", "celltype",
                                         "x", "y", "position", "cell_key"]])

    out_pos = {}
    for pos in (1, 2, 3):
        if pos_frames[pos]:
            df = pd.concat(pos_frames[pos], axis=0, ignore_index=True)
            df = df.drop_duplicates(subset=["cell_key"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["slice_index", "cell_index", "barcode", "celltype", "x", "y", "position", "cell_key"])
        out_pos[pos] = df

    return {
        "position_dfs": out_pos,
        "n_triplets": int(total_triplets),
        "n_slices_with_hits": int(n_slices_with_hits),
    }


def _exclude_interest_from_baseline(baseline_df: pd.DataFrame, interest_df: pd.DataFrame) -> pd.DataFrame:
    if baseline_df is None or baseline_df.empty:
        return baseline_df.copy() if isinstance(baseline_df, pd.DataFrame) else pd.DataFrame()
    if interest_df is None or interest_df.empty:
        return baseline_df.copy().reset_index(drop=True)
    keys = set(interest_df["cell_key"].astype(str))
    return baseline_df.loc[~baseline_df["cell_key"].astype(str).isin(keys)].reset_index(drop=True)


def _run_deg(ds: DatasetPaths, interest_df: pd.DataFrame, baseline_df: pd.DataFrame, lfc_thresh: float, padj_thresh: float) -> dict[str, Any]:
    n_int = 0 if interest_df is None else int(interest_df.shape[0])
    n_base = 0 if baseline_df is None else int(baseline_df.shape[0])
    if n_int < DEG_MIN_CELLS_PER_GROUP or n_base < DEG_MIN_CELLS_PER_GROUP:
        return {"ok": False,
                "message": f"need ≥{DEG_MIN_CELLS_PER_GROUP} cells per group; have interest={n_int}, baseline={n_base}",
                "deg_df": pd.DataFrame(), "up_df": pd.DataFrame(), "down_df": pd.DataFrame(),
                "n_interest": n_int, "n_baseline": n_base}

    X1n, X1log, genes = _collect_group_expression(ds, interest_df)
    X2n, X2log, genes2 = _collect_group_expression(ds, baseline_df)
    if X1n.shape[1] == 0 or X2n.shape[1] == 0:
        return {"ok": False, "message": "no genes after extraction",
                "deg_df": pd.DataFrame(), "up_df": pd.DataFrame(), "down_df": pd.DataFrame(),
                "n_interest": n_int, "n_baseline": n_base}
    if len(genes) != len(genes2) or np.any(genes != genes2):
        raise ValueError("gene names differ between interest and baseline")

    try:
        t = _stats.ttest_ind(X1log, X2log, axis=0, equal_var=False, nan_policy="omit")
        pvals = np.asarray(t.pvalue, dtype=float)
    except Exception:
        pvals = np.ones(X1log.shape[1], dtype=float)
        for j in range(X1log.shape[1]):
            try:
                _, p = _stats.mannwhitneyu(X1log[:, j], X2log[:, j], alternative="two-sided")
            except Exception:
                p = 1.0
            pvals[j] = float(p)
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    mean1 = X1n.mean(axis=0); mean2 = X2n.mean(axis=0)
    log2fc = np.log2((mean1 + 1e-8) / (mean2 + 1e-8))
    padj = _bh_adjust(pvals)

    deg_df = pd.DataFrame({
        "gene": np.asarray(genes, dtype=object),
        "log2fc": log2fc, "pvalue": pvals, "padj": padj,
        "pct_interest": (X1n > 0).mean(axis=0),
        "pct_baseline": (X2n > 0).mean(axis=0),
        "mean_interest": mean1, "mean_baseline": mean2,
    }).sort_values(["padj", "pvalue", "log2fc"], ascending=[True, True, False]).reset_index(drop=True)

    sig = deg_df.loc[(deg_df["padj"] <= float(padj_thresh)) & (np.abs(deg_df["log2fc"]) >= float(lfc_thresh))]
    up_df = sig.loc[sig["log2fc"] >= float(lfc_thresh)].sort_values(["log2fc", "padj"], ascending=[False, True]).reset_index(drop=True)
    down_df = sig.loc[sig["log2fc"] <= -float(lfc_thresh)].sort_values(["log2fc", "padj"], ascending=[True, True]).reset_index(drop=True)

    return {"ok": True, "deg_df": deg_df, "up_df": up_df, "down_df": down_df,
            "n_interest": n_int, "n_baseline": n_base,
            "message": f"interest={n_int}, baseline={n_base}, up={int(up_df.shape[0])}, down={int(down_df.shape[0])}"}


def _run_go(genes: list[str], organism: str, gene_sets: tuple = DEFAULT_GO_GENESETS, top_n: int = DEFAULT_GO_TOPN) -> dict[str, Any]:
    from modules.module2_subtype.service import _fetch_enrichment
    records, statuses = _fetch_enrichment(genes, ["go"], organism, top_n,
                                          library_candidates={"go": list(gene_sets)})
    status = statuses["go"]
    message = status.get("message", "Enrichr returned no terms." if status["state"] == "empty" else f"Top {len(records['go'])} terms ({organism})")
    message = message.replace("Compute DEG + GO/KEGG", "Compute DEG + GO")
    return {"ok": status["state"] == "ok", "state": status["state"], "message": message,
            "df": pd.DataFrame(records["go"]), "organism_used": organism, "library": status.get("library")}


def _build_volcano(deg_df: pd.DataFrame, lfc_thresh: float, padj_thresh: float, title: str, theme: dict) -> dict:
    if deg_df is None or deg_df.empty:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="No DEG data.", showarrow=False, font=dict(color=theme["text"]))
        fig.update_layout(paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=320, title=dict(text=title, font=dict(color=theme["text"], size=12)))
        return _to_plotly_json(fig)

    df = deg_df.copy()
    df["lfc"] = pd.to_numeric(df["log2fc"], errors="coerce")
    df["nlp"] = np.minimum(-np.log10(np.clip(pd.to_numeric(df["padj"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1e-300, 1.0)), 10.0)
    df["padj_v"] = pd.to_numeric(df["padj"], errors="coerce")
    df = df.dropna(subset=["lfc", "nlp"])

    up = (df["lfc"] >= lfc_thresh) & (df["padj_v"] <= padj_thresh)
    down = (df["lfc"] <= -lfc_thresh) & (df["padj_v"] <= padj_thresh)
    other = ~(up | down)

    fig = go.Figure()
    for sub, name, color in [(df[other], "n.s.", "rgba(150,150,150,0.5)"),
                              (df[down], "down", "rgba(60,130,200,0.85)"),
                              (df[up], "up", "rgba(220,80,80,0.85)")]:
        if sub.empty: continue
        fig.add_trace(go.Scattergl(
            x=sub["lfc"].astype(float).tolist(), y=sub["nlp"].astype(float).tolist(),
            mode="markers", name=f"{name} (n={len(sub)})",
            customdata=sub["gene"].astype(str).tolist(),
            marker=dict(size=4.5, color=color, line=dict(width=0)),
            hovertemplate="<b>%{customdata}</b><br>log2FC: %{x:.2f}<br>-log10 padj: %{y:.2f}<extra></extra>",
        ))
    fig.add_vline(x=lfc_thresh, line_dash="dot", line_color=theme["text"], opacity=0.4)
    fig.add_vline(x=-lfc_thresh, line_dash="dot", line_color=theme["text"], opacity=0.4)
    if padj_thresh > 0:
        fig.add_hline(y=-np.log10(padj_thresh), line_dash="dot", line_color=theme["text"], opacity=0.4)

    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["text"], size=12)),
        autosize=True, height=380,
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=10),
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(font=dict(color=theme["text"], size=9), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title="log2 fold change", color=theme["text"], gridcolor=theme["grid"])
    fig.update_yaxes(title="-log10 adj p", color=theme["text"], gridcolor=theme["grid"])
    return _to_plotly_json(fig)


def _build_go_bubble(go_df: pd.DataFrame, title: str, theme: dict, message: str = "No enriched GO terms.") -> dict:
    if go_df is None or go_df.empty:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="<br>".join(html.escape(line) for line in textwrap.wrap(message, 55)), showarrow=False, font=dict(color=theme["text"]))
        fig.update_layout(paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=320, title=dict(text=title, font=dict(color=theme["text"], size=12)))
        return _to_plotly_json(fig)

    df = go_df.copy()
    df["padj"] = pd.to_numeric(df.get("Adjusted P-value", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    df["neglog10"] = -np.log10(np.clip(df["padj"].to_numpy(dtype=float), 1e-300, 1.0))
    if "Overlap" in df.columns:
        df["count"] = df["Overlap"].astype(str).str.extract(r"^(\d+)").fillna(0).astype(int)
    elif "Gene count" in df.columns:
        df["count"] = df["Gene count"]
    else:
        df["count"] = 1
    df = df.iloc[::-1].reset_index(drop=True)

    terms = df.get("Term", pd.Series([""] * len(df))).astype(str).tolist()
    short = [t if len(t) <= 60 else t[:57] + "..." for t in terms]
    sizes = np.clip(df["count"].to_numpy(dtype=float) * 1.4, 6, 28)

    fig = go.Figure(go.Scatter(
        x=df["neglog10"].astype(float).tolist(),
        y=short, mode="markers",
        marker=dict(size=sizes.tolist(),
                    color=df["neglog10"].astype(float).tolist(),
                    colorscale="Viridis",
                    colorbar=dict(title=dict(text="-log10 padj", font=dict(color=theme["text"], size=10)),
                                  tickfont=dict(color=theme["text"], size=9),
                                  outlinecolor=theme["text"], outlinewidth=1.0),
                    line=dict(width=0)),
        customdata=np.column_stack([df["count"].astype(int).to_numpy(),
                                     df["padj"].astype(float).to_numpy(),
                                     terms]).tolist(),
        hovertemplate="<b>%{customdata[2]}</b><br>genes: %{customdata[0]}<br>padj: %{customdata[1]:.2e}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["text"], size=12)),
        autosize=True, height=max(280, 60 + len(df) * 20),
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(color=theme["text"], size=10),
        margin=dict(l=10, r=20, t=40, b=40),
    )
    fig.update_xaxes(title="-log10 adj p", color=theme["text"], gridcolor=theme["grid"])
    fig.update_yaxes(color=theme["text"], showgrid=False, automargin=True)
    return _to_plotly_json(fig)


def run_cascade_deggo(
    ds: DatasetPaths,
    pair_payload: dict[str, Any],
    *,
    pair_key: str,
    cell1_type: str,
    cell2_type: str,
    cell3_type: str,
    mi_threshold: float = DEFAULT_MI_THRESHOLD,
    baseline_upstream_active: bool = False,
    baseline_downstream_active: bool = True,
    lfc_thresh: float = DEFAULT_DEG_LFC,
    padj_thresh: float = DEFAULT_DEG_PADJ,
    gene_direction: str = "up",
    top_n_terms: int = DEFAULT_GO_TOPN,
    theme_mode: str = "dark",
) -> dict[str, Any]:
    direction = "down" if str(gene_direction).lower() == "down" else "up"
    organism = str(ds.config.get("SPECIES", "human"))

    cache_token = json.dumps({
        "dataset": ds.name, "cache": pair_payload.get("cache_key"),
        "pair": pair_key, "c1": cell1_type, "c2": cell2_type, "c3": cell3_type,
        "mi": round(float(mi_threshold), 6),
        "ub": bool(baseline_upstream_active), "db": bool(baseline_downstream_active),
        "lfc": round(float(lfc_thresh), 6), "padj": round(float(padj_thresh), 8),
        "dir": direction, "topn": int(top_n_terms), "theme": theme_mode,
    }, sort_keys=True)
    cache_key = hashlib.md5(cache_token.encode("utf-8")).hexdigest()
    with _5B_LOCK:
        cached = _DEGGO_CACHE.get(cache_key)
    if cached is not None:
        # Retry only failed GO requests; preserve the existing DEG calculations.
        for position in cached["positions"].values():
            if position.get("go_state") == "error":
                result = _run_go(position["enrichment_genes"], organism=organism, top_n=top_n_terms)
                position.update(go_state=result["state"], go_message=result["message"], go_library=result.get("library"))
                position["go_figure"] = _build_go_bubble(result["df"], f"GO ({direction}) · {position['label']}", _theme_spec(theme_mode), message=result["message"])
        return cached

    sig_df = pair_payload["sig_df_numeric"]
    row = sig_df.loc[sig_df["pair_key"] == pair_key]
    if row.empty:
        raise KeyError(f"pair {pair_key!r} not found")
    mi_first = int(row.iloc[0]["MI_first_int"])
    mi_second = int(row.iloc[0]["MI_second_int"])

    interest = _collect_triplet_cells_across_slices(
        ds, mi_first, mi_second, mi_threshold, cell1_type, cell2_type, cell3_type,
        upstream_active=True, downstream_active=True,
    )
    baseline = _collect_triplet_cells_across_slices(
        ds, mi_first, mi_second, mi_threshold, cell1_type, cell2_type, cell3_type,
        upstream_active=baseline_upstream_active, downstream_active=baseline_downstream_active,
    )

    theme = _theme_spec(theme_mode)
    positions: dict[int, dict[str, Any]] = {}
    for pos, ct in zip((1, 2, 3), (cell1_type, cell2_type, cell3_type)):
        i_df = interest["position_dfs"].get(pos, pd.DataFrame())
        b_df = baseline["position_dfs"].get(pos, pd.DataFrame())
        b_filtered = _exclude_interest_from_baseline(b_df, i_df)
        deg = _run_deg(ds, i_df, b_filtered, lfc_thresh=lfc_thresh, padj_thresh=padj_thresh)
        chosen_genes = (deg["up_df"]["gene"].astype(str).tolist() if direction == "up"
                        else deg["down_df"]["gene"].astype(str).tolist())
        go_res = _run_go(chosen_genes, organism=organism, top_n=top_n_terms)

        position_label = f"Cell{pos} · {ct}"
        positions[pos] = {
            "position": int(pos), "celltype": str(ct), "label": position_label,
            "interest_n": int(i_df.shape[0]) if isinstance(i_df, pd.DataFrame) else 0,
            "baseline_n": int(b_df.shape[0]) if isinstance(b_df, pd.DataFrame) else 0,
            "filtered_baseline_n": int(b_filtered.shape[0]) if isinstance(b_filtered, pd.DataFrame) else 0,
            "deg_message": deg["message"],
            "n_up": int(deg["up_df"].shape[0]),
            "n_down": int(deg["down_df"].shape[0]),
            "volcano_figure": _build_volcano(
                deg["deg_df"], lfc_thresh=lfc_thresh, padj_thresh=padj_thresh,
                title=f"Volcano · {position_label}", theme=theme,
            ),
            "go_figure": _build_go_bubble(
                go_res["df"], title=f"GO ({direction}) · {position_label}", theme=theme, message=go_res["message"],
            ),
            "go_message": go_res["message"],
            "go_state": go_res["state"], "go_library": go_res.get("library"),
            "enrichment_genes": chosen_genes,
            "top_genes": (deg["up_df"] if direction == "up" else deg["down_df"]).head(20).to_dict(orient="records"),
        }

    payload = {
        "pair_key": str(pair_key), "mi_first": mi_first, "mi_second": mi_second,
        "cell_types": {"cell1": str(cell1_type), "cell2": str(cell2_type), "cell3": str(cell3_type)},
        "mi_threshold": float(mi_threshold),
        "baseline_upstream_active": bool(baseline_upstream_active),
        "baseline_downstream_active": bool(baseline_downstream_active),
        "interest_triplets": int(interest["n_triplets"]),
        "interest_slices": int(interest["n_slices_with_hits"]),
        "baseline_triplets": int(baseline["n_triplets"]),
        "gene_direction": direction,
        "positions": positions,
        "organism": organism,
    }
    with _5B_LOCK:
        _DEGGO_CACHE[cache_key] = payload
    return payload


# ---------------------------- Public route helpers ----------------------------------

def build_insitu_response(
    ds: DatasetPaths, pair_payload: dict[str, Any],
    *, pair_key: str, cell1_type: str, cell2_type: str, cell3_type: str,
    slice_index: int, mi_threshold: float = DEFAULT_MI_THRESHOLD,
    cell_alpha: float = 0.82, cell_size: float = 6.0,
    edge_width: float = 1.2, arrow_size: float = 0.55,
    theme_mode: str = "dark",
) -> dict[str, Any]:
    return build_insitu_figure(
        ds=ds, pair_payload=pair_payload,
        pair_key=pair_key,
        cell1_type=cell1_type, cell2_type=cell2_type, cell3_type=cell3_type,
        slice_index=slice_index, mi_threshold=mi_threshold,
        cell_alpha=cell_alpha, cell_size=cell_size,
        edge_width=edge_width, arrow_size=arrow_size,
        theme_mode=theme_mode,
    )
