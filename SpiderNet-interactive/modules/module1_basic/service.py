"""Module 1 — Basic Analysis service.

Ports the spatial-canvas + top-loadings pipeline from
`spidernet_interactive_module1_basicanalysis_web.ipynb` (cells 3 + 5 + 8).

On first request for a dataset:
  1. Build a per-slice export bundle (cells.csv.gz / edges_wide.csv.gz / summary.json)
     under <run_dir>/UI_Exports/module1_in_situ_professional/, keyed by export config.
  2. Cache loaded slice tables in memory.

Each spatial render fetches the cached slice tables, applies the user's filter
controls, and returns a Plotly JSON figure spec. Top-loadings (LR pairs / sender
genes / receiver genes) are computed from the run-dir loading_*_use.npy matrices.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.datasets import DatasetPaths
from core.loaders import edge_index_array, get_core_bundle


# ============================================================================
# Constants (lifted from notebook cell 1)
# ============================================================================

EDGE_MIN_EXPORT_THRESHOLD = 0.0
ROUND_DECIMALS = 6

DEFAULT_MAX_EDGES_DISPLAY = 12000
DEFAULT_CELL_ALPHA = 0.90
DEFAULT_EDGE_WIDTH = 1.2
DEFAULT_ARROW_SIZE = 0.55
DEFAULT_MARKER_SIZE = 6
EDGE_COLOR_QUANTILE_CLIP = 0.995

BASE_FONT_SIZE = 16
SMALL_FONT_SIZE = 14

# Dark-mode edge colormap (white -> cyan). Use Plotly's Reds-like for light mode.
EDGE_COLOR_MAP_DARK = [(0.0, "#FFFFFF"), (0.55, "#B0FFFA"), (1.0, "#00F7FF")]
EDGE_COLOR_MAP_LIGHT = "YlOrRd"


# ============================================================================
# Color helpers (cell 5)
# ============================================================================

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    rgb = mcolors.to_rgb(hex_color)
    rgba = tuple(int(round(v * 255)) for v in rgb) + (alpha,)
    return f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]:.3f})"


def get_colorscale_hex_list(colorscale, n: int = 6) -> list[str]:
    if isinstance(colorscale, (list, tuple)):
        if len(colorscale) == 0:
            return ["#FFFFFF"] * n
        first = colorscale[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom_edge_map", [(float(p), c) for p, c in colorscale]
            )
        else:
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_edge_map", list(colorscale))
        sample_points = np.linspace(0.0, 1.0, n)
        return [mcolors.to_hex(cmap(v)) for v in sample_points]
    cmap = plt.get_cmap(colorscale, n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


# ============================================================================
# Theme spec (cell 8)
# ============================================================================

def theme_spec(theme_mode: str = "dark") -> dict:
    if str(theme_mode).lower() == "light":
        return {
            "mode": "light",
            "bg": "#ffffff",
            "text": "#111111",
            "border_solid": "#1f2937",
            "hover_bg": "rgba(255,255,255,0.97)",
            "hover_border": "rgba(17,17,17,0.20)",
            "hover_font": "#111111",
            "grid": "rgba(17,17,17,0.10)",
            "edge_scale": EDGE_COLOR_MAP_LIGHT,
            "edge_alpha": 0.60,
        }
    return {
        "mode": "dark",
        "bg": "#000000",
        "text": "#ffffff",
        "border_solid": "#ffffff",
        "hover_bg": "rgba(25,25,25,0.96)",
        "hover_border": "rgba(255,255,255,0.18)",
        "hover_font": "#ffffff",
        "grid": "rgba(255,255,255,0.10)",
        "edge_scale": EDGE_COLOR_MAP_DARK,
        "edge_alpha": 0.52,
    }


# ============================================================================
# Bundle export — runs once per dataset, on first access
# ============================================================================

def mi_columns(dim_envir: int) -> list[str]:
    return [f"MI-{i + 1}" for i in range(dim_envir)]


def _build_palette(categories: Iterable[str]) -> dict[str, str]:
    cats = list(pd.Index(categories).astype(str))
    n = len(cats)
    if n <= 20:
        cmap = plt.get_cmap("tab20", n)
        palette = [mcolors.to_hex(cmap(i)) for i in range(n)]
    elif n <= 40:
        cmap1 = plt.get_cmap("tab20", 20)
        cmap2 = plt.get_cmap("tab20b", 20)
        palette = [mcolors.to_hex(cmap1(i)) for i in range(20)] + [mcolors.to_hex(cmap2(i)) for i in range(n - 20)]
    else:
        cmap = plt.get_cmap("hsv", n)
        palette = [mcolors.to_hex(cmap(i)) for i in range(n)]
    return dict(zip(cats, palette))


def _get_spatial_xy(adata) -> tuple[np.ndarray, np.ndarray]:
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"])
        if xy.shape[1] < 2:
            raise ValueError("adata.obsm['spatial'] must have at least 2 columns")
        return xy[:, 0], xy[:, 1]
    raise ValueError("adata.obsm['spatial'] is required")


def _build_cell_table(adata, celltype_col: str, sample_key: str, palette: dict[str, str]) -> pd.DataFrame:
    x, y = _get_spatial_xy(adata)
    if celltype_col not in adata.obs:
        raise KeyError(f"{celltype_col!r} not in adata.obs")
    celltypes = adata.obs[celltype_col].astype(str).values
    sample_values = (
        adata.obs[sample_key].astype(str).values
        if sample_key in adata.obs
        else np.array(["slice"] * adata.n_obs)
    )
    df = pd.DataFrame({
        "cell_index_local": np.arange(adata.n_obs),
        "barcode": adata.obs_names.astype(str),
        "celltype": celltypes,
        "sample_name": sample_values,
        "x": x,
        "y": y,
    })
    df["color"] = df["celltype"].map(palette).fillna("#999999")
    return df


def _build_edge_table(
    edge_index: np.ndarray,
    factor_envir: np.ndarray,
    cell_df: pd.DataFrame,
    dim_envir: int,
    min_threshold: float,
) -> pd.DataFrame:
    edge_index = np.asarray(edge_index)
    factor_envir = np.asarray(factor_envir)
    if factor_envir.shape[1] != dim_envir:
        raise ValueError(f"Expected {dim_envir} MIs, got {factor_envir.shape[1]}")

    keep = np.max(factor_envir, axis=1) >= min_threshold
    edge_index = edge_index[keep]
    factor_envir = factor_envir[keep]

    sender_idx = edge_index[:, 0].astype(np.int32, copy=False)
    receiver_idx = edge_index[:, 1].astype(np.int32, copy=False)

    barcode = cell_df["barcode"].astype(str).to_numpy()
    celltype = cell_df["celltype"].astype(str).to_numpy()
    sample_name = cell_df["sample_name"].astype(str).to_numpy()
    x = cell_df["x"].to_numpy()
    y = cell_df["y"].to_numpy()
    color = cell_df["color"].astype(str).to_numpy()

    mi_values = np.round(factor_envir, ROUND_DECIMALS).astype(np.float32, copy=False)
    mi_argmax_idx = np.argmax(mi_values, axis=1)
    mi_max = mi_values[np.arange(mi_values.shape[0]), mi_argmax_idx]

    mi_cols = np.array(mi_columns(dim_envir), dtype=object)
    edge_df = pd.DataFrame({
        "edge_index_local": np.arange(edge_index.shape[0], dtype=np.int32),
        "sender_index_local": sender_idx,
        "receiver_index_local": receiver_idx,
        "sender_barcode": barcode[sender_idx],
        "sender_celltype": celltype[sender_idx],
        "sender_sample_name": sample_name[sender_idx],
        "sender_x": x[sender_idx],
        "sender_y": y[sender_idx],
        "sender_color": color[sender_idx],
        "receiver_barcode": barcode[receiver_idx],
        "receiver_celltype": celltype[receiver_idx],
        "receiver_sample_name": sample_name[receiver_idx],
        "receiver_x": x[receiver_idx],
        "receiver_y": y[receiver_idx],
        "receiver_color": color[receiver_idx],
    })
    mi_df = pd.DataFrame(mi_values, columns=mi_cols)
    edge_df = pd.concat([edge_df, mi_df], axis=1)
    edge_df["mi_max"] = mi_max
    edge_df["mi_argmax"] = mi_cols[mi_argmax_idx]
    return edge_df


def _export_signature(ds: DatasetPaths, min_threshold: float) -> dict:
    return {
        "celltype_column": ds.config.get("CELL_TYPE_COL"),
        "sample_key": ds.config.get("SAMPLE_ID_COL"),
        "spatial_obsm_key": ds.config.get("SPATIAL_KEY", "spatial"),
        "x_obs_key": None,
        "y_obs_key": None,
        "dim_envir": int(ds.dim_envir),
        "min_threshold": float(min_threshold),
    }


def export_dir_for(ds: DatasetPaths) -> Path:
    return ds.run_dir / "UI_Exports" / "module1_in_situ_professional"


def export_in_situ_bundle(ds: DatasetPaths, force: bool = False) -> dict:
    """Build per-slice export tables under <run_dir>/UI_Exports/module1.../

    Returns the dataset_summary.json dict. Skips work if a cached export
    matches the current config and force=False.
    """
    outdir = export_dir_for(ds)
    outdir.mkdir(parents=True, exist_ok=True)
    palette_path = outdir / "palette.json"
    summary_path = outdir / "dataset_summary.json"
    sig = _export_signature(ds, EDGE_MIN_EXPORT_THRESHOLD)

    if not force and summary_path.exists() and palette_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("export_config", {}) == sig:
                return cached
        except Exception:
            pass

    bundle = get_core_bundle(ds)
    adata_list = bundle["adata_list"]
    pyg_list = bundle["pyg_list"]
    fe_list = bundle["factor_envir_list"]

    celltype_col = sig["celltype_column"]
    sample_key = sig["sample_key"]
    dim_envir = sig["dim_envir"]
    min_threshold = sig["min_threshold"]

    all_celltypes = sorted({
        str(ct)
        for adata in adata_list
        for ct in pd.unique(adata.obs[celltype_col].astype(str))
    })
    palette = _build_palette(all_celltypes)
    compression = {"method": "gzip", "compresslevel": 1}

    dataset_summary: dict[str, Any] = {
        "dataset_name": ds.name,
        "dim_envir": dim_envir,
        "mi_list": mi_columns(dim_envir),
        "slices": [],
        "export_config": sig,
    }

    n = len(adata_list)
    for slice_idx, (adata, pyg_obj, factor_envir) in enumerate(zip(adata_list, pyg_list, fe_list)):
        print(f"[m1] Exporting slice {slice_idx + 1}/{n} for {ds.name}...")
        slice_dir = outdir / f"slice_{slice_idx:03d}"
        slice_dir.mkdir(parents=True, exist_ok=True)

        cell_df = _build_cell_table(adata, celltype_col, sample_key, palette)
        edge_index = edge_index_array(pyg_obj)
        edge_df = _build_edge_table(
            edge_index=edge_index,
            factor_envir=np.asarray(factor_envir),
            cell_df=cell_df,
            dim_envir=dim_envir,
            min_threshold=min_threshold,
        )

        cell_df.to_csv(slice_dir / "cells.csv.gz", index=False, compression=compression)
        edge_df.to_csv(slice_dir / "edges_wide.csv.gz", index=False, compression=compression)

        per_slice = {
            "slice_index": int(slice_idx),
            "sample_names": sorted(cell_df["sample_name"].unique().tolist()),
            "n_cells": int(cell_df.shape[0]),
            "n_edges": int(edge_df.shape[0]),
            "celltype_column": celltype_col,
            "available_sender_celltypes": sorted(pd.unique(edge_df["sender_celltype"].astype(str)).tolist()),
            "available_receiver_celltypes": sorted(pd.unique(edge_df["receiver_celltype"].astype(str)).tolist()),
            "default_x_range": [float(cell_df["x"].min()), float(cell_df["x"].max())],
            "default_y_range": [float(cell_df["y"].min()), float(cell_df["y"].max())],
        }
        with open(slice_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(per_slice, f, indent=2)
        dataset_summary["slices"].append(per_slice)

    with open(palette_path, "w", encoding="utf-8") as f:
        json.dump(palette, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    return dataset_summary


# ============================================================================
# Slice loading (cell 5)
# ============================================================================

@dataclass
class SliceTables:
    cells: pd.DataFrame
    edges: pd.DataFrame
    summary: dict


# (dataset_name, slice_idx) -> SliceTables
_SLICE_CACHE: dict[tuple[str, int], SliceTables] = {}
_SLICE_LOCK = Lock()


def load_slice(ds: DatasetPaths, slice_idx: int) -> SliceTables:
    key = (ds.name, int(slice_idx))
    with _SLICE_LOCK:
        cached = _SLICE_CACHE.get(key)
        if cached is not None:
            return cached
        slice_dir = export_dir_for(ds) / f"slice_{slice_idx:03d}"
        if not slice_dir.exists():
            raise FileNotFoundError(f"slice not found: {slice_dir}")
        with open(slice_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        tables = SliceTables(
            cells=pd.read_csv(slice_dir / "cells.csv.gz"),
            edges=pd.read_csv(slice_dir / "edges_wide.csv.gz"),
            summary=summary,
        )
        _SLICE_CACHE[key] = tables
        return tables


def load_palette(ds: DatasetPaths) -> dict[str, str]:
    p = export_dir_for(ds) / "palette.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_summary(ds: DatasetPaths) -> dict:
    p = export_dir_for(ds) / "dataset_summary.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Edge filter (cell 5)
# ============================================================================

def filter_edges(
    edge_df: pd.DataFrame,
    mi_name: str,
    threshold: float,
    sender_types: Optional[list[str]] = None,
    receiver_types: Optional[list[str]] = None,
    max_edges: int = DEFAULT_MAX_EDGES_DISPLAY,
) -> tuple[pd.DataFrame, bool]:
    keep = edge_df[mi_name].to_numpy(dtype=float) >= float(threshold)
    if sender_types:
        keep &= edge_df["sender_celltype"].astype(str).isin(list(sender_types)).to_numpy()
    if receiver_types:
        keep &= edge_df["receiver_celltype"].astype(str).isin(list(receiver_types)).to_numpy()
    sub = edge_df.loc[keep].copy()
    sub = sub.sort_values(mi_name, ascending=False)
    truncated = False
    if sub.shape[0] > int(max_edges):
        sub = sub.iloc[: int(max_edges)].copy()
        truncated = True
    return sub, truncated


def edge_threshold_default(edge_df: pd.DataFrame, mi_name: str) -> tuple[float, float, float]:
    values = edge_df[mi_name].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0, 0.0
    vmax = float(values.max())
    if vmax <= 0:
        return 0.0, 1.0, 0.0
    q = float(np.quantile(values, 0.90))
    return 0.0, vmax, q


# ============================================================================
# Geometry (cell 5)
# ============================================================================

def _tight_axis_ranges(cells_df: pd.DataFrame, pad_frac: float = 0.03, min_pad: float = 40.0):
    x = cells_df["x"].to_numpy(dtype=float)
    y = cells_df["y"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return None, None
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xpad = max((xmax - xmin) * pad_frac, min_pad)
    ypad = max((ymax - ymin) * pad_frac, min_pad)
    return [xmin - xpad, xmax + xpad], [ymax + ypad, ymin - ypad]


def _figure_geometry(cells_df: pd.DataFrame) -> dict:
    x_range, y_range = _tight_axis_ranges(cells_df)
    target_h = 640
    min_w, max_w = 920, 1240
    margin_l, margin_r, margin_t, margin_b = 6, 10, 20, 160

    if x_range is None or y_range is None:
        return {
            "width": min_w, "height": target_h + margin_t + margin_b,
            "margin": dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
            "x_range": x_range, "y_range": y_range,
        }
    x_span = max(float(x_range[1] - x_range[0]), 1.0)
    y_span = max(float(abs(y_range[0] - y_range[1])), 1.0)
    aspect = x_span / y_span
    plot_w = aspect * target_h
    width = int(np.clip(plot_w + margin_l + margin_r, min_w, max_w))
    plot_w_eff = max(width - margin_l - margin_r, 1.0)
    plot_h_eff = plot_w_eff / max(aspect, 1e-8)
    height = int(np.clip(plot_h_eff + margin_t + margin_b, 760, 980))
    return {
        "width": width, "height": height,
        "margin": dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        "x_range": x_range, "y_range": y_range,
    }


# ============================================================================
# Trace builders (cell 8 themed variants)
# ============================================================================

def _arrow_segments(df: pd.DataFrame, line_width: float, arrow_size: float):
    sx = df["sender_x"].to_numpy(dtype=float); sy = df["sender_y"].to_numpy(dtype=float)
    rx = df["receiver_x"].to_numpy(dtype=float); ry = df["receiver_y"].to_numpy(dtype=float)
    dx = rx - sx; dy = ry - sy
    length = np.sqrt(dx ** 2 + dy ** 2)
    length = np.where(length <= 1e-12, 1e-12, length)
    ux = dx / length; uy = dy / length
    px = -uy; py = ux
    arrow_len = np.clip(length * (0.10 * arrow_size), 4.0 * line_width * arrow_size, 11.0 * line_width * arrow_size)
    arrow_hw = np.clip(length * (0.045 * arrow_size), 2.2 * line_width * arrow_size, 6.0 * line_width * arrow_size)
    left_x = rx - ux * arrow_len + px * arrow_hw
    left_y = ry - uy * arrow_len + py * arrow_hw
    right_x = rx - ux * arrow_len - px * arrow_hw
    right_y = ry - uy * arrow_len - py * arrow_hw
    n = df.shape[0]
    arrow_x = np.column_stack([left_x, rx, np.full(n, np.nan), right_x, rx, np.full(n, np.nan)]).ravel()
    arrow_y = np.column_stack([left_y, ry, np.full(n, np.nan), right_y, ry, np.full(n, np.nan)]).ravel()
    return arrow_x, arrow_y


def _build_cells_trace(cells_df: pd.DataFrame, palette: dict[str, str], cell_alpha: float, marker_size: float, theme: dict) -> go.Scatter:
    customdata = cells_df[["barcode", "celltype", "sample_name"]].astype(str).values.tolist()
    base_colors = cells_df["celltype"].astype(str).map(palette)
    if "color" in cells_df.columns:
        base_colors = base_colors.fillna(cells_df["color"].astype(str))
    else:
        base_colors = base_colors.fillna("#999999")
    base_colors = base_colors.astype(str).tolist()
    marker_colors = [hex_to_rgba(c, cell_alpha) for c in base_colors]
    return go.Scatter(
        x=cells_df["x"].astype(float).tolist(),
        y=cells_df["y"].astype(float).tolist(),
        mode="markers",
        name="Cells",
        showlegend=False,
        customdata=customdata,
        marker=dict(size=float(marker_size), color=marker_colors, line=dict(width=0)),
        hoverlabel=dict(
            bgcolor=theme["hover_bg"], bordercolor=theme["hover_border"],
            font=dict(size=SMALL_FONT_SIZE, color=theme["hover_font"]),
        ),
        hovertemplate=(
            "<b>Cell</b><br>Barcode: %{customdata[0]}<br>"
            "Cell type: %{customdata[1]}<br>Sample: %{customdata[2]}<br>"
            "x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
        ),
    )


def _build_edge_traces(edge_sub: pd.DataFrame, mi_name: str, line_width: float, arrow_size: float, theme: dict):
    if edge_sub.shape[0] == 0:
        return [], None
    values = edge_sub[mi_name].to_numpy(dtype=float)
    vmax = float(np.quantile(values, EDGE_COLOR_QUANTILE_CLIP)) if values.size else 1.0
    vmax = max(vmax, float(values.max()), 1e-8)
    bins = np.linspace(values.min(), vmax, 7)
    if np.unique(bins).size < 2:
        bins = np.linspace(values.min(), values.min() + 1e-6, 7)

    color_list = get_colorscale_hex_list(theme["edge_scale"], n=6)
    traces = []
    for i in range(6):
        if i < 5:
            mask = (values >= bins[i]) & (values < bins[i + 1])
        else:
            mask = values >= bins[i]
        if not np.any(mask):
            continue
        sub = edge_sub.loc[mask]
        line_x = np.column_stack([
            sub["sender_x"].to_numpy(dtype=float),
            sub["receiver_x"].to_numpy(dtype=float),
            np.full(sub.shape[0], np.nan),
        ]).ravel().tolist()
        line_y = np.column_stack([
            sub["sender_y"].to_numpy(dtype=float),
            sub["receiver_y"].to_numpy(dtype=float),
            np.full(sub.shape[0], np.nan),
        ]).ravel().tolist()
        edge_color = hex_to_rgba(color_list[i], theme["edge_alpha"])
        traces.append(go.Scatter(
            x=line_x, y=line_y, mode="lines", name=f"{mi_name} edges",
            showlegend=False, hoverinfo="skip",
            line=dict(width=float(line_width), color=edge_color),
        ))
        ax, ay = _arrow_segments(sub, line_width=line_width, arrow_size=arrow_size)
        traces.append(go.Scatter(
            x=np.asarray(ax, dtype=float).tolist(),
            y=np.asarray(ay, dtype=float).tolist(),
            mode="lines", name=f"{mi_name} arrows",
            showlegend=False, hoverinfo="skip",
            line=dict(width=max(float(line_width) * 0.95, 0.5), color=edge_color),
        ))

    mid_x = ((edge_sub["sender_x"].to_numpy(dtype=float) + edge_sub["receiver_x"].to_numpy(dtype=float)) / 2.0).tolist()
    mid_y = ((edge_sub["sender_y"].to_numpy(dtype=float) + edge_sub["receiver_y"].to_numpy(dtype=float)) / 2.0).tolist()
    hover_text = (
        "<b>MI edge</b><br>"
        + "Sender: " + edge_sub["sender_barcode"].astype(str)
        + " (" + edge_sub["sender_celltype"].astype(str) + ")<br>"
        + "Receiver: " + edge_sub["receiver_barcode"].astype(str)
        + " (" + edge_sub["receiver_celltype"].astype(str) + ")<br>"
        + mi_name + ": " + edge_sub[mi_name].map(lambda v: f"{v:.4f}")
    ).tolist()

    edge_hover = go.Scatter(
        x=mid_x, y=mid_y, mode="markers", name="Edge hover",
        showlegend=False, text=hover_text, hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(
            font=dict(size=SMALL_FONT_SIZE, color=theme["hover_font"]),
            bgcolor=theme["hover_bg"], bordercolor=theme["hover_border"],
        ),
        marker=dict(
            size=8,
            color=edge_sub[mi_name].to_numpy(dtype=float).tolist(),
            colorscale=theme["edge_scale"],
            cmin=float(values.min()), cmax=float(vmax),
            opacity=0.10, showscale=True,
            colorbar=dict(
                title=dict(text=f"{mi_name} strength", side="top",
                           font=dict(size=SMALL_FONT_SIZE, color=theme["text"])),
                orientation="h", thickness=16, len=0.20,
                x=0.74, xanchor="center", y=-0.26, yanchor="top",
                tickfont=dict(size=SMALL_FONT_SIZE, color=theme["text"]),
                bgcolor=theme["bg"], outlinecolor=theme["border_solid"], outlinewidth=1.0,
            ),
            line=dict(width=0),
        ),
    )
    return traces, edge_hover


def _build_legend_traces(palette: dict[str, str], senders: set[str], receivers: set[str]):
    out = []
    for celltype, color in sorted(palette.items()):
        if celltype in senders and celltype in receivers:
            suffix, line_color, line_w = " [S/R]", "#111", 2.0
        elif celltype in senders:
            suffix, line_color, line_w = " [S]", "#1f4e79", 2.0
        elif celltype in receivers:
            suffix, line_color, line_w = " [R]", "#7a1f5c", 2.0
        else:
            suffix, line_color, line_w = "", "#888", 1.0
        out.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=f"{celltype}{suffix}", showlegend=True, hoverinfo="skip",
            marker=dict(size=10, color=color, line=dict(color=line_color, width=line_w)),
            legendgroup="celltype_legend",
        ))
    return out


# ============================================================================
# Public API
# ============================================================================

def build_spatial_figure(
    ds: DatasetPaths,
    slice_idx: int,
    mi_idx: int,
    threshold: Optional[float] = None,
    sender_types: Optional[list[str]] = None,
    receiver_types: Optional[list[str]] = None,
    cell_alpha: float = DEFAULT_CELL_ALPHA,
    max_edges: int = DEFAULT_MAX_EDGES_DISPLAY,
    marker_size: float = DEFAULT_MARKER_SIZE,
    edge_width: float = DEFAULT_EDGE_WIDTH,
    arrow_size: float = DEFAULT_ARROW_SIZE,
    theme_mode: str = "dark",
) -> dict:
    """Return a Plotly figure dict for the requested (slice, MI) view."""
    tables = load_slice(ds, slice_idx)
    palette = load_palette(ds)
    mi_name = mi_columns(ds.dim_envir)[int(mi_idx)]
    theme = theme_spec(theme_mode)

    if threshold is None:
        _, _, threshold = edge_threshold_default(tables.edges, mi_name)

    edge_sub, truncated = filter_edges(
        tables.edges, mi_name=mi_name, threshold=threshold,
        sender_types=sender_types, receiver_types=receiver_types,
        max_edges=max_edges,
    )

    fig = go.Figure()
    edge_traces, edge_hover = _build_edge_traces(
        edge_sub, mi_name=mi_name, line_width=edge_width, arrow_size=arrow_size, theme=theme,
    )
    for tr in edge_traces:
        fig.add_trace(tr)
    if edge_hover is not None:
        fig.add_trace(edge_hover)

    fig.add_trace(_build_cells_trace(
        tables.cells, palette=palette, cell_alpha=cell_alpha, marker_size=marker_size, theme=theme,
    ))

    senders = set(sender_types or [])
    receivers = set(receiver_types or [])
    for legend_trace in _build_legend_traces(palette, senders, receivers):
        fig.add_trace(legend_trace)

    geometry = _figure_geometry(tables.cells)

    fig.update_layout(
        autosize=True,
        width=None,
        height=geometry["height"],
        dragmode="pan",
        hovermode="closest",
        margin=geometry["margin"],
        title=None,
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["bg"],
        font=dict(size=BASE_FONT_SIZE, color=theme["text"]),
        hoverlabel=dict(
            font=dict(size=SMALL_FONT_SIZE, color=theme["hover_font"]),
            bgcolor=theme["hover_bg"], bordercolor=theme["hover_border"],
        ),
        uirevision="keep_zoom",
        modebar_add=["select2d", "lasso2d"],
        legend=dict(
            title=dict(text="Cell types", font=dict(size=BASE_FONT_SIZE + 1, color=theme["text"])),
            orientation="h", x=0.16, y=-0.11, xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.0)", bordercolor=theme["border_solid"], borderwidth=1,
            font=dict(size=SMALL_FONT_SIZE + 1, color=theme["text"]),
            itemsizing="constant", tracegroupgap=6,
        ),
    )
    fig.update_xaxes(
        title=None, showticklabels=False, ticks="", showgrid=False,
        zeroline=False, showline=False, constrain="domain",
        range=geometry["x_range"], automargin=False,
    )
    fig.update_yaxes(
        title=None, showticklabels=False, ticks="", showgrid=False,
        zeroline=False, showline=False, scaleanchor="x", scaleratio=1,
        constrain="domain", range=geometry["y_range"], automargin=False,
    )

    fig_json = json.loads(fig.to_json(validate=False))
    fig_json["layout"]["template"] = None  # let Plotly.js use defaults

    return {
        "figure": fig_json,
        "meta": {
            "mi_name": mi_name,
            "threshold": float(threshold),
            "n_edges_visible": int(edge_sub.shape[0]),
            "truncated": bool(truncated),
        },
    }


# ============================================================================
# Top loadings (LR pairs / sender genes / receiver genes per MI)
# ============================================================================

# dataset_name -> {"loading_lr_norm", "loading_sender_norm", "loading_receiver_norm",
#                  "lr_labels", "sender_gene_labels", "receiver_gene_labels"}
_LOADING_CACHE: dict[str, dict[str, Any]] = {}
_LOADING_LOCK = Lock()


def _format_lr_label(x: Any) -> str:
    """Format a single LR-pair entry. Handles plain strings, [[lig...],[rec...]],
    and tuples/arrays. Inner lists are joined with '+'."""
    def _flatten(item: Any) -> str:
        if isinstance(item, (list, tuple, np.ndarray)):
            return "+".join(str(v) for v in item)
        return str(item)

    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) >= 2:
            return f"{_flatten(x[0])}→{_flatten(x[1])}"
        if len(x) == 1:
            return _flatten(x[0])
    return str(x)


def _ensure_mi_by_feature(arr: np.ndarray, n_mi: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    if arr.shape[0] == n_mi:
        return arr
    if arr.shape[1] == n_mi:
        return arr.T
    raise ValueError(
        f"loading matrix shape {arr.shape} does not match n_mi={n_mi}"
    )


def _column_sum_normalize(arr: np.ndarray) -> np.ndarray:
    """Match notebook: divide each feature's column by its sum across MIs.

    Input shape is (n_mi, n_features); after normalization, each *feature column*
    sums to 1 across MIs. Picking top features for one MI by this normalized
    value surfaces features whose loading is most concentrated on that MI.
    """
    denom = np.sum(arr, axis=0, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return arr / denom


def _align_labels(labels: Any, n_features: int, prefix: str) -> list[str]:
    if labels is None:
        labels = []
    elif isinstance(labels, pd.Index):
        labels = labels.astype(str).tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif isinstance(labels, (list, tuple)):
        labels = list(labels)
    else:
        try:
            labels = list(labels)
        except TypeError:
            labels = [labels]
    labels = [str(x) for x in labels]
    if len(labels) < n_features:
        labels = labels + [f"{prefix}-{i + 1}" for i in range(len(labels), n_features)]
    if len(labels) > n_features:
        labels = labels[:n_features]
    return labels


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_gene_labels(ds: DatasetPaths, n_features: int) -> list[str]:
    candidates = [
        ds.processed_dir / "genenames_train.pkl",
        ds.processed_dir / "genenames_train.npy",
        ds.processed_dir / "genenames.pkl",
        ds.processed_dir / "genenames.npy",
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".pkl":
            return _align_labels(_load_pickle(p), n_features=n_features, prefix="Gene")
        return _align_labels(np.load(p, allow_pickle=True), n_features=n_features, prefix="Gene")
    return _align_labels(None, n_features=n_features, prefix="Gene")


def get_loading_bundle(ds: DatasetPaths) -> dict[str, Any]:
    with _LOADING_LOCK:
        cached = _LOADING_CACHE.get(ds.name)
        if cached is not None:
            return cached

        n_mi = int(ds.dim_envir)
        loading_lr = _ensure_mi_by_feature(np.load(ds.run_dir / "loading_LR_use.npy"), n_mi)
        loading_sender = _ensure_mi_by_feature(np.load(ds.run_dir / "loading_sender_use.npy"), n_mi)
        loading_receiver = _ensure_mi_by_feature(np.load(ds.run_dir / "loading_receiver_use.npy"), n_mi)

        lr_path = ds.processed_dir / "LR_list.pkl"
        lr_labels = None
        if lr_path.exists():
            lr_labels = [_format_lr_label(x) for x in _load_pickle(lr_path)]
        lr_labels = _align_labels(lr_labels, n_features=loading_lr.shape[1], prefix="LR")

        gene_labels = _load_gene_labels(
            ds, n_features=max(loading_sender.shape[1], loading_receiver.shape[1])
        )

        bundle = {
            "loading_lr_norm": _column_sum_normalize(loading_lr),
            "loading_sender_norm": _column_sum_normalize(loading_sender),
            "loading_receiver_norm": _column_sum_normalize(loading_receiver),
            "lr_labels": lr_labels,
            "sender_gene_labels": _align_labels(gene_labels, n_features=loading_sender.shape[1], prefix="Gene"),
            "receiver_gene_labels": _align_labels(gene_labels, n_features=loading_receiver.shape[1], prefix="Gene"),
        }
        _LOADING_CACHE[ds.name] = bundle
        return bundle


def top_loadings_for_mi(ds: DatasetPaths, mi_idx: int, kind: str, top_n: int = 10) -> pd.DataFrame:
    bundle = get_loading_bundle(ds)
    if kind == "lr":
        values = np.asarray(bundle["loading_lr_norm"][mi_idx], dtype=float)
        labels = bundle["lr_labels"]
    elif kind == "sender":
        values = np.asarray(bundle["loading_sender_norm"][mi_idx], dtype=float)
        labels = bundle["sender_gene_labels"]
    elif kind == "receiver":
        values = np.asarray(bundle["loading_receiver_norm"][mi_idx], dtype=float)
        labels = bundle["receiver_gene_labels"]
    else:
        raise ValueError(f"kind must be 'lr'|'sender'|'receiver', got {kind!r}")
    order = np.argsort(values)[::-1][: int(top_n)]
    return pd.DataFrame({"label": [labels[i] for i in order], "value": values[order]})


def _make_stem_figure(top_df: pd.DataFrame, title_text: str, color_hex: str, theme: dict) -> dict:
    """Plotly stem plot for one (kind, MI) top-N table."""
    if top_df is None or top_df.shape[0] == 0:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=f"No loading values for {title_text}.",
            showarrow=False, font=dict(size=14, color=theme["text"]),
        )
        fig.update_layout(
            paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=15, r=15, t=10, b=10), height=240,
        )
        return json.loads(fig.to_json(validate=False))

    x = np.arange(top_df.shape[0], dtype=float)
    values = top_df["value"].to_numpy(dtype=float)
    labels = top_df["label"].astype(str).tolist()
    display_labels = [lab if len(lab) <= 24 else lab[:21] + "..." for lab in labels]
    line_x = np.column_stack([x, x, np.full_like(x, np.nan)]).ravel().tolist()
    line_y = np.column_stack([np.zeros_like(values), values, np.full_like(values, np.nan)]).ravel().tolist()

    fig = go.Figure()
    fig.add_hline(y=0.0, line_width=1.0, line_color=theme["grid"])
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=theme["text"], width=2.0),
    ))
    fig.add_trace(go.Scatter(
        x=x.tolist(), y=values.tolist(), mode="markers", showlegend=False,
        customdata=np.array(labels, dtype=object).tolist(),
        marker=dict(size=10.0, color=color_hex, line=dict(color=theme["text"], width=1.0)),
        hovertemplate="<b>%{customdata}</b><br>Normalized loading: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color=theme["text"])),
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(size=13, color=theme["text"]),
        margin=dict(l=58, r=14, t=36, b=110),
        height=260,
        autosize=True,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=x.tolist(), ticktext=display_labels,
        tickangle=38, tickfont=dict(size=11, color=theme["text"]),
        showgrid=False, zeroline=False, showline=False,
    )
    fig.update_yaxes(
        title=dict(text="Normalized loading", font=dict(size=12, color=theme["text"])),
        tickfont=dict(size=11, color=theme["text"]),
        showgrid=True, gridcolor=theme["grid"], zeroline=False, showline=False,
    )
    return json.loads(fig.to_json(validate=False))


def _safe_column_sum_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    denom = arr.sum(axis=0, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return arr / denom


def _load_edge_index_as_e_by_2(edge_index) -> np.ndarray:
    import torch
    if torch.is_tensor(edge_index):
        arr = edge_index.detach().cpu().numpy()
    else:
        arr = np.asarray(edge_index)
    if arr.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {arr.shape}")
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"edge_index must be (E,2) or (2,E), got {arr.shape}")
    return arr.astype(int)


# ============================================================================
# Enrichment: LR-pathway heatmap (cell 5)
# ============================================================================

def _enrichment_dir(ds: DatasetPaths) -> Path:
    p = export_dir_for(ds) / "enrichment"
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_lr_loading_pathway(ds: DatasetPaths) -> pd.DataFrame:
    """Pathway × MI loading table (rows = MIs, cols = pathways), ordered the
    way the notebook orders them. Cached as `LR_loading_pathway.csv`.
    """
    out = _enrichment_dir(ds) / "LR_loading_pathway.csv"
    if out.exists():
        return pd.read_csv(out, index_col=0)

    lr_path = ds.processed_dir / "LR_list.pkl"
    lr_db_path = ds.processed_dir / "LR_list_cellchatdb.pkl"
    lr_meta_path = ds.processed_dir / "LR_meta_cellchatdb.pkl"
    fe_use_path = ds.run_dir / "Factor_envir_use.npy"
    loading_lr_path = ds.run_dir / "loading_LR_use.npy"

    for p in (lr_path, lr_db_path, lr_meta_path, fe_use_path, loading_lr_path):
        if not p.exists():
            raise FileNotFoundError(f"Required for LR-pathway enrichment: {p}")

    loading_lr = np.load(loading_lr_path)
    factor_envir_use = np.load(fe_use_path)
    n_mi = factor_envir_use.shape[1] if factor_envir_use.ndim == 2 else int(ds.dim_envir)
    loading_lr = _ensure_mi_by_feature(loading_lr, n_mi=n_mi)

    LR_list = _load_pickle(lr_path)
    LR_list_db = _load_pickle(lr_db_path)
    LR_meta_db = _load_pickle(lr_meta_path).copy()

    factor_colmax = np.nanmax(factor_envir_use, axis=0)
    factor_colmax = np.where(np.isfinite(factor_colmax), factor_colmax, 0.0)
    loading_weighted = loading_lr.astype(float).copy()
    for i in range(min(loading_weighted.shape[0], factor_colmax.shape[0])):
        loading_weighted[i, :] *= factor_colmax[i]

    LR_merged = ["+".join(lr[0]) + "->" + "+".join(lr[1]) for lr in LR_list]
    LR_db_merged = ["+".join(lr[0]) + "->" + "+".join(lr[1]) for lr in LR_list_db]
    db_idx = {name: i for i, name in enumerate(LR_db_merged)}

    matched_lr_idx, matched_db_idx = [], []
    for i, name in enumerate(LR_merged):
        j = db_idx.get(name)
        if j is not None:
            matched_lr_idx.append(i)
            matched_db_idx.append(j)

    if not matched_lr_idx:
        raise ValueError("No LR pairs overlapped with CellChatDB.")

    LR_meta_in = LR_meta_db.iloc[matched_db_idx, :].copy()
    LR_meta_in.index = np.arange(LR_meta_in.shape[0])

    LR_pretty = ["+".join(lr[0]) + " -> " + "+".join(lr[1]) for lr in LR_list]
    mi_names = [f"MI-{i + 1}" for i in range(loading_weighted.shape[0])]

    loading_norm = pd.DataFrame(
        _safe_column_sum_normalize(loading_weighted),
        index=mi_names,
        columns=LR_pretty,
    )

    pathways = pd.Index(LR_meta_in["pathway_name"].astype(str)).unique().tolist()
    pathway_scores, kept = [], []
    for pw in pathways:
        idx = LR_meta_in.index[LR_meta_in["pathway_name"].astype(str) == str(pw)].tolist()
        if not idx:
            continue
        pathway_scores.append(loading_norm.iloc[:, idx].mean(axis=1))
        kept.append(pw)

    if not pathway_scores:
        raise ValueError("No pathways with mapped LR pairs.")

    df = pd.DataFrame(pathway_scores, index=kept, columns=loading_norm.index).fillna(0).T
    df = df.loc[:, list(df.columns)[::-1]]
    df = df.iloc[np.argsort(np.array(np.max(df, axis=1)))[::-1], :]

    argmax = np.argmax(df.values, axis=0)
    maxv = np.max(df.values, axis=0)
    order = []
    for a in np.sort(np.unique(argmax)):
        idx = np.where(argmax == a)[0]
        idx = idx[np.argsort(maxv[idx])[::-1]]
        order.extend(idx.tolist())
    df = df.iloc[:, order]

    df.to_csv(out, index=True)
    return df


def build_lr_pathway_heatmap(ds: DatasetPaths, theme_mode: str = "dark") -> dict:
    df = compute_lr_loading_pathway(ds)
    theme = theme_spec(theme_mode)

    data = df.values.astype(float)
    vmax = float(min(0.4, np.nanmax(data) * 0.7 if np.nanmax(data) > 0 else 0.4))
    if vmax <= 0:
        vmax = 0.4

    fig = go.Figure(go.Heatmap(
        z=data.tolist(),
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale=[
            [0.0, "#FCF5F0"], [0.25, "#F9B2BC"], [0.5, "#F6689F"],
            [0.75, "#C31988"], [1.0, "#510269"],
        ],
        zmin=0.0,
        zmax=vmax,
        colorbar=dict(
            title=dict(text="Pathway × MI loading", side="right",
                       font=dict(size=12, color=theme["text"])),
            tickfont=dict(size=11, color=theme["text"]),
            outlinecolor=theme["border_solid"], outlinewidth=1.0,
        ),
        xgap=0.5, ygap=0.5,
        hovertemplate="MI: %{y}<br>Pathway: %{x}<br>Score: %{z:.4f}<extra></extra>",
    ))
    n_x, n_y = data.shape[1], data.shape[0]
    fig.update_layout(
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(size=12, color=theme["text"]),
        margin=dict(l=80, r=20, t=20, b=200),
        height=max(500, 60 + n_y * 28),
        autosize=True,
    )
    fig.update_xaxes(
        side="bottom", tickangle=55,
        tickfont=dict(size=10, color=theme["text"]),
        showgrid=False, automargin=True,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=11, color=theme["text"]),
        showgrid=False, automargin=True,
    )
    return {
        "figure": json.loads(fig.to_json(validate=False)),
        "table": {
            "rows": df.index.tolist(),
            "cols": df.columns.tolist(),
            "data": data.tolist(),
        },
    }


# ============================================================================
# Enrichment: Cell-type-pair × MI heatmap (cell 5)
# ============================================================================

def compute_avg_mi_cellclass_pair(
    ds: DatasetPaths, mi_threshold: float = 0.6
) -> tuple[pd.DataFrame, float]:
    """Filtered + z-scored sender→receiver × MI table. Cached as
    `Avg_MI_cellclass_pair_merge_use.csv`."""
    out = _enrichment_dir(ds) / "Avg_MI_cellclass_pair_merge_use.csv"
    threshold_path = _enrichment_dir(ds) / "mi_threshold.json"
    if out.exists() and threshold_path.exists():
        with open(threshold_path, "r") as f:
            thr = json.load(f).get("mi_threshold", mi_threshold)
        return pd.read_csv(out), float(thr)

    fe_use_path = ds.run_dir / "Factor_envir_use.npy"
    if not fe_use_path.exists():
        raise FileNotFoundError(f"Missing: {fe_use_path}")

    bundle = get_core_bundle(ds)
    pyg_list = bundle["pyg_list"]
    MI_use = np.load(fe_use_path)
    if MI_use.ndim == 1:
        MI_use = MI_use.reshape(-1, 1)

    total_edges = sum(_load_edge_index_as_e_by_2(item["edge_index"]).shape[0] for item in pyg_list)
    if MI_use.shape[0] != total_edges and MI_use.shape[1] == total_edges:
        MI_use = MI_use.T
    if MI_use.shape[0] != total_edges:
        raise ValueError(f"Factor_envir_use shape {MI_use.shape} doesn't match edge count {total_edges}")

    if "cell_class_unique" in pyg_list[0]:
        cellclass_unique = np.asarray(pyg_list[0]["cell_class_unique"]).astype(str)
    else:
        cellclass_unique = np.unique(np.hstack([
            np.asarray(item["cell_class"]).astype(str) for item in pyg_list
        ]))

    sender_list, receiver_list = [], []
    for item in pyg_list:
        edges = _load_edge_index_as_e_by_2(item["edge_index"])
        cell_class = np.asarray(item["cell_class"]).astype(str)
        sender_list.append(cell_class[edges[:, 0]])
        receiver_list.append(cell_class[edges[:, 1]])

    sender_arr = np.hstack(sender_list)
    receiver_arr = np.hstack(receiver_list)
    cellclass_edge = pd.DataFrame({"Sender": sender_arr, "Receiver": receiver_arr})

    # filter to "frequent" pairs (>10% of max neighbor count, both directions)
    keep_pairs = []
    for ct in cellclass_unique:
        sub = cellclass_edge.loc[cellclass_edge["Sender"] == ct]
        if not sub.empty:
            neigh, counts = np.unique(sub["Receiver"].values, return_counts=True)
            df = pd.DataFrame({"Sender": ct, "Receiver": neigh, "Count": counts})
            df["norm"] = df["Count"] / max(df["Count"].max(), 1)
            keep_pairs.append(df.loc[df["norm"] > 0.1, ["Sender", "Receiver"]])
        sub = cellclass_edge.loc[cellclass_edge["Receiver"] == ct]
        if not sub.empty:
            neigh, counts = np.unique(sub["Sender"].values, return_counts=True)
            df = pd.DataFrame({"Sender": neigh, "Receiver": ct, "Count": counts})
            df["norm"] = df["Count"] / max(df["Count"].max(), 1)
            keep_pairs.append(df.loc[df["norm"] > 0.1, ["Sender", "Receiver"]])
    pair_filter = pd.concat([x for x in keep_pairs if x.shape[0] > 0], ignore_index=True).drop_duplicates()

    # mean MI strength per (Sender, Receiver) per MI
    mi_long_first = None
    avg_values = []
    for mi_index in range(MI_use.shape[1]):
        avg_mat = pd.DataFrame(0.0, index=cellclass_unique, columns=cellclass_unique)
        edges = cellclass_edge.copy()
        edges["MI_strength"] = MI_use[:, mi_index].astype(float)
        pivoted = edges.pivot_table(index="Sender", columns="Receiver", values="MI_strength",
                                     aggfunc="mean", fill_value=0.0)
        avg_mat.loc[pivoted.index.astype(str), pivoted.columns.astype(str)] = pivoted.values
        flat = avg_mat.stack().reset_index()
        flat.columns = ["Sender", "Receiver", "MI_strength"]
        if mi_long_first is None:
            mi_long_first = flat[["Sender", "Receiver"]].copy()
        avg_values.append(flat["MI_strength"].values)

    Avg = pd.DataFrame(
        np.vstack(avg_values).T,
        columns=[f"MI-{i + 1}" for i in range(MI_use.shape[1])],
    )
    Avg["Sender"] = mi_long_first["Sender"].astype(str).values
    Avg["Receiver"] = mi_long_first["Receiver"].astype(str).values

    # reorder columns by LR-pathway MI order if available
    lr_path_csv = _enrichment_dir(ds) / "LR_loading_pathway.csv"
    if lr_path_csv.exists():
        lr_path_df = pd.read_csv(lr_path_csv, index_col=0)
        mi_order = [m for m in lr_path_df.index.tolist() if m in Avg.columns]
    else:
        mi_order = []
    remaining = [c for c in Avg.columns if c.startswith("MI-") and c not in mi_order]
    mi_order = mi_order + remaining
    Avg = Avg.loc[:, mi_order + ["Sender", "Receiver"]]

    # min-max z-score each MI column to [0, 1]
    vals = Avg.iloc[:, : len(mi_order)].values.astype(float)
    col_min = vals.min(axis=0); col_max = vals.max(axis=0)
    denom = np.where((col_max - col_min) == 0, 1.0, col_max - col_min)
    Avg_z = Avg.copy()
    Avg_z.iloc[:, : len(mi_order)] = (vals - col_min) / denom

    # filter to high-frequency (Sender, Receiver) pairs
    Avg_z = Avg_z.merge(pair_filter, on=["Sender", "Receiver"], how="inner")

    # auto-relax threshold
    threshold = float(mi_threshold)
    if Avg_z.shape[0] > 0:
        candidates = []
        for ct in cellclass_unique:
            mask = (Avg_z["Sender"] == ct) | (Avg_z["Receiver"] == ct)
            if mask.any():
                candidates.append(np.max(Avg_z.loc[mask, mi_order].values))
        if candidates:
            tmin = float(np.min(candidates))
            if tmin < threshold and tmin > 0.5:
                threshold = tmin

    keep_mask = np.max(Avg_z.iloc[:, : len(mi_order)].values, axis=1) >= threshold
    Avg_use = Avg_z.iloc[np.where(keep_mask)[0], :].copy()
    if Avg_use.shape[0] > 0:
        order = np.argsort(np.argmax(Avg_use.iloc[:, : len(mi_order)].values, axis=1))
        Avg_use = Avg_use.iloc[order, :]

    Avg_use.to_csv(out, index=False)
    with open(threshold_path, "w") as f:
        json.dump({"mi_threshold": float(threshold)}, f)
    return Avg_use, float(threshold)


def build_celltype_pair_heatmap(ds: DatasetPaths, theme_mode: str = "dark") -> dict:
    df, threshold = compute_avg_mi_cellclass_pair(ds)
    theme = theme_spec(theme_mode)

    if df.shape[0] == 0:
        # build a "no pairs" placeholder
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No high-strength sender→receiver pairs at the current threshold.",
            showarrow=False, font=dict(size=14, color=theme["text"]),
        )
        fig.update_layout(
            paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=10, b=10), height=320,
        )
        return {
            "figure": json.loads(fig.to_json(validate=False)),
            "threshold": threshold,
            "n_pairs": 0,
        }

    mi_cols = [c for c in df.columns if str(c).startswith("MI-")]
    senders = df["Sender"].astype(str).tolist()
    receivers = df["Receiver"].astype(str).tolist()
    pair_labels = [f"{s} → {r}" for s, r in zip(senders, receivers)]
    z = df.loc[:, mi_cols].values.astype(float).T.tolist()  # rows=MI, cols=pairs

    # mark cells above threshold via custom hover
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=pair_labels, y=mi_cols,
        colorscale=[[0.0, "#4575b4"], [0.5, "#f0f0f0"], [1.0, "#d73027"]],
        zmin=0.0, zmax=1.0,
        colorbar=dict(
            title=dict(text="Z-score", side="right",
                       font=dict(size=12, color=theme["text"])),
            tickfont=dict(size=11, color=theme["text"]),
            outlinecolor=theme["border_solid"], outlinewidth=1.0,
        ),
        xgap=0.5, ygap=0.5,
        hovertemplate="Pair: %{x}<br>%{y}<br>z: %{z:.3f}<extra></extra>",
    ))

    # Yellow rectangles around cells ≥ threshold
    shapes = []
    arr = np.asarray(z)
    above = np.argwhere(arr >= float(threshold))
    for r, c in above:
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
            line=dict(color="#FDE047", width=1.4),
            fillcolor="rgba(0,0,0,0)",
        ))

    n_pairs = len(pair_labels)
    fig.update_layout(
        paper_bgcolor=theme["bg"], plot_bgcolor=theme["bg"],
        font=dict(size=12, color=theme["text"]),
        margin=dict(l=80, r=20, t=20, b=200),
        height=max(420, 50 + len(mi_cols) * 24),
        autosize=True,
        shapes=shapes,
    )
    fig.update_xaxes(
        side="bottom", tickangle=55,
        tickfont=dict(size=10, color=theme["text"]),
        showgrid=False, automargin=True,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=11, color=theme["text"]),
        showgrid=False, automargin=True,
    )
    return {
        "figure": json.loads(fig.to_json(validate=False)),
        "threshold": float(threshold),
        "n_pairs": int(n_pairs),
    }


def build_loadings_figures(
    ds: DatasetPaths, mi_idx: int, top_n: int = 10, theme_mode: str = "dark"
) -> dict:
    """Return three Plotly figure dicts for the chosen MI: LR / sender / receiver."""
    theme = theme_spec(theme_mode)
    mi_name = mi_columns(ds.dim_envir)[int(mi_idx)]

    lr_df = top_loadings_for_mi(ds, mi_idx, kind="lr", top_n=top_n)
    sender_df = top_loadings_for_mi(ds, mi_idx, kind="sender", top_n=top_n)
    receiver_df = top_loadings_for_mi(ds, mi_idx, kind="receiver", top_n=top_n)

    return {
        "mi_name": mi_name,
        "top_n": int(top_n),
        "lr": {
            "figure": _make_stem_figure(lr_df, f"Top {top_n} LR pairs · {mi_name}", "#00F7FF", theme),
            "table": lr_df.to_dict(orient="records"),
        },
        "sender": {
            "figure": _make_stem_figure(sender_df, f"Top {top_n} sender genes · {mi_name}", "#7CFC00", theme),
            "table": sender_df.to_dict(orient="records"),
        },
        "receiver": {
            "figure": _make_stem_figure(receiver_df, f"Top {top_n} receiver genes · {mi_name}", "#FF7AB6", theme),
            "table": receiver_df.to_dict(orient="records"),
        },
    }
