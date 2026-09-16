"""Two-study orchestration and combined figures for the CCC directionality notebook."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t


FEATURE_TYPES = ["Receiver targets", "Sender regulators"]
CANONICAL_METHOD_ORDER = [
    "SpiderNet",
    "NMF-LR",
    "COMMOT",
    "ScCChain",
    "Spacia",
    "SpiderNet-reverse direction",
]


def directionality_output_dir(cfg: dict) -> Path:
    """Return the same per-study directionality directory used by the notebook."""
    result_root = cfg.get("results_path_main", str(cfg["data_path_main"]).replace("Data/", "Results/"))
    return (
        Path(result_root)
        / str(cfg["version"])
        / f"SpiderNet_Result_dim{cfg['dim_envir']}"
        / "Unified_CCC_benchmarking"
        / "Directionality_analyses"
    )


def run_additional_studies(notebook_path, datasets, current_dataset):
    """Execute the notebook once for every study not handled by the parent kernel."""
    notebook_path = Path(notebook_path).resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    for dataset in datasets:
        if dataset == current_dataset:
            continue
        print(f"\nRunning complete notebook pipeline for {dataset} ...")
        env = os.environ.copy()
        env["SPIDERNET_SINGLE_DATASET"] = dataset
        env["MPLBACKEND"] = "Agg"
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_benchmarks.py")),
            "--execute-notebook",
            str(notebook_path),
        ]
        subprocess.run(command, env=env, cwd=notebook_path.parent, check=True)
        print(f"Completed notebook pipeline for {dataset}.")


def load_two_study_outputs(dataset_config, datasets):
    """Load the two tables needed for the requested combined displays."""
    records = {}
    for dataset in datasets:
        cfg = dataset_config[dataset]
        outdir = directionality_output_dir(cfg)
        benchmark_path = outdir / "Analysis2_Benchmark1_with_reverse_direction_long.csv"
        boxplot_path = outdir / "Analysis1_MI_reverse_direction_Pearson_by_slice.csv"
        missing = [path for path in (benchmark_path, boxplot_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required {dataset} output(s): "
                + ", ".join(str(path) for path in missing)
            )
        records[dataset] = {
            "display_name": cfg["display_name"],
            "outdir": outdir,
            "benchmark": pd.read_csv(benchmark_path),
            "boxplot": pd.read_csv(boxplot_path),
        }
    return records


def plot_two_study_displays(dataset_config, datasets, method_colors, output_dir):
    """Redraw the saved tables with the explicit style of the archived displays."""
    records = load_two_study_outputs(dataset_config, datasets)
    output_dir = Path(output_dir)
    with plt.rc_context():
        plt.rcdefaults()
        sns.set_style("darkgrid")
        coupling = plot_benchmark1_with_reverse_two_studies(
            records, datasets, method_colors,
            output_dir / "Analysis2_Benchmark1_curated_gene_coupling_with_reverse_direction_both_studies",
        )
        reciprocal = plot_reciprocal_boxplots_two_studies(
            records, datasets,
            output_dir / "Analysis1_reciprocal_edge_MI_Pearson_by_slice_boxplot_both_studies",
        )
    return coupling + reciprocal


def _mean_ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    sem = np.std(values, ddof=1) / np.sqrt(n)
    half_width = float(t.ppf(0.975, df=n - 1) * sem)
    return mean, mean - half_width, mean + half_width


def _save_figure(fig, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return str(png), str(pdf)


def plot_benchmark1_with_reverse_two_studies(
    records,
    datasets,
    method_colors,
    output_stem,
):
    """Plot one study per row and one curated feature type per column."""
    plt.close("all")
    max_lr_total = 1
    for dataset in datasets:
        frame = records[dataset]["benchmark"]
        count = sum(
            frame.loc[frame["FeatureType"].eq(feature), "RepresentativeLR"].nunique()
            for feature in FEATURE_TYPES
        )
        max_lr_total = max(max_lr_total, count)

    fig_width = max(10.5, 1.8 + 0.43 * max_lr_total)
    fig, axes = plt.subplots(
        len(datasets),
        len(FEATURE_TYPES),
        figsize=(fig_width, 3.15 * len(datasets)),
        constrained_layout=True,
        squeeze=False,
    )

    methods_present = set()
    for row, dataset in enumerate(datasets):
        long_df = records[dataset]["benchmark"].copy()
        dataset_methods = [
            method for method in CANONICAL_METHOD_ORDER
            if method in set(long_df["Method"].dropna())
        ]
        methods_present.update(dataset_methods)

        for col, feature_type in enumerate(FEATURE_TYPES):
            ax = axes[row, col]
            sub = long_df[long_df["FeatureType"].eq(feature_type)].copy()
            lr_order = (
                sub[sub["Method"].eq("SpiderNet")]
                .groupby("RepresentativeLR")["Correlation"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            base = np.arange(len(lr_order), dtype=float) * 1.10 + 1
            offsets = np.linspace(-0.41, 0.41, max(1, len(dataset_methods)))
            bounds = []

            for offset, method in zip(offsets, dataset_methods):
                means, lows, highs = [], [], []
                for lr_pair in lr_order:
                    values = sub.loc[
                        sub["RepresentativeLR"].eq(lr_pair)
                        & sub["Method"].eq(method),
                        "Correlation",
                    ].to_numpy(dtype=float)
                    mean, low, high = _mean_ci95(values)
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                means = np.asarray(means, dtype=float)
                lows = np.asarray(lows, dtype=float)
                highs = np.asarray(highs, dtype=float)
                ax.errorbar(
                    base + offset,
                    means,
                    yerr=np.vstack([means - lows, highs - means]),
                    fmt="o",
                    linestyle="none",
                    color=method_colors[method]["edge"],
                    ecolor=method_colors[method]["edge"],
                    markerfacecolor=method_colors[method]["fill"],
                    markeredgecolor=method_colors[method]["edge"],
                    markeredgewidth=0.8,
                    markersize=2.7,
                    elinewidth=0.8,
                    capsize=0,
                    zorder=3,
                )
                bounds.extend([lows, highs])

            ax.set_xticks(base)
            ax.set_xticklabels(lr_order, rotation=45, ha="right")
            ax.set_xlabel("Representative LR pairs")
            if col == 0:
                ax.set_ylabel("Spearman correlation")
            if row == 0:
                ax.set_title(feature_type)
            ax.text(
                -0.15,
                0.5,
                dataset,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=9,
                fontweight="bold",
                visible=(col == 0),
            )
            ax.axhline(0, color="0.6", linestyle="--", linewidth=0.8, zorder=1)
            ax.yaxis.grid(True, linewidth=0.6, color="0.88")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out")
            if len(base):
                ax.set_xlim(base[0] - 0.55, base[-1] + 0.55)
            finite = np.concatenate(bounds) if bounds else np.array([])
            finite = finite[np.isfinite(finite)]
            if finite.size:
                ymin, ymax = float(finite.min()), float(finite.max())
                pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.05
                ax.set_ylim(ymin - pad, ymax + pad)

    legend_order = [m for m in CANONICAL_METHOD_ORDER if m in methods_present]
    handles = [
        mpatches.Patch(
            facecolor=method_colors[m]["fill"],
            edgecolor=method_colors[m]["edge"],
            label=m,
        )
        for m in legend_order
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=min(6, len(handles)),
    )
    outputs = _save_figure(fig, output_stem)
    plt.show()
    return outputs


def plot_reciprocal_boxplots_two_studies(records, datasets, output_stem):
    """Plot reciprocal-edge MI correlations with one study per row."""
    plt.close("all")
    max_mis = max(
        pd.to_numeric(records[d]["boxplot"]["MI_index_1based"], errors="coerce").nunique()
        for d in datasets
    )
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(max(8.0, 0.34 * max_mis), 3.0 * len(datasets)),
        constrained_layout=True,
        squeeze=False,
    )

    for row, dataset in enumerate(datasets):
        ax = axes[row, 0]
        frame = records[dataset]["boxplot"].copy()
        frame["MI_index_1based"] = pd.to_numeric(
            frame["MI_index_1based"], errors="coerce"
        )
        frame["pearson_r"] = pd.to_numeric(frame["pearson_r"], errors="coerce")
        frame = frame.dropna(subset=["MI_index_1based", "pearson_r"])
        frame["MI_index_1based"] = frame["MI_index_1based"].astype(int)
        mi_indices = sorted(frame["MI_index_1based"].unique())
        grouped = [
            frame.loc[frame["MI_index_1based"].eq(mi), "pearson_r"].to_numpy(dtype=float)
            for mi in mi_indices
        ]
        positions = np.arange(1, len(mi_indices) + 1, dtype=float)
        bp = ax.boxplot(
            grouped,
            positions=positions,
            widths=0.60,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            boxprops={"color": "black", "linewidth": 0.7},
            whiskerprops={"color": "black", "linewidth": 0.7},
            capprops={"color": "black", "linewidth": 0.7},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("none")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"MI-{mi}" for mi in mi_indices], rotation=45, ha="right")
        ax.set_xlabel("Meta-interaction")
        ax.set_ylabel("Pearson correlation within slice")
        ax.set_title(dataset)
        ax.set_ylim(-1.03, 1.03)
        ax.yaxis.grid(True, color="black", linewidth=0.5, alpha=0.12)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    outputs = _save_figure(fig, output_stem)
    plt.show()
    return outputs
