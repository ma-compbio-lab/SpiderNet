"""Render correlation heatmaps and robustness boxplots from analysis results."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from robustness_analysis import (
    DATASETS, FIGURES, TABLES, PAIR_RESULTS, LR_PANELS, STABILITY_TABLES,
    MISANNOTATION_RATES, USE_PUBLISHED_Y_LIMITS_WHEN_POSSIBLE,
    write_json, record, label_rank_sum_tests, stability_repeat_summary,
)

# The saved notebook figures inherited this style from their interactive session.
# Set it explicitly so a fresh command-line process reproduces that appearance.
sns.set_style("darkgrid")
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "Arial",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220})
MI_CMAP = LinearSegmentedColormap.from_list("paper_MI", ["#ffffed", "#fff5a0", "#ffd24c", "#ff8524", "#702a1d"])
LR_CMAP = LinearSegmentedColormap.from_list("paper_LR", ["#ffffff", "#e7b5b9", "#af2639"])
METRICS = ["MI strength", "LR loading", "Regulatory gene", "Target gene"]
METRIC_LABELS = ["MI activity", "LR-pair loading", "Sender-regulator loading", "Receiver-target loading"]
METRIC_COLORS = ["#FFA500", "#646491", "#84cce2", "#e43429"]

def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    print("Saved:", FIGURES / f"{name}.pdf")

def title_strip(ax, text):
    ax.add_patch(Rectangle((0, 1.08), 1, 0.13, transform=ax.transAxes,
                           facecolor="#eeeeee", edgecolor="none", clip_on=False))
    ax.text(0.5, 1.145, text, transform=ax.transAxes, ha="center", va="center", fontsize=10)

def panel_letter(ax, letter):
    ax.text(-0.14, 1.20, letter, transform=ax.transAxes, fontsize=14, weight="bold", va="top")

def draw_mi_heatmap(ax, result, row_axis, col_axis):
    corr = result["corr"]
    pairs = sorted(result["pairs"], key=lambda p: p[0])
    ro = [i for i, _ in pairs] + [i for i in range(corr.shape[0]) if i not in {p[0] for p in pairs}]
    co = [j for _, j in pairs] + [j for j in range(corr.shape[1]) if j not in {p[1] for p in pairs}]
    ax.imshow(corr[np.ix_(ro, co)], vmin=0, vmax=1, cmap=MI_CMAP, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(len(co)), [f"MI-{j+1}" for j in co], rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(ro)), [f"MI-{i+1}" for i in ro], fontsize=6.5)
    ax.set_xticks(np.arange(-0.5, len(co), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ro), 1), minor=True)
    ax.grid(which="minor", color="#bdbb9b", linewidth=0.25)
    ax.tick_params(which="both", length=0, pad=2)
    ax.set_xlabel(col_axis)
    ax.set_ylabel(row_axis)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if result.get("tag"):
        write_json(TABLES / result["tag"] / "heatmap_display_order.json", {"row_mi_ids": [i+1 for i in ro], "column_mi_ids": [j+1 for j in co]})

def draw_lr_heatmap(ax, dataset, data):
    order = data["order"]
    idx = order["ordered_indices"]
    ax.imshow(data["corr"][np.ix_(idx, idx)], vmin=0, vmax=1, cmap=LR_CMAP, interpolation="nearest")
    for start, size, _ in order["block_spans"]:
        ax.add_patch(Rectangle((start-0.5, start-0.5), size, size, fill=False, edgecolor="#e3da00", linewidth=1.3))
        ax.axhline(start+size-0.5, color="#e3da00", alpha=0.5, linewidth=0.35)
        ax.axvline(start+size-0.5, color="#e3da00", alpha=0.5, linewidth=0.35)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("LR pairs")
    ax.set_ylabel("LR pairs")
    title_strip(ax, DATASETS[dataset]["title"])
    for spine in ax.spines.values():
        spine.set_visible(False)

def s31_specs():
    return [
        ("S31b_HGSOC_M12_vs_M15", "MIs (M = 12)", "MIs (M = 15)", DATASETS["HGSOC"]["title"]),
        ("S31b_PerturbFISH_M20_vs_M23", "MIs (M = 20)", "MIs (M = 23)", DATASETS["PerturbFISH"]["title"]),
        ("S31c_HGSOC_K5_vs_K8", "MIs (K = 5)", "MIs (K = 8)", DATASETS["HGSOC"]["title"]),
        ("S31b_HGSOC_M15_vs_M18", "MIs (M = 15)", "MIs (M = 18)", None),
        ("S31b_PerturbFISH_M23_vs_M26", "MIs (M = 23)", "MIs (M = 26)", None),
        ("S31c_HGSOC_K8_vs_K10", "MIs (K = 8)", "MIs (K = 10)", None),
    ]

def plot_s31():
    for dataset, data in LR_PANELS.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_lr_heatmap(ax, dataset, data)
        save_figure(fig, f"S31a_{dataset}")
        plt.close(fig)
    specs = s31_specs()
    for key, ylabel, xlabel, title in specs:
        if key in PAIR_RESULTS:
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            draw_mi_heatmap(ax, PAIR_RESULTS[key], ylabel, xlabel)
            if title:
                title_strip(ax, title)
            fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=MI_CMAP), ax=ax, fraction=0.035, pad=0.04, label="Spearman rho")
            save_figure(fig, key)
            plt.close(fig)
    if not all(key in PAIR_RESULTS for key, *_ in specs):
        record("S31", "combined figure", "pending", "Missing heatmap results; individual available panels exported.")
        return
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.5))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.90, hspace=0.60, wspace=0.42)
    for ax, (key, ylabel, xlabel, title) in zip(axes.flat, specs):
        draw_mi_heatmap(ax, PAIR_RESULTS[key], ylabel, xlabel)
        if title:
            title_strip(ax, title)
    panel_letter(axes[0, 0], "b"); panel_letter(axes[0, 2], "c")
    cax = fig.add_axes([0.42, 0.055, 0.16, 0.012])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=MI_CMAP), cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.set_label("Spearman rho of MI strengths")
    save_figure(fig, "Figure_S31bc")
    plt.show()
    if set(DATASETS) != set(LR_PANELS):
        record("S31", "full a-c figure", "pending", "S31a requires all four LR panels.")
        return
    fig = plt.figure(figsize=(14.5, 13.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 2.1], left=0.065, right=0.97, bottom=0.13, top=0.92, hspace=0.5)
    top = outer[0].subgridspec(1, 4, wspace=0.27)
    topaxes = []
    for j, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(top[0, j]); topaxes.append(ax)
        draw_lr_heatmap(ax, dataset, LR_PANELS[dataset])
    panel_letter(topaxes[0], "a")
    lower = outer[1].subgridspec(2, 3, hspace=0.65, wspace=0.40)
    axes = []
    for j, (key, ylabel, xlabel, title) in enumerate(specs):
        ax = fig.add_subplot(lower[j//3, j%3]); axes.append(ax)
        draw_mi_heatmap(ax, PAIR_RESULTS[key], ylabel, xlabel)
        if title:
            title_strip(ax, title)
    panel_letter(axes[0], "b"); panel_letter(axes[2], "c")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=LR_CMAP), ax=topaxes, orientation="horizontal", fraction=0.04, pad=0.15, shrink=0.16, ticks=[0, 0.5, 1])
    cb.set_label("Spearman rho of LR-pair co-expression")
    cax = fig.add_axes([0.42, 0.055, 0.16, 0.010])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=MI_CMAP), cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.set_label("Spearman rho of MI strengths")
    save_figure(fig, "Figure_S31")
    plt.show()

def box_with_points(ax, values, colors, seed=91):
    rng = np.random.default_rng(seed)
    for i, (v, color) in enumerate(zip(values, colors)):
        v = np.asarray(v, dtype=float)
        if not len(v) or not np.all(np.isfinite(v)):
            raise ValueError("Boxplot input contains missing/undefined correlations. Inspect the per-MI tables.")
        ax.boxplot(v, positions=[i], widths=0.52, whis=1.5, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "white", "edgecolor": "#555555", "linewidth": 0.9},
                   medianprops={"color": "#333333", "linewidth": 1.1},
                   whiskerprops={"color": "#555555", "linewidth": 0.8},
                   capprops={"color": "#555555", "linewidth": 0.8})
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(v)), v, s=19, c=color, alpha=0.76, edgecolors="none", zorder=3)
    ax.set_xlim(-0.5, len(values)-0.5)

def set_corr_limits(ax, values, lower, upper=1.005):
    vals = np.concatenate(values)
    low = min(lower, float(vals.min()) - 0.007)
    high = max(upper, float(vals.max()) + 0.003)
    if not USE_PUBLISHED_Y_LIMITS_WHEN_POSSIBLE:
        low = float(vals.min()) - max(0.005, float(np.ptp(vals))*0.12)
    ax.set_ylim(low, high)

def draw_s32a(ax, df):
    values = [df.loc[np.isclose(df.rate, rate), "corr"].to_numpy() for rate in MISANNOTATION_RATES]
    if any(len(v) != 15 for v in values):
        raise ValueError("S32a must have one Pearson correlation per each of 15 MIs at every rate.")
    box_with_points(ax, values, ["#666666"]*3)
    ax.set_xticks(range(3), [f"{r:.0%}" for r in MISANNOTATION_RATES])
    ax.set_xlabel("Misannotation rate")
    ax.set_ylabel("Pearson correlation with original-label model")
    ax.set_title("Cell-type label misannotation robustness", fontsize=10, pad=12)
    tests = label_rank_sum_tests(df)
    tests.to_csv(TABLES / "S32a_adjacent_rate_rank_sum_tests.csv", index=False)
    top = max(float(np.max(v)) for v in values)
    span = max(0.06, top-min(float(np.min(v)) for v in values))
    for i, p in enumerate(tests.pvalue):
        y = top + span*(0.045 + 0.08*i)
        h = span*0.018
        ax.plot([i, i, i+1, i+1], [y, y+h, y+h, y], color="#555555", linewidth=0.8)
        label = "ns" if p > 0.05 else ("***" if p < 0.001 else "**" if p < 0.01 else "*")
        ax.text(i+0.5, y+h, label, ha="center", va="bottom", fontsize=8)
    set_corr_limits(ax, values, 0.90, upper=top + span*0.23)
    lo, hi = ax.get_ylim()
    ticks = np.arange(np.ceil(lo / 0.04) * 0.04, min(hi, 1.0) + 1e-8, 0.04)
    ax.set_yticks(ticks, [f"{t:.2f}" if t < 0.9999 else "1.0" for t in ticks])
    ax.set_ylim(lo, hi)

def draw_stability(ax, df, experiment):
    summary = stability_repeat_summary(df)
    values = [summary.loc[summary.metric == m, "corr"].to_numpy() for m in METRICS]
    if any(len(v) != 10 for v in values):
        raise ValueError(f"{experiment}: expected ten complete repeats per output.")
    box_with_points(ax, values, METRIC_COLORS)
    ax.set_xticks(range(4), METRIC_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("SpiderNet output")
    ax.set_ylabel("Pearson correlation with " + ("reference model" if experiment == "Random seed" else "full-slice model"))
    ax.set_title("Random-seed robustness" if experiment == "Random seed" else "Robustness to 70%-slices subsampling", fontsize=10, pad=12)
    set_corr_limits(ax, values, 0.97 if experiment == "Random seed" else 0.80)
    lo, hi = ax.get_ylim()
    step = 0.01 if experiment == "Random seed" else 0.05
    ticks = np.arange(np.ceil(lo / step) * step, min(hi, 1.0) + 1e-8, step)
    ax.set_yticks(ticks, [f"{t:.2f}" if t < 0.9999 else "1.0" for t in ticks])
    ax.set_ylim(lo, hi)

def plot_s32():
    ready = {k: v for k, v in STABILITY_TABLES.items() if v is not None}
    specs = [("S32a", "a", lambda ax, df: draw_s32a(ax, df)),
             ("S32b", "b", lambda ax, df: draw_stability(ax, df, "Random seed")),
             ("S32c", "c", lambda ax, df: draw_stability(ax, df, "70% slices"))]
    for key, letter, draw in specs:
        if key in ready:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            fig.subplots_adjust(left=0.19, bottom=0.34, right=0.98, top=0.85)
            draw(ax, ready[key]); panel_letter(ax, letter)
            save_figure(fig, key)
            plt.close(fig)
    if not all(k in ready for k, *_ in specs):
        record("S32", "combined figure", "pending", "All three complete experiments are required.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.36, top=0.84, wspace=0.36)
    for ax, (key, letter, draw) in zip(axes, specs):
        draw(ax, ready[key]); panel_letter(ax, letter)
    save_figure(fig, "Figure_S32")
    plt.show()
