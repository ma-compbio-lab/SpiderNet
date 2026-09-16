import os
import numpy as np
import matplotlib.pyplot as plt
from SpiderNet.analysis import *

def insituplot_MIcascade(
    spatial,
    celltypes,
    edge_first,
    edge_second,
    color_map,
    sample_index,
    ct1,
    ct2,
    ct3,
    MI_first_id,
    MI_second_id,
    save_dir,
):
    """
    Plot a spatial high-order MI cascade for one sample.

    This function visualizes:
    1. all cells as colored scatter points by cell type,
    2. edges belonging to the first MI,
    3. edges belonging to the second MI.

    Parameters
    ----------
    spatial : np.ndarray
        Spatial coordinates of shape (n_cells, 2).
    celltypes : np.ndarray
        Cell-type labels of shape (n_cells,).
    edge_first : np.ndarray
        Array of shape (n_edges_1, 2) containing (sender, receiver) pairs
        for the first MI.
    edge_second : np.ndarray
        Array of shape (n_edges_2, 2) containing (sender, receiver) pairs
        for the second MI.
    color_map : dict
        Mapping from cell-type name to color.
    sample_index : int
        Index of the current sample.
    ct1, ct2, ct3 : str
        Cell-type triple label for the cascade.
    MI_first_id : int
        One-based MI ID for the first MI.
    MI_second_id : int
        One-based MI ID for the second MI.
    save_dir : str
        Output directory.
    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.family"] = "arial"

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("white")
    ax.grid(False)

    # Plot all cells
    classes = np.unique(celltypes)
    for cls in classes:
        idx = (celltypes == cls)
        if np.any(idx):
            ax.scatter(
                spatial[idx, 0],
                spatial[idx, 1],
                s=6,
                c=color_map.get(cls, "gray"),
                label=cls,
                alpha=1,
                edgecolors="none",
            )

    # Plot MI-first edges
    for sender, receiver in edge_first:
        x0, y0 = spatial[sender]
        x1, y1 = spatial[receiver]
        ax.arrow(
            x0, y0, x1 - x0, y1 - y0,
            length_includes_head=True,
            lw=1,
            head_width=15,
            head_length=15,
            fc="#C71E64",
            ec="#C71E64",
            alpha=0.6,
        )

    # Plot MI-second edges
    for sender, receiver in edge_second:
        x0, y0 = spatial[sender]
        x1, y1 = spatial[receiver]
        ax.arrow(
            x0, y0, x1 - x0, y1 - y0,
            length_includes_head=True,
            lw=1,
            head_width=15,
            head_length=15,
            fc="#4D2D8C",
            ec="#4D2D8C",
            alpha=0.6,
        )

    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"{ct1} → {ct2} → {ct3}\n(MI-{MI_first_id} → MI-{MI_second_id})",
        fontsize=12
    )
    plt.tight_layout()

    out_dir = os.path.join(save_dir, "Insitu_High_order_MI")
    os.makedirs(out_dir, exist_ok=True)

    triple_name = f"{ct1} -> {ct2} -> {ct3}"
    triple_safe = safe_filename(triple_name)
    base = f"Spatial_{sample_index}_{triple_safe}_MI-{MI_first_id}_MI-{MI_second_id}"

    plt.savefig(os.path.join(out_dir, base + ".pdf"), dpi=300, facecolor="white")
    plt.savefig(os.path.join(out_dir, base + ".png"), dpi=300, facecolor="white")
    plt.close(fig)

def Heatmap_celltypetriplet_prop(data, title, xlabel, ylabel, filename, highlight_malignant=False,file_savepath_main="./results"):
    import seaborn as sns
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'arial'
    """Plot clustered heatmap for triple × MI cascade proportion."""
    # --- determine figure size dynamically ---
    fig_w = max(6, 0.6 * data.shape[0]) * 0.8
    fig_h = max(6, 0.5 * data.shape[1])

    # --- create clustermap ---
    g = sns.clustermap(
        data.T,
        cmap="Reds",
        vmin=0, vmax=0.4,
        figsize=(fig_w, fig_h),
        square=False,
        row_cluster=True, col_cluster=True,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=None,
        linecolor="lightgrey", linewidth=0.5
    )

    # --- hide dendrograms ---
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    # --- axis labels ---
    g.ax_heatmap.set_xlabel(xlabel, fontsize=14)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=14)
    g.ax_heatmap.set_title(title, fontsize=16, pad=12)

    # --- highlight "Malignant" triples in x-axis if required ---
    xticklabels = g.ax_heatmap.get_xticklabels()
    for label in xticklabels:
        label_text = label.get_text()
        label.set_color("red" if ("Malignant" in label_text and highlight_malignant) else "black")

    # --- format tick labels ---
    g.ax_heatmap.set_xticklabels(xticklabels, rotation=60, ha="right", fontsize=11)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=11)

    # --- layout & save ---
    g.fig.tight_layout()
    g.fig.subplots_adjust(bottom=0.25, left=0.25)
    g.fig.savefig(file_savepath_main + "/" + filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {filename}")

"""Visualization utilities for MI cascade analysis in SpiderNet."""

from typing import Dict, Iterable, List, Literal, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch
from scipy.stats import ranksums

GroupName = Literal["BothPos", "OnlyMIfirst", "OnlyMIsecond", "BothNeg"]

DEFAULT_PRETTY_GROUP_LABELS = {
    "BothPos": "MI cascade",
    "OnlyMIfirst": "MI-first only",
    "OnlyMIsecond": "MI-second only",
    "BothNeg": "No MI",
}

DEFAULT_TWO_GROUP_COLORS = {
    "BothPos": "#ABD58E",
    "OnlyMIsecond": "#F0F6DE",
}

def set_nature_style() -> None:
    """Set Nature-style and Illustrator-friendly plotting defaults."""
    plt.close("all")
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    sns.set_theme(style="white")

def p_to_star(p: float) -> str:
    """Convert p-value to significance star string."""
    if np.isnan(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

def add_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    lw: float = 0.8,
    fs: float = 8,
) -> None:
    """Add a significance bracket between two x positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=fs, color="black")

def compute_ranksum_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "greater",
) -> float:
    """Compute Wilcoxon rank-sum p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    _, pval = ranksums(x, y, alternative=alternative)
    return float(pval)

def _get_box_patches(ax):
    """Return box patches drawn in an axis."""
    return [p for p in ax.patches if isinstance(p, PathPatch)]

def _recolor_boxes(ax, group_order: Sequence[str], group2color: Dict[str, str]) -> None:
    """Recolor seaborn box patches using a group-to-color mapping."""
    patches = _get_box_patches(ax)
    if len(patches) < len(group_order):
        return
    for i, group in enumerate(group_order):
        patches[i].set_facecolor(group2color[group])
        patches[i].set_edgecolor("black")
        patches[i].set_linewidth(0.8)

def build_long_dataframe(
    score_dict: Dict[str, Dict[str, List[float]]],
    feature_col: str = "feature",
    group_col: str = "group",
    value_col: str = "value",
) -> pd.DataFrame:
    """Convert {feature: {group: values}} into a long-form dataframe."""
    records = []
    for feature_name, group_dict in score_dict.items():
        for group_name, values in group_dict.items():
            for value in values:
                records.append(
                    {
                        feature_col: feature_name,
                        group_col: group_name,
                        value_col: value,
                    }
                )
    df = pd.DataFrame(records)
    if len(df) == 0:
        return pd.DataFrame(columns=[feature_col, group_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, feature_col, group_col]).copy()
    return df

def plot_two_group_feature_boxes(
    score_dict: Dict[str, Dict[str, List[float]]],
    group_order: Sequence[str] = ("BothPos", "OnlyMIsecond"),
    pretty_group_labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    ylabel: str = "",
    output_path: Optional[str] = None,
    show: bool = True,
    pvalue_alternative: str = "greater",
    annotate_with: Literal["pvalue", "stars"] = "pvalue",
    figsize: tuple = (2.6, 3.2),
) -> None:
    """Draw one two-group boxplot per feature and save each as a separate panel.

    Parameters
    ----------
    score_dict
        Nested dictionary in the form {feature_name: {group_name: [values...]}}.
    """
    set_nature_style()
    pretty_group_labels = pretty_group_labels or DEFAULT_PRETTY_GROUP_LABELS
    colors = colors or DEFAULT_TWO_GROUP_COLORS

    for feature_name, group_dict in score_dict.items():
        data = [np.asarray(group_dict.get(group, []), dtype=float) for group in group_order]
        fig, ax = plt.subplots(figsize=figsize)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
            medianprops=dict(color="black", linewidth=0.8),
            boxprops=dict(edgecolor="black", linewidth=0.8),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
        )
        for box, group in zip(bp["boxes"], group_order):
            box.set_facecolor(colors[group])

        pval = compute_ranksum_pvalue(data[0], data[1], alternative=pvalue_alternative)
        text = f"p={pval:.2e}" if annotate_with == "pvalue" and np.isfinite(pval) else p_to_star(pval)

        finite_arrays = [d[np.isfinite(d)] for d in data if len(d) > 0]
        if len(finite_arrays) > 0 and sum(len(arr) for arr in finite_arrays) > 0:
            allv = np.concatenate(finite_arrays)
            y_max = np.nanmax(allv)
            y_min = np.nanmin(allv)
        else:
            y_max, y_min = 1.0, 0.0
        yrng = (y_max - y_min) + 1e-12
        y = y_max + 0.10 * yrng
        h = 0.05 * yrng
        add_sig_bracket(ax, 1, 2, y=y, h=h, text=text, lw=0.8, fs=8)
        ax.set_ylim(top=y + h * 2.2)

        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([pretty_group_labels[g] for g in group_order], rotation=20, ha="right")
        ax.set_title(str(feature_name), pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        fig.tight_layout()

        if output_path is not None:
            safe_name = str(feature_name).replace("/", "_").replace(" ", "_")
            fig.savefig(f"{output_path}_{safe_name}.pdf", bbox_inches="tight", facecolor="white")

        if show:
            plt.show()
        plt.close(fig)

def plot_facet_boxplots(
    score_dict: Dict[str, Dict[str, List[float]]],
    group_order: Sequence[str] = ("BothPos", "OnlyMIsecond"),
    pretty_group_labels: Optional[Dict[str, str]] = None,
    group2color: Optional[Dict[str, str]] = None,
    ylabel: str = "",
    output_path: Optional[str] = None,
    show: bool = True,
    pvalue_alternative: str = "greater",
    annotate_with: Literal["pvalue", "stars"] = "pvalue",
    col_wrap: int = 4,
    height: float = 2.8,
    aspect: float = 0.85,
    figure_title: Optional[str] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> None:
    """Draw faceted two-group boxplots for many features."""
    set_nature_style()
    pretty_group_labels = pretty_group_labels or DEFAULT_PRETTY_GROUP_LABELS
    group2color = group2color or DEFAULT_TWO_GROUP_COLORS

    df = build_long_dataframe(score_dict)
    if df.empty:
        raise ValueError("No data available for plotting.")

    if feature_order is None:
        feature_order = sorted(df["feature"].unique())

    df["group"] = pd.Categorical(df["group"], categories=list(group_order), ordered=True)
    df["feature"] = pd.Categorical(df["feature"], categories=list(feature_order), ordered=True)

    g = sns.catplot(
        data=df,
        x="group",
        y="value",
        col="feature",
        col_order=list(feature_order),
        kind="box",
        order=list(group_order),
        showfliers=False,
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        linewidth=0.8,
        boxprops={"edgecolor": "black"},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 0.8},
    )

    for ax, feature_name in zip(g.axes.flatten(), feature_order):
        _recolor_boxes(ax, group_order=group_order, group2color=group2color)

        subdf = df[df["feature"] == feature_name]
        v1 = subdf.loc[subdf["group"] == group_order[0], "value"].values
        v2 = subdf.loc[subdf["group"] == group_order[1], "value"].values
        pval = compute_ranksum_pvalue(v1, v2, alternative=pvalue_alternative)

        finite_arrays = [arr[np.isfinite(arr)] for arr in (v1, v2) if len(arr) > 0]
        if len(finite_arrays) > 0 and sum(len(arr) for arr in finite_arrays) > 0:
            allv = np.concatenate(finite_arrays)
            y_max = np.nanmax(allv)
            y_min = np.nanmin(allv)
            yrng = (y_max - y_min) + 1e-12
            y = y_max + 0.10 * yrng
            h = 0.05 * yrng
            text = f"p={pval:.2e}" if annotate_with == "pvalue" and np.isfinite(pval) else p_to_star(pval)
            add_sig_bracket(ax, 0, 1, y=y, h=h, text=text, lw=0.8, fs=8)
            ax.set_ylim(top=y + h * 2.2)

        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels([pretty_group_labels[gname] for gname in group_order], rotation=20, ha="right")
        ax.set_title(str(feature_name), fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    g.set_axis_labels("", ylabel)
    if figure_title is not None:
        g.fig.suptitle(figure_title, fontsize=9)
        g.fig.subplots_adjust(top=0.88, wspace=0.35, hspace=0.45)
    else:
        g.fig.subplots_adjust(wspace=0.35, hspace=0.45)

    if output_path is not None:
        g.fig.savefig(output_path, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(g.fig)

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import mannwhitneyu




def plot_mi_change_heatmap(mi_change_df, save_prefix, xlabel, ylabel, annotate_threshold=0.4, figsize=(5, 6.6),ifshow=False):
    apply_publication_style(font_size=8)
    plot_df = mi_change_df.T.astype(float)
    annot = plot_df.applymap(lambda v: "" if (pd.isna(v) or abs(v) < annotate_threshold) else f"{v:.2f}")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_df,
        annot=annot,
        fmt="",
        cmap="vlag",
        center=0,
        cbar=False,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        linecolor="#D0D0D0",
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    if ifshow:
        plt.show()
    plt.close(fig)


from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import mannwhitneyu

def apply_publication_style(font_size=8):
    rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": max(font_size - 1, 6),
        "ytick.labelsize": max(font_size - 1, 6),
        "legend.fontsize": max(font_size - 1, 6),
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })
    sns.set_style("white")

def format_pvalue(p):
    if p < 1e-4:
        return "P < 1×10$^{-4}$"
    if p < 0.001:
        a, b = f"{p:.1e}".split("e")
        return f"P = {a}×10$^{{{int(b)}}}$"
    return f"P = {p:.3f}"

def plot_gene_correlation_barplot(summary_df, save_prefix):
    apply_publication_style(font_size=9)
    plot_df = summary_df.reset_index().rename(columns={"index": "Gene"})
    plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=plot_df["Gene"].tolist(), ordered=True)

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    sns.barplot(data=plot_df, x="Gene", y="Correlation", color="#DA6E6E", errorbar=None, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("T-cell perturbation effect correlation")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_group_distribution(values_control, values_perturb, y_label, save_prefix, plot_kind="violin", group_order=("Perturb edge", "Control edge"), palette=None, ylim_top=None,
                            ifshow=False):
    apply_publication_style(font_size=8)

    values_control = np.asarray(values_control, dtype=float)
    values_perturb = np.asarray(values_perturb, dtype=float)
    values_control = values_control[np.isfinite(values_control)]
    values_perturb = values_perturb[np.isfinite(values_perturb)]

    if palette is None:
        palette = {
            "Control edge": "#F0F3FF",
            "Perturb edge": "#836FFF",
        }

    plot_df = pd.DataFrame({
        "Group": (["Control edge"] * len(values_control)) + (["Perturb edge"] * len(values_perturb)),
        "Value": np.concatenate([values_control, values_perturb]),
    })
    plot_df["Group"] = pd.Categorical(plot_df["Group"], categories=list(group_order), ordered=True)

    _, p_val = mannwhitneyu(values_control, values_perturb, alternative="two-sided")
    p_text = format_pvalue(p_val)

    def upper_whisker(x):
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        return q3 + 1.5 * iqr

    def lower_whisker_like(x):
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        return q1 - 0.5 * iqr

    y_upper = max(upper_whisker(values_control), upper_whisker(values_perturb)) + 0.1
    y_lower = min(lower_whisker_like(values_control), lower_whisker_like(values_perturb))

    fig, ax = plt.subplots(figsize=(1.8, 2.8))

    if plot_kind == "violin":
        sns.violinplot(
            data=plot_df,
            x="Group",
            y="Value",
            order=list(group_order),
            palette=palette,
            cut=0,
            inner=None,
            linewidth=0.8,
            width=0.8,
            ax=ax,
        )
    else:
        sns.boxplot(
            data=plot_df,
            x="Group",
            y="Value",
            order=list(group_order),
            palette=palette,
            width=0.55,
            showfliers=False,
            linewidth=0.8,
            ax=ax,
        )

    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(axis="x", labelrotation=0)

    if ylim_top is None:
        ax.set_ylim(y_lower, y_upper)
    else:
        ax.set_ylim(y_lower, ylim_top)

    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_bar = ax.get_ylim()[1] - 0.08 * y_range
    h = 0.03 * y_range
    ax.plot([0, 0, 1, 1], [y_bar, y_bar + h, y_bar + h, y_bar], lw=0.8, c="black")
    ax.text(0.5, y_bar + h + 0.01 * y_range, p_text, ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    if ifshow:
        plt.show()
    plt.close(fig)

def plot_lr_positive_proportion_bar(df, save_prefix,ifshow=False):
    apply_publication_style(font_size=8)

    palette = {
        "Control edge": "#F0F3FF",
        "Perturb edge": "#836FFF",
    }

    df = df.copy()
    df["LR_pair"] = df["LR_pair"].astype(str)

    x_labels = df["LR_pair"].tolist()
    control = df["Control_edge_positive_prop"].to_numpy(dtype=float)
    perturb = df["Perturb_edge_positive_prop"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.8, 2.8))
    x = np.arange(len(x_labels))
    w = 0.38

    ax.bar(
        x - w / 2, perturb, width=w,
        label="Perturb edge",
        color=palette["Perturb edge"],
        edgecolor="black", linewidth=0.6
    )
    ax.bar(
        x + w / 2, control, width=w,
        label="Control edge",
        color=palette["Control edge"],
        edgecolor="black", linewidth=0.6
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=40, ha="right")
    ax.set_ylabel("Positive proportion")
    ax.set_xlabel("LR pair")
    ax.set_ylim(0, max(np.nanmax(control), np.nanmax(perturb)) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    if ifshow:
        plt.show()
    plt.close(fig)

def plot_marker_tertile_boxplot(x_values, y_values, marker_name, xlabel, ylabel, save_prefix):
    apply_publication_style(font_size=8)

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    mask = y_values > 0
    x = x_values[mask]
    y = y_values[mask]

    q1, q2 = np.percentile(x, [33.3, 66.6])
    group = np.digitize(x, bins=[q1, q2])

    fig, ax = plt.subplots(figsize=(3.0, 3.2))
    sns.boxplot(
        x=group, y=y, ax=ax,
        palette=["#8ecae6", "#ffb703", "#fb8500"],
        showfliers=False,
    )
    sns.stripplot(
        x=group, y=y, ax=ax,
        color="black", size=1, alpha=0.5,
    )

    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or f"{marker_name} expression")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", width=0.8)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)




