"""Original loading plots and spatial diagnostics from saved simulation inputs."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


def reproduce_sample_figures(setting_index=0, experiment_index=18):
    """Replot spatial inputs and exported loadings without fitting a model."""
    import json
    import shutil
    import torch
    import simulation_benchmark_utils as sim
    from SimulationData_generation import SimulationConfig, plot_celltype_map, plot_inner_area_map, plot_edge_map

    setting = sim.SETTING_LIST[setting_index]
    data_dir = sim.resolve_data_root() / setting / f"Experiment_{experiment_index}"
    source_dir = sim.get_local_result_dir("SpiderNet", setting, experiment_index)
    model_out = sim.OUTPUT_ROOT / "SpiderNet" / setting / f"Experiment_{experiment_index}" / "SpiderNet_Result_Mode_cell_class"
    data_out = sim.OUTPUT_ROOT / "Simulation_Data" / setting / f"Experiment_{experiment_index}"
    model_out.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)
    cfg = SimulationConfig()
    location = pd.read_csv(data_dir / "spatial_location.csv", index_col=0).to_numpy()
    celltypes = pd.read_csv(data_dir / "cell_metadf.csv", index_col=0)["Celltype"].to_numpy()
    edges = pd.read_csv(data_dir / "edge_metadf.csv", index_col=0).rename(columns={"MetaItype": "MI_type"})
    # These are the saved, already-expanded coordinates. Do not expand them again.
    radius = cfg.tissue_size[0] * cfg.inner_circle_radius_ratio
    inner = (np.sqrt(location[:, 0] ** 2 + location[:, 1] ** 2) < radius).astype(int)
    plot_celltype_map(data_out, location, celltypes, cfg)
    plot_inner_area_map(data_out, location, inner)
    for mode in ("all", "MI-1", "MI-2"):
        plot_edge_map(data_out, location, celltypes, edges, mode=mode)

    def tensor(array):
        return torch.as_tensor(array, dtype=torch.float32)

    intrinsic = pd.read_csv(source_dir / "loading_intrinsic_use.csv", index_col=0).to_numpy()
    lr = np.load(source_dir / "loading_LR_use.npy", allow_pickle=False)
    sender = np.load(source_dir / "loading_sender_use.npy", allow_pickle=False)
    receiver = np.load(source_dir / "loading_receiver_use.npy", allow_pickle=False)
    genes = pd.read_csv(data_dir / "gene_exp.csv", nrows=0, index_col=0).columns.to_numpy()
    plot_trained_loadings(tensor(intrinsic), tensor(lr), tensor(sender), tensor(receiver),
                          {"genenames": genes}, str(model_out))
    # The original intrinsic-expression layer was not saved separately. Its two
    # diagnostic panels cannot be reconstructed from the mixed expression matrix.
    copied = {}
    for name in ("Gene_expression_celltype.pdf", "Gene_expression_celltype_choose.pdf"):
        source = data_dir / name
        if source.is_file():
            shutil.copy2(source, data_out / name)
            copied[name] = {"source": str(source), "sha256": sim._sha256(source),
                            "action": "copied existing diagnostic; latent expression layer unavailable"}
    for name in ("Benchmark_MacroMetrics.csv", "LoadingRank_Ratio_Summary.csv",
                 "LoadingBlockMass_Ratio_Summary.csv", "ROC_AUROC_all.csv", "PUC_AUPRC_all.csv"):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, model_out / name)
    (data_out / "copied_diagnostics.json").write_text(json.dumps(copied, indent=2), encoding="utf-8")
    print(f"Saved sample figures to {data_out} and {model_out}")

def set_figure_style():
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3

def save_ai_pdf(fig, out_path, transparent=True):
    fig.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight",
        transparent=transparent,
    )

def make_linear_cmap(color_list, cmap_name):
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N=256)

def plot_matrix_heatmap(
    mat,
    row_labels,
    out_path,
    xlabel,
    ylabel,
    cmap,
    fig_w,
    fig_h,
    x_tick_labels=None,
    max_xticks=25,
    show_x_ticks=True,
    cbar_fraction=0.03,
    cbar_pad=0.02,
):
    set_figure_style()
    plt.close("all")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)

    if show_x_ticks:
        if x_tick_labels is None:
            x_tick_labels = [str(i + 1) for i in range(mat.shape[1])]
        step = max(1, int(np.ceil(mat.shape[1] / max_xticks)))
        xticks = np.arange(0, mat.shape[1], step)
        ax.set_xticks(xticks)
        ax.set_xticklabels([x_tick_labels[i] for i in xticks], rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3)
    cbar.outline.set_linewidth(0.8)

    fig.tight_layout()
    save_ai_pdf(fig, out_path, transparent=True)
    plt.close(fig)

def plot_matrix_heatmap_vector(
    mat,
    row_labels,
    out_path,
    xlabel,
    ylabel,
    cmap,
    fig_w,
    fig_h,
    x_tick_labels=None,
    max_xticks=25,
    show_x_ticks=True,
    cbar_fraction=0.03,
    cbar_pad=0.02,
    cell_edgecolor="none",
    cell_linewidth=0.0,
):
    set_figure_style()
    plt.close("all")

    mat = np.asarray(mat, dtype=float)
    n_rows, n_cols = mat.shape

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    if vmin == vmax:
        vmax = vmin + 1e-8

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap

    # Draw each heatmap cell as an individual vector rectangle
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            facecolor = cmap_obj(norm(val))
            rect = Rectangle(
                (j, i), 1, 1,
                facecolor=facecolor,
                edgecolor=cell_edgecolor,
                linewidth=cell_linewidth
            )
            ax.add_patch(rect)

    # Set axis limits and invert y-axis to match heatmap convention
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)

    # Y-axis labels
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)

    # X-axis labels
    if show_x_ticks:
        if x_tick_labels is None:
            x_tick_labels = [str(i + 1) for i in range(n_cols)]
        step = max(1, int(np.ceil(n_cols / max_xticks)))
        xticks = np.arange(0, n_cols, step) + 0.5
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [x_tick_labels[i] for i in range(0, n_cols, step)],
            rotation=45,
            ha="right",
            fontsize=7
        )
    else:
        ax.set_xticks([])

    # Clean frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ScalarMappable only for colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3)
    cbar.outline.set_linewidth(0.8)

    fig.tight_layout()
    save_ai_pdf(fig, out_path, transparent=True)
    plt.close(fig)

def _normalize_loading_rows_for_summary(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        row_sum = np.nansum(arr)
        if (not np.isfinite(row_sum)) or row_sum <= 0:
            return np.zeros_like(arr, dtype=float)
        out = arr / row_sum
        out[~np.isfinite(out)] = 0.0
        return out

    out = arr.copy()
    if out.size == 0:
        return out
    row_sum = np.nansum(out, axis=1, keepdims=True)
    row_sum[(~np.isfinite(row_sum)) | (row_sum <= 0)] = 1.0
    out = out / row_sum
    out[~np.isfinite(out)] = 0.0
    return out

def _normalize_loading_columns_for_summary(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        col_sum = np.nansum(arr)
        if (not np.isfinite(col_sum)) or col_sum <= 0:
            return np.zeros_like(arr, dtype=float)
        out = arr / col_sum
        out[~np.isfinite(out)] = 0.0
        return out

    out = arr.copy()
    if out.size == 0:
        return out
    col_sum = np.nansum(out, axis=0, keepdims=True)
    col_sum[(~np.isfinite(col_sum)) | (col_sum <= 0)] = 1.0
    out = out / col_sum
    out[~np.isfinite(out)] = 0.0
    return out

def _build_loading_truth_index_dict(n_lr, n_gene):
    lr_split = max(1, n_lr // 2)
    return {
        "lr_idx_mi1": np.arange(0, lr_split),
        "lr_idx_mi2": np.arange(lr_split, n_lr),
        "sender_idx_mi1": np.arange(min(40, n_gene), min(50, n_gene)),
        "sender_idx_mi2": np.arange(min(50, n_gene), min(60, n_gene)),
        "receiver_idx_mi1": np.arange(min(60, n_gene), min(70, n_gene)),
        "receiver_idx_mi2": np.arange(min(70, n_gene), min(80, n_gene)),
    }

def _mean_component_share_on_block(arr_colnorm, row_idx, col_idx):
    if row_idx is None or arr_colnorm.ndim != 2 or arr_colnorm.shape[0] == 0:
        return np.nan
    if row_idx >= arr_colnorm.shape[0] or col_idx.size == 0:
        return np.nan
    vals = arr_colnorm[row_idx, col_idx]
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))

def _top1_hit_ratio_on_block(arr_colnorm, row_idx, col_idx):
    if row_idx is None or arr_colnorm.ndim != 2 or arr_colnorm.shape[0] == 0:
        return np.nan
    if row_idx >= arr_colnorm.shape[0] or col_idx.size == 0:
        return np.nan
    sub = arr_colnorm[:, col_idx]
    if sub.size == 0:
        return np.nan
    top_rows = np.nanargmax(sub, axis=0)
    return float(np.mean(top_rows == row_idx))

def _rank_ratio_on_block(arr_colnorm, row_idx, col_idx):
    mean_share = _mean_component_share_on_block(arr_colnorm, row_idx, col_idx)
    top1_ratio = _top1_hit_ratio_on_block(arr_colnorm, row_idx, col_idx)
    return float(np.nanmean([mean_share, top1_ratio]))

def _match_components_by_lr_block(loading_lr):
    loading_lr_colnorm = _normalize_loading_columns_for_summary(loading_lr)
    if loading_lr_colnorm.ndim == 1:
        loading_lr_colnorm = loading_lr_colnorm.reshape(1, -1)

    n_comp = loading_lr_colnorm.shape[0]
    truth_idx = _build_loading_truth_index_dict(n_lr=loading_lr_colnorm.shape[1], n_gene=0)
    lr_idx_mi1 = truth_idx["lr_idx_mi1"]
    lr_idx_mi2 = truth_idx["lr_idx_mi2"]

    mi1_scores = np.array([
        _rank_ratio_on_block(loading_lr_colnorm, i, lr_idx_mi1)
        for i in range(n_comp)
    ], dtype=float)
    mi2_scores = np.array([
        _rank_ratio_on_block(loading_lr_colnorm, i, lr_idx_mi2)
        for i in range(n_comp)
    ], dtype=float)

    if n_comp == 0:
        return {
            "mi1_component": None,
            "mi2_component": None,
            "mi1_scores": mi1_scores,
            "mi2_scores": mi2_scores,
        }

    if n_comp == 1:
        mi1_component = 0
        mi2_component = 0
    else:
        best_pair = None
        best_score = -np.inf
        for i in range(n_comp):
            for j in range(n_comp):
                if i == j:
                    continue
                score = np.nan_to_num(mi1_scores[i], nan=-np.inf) + np.nan_to_num(mi2_scores[j], nan=-np.inf)
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_pair is None:
            mi1_component = int(np.nanargmax(np.nan_to_num(mi1_scores, nan=-np.inf)))
            remaining = [k for k in range(n_comp) if k != mi1_component]
            if len(remaining) == 0:
                mi2_component = mi1_component
            else:
                rem_scores = mi2_scores[remaining]
                mi2_component = int(remaining[int(np.nanargmax(np.nan_to_num(rem_scores, nan=-np.inf)))])
        else:
            mi1_component, mi2_component = best_pair

    return {
        "mi1_component": int(mi1_component),
        "mi2_component": int(mi2_component),
        "mi1_scores": mi1_scores,
        "mi2_scores": mi2_scores,
    }

def _build_ordered_loading_component_labels(n_comp, mi1_component, mi2_component):
    ordered_components = []
    if mi1_component is not None:
        ordered_components.append(mi1_component)
    if mi2_component is not None and mi2_component not in ordered_components:
        ordered_components.append(mi2_component)
    ordered_components.extend([i for i in range(n_comp) if i not in ordered_components])

    labels = []
    for comp_idx in ordered_components:
        if comp_idx == mi1_component:
            labels.append("Matched MI-1")
        elif comp_idx == mi2_component:
            labels.append("Matched MI-2")
        else:
            labels.append(f"Component {comp_idx + 1}")
    return ordered_components, labels

def plot_trained_loadings(Loading_intrinsic, loading_LR, loading_sender, loading_receiver, SpiderNet_data_pyg, file_savepath_main):
    Loading_intrinsic_use = Loading_intrinsic.to("cpu").detach().numpy()
    Loading_intrinsic_use_colSums = np.sum(Loading_intrinsic_use, axis=0)
    Loading_intrinsic_use_colSums[Loading_intrinsic_use_colSums == 0] = 1
    Loading_intrinsic_norm = Loading_intrinsic_use / Loading_intrinsic_use_colSums

    loading_LR_use = loading_LR.to("cpu").detach().numpy()
    loading_sender_use = loading_sender.to("cpu").detach().numpy()
    loading_receiver_use = loading_receiver.to("cpu").detach().numpy()

    loading_match_info = _match_components_by_lr_block(loading_LR_use)
    mi1_component = loading_match_info["mi1_component"]
    mi2_component = loading_match_info["mi2_component"]
    ordered_components, ordered_component_labels = _build_ordered_loading_component_labels(
        n_comp=loading_LR_use.shape[0],
        mi1_component=mi1_component,
        mi2_component=mi2_component,
    )

    set_figure_style()
    plt.close("all")
    fig, ax = plt.subplots()
    cax = ax.imshow(Loading_intrinsic_norm, aspect="auto")
    plt.colorbar(cax)
    plt.xlabel("Gene")
    plt.ylabel("Cell type")
    plt.title("Normalized loading matrix of intrinsic genes")
    celltype_labels = ["Celltype A", "Celltype B", "Celltype C"]
    ax.set_yticks(np.arange(len(celltype_labels)))
    ax.set_yticklabels(celltype_labels)
    ax.set_xticks([])
    save_ai_pdf(fig, os.path.join(file_savepath_main, "Normalized_loading_intrinsic.pdf"), transparent=True)
    plt.close(fig)

    loading_LR_plot = loading_LR_use[ordered_components, :]
    loading_LR_use_colSums = np.sum(loading_LR_plot, axis=0)
    loading_LR_use_colSums[loading_LR_use_colSums == 0] = 1
    loading_LR_norm = loading_LR_plot / loading_LR_use_colSums
    lr_loading_cmap = make_linear_cmap(
        ["#FFFFFF", "#636491"],
        "lr_loading_cmap",
    )

    mat = loading_LR_norm
    n_rows, n_cols = mat.shape
    fig_w = max(4.5, min(12, 0.18 * n_cols))
    fig_h = 1.6 + 0.35 * n_rows
    lr_labels = [f"LR-{i+1}" for i in range(n_cols)]

    plot_matrix_heatmap(
        mat=mat,
        row_labels=ordered_component_labels,
        out_path=os.path.join(file_savepath_main, "Normalized_loading_LR.pdf"),
        xlabel="LR pair",
        ylabel="Meta-interaction",
        cmap=lr_loading_cmap,
        fig_w=fig_w,
        fig_h=fig_h,
        x_tick_labels=lr_labels,
        max_xticks=25,
        show_x_ticks=True,
    )

    loading_envir_use = np.vstack(
        [
            loading_sender_use[mi1_component, :],
            loading_receiver_use[mi1_component, :],
            loading_sender_use[mi2_component, :],
            loading_receiver_use[mi2_component, :],
        ]
    )
    loading_envir_use = pd.DataFrame(
        loading_envir_use,
        columns=SpiderNet_data_pyg["genenames"],
        index=["MI-1_sending", "MI-1_receiving", "MI-2_sending", "MI-2_receiving"],
    )

    loading_envir_use = loading_envir_use.to_numpy()
    loading_envir_use_colSums = np.sum(loading_envir_use, axis=0)
    loading_envir_use_colSums[loading_envir_use_colSums == 0] = 1
    loading_envir_use_norm = loading_envir_use / loading_envir_use_colSums
    loading_envir_use_norm = pd.DataFrame(
        loading_envir_use_norm,
        columns=SpiderNet_data_pyg["genenames"],
        index=["MI-1_sending", "MI-1_receiving", "MI-2_sending", "MI-2_receiving"],
    )

    full_loading_cmap = make_linear_cmap(["#FFFFFF", "#BDBDBD", "#4D4D4D"], "full_loading_cmap")

    plot_matrix_heatmap(
        mat=loading_envir_use_norm.values,
        row_labels=list(loading_envir_use_norm.index),
        out_path=os.path.join(file_savepath_main, "Normalized_loading_envir.pdf"),
        xlabel="Genes",
        ylabel="",
        cmap=full_loading_cmap,
        fig_w=6.0,
        fig_h=1.6,
        x_tick_labels=None,
        show_x_ticks=False,
    )

    regulator_col_start = 40
    regulator_col_end = 60
    target_col_start = 60
    target_col_end = 80

    regulator_loading_norm = loading_envir_use_norm.iloc[:, regulator_col_start:regulator_col_end].copy()
    target_loading_norm = loading_envir_use_norm.iloc[:, target_col_start:target_col_end].copy()

    regulator_cmap = make_linear_cmap(["#FFFFFF", "#43C5E3"], "regulator_loading_cmap")
    target_cmap = make_linear_cmap(["#FFFFFF", "#E5352A"], "target_loading_cmap")

    plot_matrix_heatmap(
        mat=regulator_loading_norm.values,
        row_labels=list(regulator_loading_norm.index),
        out_path=os.path.join(file_savepath_main, "Normalized_loading_envir_regulator_genes.pdf"),
        xlabel="Regulator genes",
        ylabel="",
        cmap=regulator_cmap,
        fig_w=max(4.5, 0.22 * regulator_loading_norm.shape[1]),
        fig_h=1.6,
        x_tick_labels=list(regulator_loading_norm.columns),
        max_xticks=20,
        show_x_ticks=True,
    )

    plot_matrix_heatmap(
        mat=target_loading_norm.values,
        row_labels=list(target_loading_norm.index),
        out_path=os.path.join(file_savepath_main, "Normalized_loading_envir_target_genes.pdf"),
        xlabel="Target genes",
        ylabel="",
        cmap=target_cmap,
        fig_w=max(4.5, 0.22 * target_loading_norm.shape[1]),
        fig_h=1.6,
        x_tick_labels=list(target_loading_norm.columns),
        max_xticks=20,
        show_x_ticks=True,
    )

    # Additional heatmaps for selected rows and columns of loading_envir_use_norm
    # Select columns 41-80 from loading_envir_use first, then normalize within this subset




    loading_envir_use_subset_41_80 = loading_envir_use[:, 40:80].copy()

    loading_envir_use_subset_41_80_colSums = np.sum(loading_envir_use_subset_41_80, axis=0)
    loading_envir_use_subset_41_80_colSums[loading_envir_use_subset_41_80_colSums == 0] = 1

    loading_envir_use_subset_41_80_norm = (
        loading_envir_use_subset_41_80 / loading_envir_use_subset_41_80_colSums
    )

    loading_envir_use_subset_41_80_norm = pd.DataFrame(
        loading_envir_use_subset_41_80_norm,
        columns=SpiderNet_data_pyg["genenames"][40:80],
        index=["MI-1_sending", "MI-1_receiving", "MI-2_sending", "MI-2_receiving"],
    )

    # Select rows 1 and 3 (human counting) after subset normalization
    sending_rows_loading_norm = loading_envir_use_subset_41_80_norm.iloc[[0, 2], :].copy()

    # Select rows 2 and 4 (human counting) after subset normalization
    receiving_rows_loading_norm = loading_envir_use_subset_41_80_norm.iloc[[1, 3], :].copy()

    sending_rows_cmap = make_linear_cmap(["#FFFFFF", "#43C5E3"], "sending_rows_loading_cmap")
    receiving_rows_cmap = make_linear_cmap(["#FFFFFF", "#E5352A"], "receiving_rows_loading_cmap")

    plot_matrix_heatmap_vector(
        mat=sending_rows_loading_norm.values,
        row_labels=list(sending_rows_loading_norm.index),
        out_path=os.path.join(file_savepath_main, "Normalized_loading_envir_rows1_3_cols41_80.pdf"),
        xlabel="Genes",
        ylabel="",
        cmap=sending_rows_cmap,
        fig_w=max(5.5, 0.20 * sending_rows_loading_norm.shape[1]),
        fig_h=1.25,
        x_tick_labels=list(sending_rows_loading_norm.columns),
        max_xticks=20,
        show_x_ticks=True,
        cell_edgecolor="none",
        cell_linewidth=0.0,
    )

    plot_matrix_heatmap_vector(
        mat=receiving_rows_loading_norm.values,
        row_labels=list(receiving_rows_loading_norm.index),
        out_path=os.path.join(file_savepath_main, "Normalized_loading_envir_rows2_4_cols41_80.pdf"),
        xlabel="Genes",
        ylabel="",
        cmap=receiving_rows_cmap,
        fig_w=max(5.5, 0.20 * receiving_rows_loading_norm.shape[1]),
        fig_h=1.25,
        x_tick_labels=list(receiving_rows_loading_norm.columns),
        max_xticks=20,
        show_x_ticks=True,
        cell_edgecolor="none",
        cell_linewidth=0.0,
    )
