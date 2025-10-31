import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scanpy as sc
import re
import os


def run_feature_umap(
        cellclass_choose,
        adata_choose_path,
        file_savepath_main,
        obsm_show,
        adata_copy_path,
        show=True
):
    """
    Plot UMAP with multiple annotation panels (subplots) and neighboring cell-type proportions.

    - plot_columns: shown together as one figure with subplots (each has title, legend, saved individually too)
    - prop_name_list: saved individually (not shown)
    """
    # ---------------- Load data ----------------
    adata_choose = sc.read_h5ad(adata_choose_path)
    adata_copy = sc.read_h5ad(adata_copy_path)
    adata_choose.obs.columns = adata_choose.obs.columns.str.replace('/', '-', regex=False)

    # --- Flexible key matching for obsm_show (case-insensitive)
    obsm_keys_lower = {k.lower(): k for k in adata_choose.obsm.keys()}
    if obsm_show.lower() in obsm_keys_lower:
        obsm_show = obsm_keys_lower[obsm_show.lower()]
    else:
        raise KeyError(f"{obsm_show} not found in adata_choose.obsm.keys(): {list(adata_choose.obsm.keys())}")

    MI_umap = adata_choose.obsm[obsm_show]
    prop_columns = [col for col in adata_choose.obs.columns if re.search(r'_prop$', col)]

    # ---------------- Color palettes ----------------
    OUTCOME_PALETTE = {
        'Alive': '#a1d99b',
        'Dead (disease)': '#e41a1c',
        'Dead of disease': '#d95f02',
        'Dead (other)': '#a50f15',
        'D/c to hospice (likely dead of disease)': '#77BEF0'
    }
    SITES_PALETTE = {'Adnexa': '#3a86ff', 'Omentum': '#ff006e'}
    TREATMENT_STR_PALETTE = {'Untreated': '#2a9d8f', 'Treated': '#e76f51', 'Unknown': '#000000'}

    # ---------------- Plot columns ----------------
    if "MI_louvain" in adata_choose.obs.keys():
        plot_columns = [
            ("treatment", "treatment"),
            ("stage", "stage"),
            ("patients_x", "patient"),
            ("cell.subtypes", "cellsubtypes"),
            ("sites_binary_x", "sites"),
            ("outcome", "outcome"),
            ("Malignant_C3", "Malignant_C3")
        ]
    else:
        plot_columns = [
            ("treatment", "treatment"),
            ("stage", "stage"),
            ("patients_x", "patient"),
            ("cell.subtypes", "cellsubtypes"),
            ("sites_binary_x", "sites"),
            ("outcome", "outcome"),
            ("treatment_status", "treatment_status"),
            ("sample_id", "sample_id"),
            ("annotation_subtypes", "annotation_subtypes")
        ]
        # prop_pairs = [(col, col.replace('_prop', '').strip()) for col in prop_columns]
        # plot_columns += prop_pairs

    # ==========================================================
    # Part 1: show all annotation UMAPs as subplots
    # ==========================================================
    valid_columns = [(obs_col, save_suffix) for obs_col, save_suffix in plot_columns if obs_col in adata_choose.obs.columns]
    n_plots = len(valid_columns)
    if n_plots == 0:
        print("⚠️ No valid columns found for plotting.")
        return

    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axes = axes.flatten()

    for idx, (obs_col, save_suffix) in enumerate(valid_columns):
        ax = axes[idx]
        s = adata_choose.obs[obs_col]
        size_use = 10 if cellclass_choose == 'TNK.cell' else 1

        # pick palette
        def scatter_category(s, palette=None):
            s_str = s.astype('string').fillna('Unknown')
            cats = list(s_str.unique())
            colors_use = [palette.get(c, '#636363') if palette else None for c in cats]
            for c, name in zip(colors_use, cats):
                idx_ = s_str == name
                ax.scatter(MI_umap[idx_, 0], MI_umap[idx_, 1], c=[c] if c else None, s=size_use, label=name)

        if obs_col == 'outcome':
            scatter_category(s, OUTCOME_PALETTE)
        elif obs_col == 'sites_binary_x':
            scatter_category(s, SITES_PALETTE)
        elif obs_col == 'treatment':
            scatter_category(s, TREATMENT_STR_PALETTE)
        else:
            s_cat = s.astype('category')
            codes = s_cat.cat.codes.values
            cats = list(s_cat.cat.categories)
            cmap = cm.get_cmap('tab10', len(cats))
            for i, name in enumerate(cats):
                idx_ = codes == i
                ax.scatter(MI_umap[idx_, 0], MI_umap[idx_, 1], c=[cmap(i)], s=size_use, label=name)

        ax.set_title(save_suffix, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(ax.get_legend_handles_labels()[0]) <= 15:
            ax.legend(fontsize=7, loc='upper right', frameon=False)

        # --- Save each plot individually (re-draw) ---
        fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
        scatter_category(s, OUTCOME_PALETTE if obs_col == 'outcome' else None)
        ax_ind.set_title(save_suffix, fontsize=14)
        ax_ind.set_xticks([]); ax_ind.set_yticks([])
        ax_ind.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(file_savepath_main, f'UMAP_{cellclass_choose}_{save_suffix}_{obsm_show}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_ind)

    # remove unused subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)

    # ==========================================================
    # Part 2: neighboring cell-type proportions (save only)
    # ==========================================================
    prop_name_list = [ct.replace('/', '-') + "_prop" for ct in np.unique(adata_copy.obs['cell.types'])]
    prop_name_list.append('Malignant_prop_cut')

    for prop_name_cur in prop_name_list:
        prop_name_cur_use = 'Malignant_prop' if prop_name_cur == 'Malignant_prop_cut' else prop_name_cur
        if prop_name_cur_use not in adata_choose.obs.columns:
            continue

        prop_use = adata_choose.obs[prop_name_cur_use].values
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        order = np.argsort(prop_use)
        x, y = MI_umap[order, 0], MI_umap[order, 1]
        c_plot = prop_use[order]

        if prop_name_cur == 'Malignant_prop_cut':
            prop_cat = np.where(c_plot >= 0.9, 1, 0)
            color_map = {0: '#ff7f0e', 1: '#BFBFBF'}
            for val in [1, 0]:
                idx = prop_cat == val
                ax.scatter(x[idx], y[idx], c=color_map[val], s=1, label='Tumor core' if val else 'Tumor margin')
        else:
            norm = colors.Normalize(vmin=prop_use.min(), vmax=prop_use.max())
            scmap = ax.scatter(x, y, c=c_plot, cmap='inferno', s=1, norm=norm)
            cbar = plt.colorbar(scmap, ax=ax)
            cbar.set_label(prop_name_cur_use, fontsize=12)

        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        filename = os.path.join(file_savepath_main, f"UMAP_{cellclass_choose}_neighboring{prop_name_cur}_{obsm_show}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f"✅ All annotation (shown) and neighboring-cell (saved) UMAP plots written to {file_savepath_main}")
