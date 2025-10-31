# -----------------------------
# Package imports
# -----------------------------
import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu, spearmanr
import scanpy as sc
import pickle


# =============================================================
# Core API
# =============================================================
def run_analysis(
    SpiderNet_data_pyg_list_path: str,
    Factor_envir_use_path: str,
    file_savepath_main: str,
    metadata_sample_path: str,
    LR_loading_pathway_path: str,
    dim_envir: int,
    MIlevel_agg_threshold: float,
    batch_cell_unique_path: str,
    adata_list_path: str,
    adata_copy_path: str,
    show: bool = False
):
    """
    Compute MI (meta-interaction) enrichment patterns across cell-type sender→receiver pairs,
    generate multiple summary plots (heatmaps, scatter, dot plots), and save intermediate tables.

    Parameters
    ----------
    SpiderNet_data_pyg_list_path : str
        Path to the pickled list of per-sample PyG-like dicts containing 'edge_index', 'cell_class', etc.
    Factor_envir_use_path : str
        Path to a .npy file of MI edge-level strengths (E x K).
    file_savepath_main : str
        Output directory. All figures and csv/pkl files will be saved here.
    metadata_sample_path : str
        CSV path to sample metadata; use string "None" to indicate absence.
        Must contain at least columns: ['samples', 'sites_binary'] if provided.
    LR_loading_pathway_path : str
        CSV file that provides the MI ordering / list (its index is used to select/reorder MI columns).
    dim_envir : int
        Number of MI dimensions (K).
    MIlevel_agg_threshold : float
        Threshold used to highlight strong MI values and to filter rows in some plots.
    batch_cell_unique_path : str
        Path to pickled list/array of sample names (used for correlation section if enabled).
    adata_list_path : str
        Path to pickled list of AnnData objects per sample (for correlation section if enabled).
    adata_copy_path : str
        Path to a combined AnnData .h5ad (for correlation section if enabled).
    show : bool
        If True, call plt.show() after saving each figure (useful in Notebook/PyCharm).
    """

    # -----------------------------
    # Ensure save directory exists
    # -----------------------------
    os.makedirs(file_savepath_main, exist_ok=True)

    # -----------------------------
    # Load inputs
    # -----------------------------
    CellFlowMap_data_pyg_list = pd.read_pickle(SpiderNet_data_pyg_list_path)
    Factor_envir_use = np.load(Factor_envir_use_path)

    # If metadata is provided, load and index by sample name
    if metadata_sample_path != "None":
        metadata_sample = pd.read_csv(metadata_sample_path)
        metadata_sample.index = metadata_sample['samples']

    LR_loading_pathway_show = pd.read_csv(LR_loading_pathway_path, index_col=0)
    batch_cell_unique = pd.read_pickle(batch_cell_unique_path)
    adata_list = pd.read_pickle(adata_list_path)
    adata_copy = sc.read_h5ad(adata_copy_path)

    # -----------------------------
    # Basic containers
    # -----------------------------
    cellclass_use = np.hstack([
        CellFlowMap_data_pyg_list[i]['cell_class']
        for i in range(len(CellFlowMap_data_pyg_list))
    ])
    cellclass_use_unique = np.array(CellFlowMap_data_pyg_list[0]['cell_class_unique'])
    cellclass_unique = cellclass_use_unique

    # Edge-level MI (E x K)
    MI_use = Factor_envir_use

    # Cell-type color map (not directly used in final plots but kept for completeness)
    cmap_cells = ListedColormap([
        '#1F78B5', '#FF800D', '#DE156E', '#442356', '#D08032',
        '#ABC1D2', '#68717C', '#F1947F', '#9568BE', '#A1D6CB',
        '#0F9E59', '#D82627', '#FFD801', '#A3786F', '#E575C4',
        '#BCBE1F', '#1AC1D2', '#A888B5', '#FFE6A9', '#D3F1DF',
        '#B1F0F7', '#A8CD89', '#2A3335', '#D8C4B6'
    ])

    # Encode cell-class labels if needed later
    label_encoder = LabelEncoder()
    _ = label_encoder.fit_transform(cellclass_use)

    # -----------------------------
    # Construct edge-level sender/receiver cell types
    # -----------------------------
    cellclass_sender = np.hstack([
        CellFlowMap_data_pyg_list[i]['cell_class'][
            CellFlowMap_data_pyg_list[i]['edge_index'].cpu().detach().numpy()[:, 0]
        ]
        for i in range(len(CellFlowMap_data_pyg_list))
    ])
    cellclass_receiver = np.hstack([
        CellFlowMap_data_pyg_list[i]['cell_class'][
            CellFlowMap_data_pyg_list[i]['edge_index'].cpu().detach().numpy()[:, 1]
        ]
        for i in range(len(CellFlowMap_data_pyg_list))
    ])
    cellclass_edge = pd.DataFrame({"Sender": cellclass_sender, "Receiver": cellclass_receiver})

    # -----------------------------
    # Select frequently observed cell-type pairs
    # (Keep neighbors with normalized frequency > 0.1)
    # -----------------------------
    cellclasspair_filtered_sender = []
    for i in range(len(cellclass_use_unique)):
        edge_cur = cellclass_edge.iloc[np.where(cellclass_edge.iloc[:, 0] == cellclass_use_unique[i])[0], :]
        neigh, counts = np.unique(edge_cur.iloc[:, 1].values, return_counts=True)
        df = pd.DataFrame({"Sender": cellclass_use_unique[i], "Receiver": neigh, "Count": counts})
        df['Count_norm'] = df['Count'] / np.max(df['Count'])
        df = df.loc[df['Count_norm'] > 0.1, :]
        cellclasspair_filtered_sender.append(df[['Sender', 'Receiver']])
    cellclasspair_filtered_sender = pd.concat(cellclasspair_filtered_sender, axis=0)

    cellclasspair_filtered_receiver = []
    for i in range(len(cellclass_use_unique)):
        edge_cur = cellclass_edge.iloc[np.where(cellclass_edge.iloc[:, 1] == cellclass_use_unique[i])[0], :]
        neigh, counts = np.unique(edge_cur.iloc[:, 0].values, return_counts=True)
        df = pd.DataFrame({"Sender": neigh, "Receiver": cellclass_use_unique[i], "Count": counts})
        df['Count_norm'] = df['Count'] / np.max(df['Count'])
        df = df.loc[df['Count_norm'] > 0.1, :]
        cellclasspair_filtered_receiver.append(df[['Sender', 'Receiver']])
    cellclasspair_filtered_receiver = pd.concat(cellclasspair_filtered_receiver, axis=0)

    cellclasspair_filtered = pd.concat(
        [cellclasspair_filtered_sender, cellclasspair_filtered_receiver], axis=0
    ).drop_duplicates(subset=['Sender', 'Receiver'])

    # -----------------------------
    # Pre-allocate containers
    # -----------------------------
    Avg_MI_cellclass_pair_list = []
    Avg_MI_cellclass_pair_list_Adnexa = []
    Avg_MI_cellclass_pair_list_Omentum = []
    Avg_MI_cellclass_pair_list_value = []
    Avg_MI_cellclass_pair_list_value_Adnexa = []
    Avg_MI_cellclass_pair_list_value_Omentum = []
    pivoted_sender_list = []
    pivoted_receiver_list = []

    # If metadata is available, map each edge to site label
    if metadata_sample_path != "None":
        edge_site = []
        for si in range(len(CellFlowMap_data_pyg_list)):
            num_edge = CellFlowMap_data_pyg_list[si]['edge_index'].cpu().detach().numpy().shape[0]
            sample_cur = np.unique(CellFlowMap_data_pyg_list[si]['sample_name'])[0]
            site_cur = np.unique(np.array(metadata_sample.loc[sample_cur, :]['sites_binary']))[0]
            edge_site.extend([site_cur] * num_edge)

    # -----------------------------
    # Loop over MI dimensions and compute average MI per Sender→Receiver pair
    # -----------------------------
    for MI_index in range(MI_use.shape[1]):
        # Initialize per-MI matrices (Sender x Receiver)
        Avg_MI_cellclass_pair_cur = pd.DataFrame(
            0, index=cellclass_use_unique, columns=cellclass_use_unique, dtype=np.float32
        )
        Avg_MI_cellclass_pair_Adnexa_cur = pd.DataFrame(
            0, index=cellclass_use_unique, columns=cellclass_use_unique, dtype=np.float32
        )
        Avg_MI_cellclass_pair_Omentum_cur = pd.DataFrame(
            0, index=cellclass_use_unique, columns=cellclass_use_unique, dtype=np.float32
        )
        _Avg_MI_site_pvalue = pd.DataFrame(  # retained for extensibility
            0, index=cellclass_use_unique, columns=cellclass_use_unique, dtype=np.float32
        )

        # Build a working edge frame for this MI
        edge_cur = cellclass_edge.copy()
        edge_cur['MI_strength'] = MI_use[:, MI_index]
        if metadata_sample_path != "None":
            edge_cur['Site'] = edge_site

        # Average Sender→Receiver MI
        pivoted = edge_cur.pivot_table(
            index='Sender',
            columns='Receiver',
            values='MI_strength',
            aggfunc='mean',
            fill_value=0
        )
        # Average by Sender (row mean on edges)
        pivoted_sender = edge_cur.groupby('Sender', as_index=True)['MI_strength'].mean()
        # Average by Receiver (col mean on edges)
        pivoted_receiver = edge_cur.groupby('Receiver', as_index=True)['MI_strength'].mean()

        pivoted_sender_list.append(pivoted_sender)
        pivoted_receiver_list.append(pivoted_receiver)

        # Fill back to matrix with the same shape
        Avg_MI_cellclass_pair_cur[:] = pivoted.values

        # Flatten to long format with MI index annotation
        cur_flat = Avg_MI_cellclass_pair_cur.stack().reset_index()
        cur_flat["MI_index"] = MI_index
        cur_flat.columns = ["Sender", "Receiver", "MI_strength", "MI_index"]

        # Append to collectors
        Avg_MI_cellclass_pair_list.append(cur_flat)
        Avg_MI_cellclass_pair_list_value.append(cur_flat["MI_strength"].values)

        # If metadata available, compute site-specific Sender→Receiver MI
        if metadata_sample_path != "None":
            # Adnexa
            edge_adn = edge_cur.loc[np.where(np.array(edge_cur['Site']) == "Adnexa")[0], :]
            pivoted_adn = edge_adn.pivot_table(
                index='Sender',
                columns='Receiver',
                values='MI_strength',
                aggfunc='mean',
                fill_value=0
            )
            Avg_MI_cellclass_pair_Adnexa_cur[:] = pivoted_adn.values
            cur_flat_adn = Avg_MI_cellclass_pair_Adnexa_cur.stack().reset_index()
            cur_flat_adn["MI_index"] = MI_index
            cur_flat_adn.columns = ["Sender", "Receiver", "MI_strength", "MI_index"]
            Avg_MI_cellclass_pair_list_Adnexa.append(cur_flat_adn)
            Avg_MI_cellclass_pair_list_value_Adnexa.append(cur_flat_adn["MI_strength"].values)

            # Omentum
            edge_ome = edge_cur.loc[np.where(np.array(edge_cur['Site']) == "Omentum")[0], :]
            pivoted_ome = edge_ome.pivot_table(
                index='Sender',
                columns='Receiver',
                values='MI_strength',
                aggfunc='mean',
                fill_value=0
            )
            Avg_MI_cellclass_pair_Omentum_cur[:] = pivoted_ome.values
            cur_flat_ome = Avg_MI_cellclass_pair_Omentum_cur.stack().reset_index()
            cur_flat_ome["MI_index"] = MI_index
            cur_flat_ome.columns = ["Sender", "Receiver", "MI_strength", "MI_index"]
            Avg_MI_cellclass_pair_list_Omentum.append(cur_flat_ome)
            Avg_MI_cellclass_pair_list_value_Omentum.append(cur_flat_ome["MI_strength"].values)

    # -----------------------------
    # Assemble sender/receiver summaries across all MI
    # -----------------------------
    pivoted_sender_all = pd.concat(pivoted_sender_list, axis=1)
    pivoted_receiver_all = pd.concat(pivoted_receiver_list, axis=1)
    pivoted_sender_all.columns = ["MI-" + str(i + 1) for i in range(MI_use.shape[1])]
    pivoted_receiver_all.columns = ["MI-" + str(i + 1) for i in range(MI_use.shape[1])]
    _sender_row_sum = np.sum(pivoted_sender_all, axis=1)
    _receiver_row_sum = np.sum(pivoted_receiver_all, axis=1)

    # Column-wise max normalization
    pivoted_receiver_all = pivoted_receiver_all.div(pivoted_receiver_all.max(axis=0), axis=1)
    pivoted_sender_all = pivoted_sender_all.div(pivoted_sender_all.max(axis=0), axis=1)

    # Long form for scatter
    pivoted_sender_long = (
        pivoted_sender_all
        .reset_index()
        .melt(id_vars='Sender', var_name='MI', value_name='MI_strength')
    )
    pivoted_sender_long.index = pivoted_sender_long["Sender"] + "_" + pivoted_sender_long["MI"]

    pivoted_receiver_long = (
        pivoted_receiver_all
        .reset_index()
        .melt(id_vars='Receiver', var_name='MI', value_name='MI_strength')
    )
    pivoted_receiver_long.index = pivoted_receiver_long["Receiver"] + "_" + pivoted_receiver_long["MI"]
    pivoted_receiver_long = pivoted_receiver_long.loc[pivoted_sender_long.index, :]

    pivoted_SR_long = pivoted_sender_long.copy()
    pivoted_SR_long = pivoted_SR_long.rename(columns={pivoted_SR_long.columns[2]: "Sender_MI_strength"})
    pivoted_SR_long = pivoted_SR_long.rename(columns={pivoted_SR_long.columns[0]: "Cellclass"})
    pivoted_SR_long["Receiver_MI_strength"] = pivoted_receiver_long["MI_strength"].values

    # -----------------------------
    # Merge MI matrices across all MI (wide table)
    # -----------------------------
    Avg_MI_cellclass_pair = pd.concat(Avg_MI_cellclass_pair_list, axis=0)
    Avg_MI_cellclass_pair_merge = np.vstack(Avg_MI_cellclass_pair_list_value).T
    Avg_MI_cellclass_pair_merge = pd.DataFrame(
        Avg_MI_cellclass_pair_merge,
        columns=["MI-" + str(i + 1) for i in range(MI_use.shape[1])]
    )
    Avg_MI_cellclass_pair_merge["Sender"] = Avg_MI_cellclass_pair_list[0]["Sender"]
    Avg_MI_cellclass_pair_merge["Receiver"] = Avg_MI_cellclass_pair_list[0]["Receiver"]

    # Site-specific tables if available
    if metadata_sample_path != "None":
        Avg_MI_cellclass_pair_Adnexa = pd.concat(Avg_MI_cellclass_pair_list_Adnexa, axis=0)
        Avg_MI_cellclass_pair_merge_Adnexa = np.vstack(Avg_MI_cellclass_pair_list_value_Adnexa).T
        Avg_MI_cellclass_pair_merge_Adnexa = pd.DataFrame(
            Avg_MI_cellclass_pair_merge_Adnexa,
            columns=["MI-" + str(i + 1) for i in range(MI_use.shape[1])]
        )
        Avg_MI_cellclass_pair_merge_Adnexa["Sender"] = Avg_MI_cellclass_pair_list[0]["Sender"]
        Avg_MI_cellclass_pair_merge_Adnexa["Receiver"] = Avg_MI_cellclass_pair_list[0]["Receiver"]

        Avg_MI_cellclass_pair_Omentum = pd.concat(Avg_MI_cellclass_pair_list_Omentum, axis=0)
        Avg_MI_cellclass_pair_merge_Omentum = np.vstack(Avg_MI_cellclass_pair_list_value_Omentum).T
        Avg_MI_cellclass_pair_merge_Omentum = pd.DataFrame(
            Avg_MI_cellclass_pair_merge_Omentum,
            columns=["MI-" + str(i + 1) for i in range(MI_use.shape[1])]
        )
        Avg_MI_cellclass_pair_merge_Omentum["Sender"] = Avg_MI_cellclass_pair_list[0]["Sender"]
        Avg_MI_cellclass_pair_merge_Omentum["Receiver"] = Avg_MI_cellclass_pair_list[0]["Receiver"]

        # log2 fold-change (Omentum / Adnexa)
        Avg_MI_cellclass_pair_merge_logfc_site = Avg_MI_cellclass_pair_merge_Adnexa.copy()
        Avg_MI_cellclass_pair_merge_logfc_site.iloc[:, 0:MI_use.shape[1]] = np.log2(
            (Avg_MI_cellclass_pair_merge_Omentum.iloc[:, 0:MI_use.shape[1]].values + 1e-6) /
            (Avg_MI_cellclass_pair_merge_Adnexa.iloc[:, 0:MI_use.shape[1]].values + 1e-6)
        )
        # Keep MI columns in the order of LR_loading_pathway_show index
        Avg_MI_cellclass_pair_merge_logfc_site = Avg_MI_cellclass_pair_merge_logfc_site.loc[
            :, LR_loading_pathway_show.index.tolist() + ["Sender", "Receiver"]
        ]

    # Keep MI columns in the order of LR_loading_pathway_show index for the main table
    Avg_MI_cellclass_pair_merge = Avg_MI_cellclass_pair_merge.loc[
        :, LR_loading_pathway_show.index.tolist() + ["Sender", "Receiver"]
    ]

    # Save merged wide table
    Avg_MI_cellclass_pair_merge.to_csv(
        os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_merge.csv"),
        index=False
    )

    # Column-wise min-max normalization (0-1)
    Avg_MI_cellclass_pair_merge_zscored = Avg_MI_cellclass_pair_merge.copy()
    vals = Avg_MI_cellclass_pair_merge.iloc[:, 0:MI_use.shape[1]].values
    col_min = np.min(vals, axis=0)
    col_max = np.max(vals, axis=0)
    denom = np.where((col_max - col_min) == 0, 1.0, (col_max - col_min))
    Avg_MI_cellclass_pair_merge_zscored.iloc[:, 0:MI_use.shape[1]] = (vals - col_min) / denom

    # Keep a copy with all cell-type pairs (before filtering)
    Avg_MI_cellclass_pair_merge_zscored_allctpair = Avg_MI_cellclass_pair_merge_zscored.copy()

    # Filter to frequent pairs only
    Avg_MI_cellclass_pair_merge_zscored = Avg_MI_cellclass_pair_merge_zscored.merge(
        cellclasspair_filtered, on=["Sender", "Receiver"], how="inner"
    )
    if metadata_sample_path != "None":
        Avg_MI_cellclass_pair_merge_logfc_site = Avg_MI_cellclass_pair_merge_logfc_site.merge(
            cellclasspair_filtered, on=["Sender", "Receiver"], how="inner"
        )

    # Determine a potentially tighter per-celltype threshold window
    _notfiltered = Avg_MI_cellclass_pair_merge_zscored.copy()
    MIlevel_agg_threshold_pre = ([
        np.max(
            np.max(
                _notfiltered.loc[
                    (_notfiltered['Sender'] == ct) | (_notfiltered['Receiver'] == ct),
                    :
                ].iloc[:, 0:dim_envir],
                axis=1
            )
        )
        for ct in cellclass_unique
    ])
    if (np.min(MIlevel_agg_threshold_pre) < MIlevel_agg_threshold) and (np.min(MIlevel_agg_threshold_pre) > 0.5):
        MIlevel_agg_threshold = float(np.min(MIlevel_agg_threshold_pre))

    # Keep only rows with some MI >= threshold
    mask_keep = np.max(
        Avg_MI_cellclass_pair_merge_zscored.iloc[:, 0:MI_use.shape[1]].values, axis=1
    ) >= MIlevel_agg_threshold
    Avg_MI_cellclass_pair_merge_zscored = Avg_MI_cellclass_pair_merge_zscored.iloc[np.where(mask_keep)[0], :]
    if metadata_sample_path != "None":
        Avg_MI_cellclass_pair_merge_logfc_site = Avg_MI_cellclass_pair_merge_logfc_site.iloc[np.where(mask_keep)[0], :]

    # Order rows by argmax MI column for reproducible visualization
    order_idx = np.argsort(
        np.argmax(
            Avg_MI_cellclass_pair_merge_zscored.iloc[:, 0:MI_use.shape[1]].values, axis=1
        )
    )
    if metadata_sample_path != "None":
        Avg_MI_cellclass_pair_merge_logfc_site = Avg_MI_cellclass_pair_merge_logfc_site.iloc[order_idx, :]
    Avg_MI_cellclass_pair_merge_zscored = Avg_MI_cellclass_pair_merge_zscored.iloc[order_idx, :]

    # Use zscored (min-max) table as the default for plotting
    Avg_MI_cellclass_pair_merge_use = Avg_MI_cellclass_pair_merge_zscored.copy()

    # -----------------------------
    # Save the zscored, filtered table used for heatmaps
    # -----------------------------
    Avg_MI_cellclass_pair_merge_use.to_csv(
        os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_merge_use.csv"),
        index=False
    )

    # =========================================================
    # Heatmap: MI (rows) vs Cell-type pairs (columns)
    # =========================================================
    plt.close()
    heatmap_values = Avg_MI_cellclass_pair_merge_use.iloc[:, 0:MI_use.shape[1]].values.T

    cmap = LinearSegmentedColormap.from_list(
        "custom_coolwarm_graycenter",
        ["#4575b4", "#f0f0f0", "#d73027"],
        N=256
    )
    vmin, vmax = 0, 1
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)

    n_mis, n_pairs = heatmap_values.shape
    x = np.arange(n_pairs + 1)
    y = np.arange(n_mis + 1)

    fig, ax = plt.subplots(figsize=(15 * MI_use.shape[1] / 25, 6))
    mesh = ax.pcolormesh(
        x, y, heatmap_values, cmap=cmap, norm=norm, edgecolors='#B6B9BA', linewidth=0.5
    )
    fig.colorbar(mesh, ax=ax)

    # X tick labels as "Sender → Receiver"
    cellclass_pair_names = [
        f"{s} \u2192 {r}"
        for s, r in zip(Avg_MI_cellclass_pair_merge_use["Sender"], Avg_MI_cellclass_pair_merge_use["Receiver"])
    ]
    ax.set_xticks(np.arange(n_pairs) + 0.5)
    ax.set_xticklabels(cellclass_pair_names, rotation=60, ha='right', fontsize=12)

    # Y tick labels as MI names
    mi_names = Avg_MI_cellclass_pair_merge_use.columns[0:MI_use.shape[1]]
    ax.set_yticks(np.arange(n_mis) + 0.5)
    ax.set_yticklabels(mi_names, fontsize=12)

    # Invert Y because pcolormesh puts row 0 at bottom
    ax.invert_yaxis()

    # Highlight cells with MI >= threshold
    for i in range(n_mis):
        for j in range(n_pairs):
            if heatmap_values[i, j] >= MIlevel_agg_threshold:
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='yellow', linewidth=1, zorder=10)
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(os.path.join(file_savepath_main, "Avg_MI_cellclass_pair.png"),
                format='png', bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

    # =========================================================
    # # Heatmap: Same but only outline malignant-involving pairs
    # # =========================================================
    # plt.close()
    # heatmap_values = Avg_MI_cellclass_pair_merge_use.iloc[:, 0:MI_use.shape[1]].values.T
    #
    # fig, ax = plt.subplots(figsize=(20 * MI_use.shape[1] / 25, 6))
    # mesh = ax.pcolormesh(
    #     x, y, heatmap_values, cmap=cmap, norm=norm, edgecolors='#B6B9BA', linewidth=0.5
    # )
    # fig.colorbar(mesh, ax=ax)
    #
    # ax.set_xticks(np.arange(n_pairs) + 0.5)
    # ax.set_xticklabels(cellclass_pair_names, rotation=60, ha='right', fontsize=12)
    #
    # ax.set_yticks(np.arange(n_mis) + 0.5)
    # ax.set_yticklabels(mi_names, fontsize=12)
    # ax.invert_yaxis()
    #
    # for i in range(n_mis):
    #     for j in range(n_pairs):
    #         if (heatmap_values[i, j] >= MIlevel_agg_threshold) and ("Malignant" in cellclass_pair_names[j]):
    #             rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='yellow', linewidth=1.5, zorder=10)
    #             ax.add_patch(rect)
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_onlymalignant_highlight.png"),
    #             format='png', bbox_inches='tight', dpi=300)
    # if show:
    #     plt.show()
    # plt.close()

    # # =========================================================
    # # Dot plot: site logFC (color) vs MI, sized by overall MI (use)
    # # =========================================================
    # if metadata_sample_path != "None":
    #     plt.style.use('default')
    #
    #     df_logfc = Avg_MI_cellclass_pair_merge_logfc_site.copy()
    #     df_use = Avg_MI_cellclass_pair_merge_use.copy()
    #
    #     # MI column names in original order
    #     mi_cols = [c for c in df_logfc.columns if isinstance(c, str) and c.startswith('MI-')]
    #     if len(mi_cols) == 0:
    #         raise ValueError("No columns starting with 'MI-' were found in the site logFC frame.")
    #
    #     # Keep pair order from first appearance (stable)
    #     pair_series = df_logfc['Sender'].astype(str) + '→' + df_logfc['Receiver'].astype(str)
    #     pair_order = pd.Index(pair_series).unique().tolist()
    #
    #     # Wide-to-long for logFC
    #     long_logfc = df_logfc.melt(
    #         id_vars=['Sender', 'Receiver'],
    #         value_vars=mi_cols,
    #         var_name='MI',
    #         value_name='logFC'
    #     )
    #     long_logfc['Pair'] = long_logfc['Sender'].astype(str) + '→' + long_logfc['Receiver'].astype(str)
    #
    #     # Wide-to-long for overall use (normalized MI table)
    #     long_use = df_use.melt(
    #         id_vars=['Sender', 'Receiver'],
    #         value_vars=mi_cols,
    #         var_name='MI',
    #         value_name='use_val'
    #     )
    #     long_use['Pair'] = long_use['Sender'].astype(str) + '→' + long_use['Receiver'].astype(str)
    #
    #     # Merge long tables
    #     long_df = pd.merge(
    #         long_logfc[['Sender', 'Receiver', 'Pair', 'MI', 'logFC']],
    #         long_use[['Sender', 'Receiver', 'Pair', 'MI', 'use_val']],
    #         on=['Sender', 'Receiver', 'Pair', 'MI'],
    #         how='left'
    #     )
    #     # Restore categorical order
    #     long_df['MI'] = pd.Categorical(long_df['MI'], categories=mi_cols, ordered=True)
    #     long_df['Pair'] = pd.Categorical(long_df['Pair'], categories=pair_order, ordered=True)
    #
    #     # Keep points with use above threshold
    #     long_df = long_df[long_df['use_val'] >= MIlevel_agg_threshold].copy()
    #
    #     # Map use to point sizes based on current range
    #     smin, smax = 200 * MIlevel_agg_threshold, 200
    #     use_vals = long_df['use_val'].to_numpy(dtype=float)
    #     if use_vals.size == 0 or np.all(np.isnan(use_vals)):
    #         sizes = np.array([])
    #     else:
    #         umin, umax = np.nanmin(use_vals), np.nanmax(use_vals)
    #         if umax == umin:
    #             sizes = np.full(len(use_vals), (smin + smax) / 2.0)
    #         else:
    #             sizes = smin + (use_vals - umin) / (umax - umin) * (smax - smin)
    #             sizes = np.where(np.isnan(use_vals), smin, sizes)
    #
    #     # Axes (x=pair, y=MI)
    #     x_codes = long_df['Pair'].cat.codes.to_numpy()
    #     y_codes = long_df['MI'].cat.codes.to_numpy()
    #
    #     # Color normalization for logFC in [-2, 2]
    #     vals = long_df['logFC'].to_numpy(dtype=float)
    #     norm2 = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    #
    #     fig_w = max(6, len(pair_order) * 0.6 + 3)
    #     fig_h = max(4, len(mi_cols) * 0.35 + 1)
    #     FIG_SCALE = 0.75
    #     fig, ax = plt.subplots(figsize=(fig_w * FIG_SCALE, fig_h * FIG_SCALE), facecolor='white')
    #     ax.set_facecolor('white')
    #
    #     sc_plot = ax.scatter(x_codes, y_codes, c=vals, s=sizes, cmap='coolwarm', norm=norm2) if len(x_codes) > 0 else None
    #
    #     # Emphasize points with |logFC|>1
    #     if sc_plot is not None:
    #         mask_edge = np.abs(vals) > 1
    #         if np.any(mask_edge):
    #             ax.scatter(
    #                 x_codes[mask_edge], y_codes[mask_edge],
    #                 s=sizes[mask_edge] * 1.15,
    #                 facecolors='none',
    #                 edgecolors='black',
    #                 linewidths=2,
    #                 zorder=3
    #             )
    #
    #     ax.set_xticks(np.arange(len(pair_order)))
    #     ax.set_xticklabels(pair_order, rotation=45, ha='right')
    #     ax.set_yticks(np.arange(len(mi_cols)))
    #     ax.set_yticklabels(mi_cols)
    #
    #     # Show the first MI at top
    #     ax.invert_yaxis()
    #
    #     ax.set_xlabel('Sender → Receiver')
    #     ax.set_ylabel('MI')
    #
    #     if sc_plot is not None:
    #         cbar = plt.colorbar(sc_plot, ax=ax)
    #         cbar.set_label('logFC')
    #         cbar.set_ticks([-2, -1, 0, 1, 2])
    #         cbar.ax.set_facecolor('white')
    #
    #     ax.grid(axis='y', alpha=0.2)
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_logfc_site_dotplot.png"),
    #         format='png', bbox_inches='tight', dpi=300, facecolor='white', edgecolor='white'
    #     )
    #     if show:
    #         plt.show()
    #     plt.close()
    #
    #     # Also output the selected table with |logFC|>1 sorted by absolute logFC
    #     long_df_choose = long_df.loc[np.abs(long_df['logFC']) > 1.0, :].copy()
    #     long_df_choose = long_df_choose.iloc[np.argsort(np.abs(long_df_choose['logFC'].values))[::-1], :]
    #     # Save the filtered/sorted table for convenience
    #     long_df_choose.to_csv(
    #         os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_logfc_site_dotplot_top_hits.csv"),
    #         index=False
    #     )

    # =========================================================
    # Sender-Receiver pair “arrow” plot
    # =========================================================
    cell_types = cellclass_unique.tolist()
    cell_pairs = [
        (
            Avg_MI_cellclass_pair_merge_use['Sender'].iloc[i],
            Avg_MI_cellclass_pair_merge_use['Receiver'].iloc[i]
        )
        for i in range(Avg_MI_cellclass_pair_merge_use.shape[0])
    ]

    # Reference color list for cell types (repeat if needed)
    colorct_ref = [
        '#1f77b4', '#17becf', '#c49c94', '#b22222',
        '#9467bd', '#ff7f0e', '#2ca02c'
    ]
    # Map celltype -> color (fallback to grey if we run out)
    celltype2color = {ct: colorct_ref[i % len(colorct_ref)] for i, ct in enumerate(cell_types)}

    fig, ax = plt.subplots(figsize=(max(1, len(cell_pairs)) * 0.5, len(cell_types) * 0.7 * 1.12))
    ax.set_facecolor('white')

    for col, (sender, receiver) in enumerate(cell_pairs):
        sender_y = cell_types.index(sender)
        receiver_y = cell_types.index(receiver)

        # Points for sender and receiver rows
        ax.scatter([col], [sender_y], color=celltype2color.get(sender, '#999999'), s=40, zorder=3)
        ax.scatter([col], [receiver_y], color=celltype2color.get(receiver, '#999999'), s=40, zorder=3)

        # Thick arrow from sender to receiver at the same column
        ax.annotate(
            '',
            xy=(col, receiver_y),
            xytext=(col, sender_y),
            arrowprops=dict(
                arrowstyle='->,head_width=0.6,head_length=1.0',
                color='grey',
                lw=4.5,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=10
            ),
            zorder=2
        )

    # Y-axis ticks and labels
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types, fontsize=16)

    # Set explicit axis ranges (robust if some rows have no points)
    ax.set_ylim(len(cell_types) - 0.5, -0.5)
    ax.set_xlim(-0.5, max(len(cell_pairs) - 0.5, 0.5))

    # Horizontal guide lines for each cell type row
    y_positions = range(len(cell_types))
    ax.hlines(
        y=y_positions,
        xmin=-0.5,
        xmax=max(len(cell_pairs) - 0.5, 0.5),
        colors='lightgrey',
        linestyles='--',
        linewidths=1.0,
        alpha=0.7,
        zorder=1
    )

    # Put x-axis ticks and labels at the top
    ax.set_xticks(range(len(cell_pairs)))
    ax.set_xticklabels([f"{s}→{r}" for s, r in cell_pairs], rotation=60, ha='left', fontsize=16)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', bottom=False, top=True)

    # Thicker and longer y-axis ticks
    ax.tick_params(axis='y', which='major', length=10, width=1.2)

    # Remove top and bottom spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(file_savepath_main, "Celltype_pair_example.png"),
        format='png', bbox_inches='tight', dpi=300, facecolor='white'
    )
    if show:
        plt.show()
    plt.close()

    # =========================================================
    # Aggregate per-cell-type (Sender/Receiver) across all pairs
    # =========================================================
    # Sender aggregation: for each cell type, take the max of MI columns among its outgoing pairs
    sender_agg_cols = list(Avg_MI_cellclass_pair_merge_zscored_allctpair.columns[0:dim_envir]) + ["Sender"]
    Avg_MI_cellclass_pair_merge_zscored_ctagg_sender = pd.DataFrame(0, index=cellclass_unique, columns=sender_agg_cols)
    for ct in cellclass_unique:
        idx = np.where(cellclass_unique == ct)[0]
        sub = Avg_MI_cellclass_pair_merge_zscored_allctpair.iloc[
            np.where(Avg_MI_cellclass_pair_merge_zscored_allctpair['Sender'] == ct)[0], :
        ]
        Avg_MI_cellclass_pair_merge_zscored_ctagg_sender.iloc[idx, :] = list(
            sub.iloc[:, 0:dim_envir].max(axis=0).values
        ) + [ct]

    # Receiver aggregation: for each cell type, take the max among its incoming pairs
    recv_agg_cols = list(Avg_MI_cellclass_pair_merge_zscored_allctpair.columns[0:dim_envir]) + ["Receiver"]
    Avg_MI_cellclass_pair_merge_zscored_ctagg_receiver = pd.DataFrame(0, index=cellclass_unique, columns=recv_agg_cols)
    for ct in cellclass_unique:
        idx = np.where(cellclass_unique == ct)[0]
        sub = Avg_MI_cellclass_pair_merge_zscored_allctpair.iloc[
            np.where(Avg_MI_cellclass_pair_merge_zscored_allctpair['Receiver'] == ct)[0], :
        ]
        Avg_MI_cellclass_pair_merge_zscored_ctagg_receiver.iloc[idx, :] = list(
            sub.iloc[:, 0:dim_envir].max(axis=0).values
        ) + [ct]

    Avg_MI_cellclass_pair_merge_zscored_ctagg_sender.iloc[:, 0:dim_envir] = (
        Avg_MI_cellclass_pair_merge_zscored_ctagg_sender.iloc[:, 0:dim_envir].astype(np.float32)
    )
    Avg_MI_cellclass_pair_merge_zscored_ctagg_receiver["celltype"] = (
        Avg_MI_cellclass_pair_merge_zscored_ctagg_receiver["Receiver"]
    )
    Avg_MI_cellclass_pair_merge_zscored_ctagg_sender["celltype"] = (
        Avg_MI_cellclass_pair_merge_zscored_ctagg_sender["Sender"]
    )
    Avg_MI_cellclass_pair_merge_zscored_ctagg_receiver["SR"] = "Receiver"
    Avg_MI_cellclass_pair_merge_zscored_ctagg_sender["SR"] = "Sender"

    Avg_MI_cellclass_pair_merge_zscored_ctagg = pd.concat(
        [Avg_MI_cellclass_pair_merge_zscored_ctagg_receiver, Avg_MI_cellclass_pair_merge_zscored_ctagg_sender],
        axis=0
    )
    # Remove helper columns
    Avg_MI_cellclass_pair_merge_zscored_ctagg = Avg_MI_cellclass_pair_merge_zscored_ctagg.drop(
        columns=["Sender", "Receiver"], errors='ignore'
    )

    # Order rows by celltype order (Receiver rows followed by Sender rows per cell type)
    row_idx = np.hstack([
        np.where(Avg_MI_cellclass_pair_merge_zscored_ctagg['celltype'] == cellclass_unique[i])[0]
        for i in range(len(cellclass_unique))
    ])
    Avg_MI_cellclass_pair_merge_zscored_ctagg = Avg_MI_cellclass_pair_merge_zscored_ctagg.iloc[row_idx, :]

    # -----------------------------
    # # Clustermap-like heatmap (no clustering) with SR row color bars
    # # -----------------------------
    # df_heat = Avg_MI_cellclass_pair_merge_zscored_ctagg.copy()
    # vmin, vmax = 0, 1
    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)
    # cmap = LinearSegmentedColormap.from_list(
    #     "custom_coolwarm_graycenter",
    #     ["#4575b4", "#f0f0f0", "#d73027"],
    #     N=256
    # )
    #
    # value_cols = [c for c in df_heat.columns if c.startswith('MI-')]
    # heatmap_data = df_heat[value_cols]
    #
    # sr_palette = {'Sender': '#2ca02c', 'Receiver': '#9467bd'}
    # row_colors = df_heat['SR'].map(sr_palette)
    # row_colors.name = None
    #
    # sns.set(font_scale=1.0)
    # g = sns.clustermap(
    #     heatmap_data,
    #     cmap=cmap,
    #     norm=norm,
    #     row_colors=row_colors,
    #     col_cluster=False,
    #     row_cluster=False,
    #     linewidths=0.5,
    #     linecolor='grey',
    #     figsize=(8, 3.4)
    # )
    # g.fig.set_size_inches(15, 6)  # override to a wider canvas
    #
    # # Add legend for SR color bars
    # for label, color in sr_palette.items():
    #     g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    # g.ax_col_dendrogram.legend(
    #     title='SR',
    #     loc='center',
    #     ncol=2,
    #     bbox_to_anchor=(0.5, 1.2)
    # )
    #
    # plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    # plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    #
    # # Highlight cells with MI >= threshold
    # ax_hm = g.ax_heatmap
    # data_vals = heatmap_data.values
    # n_rows, n_cols = data_vals.shape
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         if data_vals[i, j] >= MIlevel_agg_threshold:
    #             rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='yellow', linewidth=2)
    #             ax_hm.add_patch(rect)
    #
    # out_path = os.path.join(file_savepath_main, 'Avg_MI_cellclass_pair_agg.png')
    # plt.savefig(out_path, format='png', bbox_inches='tight', dpi=300)
    # if show:
    #     plt.show()
    # plt.close()

    # -----------------------------
    # Persist the final table used by plots
    # -----------------------------
    with open(os.path.join(file_savepath_main, 'Avg_MI_cellclass_pair_merge_use.pkl'), 'wb') as f:
        pickle.dump(Avg_MI_cellclass_pair_merge_use, f)

    # Also save the aggregated per-celltype sender/receiver summary (useful downstream)
    Avg_MI_cellclass_pair_merge_zscored_ctagg.to_csv(
        os.path.join(file_savepath_main, 'Avg_MI_cellclass_pair_merge_zscored_ctagg.csv'),
        index=False
    )

    # Done
    return {
        "Avg_MI_cellclass_pair_merge_use_csv": os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_merge_use.csv"),
        "Avg_MI_cellclass_pair_png": os.path.join(file_savepath_main, "Avg_MI_cellclass_pair.png"),
        "Avg_MI_cellclass_pair_onlymalignant_png": os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_onlymalignant_highlight.png"),
        "Sender_vs_Receiver_scatter_png": os.path.join(file_savepath_main, "Sender_vs_Receiver_MI_Strength.png"),
        "Celltype_pair_example_png": os.path.join(file_savepath_main, "Celltype_pair_example.png"),
        "Agg_heatmap_png": os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_agg.png"),
        "Site_dotplot_png": (
            None if metadata_sample_path == "None"
            else os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_logfc_site_dotplot.png")
        ),
        "Site_dotplot_top_hits_csv": (
            None if metadata_sample_path == "None"
            else os.path.join(file_savepath_main, "Avg_MI_cellclass_pair_logfc_site_dotplot_top_hits.csv")
        ),
        "Agg_sender_receiver_csv": os.path.join(file_savepath_main, 'Avg_MI_cellclass_pair_merge_zscored_ctagg.csv'),
        "Pickle_table_path": os.path.join(file_savepath_main, 'Avg_MI_cellclass_pair_merge_use.pkl')
    }


# =============================================================
# Optional: CLI wrapper (kept for parity with original script)
# =============================================================
if __name__ == "__main__":
    import sys

    # Expecting exactly the same argv order as the original script
    # sys.argv[1]  = CellFlowMap_data_pyg_list_path
    # sys.argv[2]  = Factor_envir_use_path
    # sys.argv[3]  = file_savepath_main
    # sys.argv[4]  = metadata_sample_path
    # sys.argv[5]  = LR_loading_pathway_show_path
    # sys.argv[6]  = dim_envir (int)
    # sys.argv[7]  = MIlevel_agg_threshold (float)
    # sys.argv[8]  = batch_cell_unique_path
    # sys.argv[9]  = adata_list_path
    # sys.argv[10] = adata_copy_path
    # (optional)   = --show to force interactive display

    if len(sys.argv) < 11:
        raise SystemExit(
            "Usage:\n"
            "  python MI_Celltypepair_enrichment_analysis.py "
            "<SpiderNet_data_pyg_list_path> <Factor_envir_use_path> <file_savepath_main> "
            "<metadata_sample_path|None> <LR_loading_pathway_path> <dim_envir> <MIlevel_agg_threshold> "
            "<batch_cell_unique_path> <adata_list_path> <adata_copy_path> [--show]"
        )

    SpiderNet_data_pyg_list_path = sys.argv[1]
    Factor_envir_use_path = sys.argv[2]
    file_savepath_main = sys.argv[3]
    metadata_sample_path = sys.argv[4]
    LR_loading_pathway_path = sys.argv[5]
    dim_envir = int(sys.argv[6])
    MIlevel_agg_threshold = float(sys.argv[7])
    batch_cell_unique_path = sys.argv[8]
    adata_list_path = sys.argv[9]
    adata_copy_path = sys.argv[10]
    show_flag = ("--show" in sys.argv[11:])

    run_analysis(
        SpiderNet_data_pyg_list_path=SpiderNet_data_pyg_list_path,
        Factor_envir_use_path=Factor_envir_use_path,
        file_savepath_main=file_savepath_main,
        metadata_sample_path=metadata_sample_path,
        LR_loading_pathway_path=LR_loading_pathway_path,
        dim_envir=dim_envir,
        MIlevel_agg_threshold=MIlevel_agg_threshold,
        batch_cell_unique_path=batch_cell_unique_path,
        adata_list_path=adata_list_path,
        adata_copy_path=adata_copy_path,
        show=show_flag
    )
