def MI_correlation(file_savepath_main, Factor_envir_use_path, show=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list
    """
    Compute and visualize correlation among MI factors.

    Args:
        file_savepath_main (str): Directory to save the result figure.
        Factor_envir_use_path (str): Path to the .npy file containing factor data.
        show (bool): Whether to display the heatmap inline (for Jupyter).
    """
    # Verify paths
    if not os.path.exists(Factor_envir_use_path):
        raise FileNotFoundError(f"File not found: {Factor_envir_use_path}")

    # Load array
    Factor_envir_use = np.load(Factor_envir_use_path)
    if Factor_envir_use.ndim == 1:
        Factor_envir_use = Factor_envir_use.reshape(-1, 1)

    # Compute correlation
    Factor_envir_use_corr = np.corrcoef(Factor_envir_use, rowvar=False)
    Factor_envir_use_corr = np.nan_to_num(Factor_envir_use_corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Clustering
    dist_matrix = 1 - Factor_envir_use_corr
    row_linkage = linkage(squareform(dist_matrix, checks=False), method='average')
    col_linkage = linkage(squareform(dist_matrix.T, checks=False), method='average')

    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    Factor_envir_use_corr = pd.DataFrame(
        Factor_envir_use_corr,
        index=[f"MI-{i+1}" for i in range(Factor_envir_use_corr.shape[0])],
        columns=[f"MI-{i+1}" for i in range(Factor_envir_use_corr.shape[0])]
    )

    # Reorder
    corr_clustered = Factor_envir_use_corr.iloc[row_order, :].iloc[:, col_order]

    # Plot
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr_clustered,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": .8},
        mask=np.eye(len(corr_clustered))
    )

    plt.title("Correlation of MI Factors (Clustered)")
    save_path = os.path.join(file_savepath_main, "MI_correlation_heatmap_clustered.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()
    print(f"Saved heatmap to {save_path}")

def LRLoading_enrichment(
        loading_LR_use_path,
        lr_list_path,
        lr_list_cellchatdb_path,
        lr_meta_cellchatdb_path,
        Factor_envir_use_path,
        file_savepath_main,
        show=True,
        min_lr_pairs_per_pathway=1,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    import os
    import pickle

    # matplotlib.use("TkAgg")  # or "Qt5Agg", "Agg" (non-interactive)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'arial'

    """
    Perform LR loading enrichment analysis and pathway visualization.

    Args:
        loading_LR_use_path (str): Path to .npy file for loading_LR_use.
        lr_list_path (str): Path to LR_list.pkl.
        lr_list_cellchatdb_path (str): Path to LR_list_cellchatdb.pkl.
        lr_meta_cellchatdb_path (str): Path to LR_meta_cellchatdb.pkl.
        Factor_envir_use_path (str): Path to Factor_envir_use.npy.
        file_savepath_main (str): Directory to save output results.
        show (bool): Whether to display the generated heatmap.
        min_lr_pairs_per_pathway (int): Minimum number of LR pairs required for a pathway
            to be included in the output heatmap and summary tables.
    """

    # Ensure save directory exists
    os.makedirs(file_savepath_main, exist_ok=True)

    # Ensure paths exist
    for path in [
        loading_LR_use_path,
        lr_list_path,
        lr_list_cellchatdb_path,
        lr_meta_cellchatdb_path,
        Factor_envir_use_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    if min_lr_pairs_per_pathway < 1:
        raise ValueError("min_lr_pairs_per_pathway must be >= 1")

    # Load data
    loading_LR_use = np.load(loading_LR_use_path)
    LR_list = pd.read_pickle(lr_list_path)
    LR_list_cellchatdb = pd.read_pickle(lr_list_cellchatdb_path)
    LR_meta_cellchatdb = pd.read_pickle(lr_meta_cellchatdb_path)
    Factor_envir_use = np.load(Factor_envir_use_path)

    # Unified MI names
    MI_names = [f"MI-{i + 1}" for i in range(loading_LR_use.shape[0])]

    # Weight by factor max
    Factor_envir_use_colmax = np.max(Factor_envir_use, axis=0)
    for i in range(loading_LR_use.shape[0]):
        loading_LR_use[i, :] *= Factor_envir_use_colmax[i]

    # Merge LR lists for comparison
    LR_list_merged = ["+".join(lr[0]) + "->" + "+".join(lr[1]) for lr in LR_list]
    LR_list_cellchatdb_merged = [
        "+".join(lr[0]) + "->" + "+".join(lr[1]) for lr in LR_list_cellchatdb
    ]

    # Identify LR pairs that exist in CellChatDB
    LR_incellchatdb_index = [
        i for i, lr in enumerate(LR_list_merged)
        if lr in LR_list_cellchatdb_merged
    ]
    LR_list_merged_incellchatdb = np.array(LR_list_merged)[LR_incellchatdb_index]

    LR_meta_incellchatdb = LR_meta_cellchatdb.iloc[
        [
            np.where(np.array(LR_list_cellchatdb_merged) == lr)[0][0]
            for lr in LR_list_merged_incellchatdb
        ],
        :
    ].copy()
    LR_meta_incellchatdb.index = np.arange(LR_meta_incellchatdb.shape[0])

    LR_meta_incellchatdb.to_csv(
        os.path.join(file_savepath_main, "LR_meta_incellchatdb.csv"),
        index=False
    )

    # Build human-readable LR pair names
    LR_list_merge = []
    for lr_pair in LR_list:
        ligand, receptor = lr_pair
        LR_list_merge.append("+".join(ligand) + " -> " + "+".join(receptor))

    # Save LR_list_merge.pkl
    LR_list_merge_path = os.path.join(file_savepath_main, "LR_list_merge.pkl")
    with open(LR_list_merge_path, "wb") as f:
        pickle.dump(LR_list_merge, f)

    # Save loading_LR_use as DataFrame csv
    loading_LR_use_df = pd.DataFrame(
        loading_LR_use,
        index=MI_names,
        columns=LR_list_merge
    )
    loading_LR_use_df_path = os.path.join(file_savepath_main, "loading_LR_use.csv")
    loading_LR_use_df.to_csv(loading_LR_use_df_path, index=True)

    # Continue for normalized matrix
    loading_LR_use = pd.DataFrame(
        loading_LR_use,
        columns=LR_list_merge,
        index=MI_names
    )
    loading_LR_use_norm = loading_LR_use.div(loading_LR_use.sum(axis=0), axis=1).fillna(0)

    # CellChat pathway enrichment
    pathway_unique_Cellchat = LR_meta_incellchatdb["pathway_name"].unique()
    LR_loading_pathway = []
    LR_idx_list_pathway = []
    pathway_summary = []

    for pathway in pathway_unique_Cellchat:
        LR_idx = LR_meta_incellchatdb.index[LR_meta_incellchatdb["pathway_name"] == pathway]
        n_lr = len(LR_idx)
        LR_cor = LR_list_merged_incellchatdb[LR_idx]

        keep_flag = n_lr >= min_lr_pairs_per_pathway
        pathway_summary.append({
            "pathway": pathway,
            "n_lr_pairs": n_lr,
            "kept": keep_flag
        })

        print(
            f"{pathway}: {n_lr} LR pairs "
            f"{'(kept)' if keep_flag else '(filtered out)'}"
        )

        if not keep_flag:
            continue

        LR_loading_pathway.append(loading_LR_use_norm.iloc[:, LR_idx].mean(axis=1))
        LR_idx_list_pathway.append(
            pd.DataFrame({
                "LR_index": LR_idx,
                "pathway": pathway,
                "LR_pair": LR_cor
            })
        )

    pathway_summary = pd.DataFrame(pathway_summary)
    pathway_summary.to_csv(
        os.path.join(file_savepath_main, "pathway_lr_pair_summary.csv"),
        index=False
    )

    if len(LR_loading_pathway) == 0:
        raise ValueError(
            f"No pathways passed the filter min_lr_pairs_per_pathway={min_lr_pairs_per_pathway}. "
            f"Try a smaller threshold."
        )

    LR_idx_list_pathway = pd.concat(LR_idx_list_pathway, ignore_index=True)
    LR_idx_list_pathway.to_csv(
        os.path.join(file_savepath_main, "LR_idx_list_pathway.csv"),
        index=False
    )

    LR_loading_pathway = pd.DataFrame(
        LR_loading_pathway,
        index=[x["pathway"] for x in pathway_summary.to_dict("records") if x["kept"]],
        columns=loading_LR_use_norm.index
    ).fillna(0).T

    # Sort columns in reverse order first
    LR_loading_pathway = LR_loading_pathway.loc[:, list(LR_loading_pathway.columns)[::-1]]

    # Sort rows by maximum pathway loading
    LR_loading_pathway = LR_loading_pathway.iloc[
        np.argsort(np.array(np.max(LR_loading_pathway, axis=1)))[::-1], :
    ]

    # Refined pathway ordering
    argmax_1 = np.argmax(np.array(LR_loading_pathway), axis=0)
    max_1 = np.max(np.array(LR_loading_pathway), axis=0)
    order_index_LRpathway = []
    for argmax_1_cur in np.sort(np.unique(argmax_1)):
        index_cur = np.where(argmax_1 == argmax_1_cur)[0]
        index_cur = index_cur[np.argsort(max_1[index_cur])[::-1]]
        index_cur = index_cur.tolist()
        order_index_LRpathway.extend(index_cur)

    LR_loading_pathway = LR_loading_pathway.iloc[:, order_index_LRpathway]

    # Visualization
    plt.close()
    fig, ax = plt.subplots(figsize=(9, 10))

    cmap = LinearSegmentedColormap.from_list(
        "white_red", ["#FCF5F0", "#F9B2BC", "#F6689F", "#C31988", "#510269"], N=256
    )

    data = LR_loading_pathway.values
    vmin = data.min()
    vmax = min(0.4, np.max(data) * 0.7)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)

    mesh = ax.pcolormesh(
        np.arange(data.shape[1] + 1),
        np.arange(data.shape[0] + 1),
        data,
        cmap=cmap,
        norm=norm,
        edgecolors="#B6B9BA",
        linewidth=1.0
    )

    ax.set_xticks(np.arange(data.shape[1]) + 0.5)
    ax.set_xticklabels(
        LR_loading_pathway.columns,
        rotation=60,
        fontsize=22,
        ha="left",
        rotation_mode="anchor"
    )
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_yticklabels(LR_loading_pathway.index, fontsize=22)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()

    plt.colorbar(mesh, ax=ax)
    plt.tight_layout()

    save_png = os.path.join(file_savepath_main, "LR_loading_pathway.png")
    plt.savefig(save_png, format="png", bbox_inches="tight", dpi=300)

    save_pdf = os.path.join(file_savepath_main, "LR_loading_pathway.pdf")
    plt.savefig(save_pdf, format="pdf", bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    plt.close()

    # Save results
    LR_loading_pathway.to_csv(os.path.join(file_savepath_main, "LR_loading_pathway.csv"))

    return {
        "loading_LR_use_df": loading_LR_use_df,
        "loading_LR_use_norm": loading_LR_use_norm,
        "LR_loading_pathway": LR_loading_pathway,
        "LR_idx_list_pathway": LR_idx_list_pathway,
        "LR_meta_incellchatdb": LR_meta_incellchatdb,
        "LR_list_merge": LR_list_merge,
        "pathway_summary": pathway_summary,
    }

def MI_Celltypepair_enrichment(
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

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.family'] = 'arial'
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
        _Avg_MI_site_pvalue = pd.DataFrame(
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
    plt.savefig(os.path.join(file_savepath_main, "Avg_MI_cellclass_pair.pdf"),
                format='pdf', bbox_inches='tight', dpi=300)
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
    plt.savefig(
        os.path.join(file_savepath_main, "Celltype_pair_example.pdf"),
        format='pdf', bbox_inches='tight', dpi=300, facecolor='white'
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

def MIinduced_cellembedding(
        cellclass_choose,
        file_savepath_main,
        Factor_envir_list_path,
        adata_copy_path,
        device,
        SpiderNet_data_pyg_list_path,
        LR_list_merge_path,
        Avg_MI_cellclass_pair_merge_use_path,
        dim_envir,
        MIlevel_agg_threshold,
        metadata_sample_path="None",
        embedding_method="PCA",
        show=True,
        cellclass_feature = 'cell.types'
):
    import numpy as np
    import pandas as pd
    import torch
    import scanpy as sc
    from torch_scatter import scatter_max
    from sklearn.decomposition import PCA, NMF
    from scipy.stats import zscore
    import umap
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    import seaborn as sns
    import os
    """
    Perform MI-induced embedding and visualization.

    Args:
        cellclass_choose (str): Target cell type for embedding.
        file_savepath_main (str): Directory to save results.
        Factor_envir_list_path (str): Path to Factor_envir_list.pkl.
        adata_copy_path (str): Path to adata_all.h5ad.
        device (str): "cuda:0" or "cpu".
        SpiderNet_data_pyg_list_path (str): Path to SpiderNet_data_pyg_list.pkl.
        LR_list_merge_path (str): Path to LR_list_merge.pkl.
        Avg_MI_cellclass_pair_merge_use_path (str): Path to Avg_MI_cellclass_pair_merge_use.pkl.
        dim_envir (int): Dimension of environmental factor.
        MIlevel_agg_threshold (float): Threshold for MI aggregation.
        metadata_sample_path (str): Optional CSV metadata file.
        embedding_method (str): "PCA" or "UMAP".
        show (bool): If True, display generated figures.
    """

    ## ---------------------------
    ## Step 1: Load data
    ## ---------------------------
    Factor_envir_list = pd.read_pickle(Factor_envir_list_path)
    adata_copy = sc.read_h5ad(adata_copy_path)
    SpiderNet_data_pyg_list = pd.read_pickle(SpiderNet_data_pyg_list_path)
    LR_list_merge = pd.read_pickle(LR_list_merge_path)
    Avg_MI_cellclass_pair_merge_use = pd.read_pickle(Avg_MI_cellclass_pair_merge_use_path)

    if metadata_sample_path != "None":
        metadata_sample = pd.read_csv(metadata_sample_path, index_col=0)
        metadata_sample.index = np.array(metadata_sample["samples"].astype(str))
    else:
        metadata_sample = None

    cellclass_unique = np.unique(adata_copy.obs[cellclass_feature])

    ## ---------------------------
    ## Step 2: Aggregate MI for each cell
    ## ---------------------------
    MI_SR_agg_cur_all = []
    MI_SR_agg_ct_all = []

    for slice_index in range(len(Factor_envir_list)):
        Factor_envir_cur = torch.tensor(Factor_envir_list[slice_index], dtype=torch.float32, device=device)
        edge_index_cur = SpiderNet_data_pyg_list[slice_index]['edge_index']
        num_cell_cur = SpiderNet_data_pyg_list[slice_index].x.shape[0]
        cellclass_cur = SpiderNet_data_pyg_list[slice_index]['cell_class']

        sender_cellclass_cur = cellclass_cur[edge_index_cur[:, 0].to(torch.int64).cpu().numpy()]
        receiver_cellclass_cur = cellclass_cur[edge_index_cur[:, 1].to(torch.int64).cpu().numpy()]

        MI_SR_agg_cur_sub_list = []

        for cellclass_unique_cur in cellclass_unique:
            sender_cellclass_cur_index = np.where(sender_cellclass_cur == cellclass_unique_cur)[0]
            receiver_cellclass_cur_index = np.where(receiver_cellclass_cur == cellclass_unique_cur)[0]

            Factor_envir_cur_useforreceiver = np.zeros(Factor_envir_cur.shape, dtype=np.float32) * np.nan
            Factor_envir_cur_useforreceiver[receiver_cellclass_cur_index, :] = (
                Factor_envir_cur[receiver_cellclass_cur_index, :].to("cpu").numpy()
            )

            Factor_envir_cur_useforsender = np.zeros(Factor_envir_cur.shape, dtype=np.float32) * np.nan
            Factor_envir_cur_useforsender[sender_cellclass_cur_index, :] = (
                Factor_envir_cur[sender_cellclass_cur_index, :].to("cpu").numpy()
            )

            MI_receiver_agg_cur_sub = scatter_max(
                torch.tensor(Factor_envir_cur_useforsender).to(device),
                edge_index_cur[:, 1].to(torch.int64).to(device),
                dim=0,
                dim_size=num_cell_cur
            )[0].to("cpu").numpy()

            MI_sender_agg_cur_sub = scatter_max(
                torch.tensor(Factor_envir_cur_useforreceiver).to(device),
                edge_index_cur[:, 0].to(torch.int64).to(device),
                dim=0,
                dim_size=num_cell_cur
            )[0].to("cpu").numpy()

            MI_SR_agg_cur_sub = np.hstack([MI_sender_agg_cur_sub, MI_receiver_agg_cur_sub])
            MI_SR_agg_cur_sub_list.append(MI_SR_agg_cur_sub)

        MI_SR_agg_ct = np.hstack(MI_SR_agg_cur_sub_list)
        MI_SR_agg_ct_all.append(MI_SR_agg_ct)

        MI_receiver_agg_cur = scatter_max(
            Factor_envir_cur.to(device),
            edge_index_cur[:, 1].to(torch.int64).to(device),
            dim=0,
            dim_size=num_cell_cur
        )[0].to("cpu").numpy()

        MI_sender_agg_cur = scatter_max(
            Factor_envir_cur.to(device),
            edge_index_cur[:, 0].to(torch.int64).to(device),
            dim=0,
            dim_size=num_cell_cur
        )[0].to("cpu").numpy()

        MI_SR_agg_cur = np.hstack([MI_sender_agg_cur, MI_receiver_agg_cur])
        MI_SR_agg_cur_all.append(MI_SR_agg_cur)

        if slice_index == 0:
            MI_agg_meta = pd.DataFrame({
                "MI": ['MI' + str(i + 1) for i in range(MI_sender_agg_cur.shape[1])] +
                      ['MI' + str(i + 1) for i in range(MI_sender_agg_cur.shape[1])],
                "SR": ['Sender'] * MI_sender_agg_cur.shape[1] + ['Receiver'] * MI_receiver_agg_cur.shape[1]
            })

            MI_agg_ct_meta = []
            for i in range(len(cellclass_unique)):
                meta_copy = MI_agg_meta.copy()
                meta_copy['celltype'] = cellclass_unique[i]
                MI_agg_ct_meta.append(meta_copy)
            MI_agg_ct_meta = pd.concat(MI_agg_ct_meta, axis=0, ignore_index=True)

    MI_SR_agg_cur_all = np.vstack(MI_SR_agg_cur_all)
    MI_SR_agg_ct_all = np.vstack(MI_SR_agg_ct_all)
    np.save(os.path.join(file_savepath_main, "MI_SR_agg_all.npy"), MI_SR_agg_cur_all)
    np.save(os.path.join(file_savepath_main, "MI_SR_agg_ct_all.npy"), MI_SR_agg_ct_all)
    ##Save the MI_agg_ct_meta and MI_agg_meta
    MI_agg_meta.to_csv(os.path.join(file_savepath_main, "MI_agg_meta.csv"), index=False)
    MI_agg_ct_meta.to_csv(os.path.join(file_savepath_main, "MI_agg_ct_meta.csv"), index=False)
    ##
    ## =============================================================
    ## Step 3: Aggregate LR neighborhood information
    ## =============================================================
    cellpair_LRpair_neigh_sender_all = []
    cellpair_LRpair_neigh_receiver_all = []

    for slice_index in range(len(Factor_envir_list)):
        cellpair_LRpair_neigh_cur = SpiderNet_data_pyg_list[slice_index]['cellpair_LRpair_neigh']
        edge_index_cur = SpiderNet_data_pyg_list[slice_index]['edge_index']
        num_cell_cur = SpiderNet_data_pyg_list[slice_index].x.shape[0]

        cellpair_LRpair_neigh_receiver_cur = scatter_max(
            cellpair_LRpair_neigh_cur.to(device),
            edge_index_cur[:, 1].to(torch.int64).to(device),
            dim=0,
            dim_size=num_cell_cur
        )[0].to("cpu").numpy()

        cellpair_LRpair_neigh_sender_cur = scatter_max(
            cellpair_LRpair_neigh_cur.to(device),
            edge_index_cur[:, 0].to(torch.int64).to(device),
            dim=0,
            dim_size=num_cell_cur
        )[0].to("cpu").numpy()

        cellpair_LRpair_neigh_sender_all.append(cellpair_LRpair_neigh_sender_cur)
        cellpair_LRpair_neigh_receiver_all.append(cellpair_LRpair_neigh_receiver_cur)

    cellpair_LRpair_neigh_sender_all = np.vstack(cellpair_LRpair_neigh_sender_all)
    cellpair_LRpair_neigh_receiver_all = np.vstack(cellpair_LRpair_neigh_receiver_all)

    df_sender = pd.DataFrame(cellpair_LRpair_neigh_sender_all,
                             index=adata_copy.obs.index,
                             columns=LR_list_merge)
    df_receiver = pd.DataFrame(cellpair_LRpair_neigh_receiver_all,
                               index=adata_copy.obs.index,
                               columns=LR_list_merge)

    df_sender.to_csv(os.path.join(file_savepath_main, "cellpair_LRpair_neigh_sender_all.csv"))
    df_receiver.to_csv(os.path.join(file_savepath_main, "cellpair_LRpair_neigh_receiver_all.csv"))

    df_sender_norm = df_sender / df_sender.max(axis=0)
    df_receiver_norm = df_receiver / df_receiver.max(axis=0)

    ## =============================================================
    ## Step 4: Select informative MI features
    ## =============================================================
    Avg_MI = Avg_MI_cellclass_pair_merge_use
    chosen = Avg_MI.iloc[np.where(~((~(Avg_MI['Sender'] == cellclass_choose)) *
                                    (~(Avg_MI['Receiver'] == cellclass_choose))))[0], :]

    num_MI_filtered = np.sum(np.sum(chosen.iloc[:, 0:dim_envir] >= MIlevel_agg_threshold, axis=0) > 0)
    chosen = chosen.drop(columns=chosen.iloc[:, 0:dim_envir].columns[
        np.sum(chosen.iloc[:, 0:dim_envir] >= MIlevel_agg_threshold, axis=0) == 0])

    chosen_further = chosen.copy()
    # chosen_further.iloc[:, 0:num_MI_filtered] = (
    #         chosen_further.iloc[:, 0:num_MI_filtered] >= MIlevel_agg_threshold
    # )
    # chosen_further.iloc[:, 0:num_MI_filtered] = (
    #     chosen_further.iloc[:, 0:num_MI_filtered].astype(bool)
    # )
    mask = (chosen_further.iloc[:, 0:num_MI_filtered] >= MIlevel_agg_threshold)
    chosen_further.iloc[:, 0:num_MI_filtered] = mask.astype(np.float32).values
    #
    # # Compute boolean mask and explicitly cast to bool before assignment
    # chosen_further.iloc[:, 0:num_MI_filtered] = (
    #     (chosen_further.iloc[:, 0:num_MI_filtered] >= MIlevel_agg_threshold)
    #     .astype(bool)
    # )
    ## Build dataframe summarizing sender/receiver directionality
    df_pairs = []
    for MI_index in range(num_MI_filtered):
        for row_index in range(chosen_further.shape[0]):
            sender_cur = chosen_further.iloc[row_index, -2]
            receiver_cur = chosen_further.iloc[row_index, -1]
            MI_cur = chosen_further.columns[MI_index]
            val = chosen_further.iloc[row_index, MI_index]
            # if val:
            if val == 1:
                df_cur = pd.DataFrame({
                    "Sender": [sender_cur],
                    "Receiver": [receiver_cur],
                    "MI": [MI_cur]
                })
                df_cur["SR"] = np.nan
                if sender_cur == cellclass_choose and receiver_cur != cellclass_choose:
                    df_cur["SR"] = "Sending"
                elif sender_cur != cellclass_choose and receiver_cur == cellclass_choose:
                    df_cur["SR"] = "Receiving"
                else:
                    df_dup = df_cur.copy()
                    df_cur["SR"] = "Sending"
                    df_dup["SR"] = "Receiving"
                    df_cur = pd.concat([df_cur, df_dup], ignore_index=True)
                df_pairs.append(df_cur)

    df_pairs = pd.concat(df_pairs, ignore_index=True)
    df_pairs.to_csv(os.path.join(file_savepath_main, f"Avg_MI_pair_{cellclass_choose}.csv"), index=False)

    ## =============================================================
    ## Step 5: Plot the MI-by-cell-type-pair heatmap
    ## =============================================================
    plt.close()
    heatmap_values = chosen.iloc[:, 0:num_MI_filtered].values.T
    cmap = LinearSegmentedColormap.from_list(
        "custom_coolwarm_graycenter",
        ["#4575b4", "#f0f0f0", "#d73027"], N=256
    )
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    n_mis, n_pairs = heatmap_values.shape
    x = np.arange(n_pairs + 1)
    y = np.arange(n_mis + 1)

    fig, ax = plt.subplots(figsize=(4, 6))
    mesh = ax.pcolormesh(x, y, heatmap_values, cmap=cmap, norm=norm,
                         edgecolors='black', linewidth=0.5)
    fig.colorbar(mesh, ax=ax)
    cellclass_pair_names = [
        f"{s} → {r}" for s, r in zip(chosen["Sender"], chosen["Receiver"])
    ]
    ax.set_xticks(np.arange(n_pairs) + 0.5)
    ax.set_xticklabels(cellclass_pair_names, rotation=60, ha='right', fontsize=12)
    ax.set_yticks(np.arange(n_mis) + 0.5)
    ax.set_yticklabels(chosen.columns[0:num_MI_filtered], fontsize=12)
    ax.invert_yaxis()

    for i in range(n_mis):
        for j in range(n_pairs):
            if heatmap_values[i, j] >= MIlevel_agg_threshold:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor='yellow', linewidth=2, zorder=10))
    plt.tight_layout()
    figpath = os.path.join(file_savepath_main, f"Avg_MI_heatmap_{cellclass_choose}.png")
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    ## =============================================================
    ## Step 6: Generate embedding
    ## =============================================================
    cellindex_choose = np.where(adata_copy.obs[cellclass_feature] == cellclass_choose)[0]
    adata_choose = adata_copy[cellindex_choose, :].copy()
    adata_choose.obs['barcode'] = adata_choose.obs_names

    if metadata_sample is not None:
        adata_choose.obs = adata_choose.obs.merge(metadata_sample, on='samples', how='left')

    MI_SR_all_choose = MI_SR_agg_cur_all[cellindex_choose, :]
    MI_SR_ct_all_choose = MI_SR_agg_ct_all[cellindex_choose, :]

    sender_subset = chosen[chosen['Sender'] == cellclass_choose]
    receiver_subset = chosen[chosen['Receiver'] == cellclass_choose]

    sender_enrichMI = sender_subset.columns[0:(sender_subset.shape[1] - 2)][
        np.max(sender_subset.iloc[:, 0:(sender_subset.shape[1] - 2)], axis=0) > MIlevel_agg_threshold
        ]
    sender_enrichMI = [x.replace('-', '') for x in sender_enrichMI]

    receiver_enrichMI = receiver_subset.columns[0:(receiver_subset.shape[1] - 2)][
        np.max(receiver_subset.iloc[:, 0:(receiver_subset.shape[1] - 2)], axis=0) > MIlevel_agg_threshold
        ]
    receiver_enrichMI = [x.replace('-', '') for x in receiver_enrichMI]

    sender_idx = np.where((MI_agg_meta['SR'] == "Sender") & (MI_agg_meta['MI'].isin(sender_enrichMI)))[0]
    receiver_idx = np.where((MI_agg_meta['SR'] == "Receiver") & (MI_agg_meta['MI'].isin(receiver_enrichMI)))[0]
    MI_agg_chooseindex = np.unique(np.hstack([sender_idx, receiver_idx]))

    MI_SR_agg_cur_all_choose_useforPCA = MI_SR_all_choose[:, MI_agg_chooseindex]

    ## =============================================================
    ## Step 7: Embedding via PCA or NMF + UMAP
    ## =============================================================
    MI_SR_z = np.nan_to_num(zscore(MI_SR_agg_cur_all_choose_useforPCA, axis=0), nan=0.0)

    if embedding_method.upper() == "PCA":
        pca = PCA(n_components=min(MI_SR_z.shape[1], 50))
        MI_pca = pca.fit_transform(MI_SR_z)
        var_ratio = np.cumsum(pca.explained_variance_ratio_)
        comp_num = np.where(var_ratio >= 0.85)[0][0]
        MI_pca = MI_pca[:, :comp_num]

    elif embedding_method.upper() == "NMF":
        n_components = min(10, MI_SR_z.shape[0], MI_SR_z.shape[1])
        model_NMF = NMF(n_components=n_components, init='nndsvda', random_state=0, max_iter=500)
        MI_pca = model_NMF.fit_transform(MI_SR_z)
    else:
        raise ValueError("embedding_method must be 'PCA' or 'NMF'")

    umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
    MI_umap = umap_model.fit_transform(MI_pca)

    adata_choose.obsm['MI_PCA'] = MI_pca
    adata_choose.obsm['MI_UMAP'] = MI_umap

    adata_choose.write_h5ad(os.path.join(file_savepath_main, f"adata_choose_{cellclass_choose}.h5ad"))
    MI_agg_meta.to_csv(os.path.join(file_savepath_main, f"MI_agg_meta_{cellclass_choose}.csv"), index=False)
    np.save(os.path.join(file_savepath_main, f"MI_agg_chooseindex_{cellclass_choose}.npy"), MI_agg_chooseindex)
    np.save(os.path.join(file_savepath_main, f"MI_SR_agg_cur_all_choose_{cellclass_choose}.npy"), MI_SR_all_choose)

    print(f"Finished embedding for {cellclass_choose}. Results saved in {file_savepath_main}.")

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# =========================================================
# Global plotting style
# =========================================================
CUSTOM_COLORS = [
    "#e6f29f", "#d6e97c",
    "#fee089", "#f6b656",
    "#645599", "#8c6bb1",
    "#ac352f", "#d95f02",
    "#90c6b6", "#64b3a9",
    "#2e87b6", "#1b5f8a",
]
BASE_COLORS = CUSTOM_COLORS[::2]

# =========================================================
# General utilities
# =========================================================
def make_categorical_palette(categories):
    """
    Build a categorical color palette as a dict: category -> color.
    Colors are assigned by cycling through BASE_COLORS.
    """
    cats = list(pd.Index(categories).astype(str))
    cols = [BASE_COLORS[i % len(BASE_COLORS)] for i in range(len(cats))]
    return dict(zip(cats, cols))

def style_umap_axes(
    ax,
    xlabel="UMAP 1",
    ylabel="UMAP 2",
    label_fontsize=18,
    hide_ticks=True,
):
    """
    Apply a clean UMAP axis style suitable for publication figures.
    """
    ax.set_facecolor("white")
    ax.grid(False)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

def save_white_bg(fig, out_png, out_pdf, dpi=300):
    """
    Save the current figure to both PNG and PDF with white background.
    """
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=dpi, facecolor="white")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=dpi, facecolor="white")

# =========================================================
# UMAP drawing helpers
# =========================================================
def draw_category_umap(
    ax,
    umap,
    series,
    *,
    size=2,
    palette=None,
    legend=True,
    legend_title=None,
    legend_fontsize=17,
    markerscale=8,
    legend_max_items=15,
):
    """
    Draw a categorical UMAP by plotting one scatter layer per category.

    Parameters
    ----------
    ax : matplotlib axis
    umap : np.ndarray
        2D embedding array of shape (n_cells, 2)
    series : pd.Series
        Categorical annotation to color
    size : float
        Marker size
    palette : dict or None
        Mapping from category -> color
    legend : bool
        Whether to draw a legend
    legend_title : str or None
        Legend title
    """
    s_str = series.astype("string").fillna("Unknown")
    cats = list(pd.Index(s_str.unique()).astype(str))

    if palette is None:
        palette = make_categorical_palette(cats)
    else:
        for cat in cats:
            if cat not in palette:
                palette[cat] = BASE_COLORS[len(palette) % len(BASE_COLORS)]

    handles, labels = [], []
    for cat in cats:
        idx = (s_str == cat).values
        h = ax.scatter(
            umap[idx, 0],
            umap[idx, 1],
            c=[palette[str(cat)]],
            s=size,
            edgecolor="none",
            rasterized=True,
            label=str(cat),
        )
        handles.append(h)
        labels.append(str(cat))

    if legend and (len(labels) <= legend_max_items):
        ax.legend(
            handles,
            labels,
            title="" if legend_title is None else legend_title,
            fontsize=legend_fontsize,
            scatterpoints=1,
            markerscale=markerscale,
            handlelength=2.0,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    return handles, labels

def draw_continuous_umap(
    ax,
    umap,
    values,
    *,
    size=2,
    cmap="inferno",
    colorbar_label=None,
    colorbar_fontsize=18,
):
    """
    Draw a continuous UMAP colored by a numeric variable.
    """
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)

    scp = ax.scatter(
        umap[order, 0],
        umap[order, 1],
        c=v[order],
        cmap=cmap,
        s=size,
        edgecolor="none",
        rasterized=True,
    )

    cbar = plt.colorbar(scp, ax=ax)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)

    return scp

# =========================================================
# Main function: feature UMAP plots
# =========================================================
def Feature_show_umap(
    cellclass_choose,
    adata_choose_path,
    file_savepath_main,
    obsm_show,
    adata_copy_path,
    show=True,
):
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # ---------------- Helper: palette ----------------
    def get_palette_for_obs(obs_col, series):
        if obs_col == "stage_x":
            stage_palette = {}
            stage_values = series.astype("string").fillna("Unknown").unique()

            for stage in stage_values:
                s = str(stage).strip()
                s_norm = s.upper().replace("STAGE", "").replace(" ", "")

                if s_norm in {"I", "1"}:
                    stage_palette[s] = "#F4CCE9"
                elif s_norm in {"III", "3"}:
                    stage_palette[s] = "#D17D98"
                elif s_norm in {"IV", "4"}:
                    stage_palette[s] = "#7D1C4A"
                else:
                    stage_palette[s] = "#636363"

            return stage_palette
        return None

    # ---------------- Helper: draw categorical ----------------
    def draw_category_umap(ax, umap, series, obs_col, size=2):

        s_str = series.astype("string").fillna("Unknown")

        # ===== SPECIAL: stage layered plotting =====
        if obs_col == "stage_x":
            stage_order = [
                ("I", 1),
                ("1", 1),
                ("III", 2),
                ("3", 2),
                ("IV", 3),
                ("4", 3),
            ]

            # normalize labels
            s_norm = (
                s_str.str.upper()
                .str.replace("STAGE", "", regex=False)
                .str.replace(" ", "", regex=False)
            )

            palette = get_palette_for_obs("stage_x", s_str)

            # assign layer
            layer_map = {}
            for raw in s_str.unique():
                raw_str = str(raw)
                norm = (
                    raw_str.upper()
                    .replace("STAGE", "")
                    .replace(" ", "")
                )

                if norm in {"I", "1"}:
                    layer_map[raw_str] = 1
                elif norm in {"III", "3"}:
                    layer_map[raw_str] = 2
                elif norm in {"IV", "4"}:
                    layer_map[raw_str] = 3
                else:
                    layer_map[raw_str] = 0  # unknown

            # plot in order
            for layer in [1, 2, 3]:
                for cat in s_str.unique():
                    if layer_map[str(cat)] != layer:
                        continue

                    idx = (s_str == cat).values
                    ax.scatter(
                        umap[idx, 0],
                        umap[idx, 1],
                        c=palette[str(cat)],
                        s=size,
                        edgecolor="none",
                        rasterized=True,
                        zorder=layer,  # Draw higher layers on top
                        label=str(cat),
                    )

            # legend
            handles = []
            labels = []
            for cat in s_str.unique():
                handles.append(
                    ax.scatter([], [], c=palette[str(cat)], s=10)
                )
                labels.append(str(cat))

            ax.legend(
                handles,
                labels,
                fontsize=14,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
            )

            return

        # ===== default =====
        cats = s_str.unique()
        palette = {c: "#636363" for c in cats}

        for i, cat in enumerate(cats):
            idx = (s_str == cat).values
            ax.scatter(
                umap[idx, 0],
                umap[idx, 1],
                c=palette[cat],
                s=size,
                edgecolor="none",
                rasterized=True,
            )

    # ---------------- Load ----------------
    adata_choose = sc.read_h5ad(adata_choose_path)
    adata_copy = sc.read_h5ad(adata_copy_path)

    MI_umap = adata_choose.obsm[obsm_show]

    os.makedirs(file_savepath_main, exist_ok=True)

    # ---------------- Plot ----------------
    for obs_col in ["stage_x"]:

        if obs_col not in adata_choose.obs:
            continue

        s = adata_choose.obs[obs_col]

        fig, ax = plt.subplots(figsize=(8, 6))

        draw_category_umap(
            ax,
            MI_umap,
            s,
            obs_col=obs_col,
            size=2,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(obs_col)

        plt.tight_layout()

        plt.savefig(
            os.path.join(file_savepath_main, f"UMAP_{obs_col}.pdf"),
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close()

# =========================================================
# Louvain / Leiden UMAP plot
# =========================================================
def plot_louvain_umap(
    adata,
    obsm_key="X_umap",
    louvain_key="X_louvain",
    out_dir=".",
    out_prefix="UMAP_Louvain",
    title=None,
    size=2,
    dpi=300,
    show=True,
    custom_colors=None,
    use_every_second_color=True,
    legend_title=None,
    legend_fontsize=17,
    label_fontsize=18,
    markerscale=8,
):
    """
    Plot Louvain or Leiden cluster labels on a UMAP embedding.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    obsm_key : str
        Key in adata.obsm for the embedding
    louvain_key : str
        Column in adata.obs storing cluster labels
    out_dir : str
        Output directory
    out_prefix : str
        Output filename prefix
    size : float
        Scatter marker size
    custom_colors : list[str] or None
        Optional custom color list
    use_every_second_color : bool
        Whether to use every second color from the custom palette
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    if obsm_key not in adata.obsm:
        raise KeyError(f"{obsm_key} not found in adata.obsm keys: {list(adata.obsm.keys())}")
    if louvain_key not in adata.obs:
        raise KeyError(f"{louvain_key} not found in adata.obs columns: {list(adata.obs.columns)}")

    umap = np.asarray(adata.obsm[obsm_key])
    lab = adata.obs[louvain_key].astype("category")
    codes = lab.cat.codes.to_numpy()
    cats = lab.cat.categories.tolist()
    n_cats = len(cats)

    # Default custom palette
    if custom_colors is None:
        custom_colors = [
            "#e6f29f", "#d6e97c",
            "#fee089", "#f6b656",
            "#645599", "#8c6bb1",
            "#ac352f", "#d95f02",
            "#90c6b6", "#64b3a9",
            "#2e87b6", "#1b5f8a",
        ]

    base_colors = custom_colors[::2] if use_every_second_color else custom_colors
    base_colors = [mcolors.to_hex(c) for c in base_colors]

    def generate_distinct_colors(n, preferred_colors):
        """
        Return n visually distinct colors.
        Use preferred_colors first if enough; otherwise extend automatically.
        """
        if len(preferred_colors) >= n:
            return preferred_colors[:n]

        # Start from preferred colors
        colors = preferred_colors.copy()

        # Add more from tab20 / tab20b / tab20c first (good categorical palettes)
        extra_pool = []
        for cmap_name in ["tab20", "tab20b", "tab20c"]:
            cmap = plt.get_cmap(cmap_name)
            extra_pool.extend([mcolors.to_hex(cmap(i)) for i in range(cmap.N)])

        # Remove duplicates while preserving order
        seen = set(colors)
        extra_pool_unique = []
        for c in extra_pool:
            if c not in seen:
                extra_pool_unique.append(c)
                seen.add(c)

        colors.extend(extra_pool_unique)

        # If still not enough, fall back to evenly spaced HSV colors
        if len(colors) < n:
            n_extra = n - len(colors)
            hsv_colors = [
                mcolors.to_hex(mcolors.hsv_to_rgb((i / n_extra, 0.65, 0.9)))
                for i in range(n_extra)
            ]
            for c in hsv_colors:
                if c not in seen:
                    colors.append(c)
                    seen.add(c)

        return colors[:n]

    colors_use = generate_distinct_colors(n_cats, base_colors)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot points
    for i, name in enumerate(cats):
        idx = (codes == i)
        if not np.any(idx):
            continue

        ax.scatter(
            umap[idx, 0],
            umap[idx, 1],
            c=[colors_use[i]],
            s=size,
            edgecolor="none",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1", fontsize=label_fontsize)
    ax.set_ylabel("UMAP 2", fontsize=label_fontsize)

    # Build cleaner legend manually
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=colors_use[i],
            markeredgecolor="none",
            markersize=max(4, markerscale),
            linestyle="None",
            label=str(cats[i]),
        )
        for i in range(n_cats)
    ]

    ax.legend(
        handles=handles,
        title="" if legend_title is None else legend_title,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
    )

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if title is None:
        title = f"{louvain_key} on {obsm_key}"
    ax.set_title(title, fontsize=label_fontsize)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{out_prefix}.png")
    out_pdf = os.path.join(out_dir, f"{out_prefix}.pdf")

    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return out_png, out_pdf

import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter_max, scatter_add

def scatter_nanmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    """
    Similar to scatter_max, but ignores NaN values when computing the group-wise maximum.
    If all elements in a group are NaN, output is set to NaN for that group.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor of shape [N, D] or compatible shape.
    index : torch.Tensor
        Group indices of shape [N].
    dim : int
        Dimension along which to scatter.
    dim_size : int or None
        Number of output groups.

    Returns
    -------
    torch.Tensor
        Group-wise maximum with NaNs ignored.
    """
    mask = (~torch.isnan(src)).float()
    src_no_nan = torch.nan_to_num(src, nan=float("-inf"))
    out, _ = scatter_max(src_no_nan, index, dim=dim, dim_size=dim_size)
    den = scatter_add(mask, index, dim=dim, dim_size=dim_size)
    out[den == 0] = float("nan")
    return out

def aggregate_mi_sender_receiver(
    Factor_envir_list,
    SpiderNet_data_pyg_list,
    cellclass_unique,
    device,
    cellclass_updated_df: pd.DataFrame | None = None,
    celltype_col: str = "cell.types.updated",
):
    """
    Aggregate sender/receiver MI factors across slices.

    For each slice:
    1. Compute cell-type-specific sender/receiver aggregated MI features
    2. Compute all-cell sender/receiver aggregated MI features
    3. Build metadata tables for aggregated features

    Parameters
    ----------
    Factor_envir_list : list
        List of environment factor matrices, one per slice.
        Each element should have shape [num_edges, num_MI].
    SpiderNet_data_pyg_list : list
        List of PyG-like data objects. Each element must contain:
        - ['edge_index']: edge index of shape [num_edges, 2]
        - .x.shape[0]: number of cells
        - ['cell_class']: class label for each edge endpoint mapping
        - ['cellnames']: barcode list
    cellclass_unique : list
        Ordered list of unique cell types to iterate over.
    device : torch.device or str
        Device for torch computation.
    cellclass_updated_df : pd.DataFrame or None
        Optional dataframe indexed by barcodes. Currently not used in aggregation,
        but kept here for compatibility/extension.
    celltype_col : str
        Column name in `cellclass_updated_df`.

    Returns
    -------
    MI_SR_agg_cur_all : np.ndarray
        Aggregated sender/receiver features across all cells, stacked across slices.
        Shape: [sum(num_cell), 2 * num_MI]
    MI_SR_agg_ct_all : np.ndarray
        Cell-type-specific aggregated sender/receiver features, stacked across slices.
        Shape: [sum(num_cell), 2 * num_MI * num_celltypes]
    MI_agg_meta : pd.DataFrame
        Metadata for all-cell sender/receiver features.
    MI_agg_ct_meta : pd.DataFrame
        Metadata for cell-type-specific sender/receiver features.
    """
    MI_SR_agg_cur_all = []
    MI_SR_agg_ct_all = []
    MI_agg_meta = None
    MI_agg_ct_meta = None

    for slice_index, Factor_envir in enumerate(Factor_envir_list):
        Factor_envir = torch.as_tensor(Factor_envir, dtype=torch.float32, device=device)

        data_cur = SpiderNet_data_pyg_list[slice_index]
        edge_index = data_cur["edge_index"]
        num_cell = data_cur.x.shape[0]
        cell_class = data_cur["cell_class"]
        barcodes = data_cur["cellnames"]

        # Optional hook: keep for compatibility, although not used below
        if cellclass_updated_df is not None:
            _ = cellclass_updated_df.loc[barcodes, celltype_col].values

        sender_class = cell_class[edge_index[:, 0].cpu().numpy()]
        receiver_class = cell_class[edge_index[:, 1].cpu().numpy()]

        # ----------------------------------------------------------
        # Cell-type-specific sender/receiver aggregation
        # ----------------------------------------------------------
        MI_SR_agg_cur_sub_list = []

        for ctype in cellclass_unique:
            send_idx = np.where(sender_class == ctype)[0]
            recv_idx = np.where(receiver_class == ctype)[0]

            F_recv = np.full(Factor_envir.shape, np.nan, dtype=np.float32)
            F_send = np.full(Factor_envir.shape, np.nan, dtype=np.float32)

            Factor_envir_np = Factor_envir.detach().cpu().numpy()
            F_recv[recv_idx, :] = Factor_envir_np[recv_idx, :]
            F_send[send_idx, :] = Factor_envir_np[send_idx, :]

            MI_recv = scatter_nanmax(
                torch.as_tensor(F_send, dtype=torch.float32, device=device),
                edge_index[:, 1].long().to(device),
                dim=0,
                dim_size=num_cell,
            ).cpu().numpy()

            MI_send = scatter_nanmax(
                torch.as_tensor(F_recv, dtype=torch.float32, device=device),
                edge_index[:, 0].long().to(device),
                dim=0,
                dim_size=num_cell,
            ).cpu().numpy()

            MI_SR_agg_cur_sub_list.append(np.hstack([MI_send, MI_recv]))

        MI_SR_agg_ct = np.hstack(MI_SR_agg_cur_sub_list)
        MI_SR_agg_ct_all.append(MI_SR_agg_ct)

        # ----------------------------------------------------------
        # All-cell sender/receiver aggregation
        # ----------------------------------------------------------
        MI_recv_all, _ = scatter_max(
            Factor_envir,
            edge_index[:, 1].long().to(device),
            dim=0,
            dim_size=num_cell,
        )
        MI_send_all, _ = scatter_max(
            Factor_envir,
            edge_index[:, 0].long().to(device),
            dim=0,
            dim_size=num_cell,
        )

        MI_SR_agg_cur_all.append(
            np.hstack([MI_send_all.cpu().numpy(), MI_recv_all.cpu().numpy()])
        )

        # ----------------------------------------------------------
        # Metadata (only need to build once)
        # ----------------------------------------------------------
        if slice_index == 0:
            n_mi = MI_send_all.shape[1]
            MI_agg_meta = pd.DataFrame({
                "MI": [f"MI{i+1}" for i in range(n_mi)] * 2,
                "SR": ["Sender"] * n_mi + ["Receiver"] * n_mi,
            })

            MI_agg_ct_meta = pd.concat(
                [MI_agg_meta.assign(celltype=ctype) for ctype in cellclass_unique],
                ignore_index=True,
            )

    MI_SR_agg_cur_all = np.vstack(MI_SR_agg_cur_all)
    MI_SR_agg_ct_all = np.vstack(MI_SR_agg_ct_all)

    return MI_SR_agg_cur_all, MI_SR_agg_ct_all, MI_agg_meta, MI_agg_ct_meta

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict

def build_hyper_edge_adj(edge_index_cur, num_edges):
    # Purpose: build an edge-to-edge adjacency matrix where edge j is connected
    # to edge i if the source node of j equals the target node of i.
    """
    Build a sparse hyper-edge adjacency matrix for one sample.

    Parameters
    ----------
    edge_index_cur : np.ndarray
        Array of shape (num_edges, 2), where each row is (src, tgt).
    num_edges : int
        Number of edges in the current sample.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix of shape (num_edges, num_edges).
    """
    src = edge_index_cur[:, 0]
    tgt = edge_index_cur[:, 1]

    src2rows = defaultdict(list)
    for j, s in enumerate(src):
        src2rows[s].append(j)

    rows, cols = [], []
    for i, t in enumerate(tgt):
        if t in src2rows:
            rows.extend([i] * len(src2rows[t]))
            cols.extend(src2rows[t])

    data = np.ones(len(rows), dtype=np.int8)
    hyper_edge_adj = csr_matrix((data, (rows, cols)), shape=(num_edges, num_edges))
    return hyper_edge_adj

def compute_observed_colocalization(F, A, MI_threshold):
    # Purpose: binarize MI activity using a threshold and compute the observed
    # MI-MI colocalization count matrix for one sample.
    """
    Compute the observed colocalization count matrix for one sample.

    Parameters
    ----------
    F : np.ndarray
        Normalized MI factor matrix of shape (num_edges, d).
    A : scipy.sparse.csr_matrix
        Hyper-edge adjacency matrix of shape (num_edges, num_edges).
    MI_threshold : float
        Threshold used to binarize MI activity.

    Returns
    -------
    F_cut : np.ndarray
        Binarized MI activity matrix of shape (num_edges, d).
    colocal_count_cur : np.ndarray
        Observed MI-MI colocalization count matrix of shape (d, d).
    """
    F_cut = np.where(F > MI_threshold, 1, 0)
    AF = A @ F_cut
    colocal_count_cur = F_cut.T @ AF
    return F_cut, colocal_count_cur

def get_celltype_pairs(edge_index_cur, adata_cur, celltype_col="cell.types"):
    # Purpose: group edge indices by sender-receiver cell-type pair so that
    # permutations can be restricted within each cell-type pair category.
    """
    Group edge indices by sender-receiver cell-type pair.

    Parameters
    ----------
    edge_index_cur : np.ndarray
        Array of shape (num_edges, 2), where each row is (src, tgt).
    adata_cur : AnnData
        AnnData object containing cell-level annotations.
    celltype_col : str
        Column name in adata_cur.obs storing cell-type labels.

    Returns
    -------
    dict
        Dictionary mapping (sender_celltype, receiver_celltype) to an array of
        edge indices belonging to that pair.
    """
    cell_types = adata_cur.obs[celltype_col].values
    sender_ct = cell_types[edge_index_cur[:, 0]]
    receiver_ct = cell_types[edge_index_cur[:, 1]]

    pair_to_indices = defaultdict(list)
    for i, pair in enumerate(zip(sender_ct, receiver_ct)):
        pair_to_indices[pair].append(i)

    pair_to_indices = {
        pair: np.asarray(idx, dtype=np.int64)
        for pair, idx in pair_to_indices.items()
    }
    return pair_to_indices

def compute_permutation_colocalization(F_cut, A, unique_pairs, nperm=100):
    # Purpose: generate permutation-based null MI-MI colocalization matrices
    # by shuffling edges only within each cell-type pair group.
    """
    Compute permutation-based null colocalization matrices.

    Parameters
    ----------
    F_cut : np.ndarray
        Binarized MI activity matrix of shape (num_edges, d).
    A : scipy.sparse.csr_matrix
        Hyper-edge adjacency matrix of shape (num_edges, num_edges).
    unique_pairs : dict
        Dictionary mapping cell-type pairs to edge indices.
    nperm : int
        Number of permutations.

    Returns
    -------
    np.ndarray
        Array of shape (nperm, d, d) containing permutation colocalization matrices.
    """
    colocal_count_perm_list = []

    for _ in range(nperm):
        F_cut_perm = np.empty_like(F_cut)

        for idx in unique_pairs.values():
            shuffled_idx = np.random.permutation(idx)
            F_cut_perm[idx, :] = F_cut[shuffled_idx, :]

        AF_perm = A @ F_cut_perm
        colocal_count_perm_cur = F_cut_perm.T @ AF_perm
        colocal_count_perm_list.append(colocal_count_perm_cur)

    return np.stack(colocal_count_perm_list, axis=0)

def permutation_upper_tail_pvalues(observed, permuted):
    """Finite-permutation upper-tail P values, counting ties as exceedances.

    The first axis of ``permuted`` indexes B random permutations. Including
    the observed configuration gives (b + 1) / (B + 1), even when b is zero.
    """
    observed = np.asarray(observed)
    permuted = np.asarray(permuted)
    if permuted.ndim != observed.ndim + 1 or permuted.shape[1:] != observed.shape:
        raise ValueError("Permutation counts must have shape (B, *observed.shape).")
    if permuted.shape[0] == 0:
        raise ValueError("At least one permutation is required.")
    exceedances = np.count_nonzero(permuted >= observed[None, ...], axis=0)
    return (exceedances + 1) / (permuted.shape[0] + 1)


def compute_statistics(colocal_count_cur, colocal_count_perm_array):
    # Purpose: compare the observed colocalization matrix against the permutation
    # null distribution and compute effect-size-like scores and p-values.
    """
    Compute per-sample colocalization statistics.

    Parameters
    ----------
    colocal_count_cur : np.ndarray
        Observed colocalization matrix of shape (d, d).
    colocal_count_perm_array : np.ndarray
        Permutation colocalization array of shape (nperm, d, d).

    Returns
    -------
    colocal_count_ratio_cur : np.ndarray
        Z-score-like normalized matrix of shape (d, d).
    pvalue_matrix : np.ndarray
        Permutation-based p-value matrix of shape (d, d).
    """
    colocal_count_ratio_cur = (
        colocal_count_cur - colocal_count_perm_array.mean(axis=0)
    ) / (colocal_count_perm_array.std(axis=0) + 1e-8)

    pvalue_matrix = permutation_upper_tail_pvalues(colocal_count_cur, colocal_count_perm_array)
    return colocal_count_ratio_cur, pvalue_matrix

def fdr_correct_pvals(pvals, alpha=0.05):
    # Purpose: apply Benjamini-Hochberg FDR correction to a matrix of p-values.
    """
    Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : np.ndarray
        Raw p-value matrix.
    alpha : float
        FDR significance threshold.

    Returns
    -------
    reject : np.ndarray
        Boolean matrix indicating whether each entry is significant after FDR correction.
    adj_matrix : np.ndarray
        Adjusted p-value matrix.
    """
    pvals_flat = pvals.flatten()
    n = len(pvals_flat)

    order = np.argsort(pvals_flat)
    ranked_pvals = pvals_flat[order]

    adj_pvals = ranked_pvals * n / (np.arange(1, n + 1))
    adj_pvals = np.minimum.accumulate(adj_pvals[::-1])[::-1]
    adj_pvals = np.clip(adj_pvals, 0, 1)

    reject = adj_pvals < alpha

    adj_matrix = np.zeros_like(pvals_flat)
    adj_matrix[order] = adj_pvals

    return reject.reshape(pvals.shape), adj_matrix.reshape(pvals.shape)

def normalize_factor_envir_list(Factor_envir_list, eps=1e-8):
    # Purpose: normalize each MI dimension across all samples using the global
    # maximum of that MI dimension.
    """
    Normalize Factor_envir matrices across all samples by global per-dimension maxima.

    Parameters
    ----------
    Factor_envir_list : list of np.ndarray
        List of MI factor matrices, each of shape (num_edges_i, d).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    dict
        Dictionary containing:
        - Factor_envir_all
        - Factor_envir_all_max
        - Factor_envir_norm_list
        - Factor_envir_norm_all
    """
    Factor_envir_all = np.concatenate(Factor_envir_list, axis=0)
    Factor_envir_all_max = np.max(Factor_envir_all, axis=0)
    Factor_envir_norm_list = [F / (Factor_envir_all_max + eps) for F in Factor_envir_list]
    Factor_envir_norm_all = np.concatenate(Factor_envir_norm_list, axis=0)

    return {
        "Factor_envir_all": Factor_envir_all,
        "Factor_envir_all_max": Factor_envir_all_max,
        "Factor_envir_norm_list": Factor_envir_norm_list,
        "Factor_envir_norm_all": Factor_envir_norm_all,
    }

def precompute_hyper_edge_adj_list(SpiderNet_data_pyg_list):
    # Purpose: precompute one hyper-edge adjacency matrix for each sample so
    # it does not need to be rebuilt during the main analysis loop.
    """
    Precompute hyper-edge adjacency matrices for all samples.

    Parameters
    ----------
    SpiderNet_data_pyg_list : list
        List of PyG-like objects containing edge_index.

    Returns
    -------
    list
        List of scipy.sparse.csr_matrix objects, one per sample.
    """
    hyper_edge_adj_list = []

    for sample_index in range(len(SpiderNet_data_pyg_list)):
        edge_index_cur = SpiderNet_data_pyg_list[sample_index].edge_index.cpu().numpy()
        A = build_hyper_edge_adj(edge_index_cur, edge_index_cur.shape[0])
        hyper_edge_adj_list.append(A)

    return hyper_edge_adj_list

def MI_colocalization_analysis(
    Factor_envir_list,
    SpiderNet_data_pyg_list,
    adata_list,
    MI_threshold=0.7,
    nperm=100,
    fdr_alpha=0.05,
    celltype_col="cell.types",
    progress_every=5,
    min_pvalue=None,
    pvalue_adjusted_threshold = 0.001,
    zscore_countcolocal_threshold = 1.3
):
    # Purpose: run the full MI-MI colocalization analysis pipeline, including
    # normalization, per-sample observed/permutation analysis, merging across
    # samples, FDR correction, and summary reporting.
    """
    Run the full MI-MI colocalization analysis pipeline.

    Parameters
    ----------
    Factor_envir_list : list of np.ndarray
        List of MI factor matrices, each of shape (num_edges_i, d).
    SpiderNet_data_pyg_list : list
        List of PyG-like graph objects containing edge_index.
    adata_list : list
        List of AnnData objects, one per sample.
    MI_threshold : float
        Threshold for binarizing MI activity.
    nperm : int
        Number of permutations for each sample.
    fdr_alpha : float
        FDR threshold used in Benjamini-Hochberg correction.
    celltype_col : str
        Column name in adata.obs storing cell-type labels.
    progress_every : int
        Print progress every `progress_every` samples.
    min_pvalue : float or None
        Deprecated compatibility argument, ignored. P-values now use the
        finite-permutation correction (b + 1) / (B + 1), without a floor.

    Returns
    -------
    dict
        Dictionary containing normalized inputs, per-sample results, merged
        statistics, and summary values.
    """
    # -----------------------------
    # Normalize MI factors
    # -----------------------------
    norm_res = normalize_factor_envir_list(Factor_envir_list)
    Factor_envir_norm_list = norm_res["Factor_envir_norm_list"]

    nsample = len(SpiderNet_data_pyg_list)
    d = Factor_envir_list[0].shape[1]

    # -----------------------------
    # Precompute adjacency matrices
    # -----------------------------
    hyper_edge_adj_list = precompute_hyper_edge_adj_list(SpiderNet_data_pyg_list)

    # -----------------------------
    # Per-sample analysis
    # -----------------------------
    colocal_count_list = []
    colocal_count_perm_array_list = []
    colocal_count_ratio_list = []
    pvalue_matrix_list = []

    for sample_index in range(nsample):
        if progress_every is not None and sample_index % progress_every == 0:
            print(f"Processing sample {sample_index + 1}/{nsample} ({sample_index / nsample:.2%})")

        F = Factor_envir_norm_list[sample_index]
        A = hyper_edge_adj_list[sample_index]
        adata_cur = adata_list[sample_index]
        edge_index_cur = SpiderNet_data_pyg_list[sample_index].edge_index.cpu().numpy()

        # Compute observed colocalization
        F_cut, colocal_count_cur = compute_observed_colocalization(F, A, MI_threshold)
        colocal_count_list.append(colocal_count_cur)

        # Build permutation groups and compute null distribution
        unique_pairs = get_celltype_pairs(edge_index_cur, adata_cur, celltype_col=celltype_col)
        colocal_count_perm_array = compute_permutation_colocalization(
            F_cut, A, unique_pairs, nperm=nperm
        )
        colocal_count_perm_array_list.append(colocal_count_perm_array)

        # Compute per-sample statistics
        colocal_count_ratio_cur, pvalue_matrix = compute_statistics(
            colocal_count_cur, colocal_count_perm_array
        )
        colocal_count_ratio_list.append(colocal_count_ratio_cur)
        pvalue_matrix_list.append(pvalue_matrix)

    # -----------------------------
    # Merge results across samples
    # -----------------------------
    colocal_count_merge = np.stack(colocal_count_list, axis=0)                  # (nsample, d, d)
    colocal_count_merge_sum = colocal_count_merge.sum(axis=0)                   # (d, d)

    colocal_count_perm_array_merge = np.stack(colocal_count_perm_array_list, axis=0)   # (nsample, nperm, d, d)
    colocal_count_perm_array_merge_sum = colocal_count_perm_array_merge.sum(axis=0)    # (nperm, d, d)

    # -----------------------------
    # Compute merged effect-size-like score
    # -----------------------------
    colocal_count_merge_sum_zscore = (
        colocal_count_merge_sum / (colocal_count_perm_array_merge_sum.mean(axis=0) + 1e-8)
    )
    colocal_count_merge_sum_zscore = np.log2(colocal_count_merge_sum_zscore + 1e-3)

    # -----------------------------
    # Compute merged permutation p-values
    # -----------------------------
    colocal_count_merge_sum_pvalue = permutation_upper_tail_pvalues(
        colocal_count_merge_sum, colocal_count_perm_array_merge_sum
    )

    # -----------------------------
    # Convert to DataFrame
    # -----------------------------
    mi_labels = [f"MI-{i}" for i in range(1, d + 1)]

    colocal_count_merge_sum_zscore_df = pd.DataFrame(
        colocal_count_merge_sum_zscore, index=mi_labels, columns=mi_labels
    )
    colocal_count_merge_sum_pvalue_df = pd.DataFrame(
        colocal_count_merge_sum_pvalue, index=mi_labels, columns=mi_labels
    )

    # -----------------------------
    # FDR correction
    # -----------------------------
    reject_cur, pvals_fdr_cur = fdr_correct_pvals(
        colocal_count_merge_sum_pvalue_df.values,
        alpha=fdr_alpha
    )

    colocal_count_merge_sum_pvalue_fdr_df = pd.DataFrame(
        pvals_fdr_cur, index=mi_labels, columns=mi_labels
    )

    # -----------------------------
    # Summary
    # -----------------------------
    sig_count = np.sum(
        (colocal_count_merge_sum_pvalue_fdr_df.values < pvalue_adjusted_threshold) *
        (colocal_count_merge_sum_zscore_df.values > np.log2(zscore_countcolocal_threshold))
    )
    print(f"Significant MI-MI co-localizations: {sig_count}")

    return {
        # normalized data
        "Factor_envir_all": norm_res["Factor_envir_all"],
        "Factor_envir_all_max": norm_res["Factor_envir_all_max"],
        "Factor_envir_norm_list": norm_res["Factor_envir_norm_list"],
        "Factor_envir_norm_all": norm_res["Factor_envir_norm_all"],

        # precomputed adjacency
        "hyper_edge_adj_list": hyper_edge_adj_list,

        # per-sample results
        "colocal_count_list": colocal_count_list,
        "colocal_count_perm_array_list": colocal_count_perm_array_list,
        "colocal_count_ratio_list": colocal_count_ratio_list,
        "pvalue_matrix_list": pvalue_matrix_list,

        # merged arrays
        "colocal_count_merge": colocal_count_merge,
        "colocal_count_merge_sum": colocal_count_merge_sum,
        "colocal_count_perm_array_merge": colocal_count_perm_array_merge,
        "colocal_count_perm_array_merge_sum": colocal_count_perm_array_merge_sum,

        # merged dataframes
        "colocal_count_merge_sum_zscore": colocal_count_merge_sum_zscore_df,
        "colocal_count_merge_sum_pvalue_raw": pd.DataFrame(
            colocal_count_merge_sum_pvalue, index=mi_labels, columns=mi_labels
        ),
        "colocal_count_merge_sum_pvalue_fdr": colocal_count_merge_sum_pvalue_fdr_df,

        # FDR results
        "reject_fdr": reject_cur,

        # summary
        "sig_count": sig_count,
        "mi_labels": mi_labels,
        "MI_threshold": MI_threshold,
        "nperm": nperm,
        "fdr_alpha": fdr_alpha,
        "pvalue_method": "plus1",
    }

import numpy as np
import pandas as pd

def compute_celltype_triples_for_one_sample(
    F,
    A,
    edge_index,
    adata_cur,
    MI_first_idx,
    MI_second_idx,
    MI_threshold,
    celltype_col="cell.types",
    sample_col="samples",
):
    # Purpose: for one sample and one ordered MI pair, identify all cell triples
    # (cell1 -> cell2 -> cell3) connected through MI_first and MI_second, and
    # summarize the counts and proportions of cell-type triples.
    """
    Compute cell-type triple summaries for one sample and one ordered MI pair.

    Parameters
    ----------
    F : np.ndarray
        Normalized MI factor matrix of shape (E, d).
    A : scipy.sparse matrix
        Hyper-edge adjacency matrix of shape (E, E).
    edge_index : np.ndarray
        Edge index array of shape (E, 2), where each row is (src, tgt).
    adata_cur : AnnData
        AnnData object containing cell-level annotations.
    MI_first_idx : int
        Zero-based index of the first MI.
    MI_second_idx : int
        Zero-based index of the second MI.
    MI_threshold : float
        Threshold used to binarize MI activity.
    celltype_col : str
        Column name in adata_cur.obs containing cell-type labels.
    sample_col : str
        Column name in adata_cur.obs containing sample identifiers.

    Returns
    -------
    triple_counts : pd.DataFrame
        Summary table for the current sample. Empty DataFrame if no valid triples are found.
    celltriple_index : np.ndarray
        Integer array of shape (n_triples, 3), containing cell indices for
        (cell1, cell2, cell3). Empty array if no valid triples are found.
    """
    celltypes_all = adata_cur.obs[celltype_col].values
    edge_celltypes = celltypes_all[edge_index]  # shape (E, 2)

    F_cut = (F > MI_threshold).astype(int)

    # Select edges with active MI_first and MI_second
    rel_first = np.where(F_cut[:, MI_first_idx])[0]
    rel_second = np.where(F_cut[:, MI_second_idx])[0]

    if len(rel_first) == 0 or len(rel_second) == 0:
        empty_df = pd.DataFrame(
            columns=[
                "Celltype_triple", "Count", "ctp_MI_first", "ctp_MI_second",
                "Sample", "prop", "ratio"
            ]
        )
        return empty_df, np.empty((0, 3), dtype=np.int64)

    # Find connected edge pairs: edge_first target == edge_second source
    A_sub = A[rel_first, :][:, rel_second].toarray()
    a_i, a_j = np.where(A_sub == 1)

    if len(a_i) == 0:
        empty_df = pd.DataFrame(
            columns=[
                "Celltype_triple", "Count", "ctp_MI_first", "ctp_MI_second",
                "Sample", "prop", "ratio"
            ]
        )
        return empty_df, np.empty((0, 3), dtype=np.int64)

    paired_idx = np.vstack([rel_first[a_i], rel_second[a_j]]).T

    # Build cell triples: (src_first, tgt_first=src_second, tgt_second)
    pair_first = edge_index[paired_idx[:, 0], :]
    pair_second = edge_index[paired_idx[:, 1], :]
    celltriple_index = np.hstack([pair_first, pair_second[:, 1][:, None]])

    triple_ct = np.asarray(celltypes_all[celltriple_index], dtype=object)
    triple_label = np.array(
        [f"{a} -> {b} -> {c}" for a, b, c in triple_ct],
        dtype=object
    )

    triple_counts = pd.Series(triple_label).value_counts().reset_index()
    triple_counts.columns = ["Celltype_triple", "Count"]

    if triple_counts.empty:
        empty_df = pd.DataFrame(
            columns=[
                "Celltype_triple", "Count", "ctp_MI_first", "ctp_MI_second",
                "Sample", "prop", "ratio"
            ]
        )
        return empty_df, celltriple_index

    # Parse cell-type pair metadata
    parts = triple_counts["Celltype_triple"].str.split("->", expand=True)
    triple_counts["ctp_MI_first"] = parts[0].str.strip() + " -> " + parts[1].str.strip()
    triple_counts["ctp_MI_second"] = parts[1].str.strip() + " -> " + parts[2].str.strip()
    triple_counts["Sample"] = adata_cur.obs[sample_col].unique()[0]
    triple_counts["prop"] = triple_counts["Count"] / triple_counts["Count"].sum()

    # Compute adjusted ratio relative to the abundance of the two component edge types
    def calc_ratio(row):
        ctp1 = row["ctp_MI_first"]
        ctp2 = row["ctp_MI_second"]
        cnt = row["Count"]

        ct1a, ct1b = ctp1.split(" -> ")
        ct2a, ct2b = ctp2.split(" -> ")

        r1 = np.sum((edge_celltypes[:, 0] == ct1a) & (edge_celltypes[:, 1] == ct1b))
        r2 = np.sum((edge_celltypes[:, 0] == ct2a) & (edge_celltypes[:, 1] == ct2b))

        denom = (float(r1) * float(r2)) / float(F_cut.shape[0])
        return float(cnt) / denom if denom > 0 else np.nan

    triple_counts["ratio"] = triple_counts.apply(calc_ratio, axis=1)

    return triple_counts, celltriple_index

def summarize_one_mi_pair_across_samples(
    MI_first,
    MI_second,
    Factor_envir_norm_list,
    hyper_edge_adj_list,
    SpiderNet_data_pyg_list,
    adata_list,
    MI_threshold,
    celltype_col="cell.types",
    sample_col="samples",
):
    # Purpose: for one ordered MI pair, aggregate cell-type triple summaries
    # across all samples and compute the average triple proportion across samples.
    """
    Summarize one ordered MI pair across all samples.

    Parameters
    ----------
    MI_first : int
        One-based index of the first MI.
    MI_second : int
        One-based index of the second MI.
    Factor_envir_norm_list : list of np.ndarray
        List of normalized MI factor matrices, one per sample.
    hyper_edge_adj_list : list
        List of hyper-edge adjacency matrices, one per sample.
    SpiderNet_data_pyg_list : list
        List of graph objects containing edge_index.
    adata_list : list
        List of AnnData objects, one per sample.
    MI_threshold : float
        Threshold used to binarize MI activity.
    celltype_col : str
        Column name in adata.obs containing cell-type labels.
    sample_col : str
        Column name in adata.obs containing sample identifiers.

    Returns
    -------
    dict
        Dictionary containing per-sample triple indices, merged summary tables,
        and average triple proportions across samples.
    """
    MI_first_idx = MI_first - 1
    MI_second_idx = MI_second - 1

    all_summary_tables = []
    celltriple_index_all = []

    nsample = len(Factor_envir_norm_list)

    for sample_index in range(nsample):
        F = Factor_envir_norm_list[sample_index]
        A = hyper_edge_adj_list[sample_index]
        edge_index = SpiderNet_data_pyg_list[sample_index].edge_index.cpu().numpy()
        adata_cur = adata_list[sample_index]

        triple_counts, celltriple_index = compute_celltype_triples_for_one_sample(
            F=F,
            A=A,
            edge_index=edge_index,
            adata_cur=adata_cur,
            MI_first_idx=MI_first_idx,
            MI_second_idx=MI_second_idx,
            MI_threshold=MI_threshold,
            celltype_col=celltype_col,
            sample_col=sample_col,
        )

        celltriple_index_all.append(celltriple_index)

        if not triple_counts.empty:
            all_summary_tables.append(triple_counts)

    if len(all_summary_tables) == 0:
        combined = pd.DataFrame(
            columns=[
                "Celltype_triple", "Count", "ctp_MI_first", "ctp_MI_second",
                "Sample", "prop", "ratio"
            ]
        )
        prop_avg = pd.DataFrame(
            columns=["Celltype_triple", "prop_avg_allsample", "MI_first", "MI_second"]
        )
    else:
        combined = pd.concat(all_summary_tables, axis=0, ignore_index=True)
        total_count = combined["Count"].sum()

        prop_avg = (
            combined.groupby("Celltype_triple")["Count"]
            .sum()
            .div(total_count)
            .reset_index(name="prop_avg_allsample")
        )
        prop_avg["MI_first"] = MI_first
        prop_avg["MI_second"] = MI_second

    return {
        "MI_first": MI_first,
        "MI_second": MI_second,
        "celltriple_index_all": celltriple_index_all,
        "summary_table_all_samples": combined,
        "prop_avg_allsample": prop_avg,
    }

def summarize_mi_cascade_celltype_triples(
    MI_colocal_summary_significant,
    Factor_envir_norm_list,
    hyper_edge_adj_list,
    SpiderNet_data_pyg_list,
    adata_list,
    MI_threshold,
    celltype_col="cell.types",
    sample_col="samples",
    select_triple_prop_threshold=0.1,
    progress_every=10,
):
    # Purpose: iterate over all significant MI pairs, summarize their dominant
    # cell-type triples across samples, and generate a pivot table showing which
    # triples are enriched in which MI cascades.
    """
    Summarize cell-type triples for all significant MI cascades.

    Parameters
    ----------
    MI_colocal_summary_significant : pd.DataFrame
        DataFrame containing at least two columns: 'MI_first' and 'MI_second'.
        MI indices should be one-based.
    Factor_envir_norm_list : list of np.ndarray
        List of normalized MI factor matrices, one per sample.
    hyper_edge_adj_list : list
        List of hyper-edge adjacency matrices, one per sample.
    SpiderNet_data_pyg_list : list
        List of graph objects containing edge_index.
    adata_list : list
        List of AnnData objects, one per sample.
    MI_threshold : float
        Threshold used to binarize MI activity.
    celltype_col : str
        Column name in adata.obs containing cell-type labels.
    sample_col : str
        Column name in adata.obs containing sample identifiers.
    select_triple_prop_threshold : float
        Keep only cell-type triples whose maximum average proportion across MI
        cascades exceeds this threshold.
    progress_every : int
        Print progress every `progress_every` MI pairs.

    Returns
    -------
    dict
        Dictionary containing merged summary tables, per-MI-pair results,
        cell triple indices, and selected pivot table.
    """
    summary_table_choose_df_list = []
    celltriple_index_sample_list = []
    prop_avg_allsample_filter_list = []
    per_pair_results = []

    n_pairs = MI_colocal_summary_significant.shape[0]

    for idx in range(n_pairs):
        if progress_every is not None and idx % progress_every == 0:
            print(f"Processing MI pair {idx + 1}/{n_pairs}")

        MI_first = int(MI_colocal_summary_significant.iloc[idx]["MI_first"])
        MI_second = int(MI_colocal_summary_significant.iloc[idx]["MI_second"])

        pair_res = summarize_one_mi_pair_across_samples(
            MI_first=MI_first,
            MI_second=MI_second,
            Factor_envir_norm_list=Factor_envir_norm_list,
            hyper_edge_adj_list=hyper_edge_adj_list,
            SpiderNet_data_pyg_list=SpiderNet_data_pyg_list,
            adata_list=adata_list,
            MI_threshold=MI_threshold,
            celltype_col=celltype_col,
            sample_col=sample_col,
        )

        per_pair_results.append(pair_res)
        celltriple_index_sample_list.append(pair_res["celltriple_index_all"])
        summary_table_choose_df_list.append(pair_res["summary_table_all_samples"])

        if not pair_res["prop_avg_allsample"].empty:
            prop_avg_allsample_filter_list.append(pair_res["prop_avg_allsample"])

    # Merge average proportions across all MI pairs
    if len(prop_avg_allsample_filter_list) == 0:
        prop_avg_allsample_filter_merge = pd.DataFrame(
            columns=["Celltype_triple", "prop_avg_allsample", "MI_first", "MI_second", "MI cascade"]
        )
        pivot = pd.DataFrame()
        prop_avg_allsample_filter_merge_pivot_select = pd.DataFrame()
    else:
        prop_avg_allsample_filter_merge = pd.concat(
            prop_avg_allsample_filter_list,
            axis=0,
            ignore_index=True
        )

        prop_avg_allsample_filter_merge["MI cascade"] = (
            "MI-"
            + prop_avg_allsample_filter_merge["MI_first"].astype(str)
            + " -> MI-"
            + prop_avg_allsample_filter_merge["MI_second"].astype(str)
        )

        pivot = prop_avg_allsample_filter_merge.pivot(
            index="Celltype_triple",
            columns="MI cascade",
            values="prop_avg_allsample"
        ).fillna(0)

        prop_avg_allsample_filter_merge_pivot_select = pivot.loc[
            pivot.max(axis=1) > select_triple_prop_threshold, :
        ]

    print("Summary completed.")
    print(f"Selected pivot shape: {prop_avg_allsample_filter_merge_pivot_select.shape}")

    return {
        "summary_table_choose_df_list": summary_table_choose_df_list,
        "celltriple_index_sample_list": celltriple_index_sample_list,
        "prop_avg_allsample_filter_list": prop_avg_allsample_filter_list,
        "prop_avg_allsample_filter_merge": prop_avg_allsample_filter_merge,
        "pivot": pivot,
        "prop_avg_allsample_filter_merge_pivot_select": prop_avg_allsample_filter_merge_pivot_select,
        "per_pair_results": per_pair_results,
    }

import os
import re
import numpy as np
import matplotlib.pyplot as plt

def safe_filename(name: str) -> str:
    """Replace invalid filename characters with underscores."""
    return re.sub(r'[\\/:*?"<>|]', "_", name.replace(" -> ", "-"))

##DEG for MI cascade

"""Core analysis utilities for MI cascade analysis in SpiderNet.

This module is designed to be reusable across datasets. The workflow is:

1. Define a cascade structure: cell1 -[MI-i]-> cell2 -[MI-j]-> cell3
2. Define pathway-induced ligand-receptor (LR) programs and gene programs
3. Run group-wise analysis and collect per-cell / per-edge scores for visualization

The code assumes the same gene order (`adata.var_names`) across samples.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

GroupName = Literal["BothPos", "OnlyMIfirst", "OnlyMIsecond", "BothNeg"]
EdgeName = Literal["upstream", "downstream"]

DEFAULT_GROUPS: Tuple[GroupName, ...] = (
    "BothPos",
    "OnlyMIfirst",
    "OnlyMIsecond",
    "BothNeg",
)

@dataclass
class CascadeDefinition:
    """Definition of an MI cascade structure.

    Attributes
    ----------
    cell_types
        Ordered cell type pattern in the cascade: (cell1_type, cell2_type, cell3_type).
    mi_first_index
        Index of the upstream MI (cell1 -> cell2).
    mi_second_index
        Index of the downstream MI (cell2 -> cell3).
    mi_threshold
        Threshold used to binarize MI activation.
    cell_type_obs_key
        Column name in `adata.obs` storing cell type labels.
    """

    cell_types: Tuple[str, str, str]
    mi_first_index: int
    mi_second_index: int
    mi_threshold: float = 0.5
    cell_type_obs_key: str = "cell.types"

@dataclass
class PathwaySpec:
    """Specification of one pathway/LR program to score.

    Attributes
    ----------
    name
        Display name for the pathway program.
    pathway_names
        Pathway names to be matched in LR metadata.
    edge
        Which edge in the cascade this pathway belongs to:
        - "upstream": cell1 -> cell2
        - "downstream": cell2 -> cell3
    source_position
        Position in the triplet used for ligand expression. Usually:
        - upstream  : 0
        - downstream: 1
    target_position
        Position in the triplet used for receptor expression. Usually:
        - upstream  : 1
        - downstream: 2
    """

    name: str
    pathway_names: Sequence[str]
    edge: EdgeName
    source_position: int
    target_position: int

@dataclass
class GeneProgramSpec:
    """Specification of one gene program scored on one triplet position.

    Attributes
    ----------
    name
        Program name, e.g. "myCAF", "EMT", or a GO term.
    genes
        Gene list for the program.
    cell_position
        Which cell in the triplet should be scored: 0 / 1 / 2.
    category
        Higher-level category for plotting/organization, e.g. "CAF", "Functional", "GO".
    """

    name: str
    genes: Sequence[str]
    cell_position: int
    category: str = "GeneProgram"

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

@dataclass
class AnalysisConfig:
    """Configuration controlling normalization and cell selection behavior."""

    expression_layer: Optional[str] = None
    score_on_normalized_expression: bool = True
    deduplicate_cells_within_group: bool = True

    exclude_overlap_mode: Optional[
        Literal["shared_triplet_nodes", "same_position_cells_vs_reference"]
    ] = None

    # Used only when exclude_overlap_mode == "same_position_cells_vs_reference"
    overlap_reference_group: str = "BothPos"
    overlap_target_groups: Sequence[str] = ("OnlyMIfirst", "OnlyMIsecond", "BothNeg")
    overlap_cell_position: int = 2

@dataclass
class PathwayIndexInfo:
    """Resolved LR indices and gene indices for one pathway spec."""

    lr_pair_indices: np.ndarray
    ligand_gene_indices: np.ndarray
    receptor_gene_indices: np.ndarray

@dataclass
class GeneProgramIndexInfo:
    """Resolved gene indices for one gene program spec."""

    gene_indices: np.ndarray

@dataclass
class SamplePreparedData:
    """Prepared per-sample matrices used during analysis."""

    edge_index: np.ndarray
    factor_binary: np.ndarray
    hyper_edge_adj: sparse.spmatrix | np.ndarray
    expression: np.ndarray
    lr_matrix: np.ndarray
    cell_types: np.ndarray
    obs_names: np.ndarray

@dataclass
class AnalysisResults:
    """Container of all outputs from cascade analysis."""

    cascade_definition: CascadeDefinition
    pathway_scores: Dict[str, Dict[str, Dict[GroupName, List[float]]]]
    gene_program_scores: Dict[str, Dict[str, Dict[GroupName, List[float]]]]
    selected_cells: Dict[GroupName, Dict[int, List[dict]]] = field(default_factory=dict)
    triplet_counts: Dict[GroupName, int] = field(default_factory=dict)

    def pathway_metric_dict(self, metric: str) -> Dict[str, Dict[GroupName, List[float]]]:
        """Return {pathway_name: {group: values}} for one metric."""
        out: Dict[str, Dict[GroupName, List[float]]] = {}
        for pathway_name, metric_dict in self.pathway_scores.items():
            if metric not in metric_dict:
                raise KeyError(f"Metric '{metric}' not found for pathway '{pathway_name}'.")
            out[pathway_name] = metric_dict[metric]
        return out

    def gene_program_category_dict(self, category: str) -> Dict[str, Dict[GroupName, List[float]]]:
        """Return {program_name: {group: values}} for one category."""
        if category not in self.gene_program_scores:
            raise KeyError(f"Category '{category}' not found.")
        return self.gene_program_scores[category]

def expand_genes(genes: Sequence[str]) -> np.ndarray:
    """Split combined symbols like 'COL1A1_COL1A2' and return unique genes."""
    expanded: List[str] = []
    for gene in genes:
        if pd.isna(gene):
            continue
        expanded.extend(str(gene).split("_"))
    return np.unique(expanded)

def standardize_edge_index(edge_index: np.ndarray) -> np.ndarray:
    """Return edge_index in shape (n_edges, 2)."""
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape={edge_index.shape}")
    if edge_index.shape[1] == 2:
        return edge_index.astype(int, copy=False)
    if edge_index.shape[0] == 2:
        return edge_index.T.astype(int, copy=False)
    raise ValueError(
        "edge_index must have shape (n_edges, 2) or (2, n_edges); "
        f"got {edge_index.shape}"
    )

def to_dense_array(x) -> np.ndarray:
    """Convert sparse/dense matrix-like input to a NumPy array."""
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)

def get_expression_matrix(adata, layer: Optional[str] = None) -> np.ndarray:
    """Get a dense expression matrix from AnnData."""
    if layer is not None and hasattr(adata, "layers") and layer in adata.layers:
        return to_dense_array(adata.layers[layer])
    return to_dense_array(adata.X)

def compute_global_expression_max(
    adata_list: Sequence,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Compute per-gene global max across samples."""
    maxima = []
    for adata in adata_list:
        x = get_expression_matrix(adata, layer=layer)
        maxima.append(np.max(x, axis=0))
    return np.max(np.vstack(maxima), axis=0)

def compute_global_lr_max(spidernet_data_list: Sequence[dict], key: str = "cellpair_LRpair_neigh") -> np.ndarray:
    """Compute per-LR-pair global max across samples."""
    maxima = []
    for data in spidernet_data_list:
        arr = data[key]
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        else:
            arr = np.asarray(arr)
        maxima.append(np.max(arr, axis=0))
    return np.max(np.vstack(maxima), axis=0)

def resolve_pathway_indices(
    lr_meta: pd.DataFrame,
    var_names: Sequence[str],
    pathway_specs: Sequence[PathwaySpec],
    pathway_col: str = "pathway_name",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
) -> Dict[str, PathwayIndexInfo]:
    """Resolve LR pair indices and ligand/receptor gene indices for pathway specs."""
    var_names = np.asarray(var_names)
    out: Dict[str, PathwayIndexInfo] = {}
    for spec in pathway_specs:
        idx = np.where(lr_meta[pathway_col].isin(spec.pathway_names))[0]
        ligands = expand_genes(np.unique(lr_meta.loc[idx, ligand_col]))
        receptors = expand_genes(np.unique(lr_meta.loc[idx, receptor_col]))
        ligand_gene_indices = np.where(np.isin(var_names, ligands))[0]
        receptor_gene_indices = np.where(np.isin(var_names, receptors))[0]
        out[spec.name] = PathwayIndexInfo(
            lr_pair_indices=np.asarray(idx, dtype=int),
            ligand_gene_indices=np.asarray(ligand_gene_indices, dtype=int),
            receptor_gene_indices=np.asarray(receptor_gene_indices, dtype=int),
        )
    return out

def resolve_gene_program_indices(
    var_names: Sequence[str],
    gene_program_specs: Sequence[GeneProgramSpec],
) -> Dict[str, GeneProgramIndexInfo]:
    """Resolve gene indices for gene program specs."""
    var_names = np.asarray(var_names)
    out: Dict[str, GeneProgramIndexInfo] = {}
    for spec in gene_program_specs:
        idx = np.where(np.isin(var_names, np.asarray(spec.genes)))[0]
        out[spec.name] = GeneProgramIndexInfo(gene_indices=np.asarray(idx, dtype=int))
    return out

def build_gene_program_specs_from_dict(
    program_dict: Dict[str, Sequence[str]],
    cell_position: int,
    category: str,
) -> List[GeneProgramSpec]:
    """Convenience helper to create gene program specs from a dict."""
    return [
        GeneProgramSpec(name=name, genes=list(genes), cell_position=cell_position, category=category)
        for name, genes in program_dict.items()
    ]

def build_gene_program_specs_from_dataframe(
    df: pd.DataFrame,
    term_col: str,
    gene_col: str,
    cell_position: int,
    category: str,
) -> List[GeneProgramSpec]:
    """Create gene program specs from a long-form dataframe."""
    specs = []
    for term, subdf in df.groupby(term_col):
        genes = subdf[gene_col].dropna().astype(str).unique().tolist()
        specs.append(
            GeneProgramSpec(
                name=str(term),
                genes=genes,
                cell_position=cell_position,
                category=category,
            )
        )
    return specs


def filter_triples_no_shared_node(triples: np.ndarray, reference_triples: np.ndarray) -> np.ndarray:
    """Keep triples whose nodes share no node with any node in reference_triples."""
    triples = np.asarray(triples)
    reference_triples = np.asarray(reference_triples)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"`triples` must be (N, 3), got {triples.shape}")
    if reference_triples.ndim != 2 or reference_triples.shape[1] != 3:
        raise ValueError(f"`reference_triples` must be (M, 3), got {reference_triples.shape}")
    reference_nodes = np.unique(reference_triples.reshape(-1))
    keep = ~np.isin(triples, reference_nodes).any(axis=1)
    return triples[keep]

def build_sample_prepared_data(
    adata,
    factor_envir_norm: np.ndarray,
    hyper_edge_adj,
    spidernet_data: dict,
    cascade_definition: CascadeDefinition,
    expression_max: np.ndarray,
    lr_max: np.ndarray,
    config: AnalysisConfig,
    lr_key: str = "cellpair_LRpair_neigh",
) -> SamplePreparedData:
    """Prepare normalized per-sample data for analysis."""
    edge_index = standardize_edge_index(
        spidernet_data["edge_index"].cpu().numpy()
        if hasattr(spidernet_data["edge_index"], "cpu")
        else np.asarray(spidernet_data["edge_index"])
    )
    factor_binary = (np.asarray(factor_envir_norm) > cascade_definition.mi_threshold).astype(int)

    expression = get_expression_matrix(adata, layer=config.expression_layer)
    if config.score_on_normalized_expression:
        expression = expression / (expression_max + 1e-8)

    lr_matrix = spidernet_data[lr_key]
    if hasattr(lr_matrix, "cpu"):
        lr_matrix = lr_matrix.cpu().numpy()
    else:
        lr_matrix = np.asarray(lr_matrix)
    lr_matrix = lr_matrix / (lr_max + 1e-8)

    cell_types = adata.obs[cascade_definition.cell_type_obs_key].astype(str).values
    obs_names = np.asarray(adata.obs_names)

    return SamplePreparedData(
        edge_index=edge_index,
        factor_binary=factor_binary,
        hyper_edge_adj=hyper_edge_adj,
        expression=np.asarray(expression),
        lr_matrix=np.asarray(lr_matrix),
        cell_types=cell_types,
        obs_names=obs_names,
    )

def build_group_conditions(
    factor_binary: np.ndarray,
    cascade_definition: CascadeDefinition,
) -> Dict[GroupName, Tuple[np.ndarray, np.ndarray]]:
    """Build edge-level boolean conditions for each cascade group."""
    first_on = factor_binary[:, cascade_definition.mi_first_index] == 1
    second_on = factor_binary[:, cascade_definition.mi_second_index] == 1
    return {
        "BothPos": (first_on, second_on),
        "OnlyMIfirst": (first_on, ~second_on),
        "OnlyMIsecond": (~first_on, second_on),
        "BothNeg": (~first_on, ~second_on),
    }

def extract_triplets(
    hyper_edge_adj,
    edge_index: np.ndarray,
    cell_types: np.ndarray,
    cond_first: np.ndarray,
    cond_second: np.ndarray,
    expected_cell_types: Tuple[str, str, str],
) -> np.ndarray:
    """Extract valid triplets matching the expected cell type pattern."""
    r1 = np.where(cond_first)[0]
    r2 = np.where(cond_second)[0]
    if len(r1) == 0 or len(r2) == 0:
        return np.empty((0, 3), dtype=int)

    adjacency_sub = hyper_edge_adj[r1, :][:, r2]
    rows, cols = adjacency_sub.nonzero()
    if len(rows) == 0:
        return np.empty((0, 3), dtype=int)

    first_edge_ids = r1[rows]
    second_edge_ids = r2[cols]
    triplets = np.column_stack(
        [
            edge_index[first_edge_ids, 0],
            edge_index[first_edge_ids, 1],
            edge_index[second_edge_ids, 1],
        ]
    )

    triplet_cell_types = cell_types[triplets]
    mask = (
        (triplet_cell_types[:, 0] == expected_cell_types[0]) &
        (triplet_cell_types[:, 1] == expected_cell_types[1]) &
        (triplet_cell_types[:, 2] == expected_cell_types[2])
    )
    return triplets[mask]

def build_edge_lookup(edge_index: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Build mapping from (src, dst) -> edge_id."""
    return {(int(src), int(dst)): i for i, (src, dst) in enumerate(edge_index)}

def triplets_to_edge_indices(
    triplets: np.ndarray,
    edge_lookup: Dict[Tuple[int, int], int],
    edge: EdgeName,
) -> np.ndarray:
    """Map triplets to edge indices for upstream or downstream edges."""
    if triplets.size == 0:
        return np.empty(0, dtype=int)

    if edge == "upstream":
        pairs = triplets[:, [0, 1]]
    elif edge == "downstream":
        pairs = triplets[:, [1, 2]]
    else:
        raise ValueError(f"Unsupported edge='{edge}'")

    edge_ids = [edge_lookup[(int(src), int(dst))] for src, dst in pairs if (int(src), int(dst)) in edge_lookup]
    return np.unique(np.asarray(edge_ids, dtype=int))

def select_triplet_nodes(
    triplets: np.ndarray,
    cell_position: int,
    unique_only: bool = True,
) -> np.ndarray:
    """Select node indices at one triplet position."""
    if triplets.size == 0:
        return np.empty(0, dtype=int)
    nodes = np.asarray(triplets[:, cell_position], dtype=int)
    return np.unique(nodes) if unique_only else nodes

def compute_mean_score_per_unit(
    matrix: np.ndarray,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
) -> np.ndarray:
    """Compute mean score across selected features for each selected row."""
    row_indices = np.asarray(row_indices, dtype=int)
    feature_indices = np.asarray(feature_indices, dtype=int)

    if len(row_indices) == 0 or len(feature_indices) == 0:
        return np.empty(0, dtype=float)

    values = matrix[row_indices][:, feature_indices]
    values = np.asarray(values)
    return values.mean(axis=1)

def initialize_results(
    cascade_definition: CascadeDefinition,
    pathway_specs: Sequence[PathwaySpec],
    gene_program_specs: Sequence[GeneProgramSpec],
) -> AnalysisResults:
    """Initialize empty analysis result containers."""
    pathway_scores: Dict[str, Dict[str, Dict[GroupName, List[float]]]] = {}
    for spec in pathway_specs:
        pathway_scores[spec.name] = {
            "lr_strength": {group: [] for group in DEFAULT_GROUPS},
            "ligand_expression": {group: [] for group in DEFAULT_GROUPS},
            "receptor_expression": {group: [] for group in DEFAULT_GROUPS},
        }

    gene_program_scores: Dict[str, Dict[str, Dict[GroupName, List[float]]]] = {}
    for spec in gene_program_specs:
        gene_program_scores.setdefault(spec.category, {})
        gene_program_scores[spec.category][spec.name] = {group: [] for group in DEFAULT_GROUPS}

    selected_cells = {
        group: {0: [], 1: [], 2: []}
        for group in DEFAULT_GROUPS
    }

    triplet_counts = {group: 0 for group in DEFAULT_GROUPS}

    return AnalysisResults(
        cascade_definition=cascade_definition,
        pathway_scores=pathway_scores,
        gene_program_scores=gene_program_scores,
        selected_cells=selected_cells,
        triplet_counts=triplet_counts,
    )

import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence

class MICascadeAnalyzer:
    """Reusable analyzer for SpiderNet MI cascade downstream scoring."""

    def __init__(
        self,
        adata_list: Sequence,
        factor_envir_norm_list: Sequence[np.ndarray],
        hyper_edge_adj_list: Sequence,
        spidernet_data_list: Sequence[dict],
        lr_meta: pd.DataFrame,
        cascade_definition,
        pathway_specs: Sequence,
        gene_program_specs: Sequence,
        config: Optional[object] = None,
    ) -> None:
        if len(adata_list) == 0:
            raise ValueError("adata_list is empty.")
        if not (
            len(adata_list)
            == len(factor_envir_norm_list)
            == len(hyper_edge_adj_list)
            == len(spidernet_data_list)
        ):
            raise ValueError("All sample-wise input lists must have the same length.")

        self.adata_list = list(adata_list)
        self.factor_envir_norm_list = list(factor_envir_norm_list)
        self.hyper_edge_adj_list = list(hyper_edge_adj_list)
        self.spidernet_data_list = list(spidernet_data_list)
        self.lr_meta = lr_meta.copy()
        self.cascade_definition = cascade_definition
        self.pathway_specs = list(pathway_specs)
        self.gene_program_specs = list(gene_program_specs)
        self.config = config or AnalysisConfig()

        self.var_names = np.asarray(self.adata_list[0].var_names)
        self.pathway_index_info = resolve_pathway_indices(
            lr_meta=self.lr_meta,
            var_names=self.var_names,
            pathway_specs=self.pathway_specs,
        )
        self.gene_program_index_info = resolve_gene_program_indices(
            var_names=self.var_names,
            gene_program_specs=self.gene_program_specs,
        )

        self.expression_max = compute_global_expression_max(
            self.adata_list,
            layer=self.config.expression_layer,
        )
        self.lr_max = compute_global_lr_max(self.spidernet_data_list)

    def _collect_selected_cell_records(
        self,
        results,
        sample_index: int,
        triplets: np.ndarray,
        obs_names: np.ndarray,
        cell_types: np.ndarray,
        group: str,
    ) -> None:
        """Store selected cells for each position and group."""
        for cell_position in (0, 1, 2):
            node_ids = select_triplet_nodes(
                triplets,
                cell_position=cell_position,
                unique_only=self.config.deduplicate_cells_within_group,
            )
            if len(node_ids) == 0:
                continue
            for nid in node_ids:
                results.selected_cells[group][cell_position].append(
                    {
                        "sample_index": sample_index,
                        "cell_index": int(nid),
                        "cell_id": str(obs_names[nid]),
                        "cell_type": str(cell_types[nid]),
                        "cell_position": cell_position,
                    }
                )

    def _apply_overlap_filter(
        self,
        triplets: np.ndarray,
        group: str,
        reference_triplets: Dict[str, np.ndarray],
        reference_nodes_by_pos: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Apply optional overlap filtering to triplets.

        Supported modes
        ---------------
        - None:
            No overlap filtering.
        - "shared_triplet_nodes":
            Original behavior: remove triplets sharing any node with reference triplets.
        - "same_position_cells_vs_reference":
            New behavior: remove triplets whose node at a specified cell position
            overlaps with the selected cells from a reference group.
            This is closer to the old notebook logic for comparing cell3 between
            BothPos and OnlyMIsecond.
        """
        mode = getattr(self.config, "exclude_overlap_mode", None)

        if mode is None:
            return triplets

        # ------------------------------------------------------------------
        # Original behavior: remove triplets that share any node with reference
        # ------------------------------------------------------------------
        if mode == "shared_triplet_nodes":
            if group == "BothPos":
                reference_triplets["BothPos"] = triplets

            elif group == "OnlyMIfirst" and "BothPos" in reference_triplets:
                triplets = filter_triples_no_shared_node(
                    triplets, reference_triplets["BothPos"]
                )
                reference_triplets["OnlyMIfirst"] = triplets

            elif group == "OnlyMIsecond" and "BothPos" in reference_triplets:
                triplets = filter_triples_no_shared_node(
                    triplets, reference_triplets["BothPos"]
                )
                reference_triplets["OnlyMIsecond"] = triplets

            elif group == "BothNeg":
                for ref_name in ("BothPos", "OnlyMIfirst", "OnlyMIsecond"):
                    if ref_name in reference_triplets:
                        triplets = filter_triples_no_shared_node(
                            triplets, reference_triplets[ref_name]
                        )

            return triplets

        # ------------------------------------------------------------------
        # New behavior: remove overlap only at one chosen cell position
        # ------------------------------------------------------------------
        if mode == "same_position_cells_vs_reference":
            ref_group = getattr(self.config, "overlap_reference_group", "BothPos")
            target_groups = tuple(
                getattr(
                    self.config,
                    "overlap_target_groups",
                    ("OnlyMIfirst", "OnlyMIsecond", "BothNeg"),
                )
            )
            overlap_cell_position = int(
                getattr(self.config, "overlap_cell_position", 2)
            )

            # Store reference nodes at the chosen position
            if group == ref_group:
                if len(triplets) > 0:
                    reference_nodes_by_pos[overlap_cell_position] = np.unique(
                        triplets[:, overlap_cell_position]
                    ).astype(int)
                else:
                    reference_nodes_by_pos[overlap_cell_position] = np.array(
                        [], dtype=int
                    )
                return triplets

            # Filter target groups using overlap at the chosen position only
            if (
                group in target_groups
                and overlap_cell_position in reference_nodes_by_pos
                and len(triplets) > 0
            ):
                ref_nodes = reference_nodes_by_pos[overlap_cell_position]
                if len(ref_nodes) > 0:
                    keep_mask = ~np.isin(triplets[:, overlap_cell_position], ref_nodes)
                    triplets = triplets[keep_mask]

            return triplets

        raise ValueError(
            f"Unsupported exclude_overlap_mode: {mode}. "
            "Supported modes are: None, 'shared_triplet_nodes', "
            "'same_position_cells_vs_reference'."
        )

    def run(self):
        """Run group-wise MI cascade analysis."""
        results = initialize_results(
            cascade_definition=self.cascade_definition,
            pathway_specs=self.pathway_specs,
            gene_program_specs=self.gene_program_specs,
        )

        for sample_index, (adata, factor_arr, hyper_adj, sp_data) in enumerate(
            zip(
                self.adata_list,
                self.factor_envir_norm_list,
                self.hyper_edge_adj_list,
                self.spidernet_data_list,
            )
        ):
            sample = build_sample_prepared_data(
                adata=adata,
                factor_envir_norm=factor_arr,
                hyper_edge_adj=hyper_adj,
                spidernet_data=sp_data,
                cascade_definition=self.cascade_definition,
                expression_max=self.expression_max,
                lr_max=self.lr_max,
                config=self.config,
            )

            group_conditions = build_group_conditions(
                sample.factor_binary, self.cascade_definition
            )
            edge_lookup = build_edge_lookup(sample.edge_index)

            # Keep overlap references within each sample
            reference_triplets: Dict[str, np.ndarray] = {}
            reference_nodes_by_pos: Dict[int, np.ndarray] = {}

            for group in DEFAULT_GROUPS:
                cond_first, cond_second = group_conditions[group]
                triplets = extract_triplets(
                    hyper_edge_adj=sample.hyper_edge_adj,
                    edge_index=sample.edge_index,
                    cell_types=sample.cell_types,
                    cond_first=cond_first,
                    cond_second=cond_second,
                    expected_cell_types=self.cascade_definition.cell_types,
                )

                triplets = self._apply_overlap_filter(
                    triplets=triplets,
                    group=group,
                    reference_triplets=reference_triplets,
                    reference_nodes_by_pos=reference_nodes_by_pos,
                )

                results.triplet_counts[group] += int(len(triplets))
                self._collect_selected_cell_records(
                    results=results,
                    sample_index=sample_index,
                    triplets=triplets,
                    obs_names=sample.obs_names,
                    cell_types=sample.cell_types,
                    group=group,
                )

                # Score pathway-level metrics
                for pathway_spec in self.pathway_specs:
                    index_info = self.pathway_index_info[pathway_spec.name]
                    edge_ids = triplets_to_edge_indices(
                        triplets=triplets,
                        edge_lookup=edge_lookup,
                        edge=pathway_spec.edge,
                    )

                    lr_scores = compute_mean_score_per_unit(
                        matrix=sample.lr_matrix,
                        row_indices=edge_ids,
                        feature_indices=index_info.lr_pair_indices,
                    )
                    source_nodes = select_triplet_nodes(
                        triplets,
                        cell_position=pathway_spec.source_position,
                        unique_only=self.config.deduplicate_cells_within_group,
                    )
                    target_nodes = select_triplet_nodes(
                        triplets,
                        cell_position=pathway_spec.target_position,
                        unique_only=self.config.deduplicate_cells_within_group,
                    )
                    ligand_scores = compute_mean_score_per_unit(
                        matrix=sample.expression,
                        row_indices=source_nodes,
                        feature_indices=index_info.ligand_gene_indices,
                    )
                    receptor_scores = compute_mean_score_per_unit(
                        matrix=sample.expression,
                        row_indices=target_nodes,
                        feature_indices=index_info.receptor_gene_indices,
                    )

                    results.pathway_scores[pathway_spec.name]["lr_strength"][group].extend(
                        lr_scores.tolist()
                    )
                    results.pathway_scores[pathway_spec.name]["ligand_expression"][group].extend(
                        ligand_scores.tolist()
                    )
                    results.pathway_scores[pathway_spec.name]["receptor_expression"][group].extend(
                        receptor_scores.tolist()
                    )

                # Score gene programs
                for program_spec in self.gene_program_specs:
                    index_info = self.gene_program_index_info[program_spec.name]
                    node_ids = select_triplet_nodes(
                        triplets,
                        cell_position=program_spec.cell_position,
                        unique_only=self.config.deduplicate_cells_within_group,
                    )
                    program_scores = compute_mean_score_per_unit(
                        matrix=sample.expression,
                        row_indices=node_ids,
                        feature_indices=index_info.gene_indices,
                    )
                    results.gene_program_scores[program_spec.category][program_spec.name][group].extend(
                        program_scores.tolist()
                    )

        return results

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import gseapy as gp



import numpy as np
import pandas as pd

def _broadcast_sample_index(sample_value, n, default_sample_index=None):
    """Broadcast sample index to length n."""
    if sample_value is None:
        if default_sample_index is None:
            return [None] * n
        return [int(default_sample_index)] * n

    arr = np.asarray(sample_value)
    if arr.ndim == 0:
        return [int(arr)] * n

    arr = arr.reshape(-1)
    if len(arr) != n:
        raise ValueError(
            f"Length mismatch between sample_index ({len(arr)}) and cell records ({n})."
        )
    return [int(x) for x in arr]

def _iter_cell_records_from_group_data(
    group_data,
    target_cell_position,
    default_group,
    default_sample_index=None,
):
    """
    Yield records with keys: group, sample_index, cell_index.

    Supported formats include:
    - DataFrame with columns like [sample_index, cell1, cell2, cell3]
    - DataFrame with columns like [sample_index, cell_index]
    - dict with keys like [sample_index, cell1, cell2, cell3]
    - dict with keys like [sample_index, cell_index]
    - dict/list/numpy arrays storing triplets
    """
    if group_data is None:
        return

    # -------------------------
    # Case 1: DataFrame input
    # -------------------------
    if isinstance(group_data, pd.DataFrame):
        col_map = {str(c).lower(): c for c in group_data.columns}

        sample_col = next(
            (col_map[k] for k in ["sample_index", "sample", "sample_id", "batch"] if k in col_map),
            None,
        )
        cell_col = next(
            (col_map[k] for k in ["cell_index", "cell", "selected_cell"] if k in col_map),
            None,
        )
        triplet_cols = [col_map[k] for k in ["cell1", "cell2", "cell3"] if k in col_map]

        # Direct cell records
        if cell_col is not None:
            for _, row in group_data.iterrows():
                yield {
                    "group": row["group"] if "group" in group_data.columns else default_group,
                    "sample_index": int(row[sample_col]) if sample_col is not None else default_sample_index,
                    "cell_index": int(row[cell_col]),
                }
            return

        # Triplet records
        if len(triplet_cols) == 3:
            cell_col_use = triplet_cols[target_cell_position]
            for _, row in group_data.iterrows():
                yield {
                    "group": row["group"] if "group" in group_data.columns else default_group,
                    "sample_index": int(row[sample_col]) if sample_col is not None else default_sample_index,
                    "cell_index": int(row[cell_col_use]),
                }
            return

    # -------------------------
    # Case 2: dict input
    # -------------------------
    if isinstance(group_data, dict):
        lower_map = {str(k).lower(): k for k in group_data.keys()}

        sample_key = next(
            (lower_map[k] for k in ["sample_index", "sample", "sample_id", "batch"] if k in lower_map),
            None,
        )
        cell_key = next(
            (lower_map[k] for k in ["cell_index", "cell", "selected_cell"] if k in lower_map),
            None,
        )

        triplet_keys = []
        for k in ["cell1", "cell2", "cell3"]:
            if k in lower_map:
                triplet_keys.append(lower_map[k])

        # Direct cell arrays
        if cell_key is not None:
            cells = np.asarray(group_data[cell_key]).reshape(-1)
            sample_vals = _broadcast_sample_index(
                group_data.get(sample_key, None),
                len(cells),
                default_sample_index=default_sample_index,
            )
            for s, c in zip(sample_vals, cells):
                if s is None:
                    continue
                yield {
                    "group": group_data.get("group", default_group),
                    "sample_index": int(s),
                    "cell_index": int(c),
                }
            return

        # Parallel triplet arrays: {sample_index: [...], cell1: [...], cell2: [...], cell3: [...]}
        if len(triplet_keys) == 3:
            arrs = [np.asarray(group_data[k]).reshape(-1) for k in triplet_keys]
            n = len(arrs[0])
            if not all(len(a) == n for a in arrs):
                raise ValueError("cell1/cell2/cell3 arrays do not have the same length.")
            sample_vals = _broadcast_sample_index(
                group_data.get(sample_key, None),
                n,
                default_sample_index=default_sample_index,
            )
            target_arr = arrs[target_cell_position]
            for s, c in zip(sample_vals, target_arr):
                if s is None:
                    continue
                yield {
                    "group": group_data.get("group", default_group),
                    "sample_index": int(s),
                    "cell_index": int(c),
                }
            return

        # Single-record triplet dict
        recognized_triplet_keys = [
            "triplet", "triplets", "cell_indices", "triplet_indices", "node_indices", "cells"
        ]
        for key in recognized_triplet_keys:
            if key in group_data:
                arr = np.asarray(group_data[key])
                sample_value = group_data.get(sample_key, default_sample_index)

                if arr.ndim == 1 and arr.size >= 3:
                    if sample_value is not None:
                        yield {
                            "group": group_data.get("group", default_group),
                            "sample_index": int(sample_value),
                            "cell_index": int(arr[target_cell_position]),
                        }
                    return

                if arr.ndim == 2 and arr.shape[1] >= 3:
                    sample_vals = _broadcast_sample_index(
                        sample_value,
                        arr.shape[0],
                        default_sample_index=default_sample_index,
                    )
                    for s, row in zip(sample_vals, arr):
                        if s is None:
                            continue
                        yield {
                            "group": group_data.get("group", default_group),
                            "sample_index": int(s),
                            "cell_index": int(row[target_cell_position]),
                        }
                    return

        # Recursive dict: e.g. {sample_index: ...} or nested structures
        for maybe_sample_index, sub_value in group_data.items():
            sample_idx = default_sample_index
            try:
                sample_idx = int(maybe_sample_index)
            except Exception:
                pass

            yield from _iter_cell_records_from_group_data(
                sub_value,
                target_cell_position=target_cell_position,
                default_group=default_group,
                default_sample_index=sample_idx,
            )
        return

    # -------------------------
    # Case 3: ndarray input
    # -------------------------
    if isinstance(group_data, np.ndarray):
        arr = np.asarray(group_data)

        # Already direct cell indices
        if arr.ndim == 1 and arr.size != 3:
            if default_sample_index is None:
                return
            for c in arr:
                yield {
                    "group": default_group,
                    "sample_index": int(default_sample_index),
                    "cell_index": int(c),
                }
            return

        # One triplet
        if arr.ndim == 1 and arr.size >= 3:
            if default_sample_index is None:
                return
            yield {
                "group": default_group,
                "sample_index": int(default_sample_index),
                "cell_index": int(arr[target_cell_position]),
            }
            return

        # Multiple triplets
        if arr.ndim == 2 and arr.shape[1] >= 3:
            if default_sample_index is None:
                return
            for row in arr:
                yield {
                    "group": default_group,
                    "sample_index": int(default_sample_index),
                    "cell_index": int(row[target_cell_position]),
                }
            return

    # -------------------------
    # Case 4: list / tuple / set
    # -------------------------
    if isinstance(group_data, (list, tuple, set)):
        for item in group_data:
            yield from _iter_cell_records_from_group_data(
                item,
                target_cell_position=target_cell_position,
                default_group=default_group,
                default_sample_index=default_sample_index,
            )
        return

def extract_group_cell_records_from_results(
    results,
    groups=("BothPos", "OnlyMIsecond"),
    target_cell_position=2,
    deduplicate=True,
):
    """
    Extract one cell position from grouped triplets or grouped selected cells
    stored in a MICascadeAnalyzer result object.
    """
    candidate_attrs = [
        "selected_cells",      # <-- this is the key fix for your current results object
        "group_triplets",
        "triplets_by_group",
        "group_to_triplets",
        "triplet_records_by_group",
        "triplet_group_records",
        "group_records",
        "records_by_group",
        "grouped_triplets",
        "triplets",
        "triplet_df",
        "triplets_df",
        "triplet_table",
        "records_df",
    ]

    source = None
    source_name = None
    for attr in candidate_attrs:
        if hasattr(results, attr):
            source = getattr(results, attr)
            source_name = attr
            break

    if source is None:
        available = [name for name in dir(results) if not name.startswith("_")]
        raise AttributeError(
            "Could not find grouped triplet/cell records inside `results`. "
            f"Checked: {candidate_attrs}. Available public attributes include: {available[:50]}"
        )

    rows = []

    # DataFrame source
    if isinstance(source, pd.DataFrame):
        if "group" not in source.columns:
            raise ValueError(
                f"DataFrame attribute `{source_name}` exists but does not contain a `group` column."
            )
        for group_name in groups:
            sub = source[source["group"] == group_name].copy()
            for rec in _iter_cell_records_from_group_data(
                sub,
                target_cell_position=target_cell_position,
                default_group=group_name,
            ):
                rows.append(rec)

    # dict-like or object-like source
    else:
        for group_name in groups:
            group_data = None
            if isinstance(source, dict):
                group_data = source.get(group_name, None)
            else:
                group_data = getattr(source, group_name, None)

            for rec in _iter_cell_records_from_group_data(
                group_data,
                target_cell_position=target_cell_position,
                default_group=group_name,
            ):
                rows.append(rec)

    group_cell_df = pd.DataFrame(rows)
    if group_cell_df.empty:
        return group_cell_df

    if deduplicate:
        group_cell_df = (
            group_cell_df
            .drop_duplicates(subset=["group", "sample_index", "cell_index"])
            .reset_index(drop=True)
        )

    return group_cell_df

def build_deg_adata_from_group_cells(
    adata_list,
    group_cell_df,
    prefer_layer="counts",
    normalize_log1p=True,
):
    """
    Build a concatenated AnnData for DEG from selected cells across groups and samples.
    """
    if group_cell_df is None or group_cell_df.empty:
        return None

    adata_chunks = []
    for (group_name, sample_index), sub in group_cell_df.groupby(["group", "sample_index"], sort=False):
        ad = adata_list[int(sample_index)]
        cell_indices = pd.Index(sub["cell_index"].astype(int)).unique().to_numpy()
        ad_sub = ad[cell_indices].copy()
        ad_sub.obs["group"] = str(group_name)
        ad_sub.obs["sample_index"] = int(sample_index)
        ad_sub.obs["cell_index"] = cell_indices.astype(int)

        if prefer_layer is not None and prefer_layer in ad_sub.layers:
            ad_sub.X = ad_sub.layers[prefer_layer].copy()

        adata_chunks.append(ad_sub)

    if len(adata_chunks) == 0:
        return None

    adata_deg = sc.concat(
        adata_chunks,
        join="outer",
        label="batch",
        keys=None,
        index_unique=None,
    )

    if normalize_log1p:
        sc.pp.normalize_total(adata_deg, target_sum=1e4)
        sc.pp.log1p(adata_deg)

    return adata_deg

def run_two_group_deg(
    adata_deg,
    test_group="BothPos",
    reference_group="OnlyMIsecond",
    method="wilcoxon",
    pts=True,
    key_added=None,
    save_path=None,
):
    """
    Run Scanpy DEG for one target group against one reference group.
    """
    if adata_deg is None:
        return None

    if key_added is None:
        key_added = f"rank_genes_{test_group}_vs_{reference_group}"

    sc.tl.rank_genes_groups(
        adata_deg,
        groupby="group",
        groups=[test_group],
        reference=reference_group,
        method=method,
        pts=pts,
        key_added=key_added,
    )

    deg_df = sc.get.rank_genes_groups_df(adata_deg, group=test_group, key=key_added)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        deg_df.to_csv(save_path, index=False)
        print(f"[Saved DEG table] {save_path}")

    return deg_df

def run_go_enrichment_from_deg(
    deg_df,
    output_dir,
    prefix,
    organism="Human",
    gene_sets=("GO_Biological_Process_2021",),
    lfc_thresh=0.5,
    padj_thresh=0.05,
    top_term=20,
    show=True,
):
    """
    Run GO enrichment on upregulated genes from a DEG result table.
    """
    if deg_df is None or deg_df.empty:
        return {"deg_up": pd.DataFrame(), "up_genes": [], "enr": None, "go_res": pd.DataFrame()}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deg_up = deg_df[(deg_df["logfoldchanges"] > lfc_thresh) & (deg_df["pvals_adj"] < padj_thresh)].copy()
    up_genes = deg_up["names"].dropna().astype(str).unique().tolist()
    print(f"#Upregulated genes in {prefix}: {len(up_genes)}")

    if len(up_genes) == 0:
        print("No upregulated genes passed the thresholds. GO enrichment was skipped.")
        return {"deg_up": deg_up, "up_genes": up_genes, "enr": None, "go_res": pd.DataFrame()}

    enr = gp.enrichr(
        gene_list=up_genes,
        gene_sets=list(gene_sets),
        organism=organism,
        outdir=str(output_dir / f"GO_enrichment_{prefix}"),
        cutoff=0.05,
    )

    go_res = enr.results.sort_values("Adjusted P-value").reset_index(drop=True)
    go_save_path = output_dir / f"GO_enrichment_{prefix}.csv"
    go_res.to_csv(go_save_path, index=False)
    print(f"[Saved GO results] {go_save_path}")

    if not go_res.empty:
        gp.barplot(
            enr.res2d,
            column="Adjusted P-value",
            title=f"GO enrichment ({prefix})",
            top_term=top_term,
            figsize=(6, 6),
        )
        plt.tight_layout()
        fig_path = output_dir / f"GO_barplot_{prefix}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved GO barplot] {fig_path}")

    return {"deg_up": deg_up, "up_genes": up_genes, "enr": enr, "go_res": go_res}

def _pick_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of {candidates} found in columns: {df.columns.tolist()}")

def _split_enrichr_genes(text):
    if pd.isna(text):
        return []
    parts = str(text).replace(",", ";").split(";")
    return [g.strip() for g in parts if g.strip()]

def build_term2genes_from_enrichr(
    go_res,
    panel_genes,
    top_n=20,
    deduplicate_signatures=True,
):
    """
    Convert top GO terms into a compact term-to-genes dictionary.
    Only genes present in `panel_genes` are retained.
    """
    if go_res is None or len(go_res) == 0:
        return pd.DataFrame(columns=[
            "Term", "Adjusted_P", "HitGenes_raw", "HitGenes_in_panel", "n_hit_raw", "n_hit_in_panel"
        ]), {}

    term_col = _pick_first_existing_column(go_res, ["Term", "term"])
    p_col = _pick_first_existing_column(go_res, ["Adjusted P-value", "Adjusted_P-value", "Adjusted P-value "])
    genes_col = _pick_first_existing_column(go_res, ["Genes", "genes"])

    panel_genes = set(map(str, panel_genes))
    top_terms = go_res.sort_values(p_col, ascending=True).head(top_n).copy()

    rows = []
    for _, row in top_terms.iterrows():
        hit_genes = _split_enrichr_genes(row[genes_col])
        hit_genes_in_panel = [g for g in hit_genes if g in panel_genes]
        rows.append(
            {
                "Term": row[term_col],
                "Adjusted_P": row[p_col],
                "HitGenes_raw": ";".join(hit_genes),
                "HitGenes_in_panel": ";".join(hit_genes_in_panel),
                "n_hit_raw": len(hit_genes),
                "n_hit_in_panel": len(hit_genes_in_panel),
            }
        )
    term_df = pd.DataFrame(rows)

    if deduplicate_signatures and not term_df.empty:
        def signature(text):
            genes = [g.strip() for g in str(text).split(";") if g.strip()]
            return tuple(sorted(set(genes)))

        term_df = (
            term_df.assign(_sig=term_df["HitGenes_in_panel"].apply(signature))
            .sort_values(["Adjusted_P", "Term"], ascending=[True, True])
            .drop_duplicates(subset=["_sig"], keep="first")
            .drop(columns=["_sig"])
            .reset_index(drop=True)
        )

    term2genes = (
        term_df
        .assign(_genes=term_df["HitGenes_in_panel"].fillna("").astype(str).str.split(";"))
        .set_index("Term")["_genes"]
        .apply(lambda genes: [g.strip() for g in genes if g is not None and g.strip() != ""])
        .to_dict()
    )

    return term_df, term2genes

def run_triplet_cell_deg_go_pipeline(
    results,
    adata_list,
    output_dir,
    prefix,
    groups=("BothPos", "OnlyMIsecond"),
    target_cell_position=2,
    prefer_layer="counts",
    normalize_log1p=True,
    organism="Human",
    gene_sets=("GO_Biological_Process_2021",),
    lfc_thresh=0.5,
    padj_thresh=0.05,
    top_n_terms=20,
    show=True,
):
    """
    End-to-end pipeline for:
    1) extracting one triplet cell position from selected groups,
    2) running DEG between two groups,
    3) running GO enrichment on upregulated genes,
    4) building and saving `term2genes`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cell_df = extract_group_cell_records_from_results(
        results=results,
        groups=groups,
        target_cell_position=target_cell_position,
        deduplicate=True,
    )
    if group_cell_df.empty:
        print("No cells were extracted from the selected triplet groups.")
        return {
            "group_cell_df": group_cell_df,
            "adata_deg": None,
            "deg_df": None,
            "deg_term_df": pd.DataFrame(),
            "term2genes": {},
        }

    print(group_cell_df.groupby("group")["cell_index"].nunique())

    adata_deg = build_deg_adata_from_group_cells(
        adata_list=adata_list,
        group_cell_df=group_cell_df,
        prefer_layer=prefer_layer,
        normalize_log1p=normalize_log1p,
    )

    test_group, reference_group = groups[0], groups[1]
    deg_path = output_dir / f"DEG_cell{target_cell_position + 1}_{test_group}_vs_{reference_group}_{prefix}.csv"
    deg_df = run_two_group_deg(
        adata_deg=adata_deg,
        test_group=test_group,
        reference_group=reference_group,
        method="wilcoxon",
        pts=True,
        save_path=deg_path,
    )

    go_out = run_go_enrichment_from_deg(
        deg_df=deg_df,
        output_dir=output_dir,
        prefix=f"cell{target_cell_position + 1}_{test_group}_up_{prefix}",
        organism=organism,
        gene_sets=gene_sets,
        lfc_thresh=lfc_thresh,
        padj_thresh=padj_thresh,
        top_term=top_n_terms,
        show=show,
    )

    deg_term_df, term2genes = build_term2genes_from_enrichr(
        go_res=go_out["go_res"],
        panel_genes=adata_deg.var_names if adata_deg is not None else [],
        top_n=top_n_terms,
        deduplicate_signatures=True,
    )

    term_df_path = output_dir / f"TopTerms_term2genes_cell{target_cell_position + 1}_{test_group}_up_{prefix}.csv"
    term_pkl_path = output_dir / f"term2genes_cell{target_cell_position + 1}_{test_group}_up_{prefix}.pkl"
    deg_term_df.to_csv(term_df_path, index=False)
    with open(term_pkl_path, "wb") as handle:
        pickle.dump(term2genes, handle)
    print(f"[Saved term table] {term_df_path}")
    print(f"[Saved term2genes] {term_pkl_path}")

    return {
        "group_cell_df": group_cell_df,
        "adata_deg": adata_deg,
        "deg_df": deg_df,
        "deg_up_df": go_out["deg_up"],
        "go_res": go_out["go_res"],
        "deg_term_df": deg_term_df,
        "term2genes": term2genes,
    }

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from torch_scatter import scatter_mean, scatter_max






















def compute_mi_receiver_agg_max(Factor_envir, edge_index_all, num_cells, device):
    edge_index_tensor = torch.tensor(edge_index_all, dtype=torch.long, device=device)
    factor_tensor = torch.tensor(Factor_envir, dtype=torch.float32, device=device)

    agg, _ = scatter_max(
        factor_tensor,
        edge_index_tensor[:, 1].to(torch.int64),
        dim=0,
        dim_size=num_cells,
    )
    agg = agg.detach().cpu().numpy()
    agg[np.isneginf(agg)] = np.nan
    return agg

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr, mannwhitneyu
from torch_scatter import scatter_mean, scatter_max
from statsmodels.stats.multitest import multipletests

def sanity_check_processed(processed):
    check_df = pd.DataFrame(
        {
            "n_cells": [processed.adata_list[i].obs.shape[0] for i in range(len(processed.adata_list))],
            "edge_index_max": [
                torch.max(processed.spidernet_data[i]["edge_index"]).cpu().item()
                for i in range(len(processed.adata_list))
            ],
        }
    )
    mismatch = check_df["n_cells"] != (check_df["edge_index_max"] + 1)
    return check_df, mismatch

def resolve_part0_meta_output_paths(base_run_dir: str | Path):
    base_run_dir = Path(base_run_dir)
    meta_json = base_run_dir / "meta_output_paths.json"
    if meta_json.exists():
        with open(meta_json, "r", encoding="utf-8") as handle:
            meta_paths = json.load(handle)
        return {k: Path(v) for k, v in meta_paths.items()}

    return {
        "loading_receiver_use_df_path": base_run_dir / "loading_receiver_use.csv",
        "loading_sender_use_df_path": base_run_dir / "loading_sender_use.csv",
        "Factor_envir_use_path": base_run_dir / "Factor_envir_use.npy",
        "Factor_envir_list_path": base_run_dir / "Factor_envir_list.pkl",
        "loading_LR_use_path": base_run_dir / "loading_LR_use.npy",
    }

def load_part0_outputs(base_run_dir: str | Path):
    meta_paths = resolve_part0_meta_output_paths(base_run_dir)

    loading_receiver_use_df = pd.read_csv(meta_paths["loading_receiver_use_df_path"], index_col=0)
    loading_sender_use_df = pd.read_csv(meta_paths["loading_sender_use_df_path"], index_col=0)
    loading_LR_use = np.load(meta_paths["loading_LR_use_path"])
    Factor_envir_use = np.load(meta_paths["Factor_envir_use_path"])

    factor_envir_list = None
    if meta_paths["Factor_envir_list_path"].exists():
        with open(meta_paths["Factor_envir_list_path"], "rb") as handle:
            factor_envir_list = pickle.load(handle)

    return {
        "meta_paths": meta_paths,
        "loading_receiver_use_df": loading_receiver_use_df,
        "loading_sender_use_df": loading_sender_use_df,
        "loading_LR_use": loading_LR_use,
        "Factor_envir_use": Factor_envir_use,
        "Factor_envir_list": factor_envir_list,
    }

def concatenate_adata_list(adata_list):
    if len(adata_list) == 1:
        return adata_list[0].copy()
    return adata_list[0].concatenate(adata_list[1:])

def normalize_edge_index(edge_index):
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()

    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {edge_index.shape}")
    if edge_index.shape[1] == 2:
        return edge_index
    if edge_index.shape[0] == 2:
        return edge_index.T
    raise ValueError(f"Unsupported edge_index shape: {edge_index.shape}")

def build_perturbation_annotation(adata_all):
    perturb_series = adata_all.obs["perturbation"].astype(str).fillna("")
    perturb_annotation = perturb_series.str.get_dummies(sep="_")

    drop_tokens = {"Control", "nan", "None", "none", ""}
    cols_to_keep = [c for c in perturb_annotation.columns if c not in drop_tokens]
    perturb_annotation = perturb_annotation[cols_to_keep]

    perturb_annotation.index = adata_all.obs.index
    perturb_annotation = perturb_annotation.astype("int8")

    is_cancer = adata_all.obs["celltype2"].astype(str).eq("cancer").values
    if perturb_annotation.shape[1] > 0:
        perturb_annotation.loc[~is_cancer, :] = 0
    return perturb_annotation

def to_gene_str(x, sep="|"):
    while isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]

    if isinstance(x, str):
        return x

    if isinstance(x, (list, tuple)):
        flat = []
        for e in x:
            if isinstance(e, (list, tuple)):
                flat.extend(list(e))
            else:
                flat.append(e)
        return sep.join([str(e) for e in flat])

    return str(x)

def build_lr_feature_names(LR_list):
    lr_merge = [f"{to_gene_str(lr[0])}_{to_gene_str(lr[1])}" for lr in LR_list]
    lr_merge_sr = [f"{x}_S" for x in lr_merge] + [f"{x}_R" for x in lr_merge]
    return lr_merge, lr_merge_sr

def compute_lr_edge_coexpression(expr_matrix, edge_index, genenames, LR_list):
    if torch.is_tensor(expr_matrix):
        expr_matrix = expr_matrix.detach().cpu().numpy()
    expr_matrix = np.asarray(expr_matrix)
    edge_index_np = normalize_edge_index(edge_index)

    lr_edge = np.zeros((edge_index_np.shape[0], len(LR_list)), dtype=np.float32)

    for LR_idx, LR_pair in enumerate(LR_list):
        ligand_idx = np.where(np.isin(genenames, LR_pair[0]))[0]
        receptor_idx = np.where(np.isin(genenames, LR_pair[1]))[0]

        exp_ligand = expr_matrix[:, ligand_idx]
        exp_receptor = expr_matrix[:, receptor_idx]

        exp_ligand = (
            np.power(np.prod(exp_ligand, axis=1), 1 / exp_ligand.shape[1])
            if exp_ligand.shape[1] > 1 else exp_ligand[:, 0]
        )
        exp_receptor = (
            np.power(np.prod(exp_receptor, axis=1), 1 / exp_receptor.shape[1])
            if exp_receptor.shape[1] > 1 else exp_receptor[:, 0]
        )

        lr_edge[:, LR_idx] = np.sqrt(
            exp_ligand[edge_index_np[:, 0]] * exp_receptor[edge_index_np[:, 1]]
        )

    return lr_edge.astype(np.float32)

def compute_lr_sender_receiver_agg(expr_matrix, edge_index, genenames, LR_list, device):
    lr_edge = compute_lr_edge_coexpression(
        expr_matrix=expr_matrix,
        edge_index=edge_index,
        genenames=genenames,
        LR_list=LR_list,
    )
    edge_index_np = normalize_edge_index(edge_index)
    n_cells = np.asarray(expr_matrix).shape[0]

    edge_index_tensor = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
    lr_edge_tensor = torch.tensor(lr_edge, dtype=torch.float32, device=device)

    lr_receiver_agg = scatter_mean(
        lr_edge_tensor,
        edge_index_tensor[:, 1].to(torch.int64),
        dim=0,
        dim_size=n_cells,
    ).detach().cpu().numpy()

    lr_sender_agg = scatter_mean(
        lr_edge_tensor,
        edge_index_tensor[:, 0].to(torch.int64),
        dim=0,
        dim_size=n_cells,
    ).detach().cpu().numpy()

    lr_sender_receiver_agg = np.hstack([lr_sender_agg, lr_receiver_agg]).astype(np.float32)
    return lr_edge, lr_sender_receiver_agg

def compute_tcell_neighboring_perturbation_matrix(
    adata_all,
    edge_index,
    perturb_annotation,
    perturb_gene_OI,
):
    edge_index_np = normalize_edge_index(edge_index)
    Tcell_index = np.where(adata_all.obs["celltype2"].astype(str).eq("T cells").values)[0]

    neighboring_counts = np.zeros((len(Tcell_index), len(perturb_gene_OI)), dtype=np.float32)

    for row_idx, tcell_idx in enumerate(Tcell_index):
        neighbors_all = edge_index_np[np.where(edge_index_np[:, 1] == tcell_idx)[0], 0]
        neighbors_perturbation = perturb_annotation.iloc[neighbors_all, :]
        for i, perturb_gene_cur in enumerate(perturb_gene_OI):
            neighboring_counts[row_idx, i] = np.sum(neighbors_perturbation[perturb_gene_cur])

    neighboring_binary = (neighboring_counts > 0).astype(int)
    return Tcell_index, neighboring_counts, neighboring_binary

def compute_golden_lfc(
    adata_all,
    Tcell_index,
    neighboring_binary,
    perturb_gene_OI,
    gene_names,
    lr_sender_receiver_agg_ori,
    lr_merge_sr,
):
    LFC_Tcell_dict = {}
    LFC_LRagg_dict = {}

    for i, perturb_gene_cur in enumerate(perturb_gene_OI):
        near_mask = neighboring_binary[:, i] == 1
        notnear_mask = neighboring_binary[:, i] == 0

        Tcell_index_near = Tcell_index[np.where(near_mask)[0]]
        Tcell_index_notnear = Tcell_index[np.where(notnear_mask)[0]]

        meanexp_near = np.mean(adata_all[Tcell_index_near].X, axis=0)
        meanexp_notnear = np.mean(adata_all[Tcell_index_notnear].X, axis=0)
        lfc_gene = np.log2(meanexp_near / meanexp_notnear)
        LFC_Tcell_dict[perturb_gene_cur] = pd.DataFrame(lfc_gene, index=gene_names, columns=[perturb_gene_cur])

        meanagg_near = np.mean(lr_sender_receiver_agg_ori[Tcell_index_near, :], axis=0)
        meanagg_notnear = np.mean(lr_sender_receiver_agg_ori[Tcell_index_notnear, :], axis=0)
        lfc_lr = np.log2(meanagg_near / meanagg_notnear)
        LFC_LRagg_dict[perturb_gene_cur] = pd.DataFrame(lfc_lr, index=lr_merge_sr, columns=[perturb_gene_cur])

    LFC_Tcell_golden = pd.concat([LFC_Tcell_dict[g] for g in perturb_gene_OI], axis=1)
    LFC_LRagg_golden = pd.concat([LFC_LRagg_dict[g] for g in perturb_gene_OI], axis=1)
    return LFC_Tcell_golden, LFC_LRagg_golden

def predict_batches_spidernet(model, data_list, device):
    exp_recon_list = []
    factor_envir_list = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_data in data_list:
            exp_reconcur, _, _, _, factor_envircur, _, _, _ = model(batch_data.to(device))
            exp_recon_list.append(exp_reconcur.detach().cpu().numpy())
            factor_envir_list.append(factor_envircur.detach().cpu().numpy())

    exp_recon = np.concatenate(exp_recon_list, axis=0).astype(np.float32)
    factor_envir = np.concatenate(factor_envir_list, axis=0).astype(np.float32)
    return exp_recon, factor_envir

def get_control_and_perturbed_cancer_indices(adata_all, perturb_annotation, perturb_gene_OI, perturb_gene_cur):
    idx = adata_all.obs.index[
        adata_all.obs["perturbation"].astype(str).str.contains(perturb_gene_cur)
    ]
    cellindex_perturbgene = np.where(adata_all.obs.index.isin(idx))[0]

    cellindex_all_perturbed = []
    for pg in perturb_gene_OI:
        idx_pg = adata_all.obs.index[
            (adata_all.obs["celltype2"] == "cancer")
            & (adata_all.obs["perturbation"].astype(str).str.contains(pg))
        ]
        cellindex_pg = np.where(adata_all.obs.index.isin(idx_pg))[0]
        cellindex_all_perturbed.extend(cellindex_pg.tolist())

    cellindex_all_perturbed = np.unique(cellindex_all_perturbed)
    cellindex_control = np.setdiff1d(
        np.where(adata_all.obs["celltype2"] == "cancer")[0],
        cellindex_all_perturbed,
    )

    cancercell_perturbed_index = np.where(
        (np.array(adata_all.obs["celltype2"]) == "cancer")
        & (np.array(perturb_annotation[perturb_gene_cur]) == 1)
    )[0]

    return cellindex_perturbgene, cellindex_control, cancercell_perturbed_index

def apply_in_silico_replacement(data_list, nearcellindex_cancer_cur, source_perturbed_index):
    data_list_perturbed = [copy.deepcopy(d) for d in data_list]

    for batch_index_cur in range(len(data_list_perturbed)):
        sampling_index = np.random.choice(
            source_perturbed_index,
            size=len(nearcellindex_cancer_cur),
            replace=True,
        )
        data_list_perturbed[batch_index_cur].x[
            nearcellindex_cancer_cur[:, None], :
        ] = data_list[batch_index_cur].x[
            sampling_index[:, None], :
        ]

    return data_list_perturbed


def evaluate_prediction_against_golden(LFC_golden_cur, LFC_predicted_cur):
    common_idx = LFC_golden_cur.index.intersection(LFC_predicted_cur.index)
    corr, pvalue = spearmanr(
        LFC_golden_cur.loc[common_idx],
        LFC_predicted_cur.loc[common_idx],
    )
    mse = np.mean((LFC_golden_cur.loc[common_idx] - LFC_predicted_cur.loc[common_idx]) ** 2)
    return corr, pvalue, mse

def run_insilico_spatial_perturbation(
    model,
    processed,
    LR_list,
    perturb_gene_OI,
    device,
    save_dir=None,
    random_seed=42,
):
    np.random.seed(random_seed)

    adata_all = concatenate_adata_list(processed.adata_list)
    gene_names = adata_all.var_names
    edge_index_all = normalize_edge_index(processed.spidernet_data[0].edge_index)
    perturb_annotation = build_perturbation_annotation(adata_all)
    _, lr_merge_sr = build_lr_feature_names(LR_list)

    lr_coexpression, lr_sender_receiver_agg_ori = compute_lr_sender_receiver_agg(
        expr_matrix=processed.spidernet_data[0]["x"],
        edge_index=processed.spidernet_data[0]["edge_index"],
        genenames=np.array(processed.spidernet_data[0]["genenames"]),
        LR_list=LR_list,
        device=device,
    )

    Tcell_index, neighboring_counts, neighboring_binary = compute_tcell_neighboring_perturbation_matrix(
        adata_all=adata_all,
        edge_index=edge_index_all,
        perturb_annotation=perturb_annotation,
        perturb_gene_OI=perturb_gene_OI,
    )

    LFC_Tcell_golden, LFC_LRagg_golden = compute_golden_lfc(
        adata_all=adata_all,
        Tcell_index=Tcell_index,
        neighboring_binary=neighboring_binary,
        perturb_gene_OI=perturb_gene_OI,
        gene_names=gene_names,
        lr_sender_receiver_agg_ori=lr_sender_receiver_agg_ori,
        lr_merge_sr=lr_merge_sr,
    )

    exp_recon, factor_envir = predict_batches_spidernet(
        model=model,
        data_list=processed.spidernet_data,
        device=device,
    )

    senders_all = edge_index_all[:, 0].astype(np.int64)
    receivers_all = edge_index_all[:, 1].astype(np.int64)

    LFC_predicted_dict = {}
    factor_envir_dict = {}
    perturbcancerTOTcell_edgerelindex_dict = {}
    TcellTOTperturbcancer_edgerelindex_dict = {}
    summary_rows = []

    for i, perturb_gene_cur in enumerate(perturb_gene_OI):
        cellindex_perturbgene, cellindex_control, cancercell_perturbed_index = \
            get_control_and_perturbed_cancer_indices(
                adata_all=adata_all,
                perturb_annotation=perturb_annotation,
                perturb_gene_OI=perturb_gene_OI,
                perturb_gene_cur=perturb_gene_cur,
            )

        near_mask = neighboring_binary[:, i].astype(bool)
        Tcell_index_near = Tcell_index[near_mask]
        Tcell_index_notnear = Tcell_index[~near_mask]

        exp_recon_Tcell_notnear = exp_recon[Tcell_index_notnear]

        nearcellindex_cur = [
            edge_index_all[:, 1][np.isin(edge_index_all[:, 0], Tcell_index_notnear)],
            edge_index_all[:, 0][np.isin(edge_index_all[:, 1], Tcell_index_notnear)],
        ]
        nearcellindex_cur = np.unique(np.hstack(nearcellindex_cur))
        nearcellindex_cancer_cur = np.intersect1d(nearcellindex_cur, cellindex_control)

        if len(nearcellindex_cancer_cur) == 0 or len(cancercell_perturbed_index) == 0:
            factor_envir_dict[perturb_gene_cur] = factor_envir.copy()
            perturbcancerTOTcell_edgerelindex_dict[perturb_gene_cur] = np.array([], dtype=np.int64)
            TcellTOTperturbcancer_edgerelindex_dict[perturb_gene_cur] = np.array([], dtype=np.int64)

            empty_lfc = pd.DataFrame(
                np.full(len(gene_names), np.nan, dtype=float),
                index=gene_names,
                columns=[perturb_gene_cur],
            )
            LFC_predicted_dict[perturb_gene_cur] = empty_lfc

            summary_rows.append(
                {
                    "Gene": perturb_gene_cur,
                    "Correlation": np.nan,
                    "PValue": np.nan,
                    "MSE": np.nan,
                    "n_tcell_near": len(Tcell_index_near),
                    "n_tcell_notnear": len(Tcell_index_notnear),
                    "n_control_cancer_replaced": 0,
                    "n_perturbed_cancer_donor": len(cancercell_perturbed_index),
                }
            )
            continue

        data_list_perturbed = apply_in_silico_replacement(
            data_list=processed.spidernet_data,
            nearcellindex_cancer_cur=nearcellindex_cancer_cur,
            source_perturbed_index=cancercell_perturbed_index,
        )

        exp_recon_perturb, factor_envir_perturb = predict_batches_spidernet(
            model=model,
            data_list=data_list_perturbed,
            device=device,
        )
        factor_envir_dict[perturb_gene_cur] = factor_envir_perturb

        exp_recon_Tcell_notnear_perturb = exp_recon_perturb[Tcell_index_notnear]
        mean_notnear = np.mean(exp_recon_Tcell_notnear, axis=0)
        mean_notnear_perturb = np.mean(exp_recon_Tcell_notnear_perturb, axis=0)

        lfc_pred = np.log2((mean_notnear_perturb + 1e-8) / (mean_notnear + 1e-8))
        LFC_predicted_dict[perturb_gene_cur] = pd.DataFrame(
            lfc_pred,
            index=gene_names,
            columns=[perturb_gene_cur],
        )

        perturbcancerTOTcell_edgerelindex = np.where(
            (np.isin(senders_all, nearcellindex_cancer_cur))
            & (np.isin(receivers_all, Tcell_index_notnear))
        )[0]
        TcellTOTperturbcancer_edgerelindex = np.where(
            (np.isin(receivers_all, nearcellindex_cancer_cur))
            & (np.isin(senders_all, Tcell_index_notnear))
        )[0]

        perturbcancerTOTcell_edgerelindex_dict[perturb_gene_cur] = perturbcancerTOTcell_edgerelindex
        TcellTOTperturbcancer_edgerelindex_dict[perturb_gene_cur] = TcellTOTperturbcancer_edgerelindex

        corr_cur, pvalue_cur, mse_cur = evaluate_prediction_against_golden(
            LFC_golden_cur=LFC_Tcell_golden[perturb_gene_cur],
            LFC_predicted_cur=LFC_predicted_dict[perturb_gene_cur][perturb_gene_cur],
        )

        summary_rows.append(
            {
                "Gene": perturb_gene_cur,
                "Correlation": corr_cur,
                "PValue": pvalue_cur,
                "MSE": mse_cur,
                "n_tcell_near": len(Tcell_index_near),
                "n_tcell_notnear": len(Tcell_index_notnear),
                "n_control_cancer_replaced": len(nearcellindex_cancer_cur),
                "n_perturbed_cancer_donor": len(cancercell_perturbed_index),
            }
        )

    LFC_Tcell_predicted = pd.concat(
        [LFC_predicted_dict[g] for g in perturb_gene_OI],
        axis=1,
    )
    summary_df = pd.DataFrame(summary_rows).set_index("Gene")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_dir / "Part1_insilico_spatial_perturbation_summary.csv")
        LFC_Tcell_golden.to_csv(save_dir / "Part1_LFC_Tcell_golden.csv")
        LFC_Tcell_predicted.to_csv(save_dir / "Part1_LFC_Tcell_predicted.csv")

    return {
        "adata_all": adata_all,
        "gene_names": gene_names,
        "edge_index_all": edge_index_all,
        "perturb_annotation": perturb_annotation,
        "Tcell_index": Tcell_index,
        "neighboring_counts": neighboring_counts,
        "neighboring_binary": neighboring_binary,
        "LFC_Tcell_golden": LFC_Tcell_golden,
        "LFC_LRagg_golden": LFC_LRagg_golden,
        "LFC_Tcell_predicted": LFC_Tcell_predicted,
        "summary_df": summary_df,
        "Factor_envir": factor_envir,
        "Factor_envir_dict": factor_envir_dict,
        "LR_coexpression": lr_coexpression,
        "LRcoexp_sender_receiver_agg_ori": lr_sender_receiver_agg_ori,
        "perturbcancerTOTcell_edgerelindex_dict": perturbcancerTOTcell_edgerelindex_dict,
        "TcellTOTperturbcancer_edgerelindex_dict": TcellTOTperturbcancer_edgerelindex_dict,
        "lr_merge_sr": lr_merge_sr,
    }

def compute_mi_change_tables(Factor_envir, Factor_envir_dict, perturb_gene_OI, perturbcancerTOTcell_edgerelindex_dict, TcellTOTperturbcancer_edgerelindex_dict, eps=1e-8):
    mi_cols = [f"MI-{i + 1}" for i in range(Factor_envir.shape[1])]
    mi_change_pc2t = pd.DataFrame(index=perturb_gene_OI, columns=mi_cols, dtype=float)
    mi_change_t2pc = pd.DataFrame(index=perturb_gene_OI, columns=mi_cols, dtype=float)

    for perturb_gene_cur in perturb_gene_OI:
        factor_envir_perturb = Factor_envir_dict[perturb_gene_cur]
        idx_pc2t = perturbcancerTOTcell_edgerelindex_dict[perturb_gene_cur]
        idx_t2pc = TcellTOTperturbcancer_edgerelindex_dict[perturb_gene_cur]

        if len(idx_pc2t) > 0:
            num = np.mean(factor_envir_perturb[idx_pc2t, :], axis=0)
            den = np.mean(Factor_envir[idx_pc2t, :], axis=0)
            mi_change_pc2t.loc[perturb_gene_cur, :] = np.log2((num + eps) / (den + eps))
        else:
            mi_change_pc2t.loc[perturb_gene_cur, :] = np.nan

        if len(idx_t2pc) > 0:
            num = np.mean(factor_envir_perturb[idx_t2pc, :], axis=0)
            den = np.mean(Factor_envir[idx_t2pc, :], axis=0)
            mi_change_t2pc.loc[perturb_gene_cur, :] = np.log2((num + eps) / (den + eps))
        else:
            mi_change_t2pc.loc[perturb_gene_cur, :] = np.nan

    order_pc2t = np.argsort(np.nansum(mi_change_pc2t.values, axis=0))[::-1]
    mi_change_pc2t = mi_change_pc2t.iloc[:, order_pc2t]

    order_t2pc = np.argsort(np.nansum(mi_change_t2pc.values, axis=0))[::-1]
    mi_change_t2pc = mi_change_t2pc.iloc[:, order_t2pc]

    return mi_change_pc2t, mi_change_t2pc

def normalize_loading_dataframe(df):
    colsum = df.abs().sum(axis=0)
    return df.div(colsum, axis=1).fillna(0)

def select_top_features_for_mi(MIOI, loading_receiver_use_df, loading_sender_use_df, loading_LR_use, LR_list, lr_threshold_map=None, target_threshold_map=None, regulator_threshold_map=None, top_n=5):
    # if lr_threshold_map is None:
    #     lr_threshold_map = {"MI6": 0.30}
    # if target_threshold_map is None:
    #     target_threshold_map = {"MI10": 0.40}
    # if regulator_threshold_map is None:
    #     regulator_threshold_map = {}

    loading_receiver_norm = normalize_loading_dataframe(loading_receiver_use_df)
    loading_sender_norm = normalize_loading_dataframe(loading_sender_use_df)

    loading_LR_use_df = pd.DataFrame(
        loading_LR_use,
        index=[f"MI{i + 1}" for i in range(loading_LR_use.shape[0])],
    )
    loading_LR_norm = normalize_loading_dataframe(loading_LR_use_df)

    loading_LR_norm_mi = loading_LR_norm.loc[[MIOI]].T.sort_values(by=MIOI, ascending=False)
    # lr_threshold = lr_threshold_map.get(MIOI, 0.40)
    # LR_index_top = loading_LR_norm_mi.loc[loading_LR_norm_mi[MIOI] > lr_threshold].index.tolist()[:top_n]
    LR_index_top = loading_LR_norm_mi.loc[loading_LR_norm_mi[MIOI] > 0.2].index.tolist()
    LR_index_top = [int(x) for x in LR_index_top]
    LR_list_top = [LR_list[i] for i in LR_index_top]

    # target_threshold = target_threshold_map.get(MIOI, 0.05)
    targetgene_top0 = loading_receiver_norm.loc[MIOI, :].sort_values(ascending=False)
    # targetgene_top = targetgene_top[targetgene_top > target_threshold].index.tolist()[:top_n]
    targetgene_top = targetgene_top0[targetgene_top0 > 0.2].index.tolist()
    if len(targetgene_top) ==0:
        targetgene_top = targetgene_top0[targetgene_top0 > 0.05].index.tolist()

    # regulator_threshold = regulator_threshold_map.get(MIOI, 0.05)
    regulatorgene_top0 = loading_sender_norm.loc[MIOI, :].sort_values(ascending=False)
    # regulatorgene_top = regulatorgene_top[regulatorgene_top > regulator_threshold].index.tolist()[:top_n]
    regulatorgene_top = regulatorgene_top0[regulatorgene_top0 > 0.2].index.tolist()
    if len(regulatorgene_top) == 0:
        regulatorgene_top = regulatorgene_top0[regulatorgene_top0 > 0.05].index.tolist()

    return {
        "loading_receiver_norm": loading_receiver_norm,
        "loading_sender_norm": loading_sender_norm,
        "loading_LR_norm": loading_LR_norm,
        "LR_index_top": LR_index_top,
        "LR_list_top": LR_list_top,
        "targetgene_top": targetgene_top,
        "regulatorgene_top": regulatorgene_top,
    }

def select_perturb_genes_for_mi(MIOI, mi_change_pc2t, threshold_map=None):
    if threshold_map is None:
        threshold_map = {"MI10": (">", 0.40), "MI6": ("<", -0.30), "MI8": (">", 0.40)}

    MIOI_modified = MIOI.replace("MI", "MI-")
    sign, threshold = threshold_map.get(MIOI, (">", 0.40))
    values = mi_change_pc2t.loc[:, MIOI_modified]

    if sign == ">":
        return values.index[values > threshold].tolist()
    return values.index[values < threshold].tolist()

def build_edge_group_masks_for_selected_genes(adata_all, edge_index_all, perturb_annotation, perturb_gene_OI_choose):
    cellclass_edge_index_all = np.array(adata_all.obs["celltype2"].values)[edge_index_all]
    perturb_annotation_edge_index_all = perturb_annotation.iloc[edge_index_all[:, 0], :]
    perturb_annotation_edge_index_all_OI = perturb_annotation_edge_index_all[perturb_gene_OI_choose]
    perturb_all_rowsum = np.sum(perturb_annotation_edge_index_all, axis=1)
    perturb_rowsum = np.sum(perturb_annotation_edge_index_all_OI, axis=1)

    edge_control = np.where(
        (cellclass_edge_index_all[:, 0] == "cancer")
        & (cellclass_edge_index_all[:, 1] == "T cells")
        & (perturb_all_rowsum == 0)
    )[0]

    edge_perturb = np.where(
        (cellclass_edge_index_all[:, 0] == "cancer")
        & (cellclass_edge_index_all[:, 1] == "T cells")
        & (perturb_rowsum > 0)
    )[0]

    return edge_control, edge_perturb

def build_tcell_subset_from_edge_groups(adata_all, edge_index_all, edge_control_index_fromcancer_to_Tcell, edge_perturb_index_fromcancer_to_Tcell):
    Tcell_in_edge_perturb = edge_index_all[edge_perturb_index_fromcancer_to_Tcell, 1]
    Tcell_in_edge_control = edge_index_all[edge_control_index_fromcancer_to_Tcell, 1]
    Tcell_in_edge_control = np.setdiff1d(Tcell_in_edge_control, Tcell_in_edge_perturb)
    ##
    Tcell_in_edge_perturb = np.unique(Tcell_in_edge_perturb)
    Tcell_in_edge_control = np.unique(Tcell_in_edge_control)

    barcode_control = adata_all.obs_names[Tcell_in_edge_control].tolist()
    barcode_perturb = adata_all.obs_names[Tcell_in_edge_perturb].tolist()

    adata_all = adata_all.copy()
    adata_all.obs["Tcell_group"] = "NA"
    adata_all.obs.loc[barcode_control, "Tcell_group"] = "InControl_edge"
    adata_all.obs.loc[barcode_perturb, "Tcell_group"] = "InPerturb_edge"
    adata_subset = adata_all[adata_all.obs["Tcell_group"].isin(["InControl_edge", "InPerturb_edge"]), :].copy()
    return adata_subset


