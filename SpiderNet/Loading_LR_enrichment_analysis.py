## Package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import os
import pickle
#
# matplotlib.use("TkAgg")  # or "Qt5Agg", "Agg" (non-interactive)


def run_analysis(
        loading_LR_use_path,
        lr_list_path,
        lr_list_cellchatdb_path,
        lr_meta_cellchatdb_path,
        Factor_envir_use_path,
        file_savepath_main,
        show=True,
):
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
    """

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

    # Load data
    loading_LR_use = np.load(loading_LR_use_path)
    LR_list = pd.read_pickle(lr_list_path)
    LR_list_cellchatdb = pd.read_pickle(lr_list_cellchatdb_path)
    LR_meta_cellchatdb = pd.read_pickle(lr_meta_cellchatdb_path)
    Factor_envir_use = np.load(Factor_envir_use_path)

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
                           ]
    LR_meta_incellchatdb.to_csv(os.path.join(file_savepath_main, "LR_meta_incellchatdb.csv"), index=False)

    # Normalize loading matrix
    LR_list_merge = []
    for lr_pair in LR_list:
        ligand, receptor = lr_pair
        LR_list_merge.append("+".join(ligand) + " -> " + "+".join(receptor))

    loading_LR_use = pd.DataFrame(
        loading_LR_use,
        columns=LR_list_merge,
        index=[f"MI-{i + 1}" for i in range(loading_LR_use.shape[0])]
    )
    loading_LR_use_norm = loading_LR_use.div(loading_LR_use.sum(axis=0), axis=1).fillna(0)

    # CellChat pathway enrichment
    LR_meta_incellchatdb.index = np.arange(LR_meta_incellchatdb.shape[0])
    pathway_unique_Cellchat = LR_meta_incellchatdb['pathway_name'].unique()
    LR_loading_pathway = []

    for pathway in pathway_unique_Cellchat:
        LR_idx = LR_meta_incellchatdb.index[LR_meta_incellchatdb['pathway_name'] == pathway]
        print(f"{pathway}: {len(LR_idx)} LR pairs")
        LR_loading_pathway.append(loading_LR_use_norm.iloc[:, LR_idx].mean(axis=1))

    LR_loading_pathway = pd.DataFrame(
        LR_loading_pathway, index=pathway_unique_Cellchat, columns=loading_LR_use_norm.index
    ).fillna(0).T

    ##
    LR_loading_pathway = LR_loading_pathway.loc[:, list(LR_loading_pathway.columns)[::-1]]
    LR_loading_pathway = LR_loading_pathway.iloc[np.argsort(np.array(np.max(LR_loading_pathway, axis=1)))[::-1], :]

    # # Sort by maximum values (rows and columns)
    # LR_loading_pathway = LR_loading_pathway.iloc[
    #     np.argsort(np.array(np.max(LR_loading_pathway, axis=1)))[::-1]
    # ]
    ##
    argmax_1 = np.argmax(np.array(LR_loading_pathway), axis=0)
    max_1 = np.max(np.array(LR_loading_pathway), axis=0)
    order_index_LRpathway = []
    for argmax_1_cur in np.sort(np.unique(argmax_1)):
        index_cur = np.where(argmax_1 == argmax_1_cur)[0]
        index_cur = index_cur[np.argsort(max_1[index_cur])[::-1]]
        index_cur = index_cur.tolist()
        order_index_LRpathway.extend(index_cur)
    # LR_loading_pathway = LR_loading_pathway.iloc[:,np.argsort(np.argmax(np.array(LR_loading_pathway),axis = 0))]
    LR_loading_pathway = LR_loading_pathway.iloc[:, order_index_LRpathway]

    # LR_loading_pathway = LR_loading_pathway.iloc[
    #     :, np.argsort(np.max(LR_loading_pathway, axis=0))[::-1]
    # ]

    # Visualization
    plt.close()
    fig, ax = plt.subplots(figsize=(12 * LR_loading_pathway.shape[1] / 20, 9.3))
    cmap = LinearSegmentedColormap.from_list(
        "white_red", ["white", "#FFDFEF", "#EABDE6", "#D69ADE", "#AA60C8"], N=256
    )

    data = LR_loading_pathway.values
    vmin, vmax = data.min(), min(0.4, np.max(data) * 0.7)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)

    mesh = ax.pcolormesh(
        np.arange(data.shape[1] + 1),
        np.arange(data.shape[0] + 1),
        data,
        cmap=cmap,
        norm=norm,
        edgecolors='#B6B9BA',
        linewidth=1.0
    )

    ax.set_xticks(np.arange(data.shape[1]) + 0.5)
    ax.set_xticklabels(LR_loading_pathway.columns, rotation=90, fontsize=22)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_yticklabels(LR_loading_pathway.index, fontsize=18)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()

    plt.colorbar(mesh, ax=ax)
    plt.tight_layout()

    save_png = os.path.join(file_savepath_main, "LR_loading_pathway.png")
    plt.savefig(save_png, format='png', bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

    # Save results
    LR_loading_pathway.to_csv(os.path.join(file_savepath_main, "LR_loading_pathway.csv"))
    with open(file_savepath_main + 'LR_list_merge.pkl', 'wb') as f:
        pickle.dump(LR_list_merge, f)


# Allow both import and direct execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 7:
        raise ValueError(
            "Usage: python Loading_LR_enrichment_analysis.py "
            "<loading_LR_use_path> <lr_list_path> <lr_list_cellchatdb_path> "
            "<lr_meta_cellchatdb_path> <Factor_envir_use_path> <file_savepath_main>"
        )
    run_analysis(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
    )
