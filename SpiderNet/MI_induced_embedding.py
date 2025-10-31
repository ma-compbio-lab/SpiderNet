## =============================================================
## MI_induced_embedding.py
## Refactored to be callable as a function from Python scripts
## =============================================================

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


def run_analysis(
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
        show=True
):
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

    cellclass_unique = np.unique(adata_copy.obs['cell.types'])

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
    ## Step 5: Plot MI–celltype pair heatmap
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
    cellindex_choose = np.where(adata_copy.obs['cell.types'] == cellclass_choose)[0]
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

    print(f"✅ Finished embedding for {cellclass_choose}. Results saved in {file_savepath_main}.")


## =============================================================
## CLI entry point
## =============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 13:
        raise ValueError(
            "Usage: python MI_induced_embedding.py "
            "<cellclass_choose> <file_savepath_main> <Factor_envir_list_path> "
            "<adata_copy_path> <device> <SpiderNet_data_pyg_list_path> "
            "<LR_list_merge_path> <Avg_MI_cellclass_pair_merge_use_path> "
            "<dim_envir> <MIlevel_agg_threshold> <metadata_sample_path> <embedding_method>"
        )

    run_analysis(
        cellclass_choose=sys.argv[1],
        file_savepath_main=sys.argv[2],
        Factor_envir_list_path=sys.argv[3],
        adata_copy_path=sys.argv[4],
        device=sys.argv[5],
        SpiderNet_data_pyg_list_path=sys.argv[6],
        LR_list_merge_path=sys.argv[7],
        Avg_MI_cellclass_pair_merge_use_path=sys.argv[8],
        dim_envir=int(sys.argv[9]),
        MIlevel_agg_threshold=float(sys.argv[10]),
        metadata_sample_path=sys.argv[11],
        embedding_method=sys.argv[12],
        show=True
    )
