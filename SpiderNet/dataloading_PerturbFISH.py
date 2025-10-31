## ================================
## Package imports
## ================================
import sys
import os
import scanpy as sc
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import pickle
from SpiderNet.utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np

## ================================
## Input arguments
## ================================
data_path_main = sys.argv[1]
nHVG = int(sys.argv[2])
nHVG_LR = int(sys.argv[3])
ligand_receptor_filedir_cellchatdb = sys.argv[4]
ligand_receptor_filedir_scSeqComm = sys.argv[5]
num_neigh = int(sys.argv[6])
CC_prop_threshold = float(sys.argv[7])
save_path_main = sys.argv[8]
file_savepath_main = sys.argv[9]
if_subsetLRpair = str(sys.argv[10]).lower() in ["true", "1", "yes"]
num_subsetLRpair_ratio = float(sys.argv[11])

#################################################################
## Step 1: Load data and preprocessing
#################################################################
print("Step 1: Load the data and perform preprocessing")

adata_path = os.path.join(data_path_main, "adata/")
adata_files = [f for f in os.listdir(adata_path) if f.endswith('.h5ad')]
adata_list = []
genename_list = []
genename_intersect = None
sample_name_list = []

## Process each .h5ad file
for adata_files_cur in adata_files:
    adata_cur = sc.read_h5ad(os.path.join(adata_path, adata_files_cur))

    # ## Use raw counts layer
    # adata_cur.X = adata_cur.layers['counts']
    #
    # ## Normalize and log-transform
    # sc.pp.normalize_total(adata_cur, target_sum=1e4)
    # sc.pp.log1p(adata_cur)

    ## Store object and gene names
    adata_list.append(adata_cur)
    genename_list.append(adata_cur.var_names)
    if genename_intersect is None:
        genename_intersect = adata_cur.var_names
    else:
        genename_intersect = np.intersect1d(genename_intersect, adata_cur.var_names)
    del adata_cur

## ------------------------------
## Subset to common genes
## ------------------------------
for i in range(len(adata_list)):
    adata_list[i] = adata_list[i][:, genename_intersect].copy()

## Merge all samples into one AnnData
adata = sc.AnnData.concatenate(*adata_list, batch_key='sample_name', index_unique=None)
adata_copy = adata.copy()

## ------------------------------
## Select highly variable genes
## ------------------------------
# sc.pp.highly_variable_genes(adata, flavor='seurat_v3',
#                             n_top_genes=max(nHVG, nHVG_LR), subset=True)
# sc.pp.highly_variable_genes(adata_copy, flavor='seurat_v3',
#                             n_top_genes=nHVG, subset=True)

genenames = adata.var_names
genenames_train = adata_copy.var_names
genenames_train_index = np.where(np.isin(genenames, genenames_train))[0]
genenames_train = genenames[genenames_train_index]

# adata.obs['sample_name'] = adata.obs['samples']
##check if 'sample_name' in the adata.obs
if 'sample_name' not in adata.obs.columns:
    adata.obs['sample_name'] = "Sample1"

batch_cell = adata.obs['sample_name'].values
batch_cell_unique = np.sort(np.unique(batch_cell))

## ------------------------------
## Prepare sample-level metadata
## ------------------------------
## Extract ligand–receptor pairs
## ------------------------------
LR_list_cellchatdb, LR_meta_cellchatdb = Ligand_Receptor_gene_extraction_CellChatdb(
    ligand_receptor_filedir_cellchatdb, adata
)

if len(LR_list_cellchatdb) >= 300:
    LR_list = LR_list_cellchatdb
    LR_list_meta = LR_meta_cellchatdb

    ## Remove duplicate LR pairs
    LR_pairs_str = ["+".join(pair[0]) + "->" + "+".join(pair[1]) for pair in LR_list]
    LR_unique_str = np.unique(LR_pairs_str)
    LR_unique_idx = [np.where(np.array(LR_pairs_str) == s)[0][0] for s in LR_unique_str]
    LR_list = [LR_list[i] for i in LR_unique_idx]
    del LR_pairs_str, LR_unique_str, LR_unique_idx
else:
    LR_list_all = Ligand_Receptor_gene_extraction_scSeqComm(ligand_receptor_filedir_scSeqComm, adata)
    LR_list = LR_list_all + LR_list_cellchatdb

LR_list_val = []

## Save all LR pairs
with open(os.path.join(file_savepath_main, 'LR_list_all.pkl'), 'wb') as f:
    pickle.dump(LR_list, f)

## ------------------------------
## Organize data by sample
## ------------------------------
SpiderNet_data_list = []
for sample in batch_cell_unique:
    adata_cur = adata[adata.obs['sample_name'] == sample, :].copy()

    data_dict = {
        'num_cells': adata_cur.n_obs,
        'cellnames': adata_cur.obs_names,
        'num_genes': adata_cur.n_vars,
        'genenames': adata_cur.var_names,
        'spatial_location': adata_cur.obsm['spatial'],
        'expression_normalized': adata_cur.X.toarray() if sparse.issparse(adata_cur.X) else adata_cur.X,
        'cell_class': adata_cur.obs['celltype2'],
        'sample_name': adata_cur.obs['sample_name'],
    }

    SpiderNet_data_list.append(data_dict)
    del adata_cur, data_dict

del adata

#################################################################
## Step 2: Build spatial neighbor graph
#################################################################
for sample in batch_cell_unique:
    idx = np.where(batch_cell_unique == sample)[0][0]
    _, edge_index = spatial_neighborindex_generation(
        cell_spatial=SpiderNet_data_list[idx]['spatial_location'],
        num_neighbor_available=num_neigh
    )
    SpiderNet_data_list[idx]['edge_index'] = edge_index
    del edge_index

num_edges = np.sum([SpiderNet_data_list[i]['edge_index'].shape[0]
                    for i in range(len(SpiderNet_data_list))])
print(f"Total number of edges across all samples: {num_edges}")


#################################################################
## Step 3: One-hot encoding of cell types and PyG object creation
#################################################################

## Create one-hot encoding for cell classes
cellclass = np.hstack([
    SpiderNet_data_list[i]['cell_class'] for i in range(len(SpiderNet_data_list))
])
cellclass_unique = np.unique(cellclass)

for idx in range(len(batch_cell_unique)):
    cellclass_curbatch = SpiderNet_data_list[idx]['cell_class']
    num_cells = cellclass_curbatch.shape[0]

    cellclass_onehot = np.zeros((num_cells, cellclass_unique.shape[0]))
    for i, ct in enumerate(cellclass_unique):
        cellclass_onehot[cellclass_curbatch == ct, i] = 1

    cellclass_onehot_pd = pd.DataFrame(cellclass_onehot, columns=cellclass_unique)
    cellclass_onehot_pd.index = SpiderNet_data_list[idx]['cellnames']
    SpiderNet_data_list[idx]['cell_class_onehot'] = cellclass_onehot_pd

    del cellclass_curbatch, num_cells, cellclass_onehot, cellclass_onehot_pd
del cellclass


## Convert to PyTorch Geometric (PyG) Data format
SpiderNet_data_pyg_list = []
for idx in range(len(batch_cell_unique)):
    data_pyg = Data(
        x=torch.tensor(SpiderNet_data_list[idx]['expression_normalized'], dtype=torch.float),
        edge_index=torch.tensor(SpiderNet_data_list[idx]['edge_index'], dtype=torch.long),
        pos=torch.tensor(SpiderNet_data_list[idx]['spatial_location'], dtype=torch.float),
        cell_class_onehot=torch.tensor(np.array(SpiderNet_data_list[idx]['cell_class_onehot']), dtype=torch.float),
        cell_class_unique=cellclass_unique.tolist(),
        cellnames=SpiderNet_data_list[idx]['cellnames'],
        genenames=SpiderNet_data_list[idx]['genenames'],
        num_cells=SpiderNet_data_list[idx]['num_cells'],
        num_genes=SpiderNet_data_list[idx]['num_genes'],
        sample_name=SpiderNet_data_list[idx]['sample_name']
    )
    SpiderNet_data_pyg_list.append(data_pyg)
    del data_pyg


#################################################################
## Step 4: Compute cell–cell–LRpair tensor
#################################################################

for idx in range(len(batch_cell_unique)):
    edge_index = SpiderNet_data_pyg_list[idx].edge_index.cpu().numpy()
    cellpair_LRpair = np.zeros((edge_index.shape[0], len(LR_list)))

    for LR_idx, LR_pair in enumerate(LR_list):
        # Extract ligand expression
        ligand_idx = np.where(np.isin(
            SpiderNet_data_list[idx]['genenames'], LR_pair[0]
        ))[0]
        exp_ligand = np.array(SpiderNet_data_list[idx]['expression_normalized'][:, ligand_idx])
        exp_ligand = np.power(np.prod(exp_ligand, axis=1), 1 / exp_ligand.shape[1]) if exp_ligand.shape[1] > 1 else exp_ligand[:, 0]

        # Extract receptor expression
        receptor_idx = np.where(np.isin(
            SpiderNet_data_list[idx]['genenames'], LR_pair[1]
        ))[0]
        exp_receptor = np.array(SpiderNet_data_list[idx]['expression_normalized'][:, receptor_idx])
        exp_receptor = np.power(np.prod(exp_receptor, axis=1), 1 / exp_receptor.shape[1]) if exp_receptor.shape[1] > 1 else exp_receptor[:, 0]

        # Compute LR co-expression for each edge (sender→receiver)
        cellpair_LRpair[:, LR_idx] = np.sqrt(
            exp_ligand[edge_index[:, 0]] * exp_receptor[edge_index[:, 1]]
        )
        del ligand_idx, receptor_idx, exp_ligand, exp_receptor

    SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'] = torch.tensor(cellpair_LRpair)
    del edge_index, cellpair_LRpair

del SpiderNet_data_list


#################################################################
## Step 5: Filter LR pairs based on coverage across edges
#################################################################

# Compute the proportion of edges with nonzero LR expression
cellpair_LRpair_prop = np.array([
    (torch.sum(d['cellpair_LRpair_neigh'] > 0, dim=0) /
     d['cellpair_LRpair_neigh'].shape[0]).numpy()
    for d in SpiderNet_data_pyg_list
])

cellpair_LRpair_prop_max = np.max(cellpair_LRpair_prop, axis=0)
print("Quantiles of LR pair activation proportion:",
      np.quantile(cellpair_LRpair_prop_max, (0, 0.2, 0.5, 0.8, 1.0)))

selected_LR_idx = np.where(cellpair_LRpair_prop_max > CC_prop_threshold)[0]
selected_LR_idx = torch.tensor(selected_LR_idx, dtype=torch.long)

for idx in range(len(SpiderNet_data_pyg_list)):
    SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'] = (
        SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'][:, selected_LR_idx]
    )

LR_list = [LR_list[i] for i in selected_LR_idx.numpy()]


#################################################################
## Step 6: Compute pairwise LR correlation across samples
#################################################################

cellpair_LRpair_corr = np.zeros((len(LR_list), len(LR_list)))
for idx in range(len(SpiderNet_data_pyg_list)):
    corr_cur = np.corrcoef(
        SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'].cpu().numpy(),
        rowvar=False
    )
    corr_cur[np.isnan(corr_cur)] = 0  # Replace NaN with 0
    cellpair_LRpair_corr = np.maximum(cellpair_LRpair_corr, corr_cur)  # element-wise max

cellpair_LRpair_corr = pd.DataFrame(
    cellpair_LRpair_corr,
    index=[f"LR{i}" for i in range(len(LR_list))],
    columns=[f"LR{i}" for i in range(len(LR_list))]
)

## Filter by correlation
corr = cellpair_LRpair_corr.copy()
np.fill_diagonal(corr.values, np.nan)
choose_LRindex = np.nanmax(corr.values, axis=0) > 0.5
choose_LRindex_tensor = torch.tensor(choose_LRindex, dtype=torch.bool)

for idx in range(len(SpiderNet_data_pyg_list)):
    SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'] = (
        SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'][:, choose_LRindex_tensor]
    )

LR_list = [LR_list[i] for i in np.where(choose_LRindex)[0]]


#################################################################

#################################################################
## Step 9: Hierarchical clustering of LR correlations
#################################################################

corr = corr.loc[choose_LRindex, choose_LRindex]

# Convert correlation to binary adjacency (for visualization)
corr_binary = (corr > 0.5).astype(int)
np.fill_diagonal(np.array(corr_binary), 1)
corr_binary = pd.DataFrame(corr_binary, index=corr.index, columns=corr.columns)

# Compute distance matrix (1 - correlation)
dist_matrix = 1 - corr.fillna(1).values
row_linkage = linkage(squareform(dist_matrix, checks=False), method='average')
col_linkage = linkage(squareform(dist_matrix.T, checks=False), method='average')

row_order = leaves_list(row_linkage)
col_order = leaves_list(col_linkage)
corr_clustered = corr.iloc[row_order, col_order]

plt.figure(figsize=(10, 10))
sns.heatmap(
    corr_clustered, cmap="viridis", annot=False, fmt=".2f",
    square=True, cbar_kws={"shrink": 0.8},
    mask=np.eye(len(corr_clustered))
)
plt.title("Correlation of LR pairs (Clustered)")
plt.savefig(save_path_main + "LR_pairs_correlation_heatmap_clustered.png",
            dpi=300, bbox_inches='tight')
plt.close()

# ## Select the optimal number of clusters
#
# # Assume dist_matrix is the distance matrix converted from the correlation matrix
# dist_matrix = np.array(dist_matrix)
#
# # Ensure that the diagonal is set to 0
# np.fill_diagonal(dist_matrix, 0)
#
# scores = []
# ks = range(2, (dist_matrix).shape[0])
#
# for k in ks:
#     cluster = AgglomerativeClustering(
#         n_clusters=k,
#         metric='precomputed',
#         linkage='average'
#     )
#     labels = cluster.fit_predict(dist_matrix)
#
#     # The diagonal must also be 0 here
#     # score = silhouette_score(dist_matrix, labels, metric='precomputed')
#     score = silhouette_score(dist_matrix, labels)
#     scores.append(score)
#
# plt.close()
# plt.figure(figsize=(10, 6))
# plt.plot(ks, scores, marker='o')
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette score")
# plt.title("Optimal number of clusters")
# plt.savefig(save_path_main + "Optimal_number_of_clusters.png", dpi=300, bbox_inches='tight')
#
# optim_clusters_num = ks[np.argmax(scores)]
# ## Obtain the cluster assignments for the optimal number of clusters
# cluster_model = AgglomerativeClustering(
#     n_clusters=optim_clusters_num,
#     metric='precomputed',
#     linkage='average'
# )
# labels = cluster_model.fit_predict(dist_matrix)
# if dim_envir is None:
#     dim_envir = optim_clusters_num
#     ## Set up the save paths for the results
#     file_savepath_main = save_path_main_cur + "SpiderNet_Result_Mode_" + str(Factor_mode) + "_optidim" + str(
#         dim_envir) + "/"
#     file_savepath_MI_main = save_path_main_cur + "SpiderNet_Result_Mode_" + str(Factor_mode) + "_optidim" + str(
#         dim_envir) + "/MetaInteraction/"
#     file_savepath_model_main = save_path_main_cur + "SpiderNet_Result_Mode_" + str(Factor_mode) + "_optidim" + str(
#         dim_envir) + "/Model/"
# else:
#     ## Set up the save paths for the results (user-defined dimension)
#     file_savepath_main = save_path_main_cur + "SpiderNet_Result_Mode_" + str(Factor_mode) + "_selfsetdim" + str(
#         dim_envir) + "/"
#     file_savepath_MI_main = save_path_main_cur + "SpiderNet_Result_Mode_" + str(Factor_mode) + "_selfsetdim" + str(
#         dim_envir) + "/MetaInteraction/"
#     file_savepath_model_main = save_path_main_cur + "SpiderNet_Result_Mode_" + str(Factor_mode) + "_selfsetdim" + str(
#         dim_envir) + "/Model/"
# print("dim_envir: " + str(dim_envir))
##
# if not os.path.exists(file_savepath_main):
#     os.makedirs(file_savepath_main)
# if not os.path.exists(file_savepath_MI_main):
#     os.makedirs(file_savepath_MI_main)
# if not os.path.exists(file_savepath_model_main):
#     os.makedirs(file_savepath_model_main)

####################################################################################################################################
#################################################################
## Step 10: Extract and save processed data
#################################################################

## Subset gene expression to the selected training genes
for idx in range(len(batch_cell_unique)):
    # SpiderNet_data_pyg_list[idx]['x'] = SpiderNet_data_pyg_list[idx]['x'][
    #     :, torch.tensor(genenames_train_index, dtype=torch.long)
    # ]
    SpiderNet_data_pyg_list[idx]['x'] = SpiderNet_data_pyg_list[idx]['x']

## Assign each cell its corresponding class label
for idx in range(len(batch_cell_unique)):
    SpiderNet_data_pyg_list[idx]['cell_class'] = np.array(
        SpiderNet_data_pyg_list[idx]['cell_class_unique']
    )[np.where(SpiderNet_data_pyg_list[idx]['cell_class_onehot'] == 1)[1]]


#################################################################
## Step 11: Compute neighboring cell-type proportions
#################################################################

# All unique cell types
cell_types = np.unique(adata_copy.obs['celltype2'])

# Dictionary to store per-cell neighbor composition for each cell type
neighbor_props = {ct: [] for ct in cell_types}

for idx in range(len(batch_cell_unique)):
    SpiderNet_data_pyg_cur = SpiderNet_data_pyg_list[idx]

    # Get current sample’s cell type labels
    cellclass_cur = SpiderNet_data_pyg_cur['cell_class']
    edge_index_cur = SpiderNet_data_pyg_cur['edge_index'].cpu().numpy()
    src, dst = edge_index_cur[:, 0], edge_index_cur[:, 1]
    n_nodes = len(cellclass_cur)

    # Count neighbors per node
    neigh_cnt = np.bincount(src, minlength=n_nodes).astype(float)
    safe_div = lambda num, den: np.divide(num, np.where(den == 0, 1.0, den))

    # Compute neighbor-type proportion for each cell type
    for ct in neighbor_props.keys():
        ct_cnt = np.bincount(
            src, weights=(cellclass_cur[dst] == ct).astype(float), minlength=n_nodes
        )
        ct_prop = safe_div(ct_cnt, neigh_cnt)
        neighbor_props[ct].extend(ct_prop.tolist())

# Add neighbor proportions to adata_copy.obs
for ct, values in neighbor_props.items():
    adata_copy.obs[f"{ct}_prop"] = values


#################################################################
## Step 12: (Optional) Randomly subset LR pairs for training/validation
#################################################################

if if_subsetLRpair:
    num_subsetLRpair = int(num_subsetLRpair_ratio * len(LR_list))
    np.random.seed(0)
    sample_index = np.random.choice(range(len(LR_list)), num_subsetLRpair, replace=False)
    unsample_index = np.setdiff1d(range(len(LR_list)), sample_index)

    LR_list_val = [LR_list[i] for i in unsample_index]
    LR_list = [LR_list[i] for i in sample_index]

    print(
        f"Subset the LR pairs. Number of Ligand–Receptor pairs for training: {num_subsetLRpair}"
    )

    # Save validation LR list
    with open(file_savepath_main + 'LR_list_val.pkl', 'wb') as f:
        pickle.dump(LR_list_val, f)

    # Subset LR features in PyG objects
    for idx in range(len(batch_cell_unique)):
        SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'] = (
            SpiderNet_data_pyg_list[idx]['cellpair_LRpair_neigh'][
                :, torch.tensor(sample_index, dtype=torch.long)
            ]
        )


#################################################################
## Step 13: Save processed objects for downstream analyses
#################################################################

# Save integrated annotated AnnData
adata_copy.write_h5ad(file_savepath_main + "adata_all.h5ad", compression='gzip')

# Save PyG-formatted data list
with open(file_savepath_main + 'SpiderNet_data_pyg_list.pkl', 'wb') as f:
    pickle.dump(SpiderNet_data_pyg_list, f)

# Save ligand–receptor lists and metadata
with open(file_savepath_main + 'LR_list.pkl', 'wb') as f:
    pickle.dump(LR_list, f)

with open(file_savepath_main + 'LR_list_cellchatdb.pkl', 'wb') as f:
    pickle.dump(LR_list_cellchatdb, f)

with open(file_savepath_main + 'LR_meta_cellchatdb.pkl', 'wb') as f:
    pickle.dump(LR_meta_cellchatdb, f)

# Save sample/batch information
with open(file_savepath_main + 'batch_cell_unique.pkl', 'wb') as f:
    pickle.dump(batch_cell_unique, f)

with open(file_savepath_main + 'batch_cell.pkl', 'wb') as f:
    pickle.dump(batch_cell, f)

# Save gene list used for model training
with open(file_savepath_main + 'genenames_train.pkl', 'wb') as f:
    pickle.dump(genenames_train, f)

# Save list of AnnData objects per sample
with open(file_savepath_main + 'adata_list.pkl', 'wb') as f:
    pickle.dump(adata_list, f)
