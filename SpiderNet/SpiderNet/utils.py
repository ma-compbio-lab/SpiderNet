# Package imports
import copy
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from torch_scatter import scatter_add
import torch
from torch_scatter import scatter_max

def spatial_neighborindex_generation(cell_spatial, num_neighbor_available):
    spatialnn_use = NearestNeighbors(n_neighbors=num_neighbor_available + 1, algorithm='auto')
    spatialnn_use.fit(cell_spatial)
    spatialdist, spatial_neighborindex = spatialnn_use.kneighbors(cell_spatial)
    for i in range(spatial_neighborindex.shape[0]):
        if not spatial_neighborindex[i, 0] == i:
            spatial_neighborindex[i, :] = np.hstack((i, np.setdiff1d(spatial_neighborindex[i, :], i)))
    ##Trans into the edge_index form
    spatial_neighborindex_further = spatial_neighborindex[:, 1:]
    centralindex = spatial_neighborindex[:, 0]
    edge_index = np.vstack((np.repeat(centralindex,spatial_neighborindex_further.shape[1]), spatial_neighborindex_further.flatten()))

    return spatial_neighborindex, edge_index.T


def Ligand_Receptor_gene_extraction_CellChatdb(ligand_receptor_filedir, adata):
    genenames = adata.var_names
    ## Check whether adata.X is sparse or dense
    if sparse.issparse(adata.X):
        genenames = genenames[np.sum(np.array(adata.X.todense()), axis=0) > 0]
    else:
        genenames = genenames[np.sum(np.array(adata.X), axis=0) > 0]
    genenames_set = np.unique(np.array(genenames))  # Convert gene names to a set for fast lookup
    ## Load the ligand–receptor pairs from the database
    db_ligand_receptor = pd.read_csv(ligand_receptor_filedir, header=0)
    LR_list_pre = []
    ifin_geneset = []
    ligand_all = []
    receptor_all = []
    for i in range(db_ligand_receptor.shape[0]):
        lr_name_cur = db_ligand_receptor['interaction_name_2'].iloc[i]
        ## Separate lr_name_cur using " - " to get ligand and receptor
        ligand_cur, receptor_cur = lr_name_cur.split(" - ")
        ## Remove spaces in ligand and receptor names
        ligand_cur = ligand_cur.replace(" ", "")
        receptor_cur = receptor_cur.replace(" ", "")
        ## Check if there is "+" in the ligand name; if yes, split by "+"
        if '+' in ligand_cur:
            ligand_cur = ligand_cur.replace("(", "").replace(")", "").split("+")
        else:
            ligand_cur = [ligand_cur]
        if '+' in receptor_cur:
            receptor_cur = receptor_cur.replace("(", "").replace(")", "").split("+")
        else:
            receptor_cur = [receptor_cur]
        lr_name_use = [ligand_cur, receptor_cur]
        ligand_all.extend(ligand_cur)
        receptor_all.extend(receptor_cur)
        ##
        # if np.cumprod(np.isin(np.array(ligand_cur), genenames_set)) == 1 and np.cumprod(np.isin(np.array(receptor_cur), genenames_set)) == 1:
        if np.sum(np.isin(np.array(ligand_cur), genenames_set)) > 0 and np.sum(
                np.isin(np.array(receptor_cur), genenames_set)) > 0:
            ifin_geneset.append(True)
            ## Further remove genes in lr_name_use that are not in genenames_set
            lr_name_use[0] = list(np.array(lr_name_use[0])[np.isin(np.array(lr_name_use[0]), genenames_set)])
            lr_name_use[1] = list(np.array(lr_name_use[1])[np.isin(np.array(lr_name_use[1]), genenames_set)])
        else:
            ifin_geneset.append(False)
        ##
        LR_list_pre.append(lr_name_use)
    ligand_all = np.unique(np.array(ligand_all))
    receptor_all = np.unique(np.array(receptor_all))
    db_ligand_receptor_choose = db_ligand_receptor.loc[ifin_geneset, :]
    LR_list_use = []
    for i in range(len(ifin_geneset)):
        if ifin_geneset[i]:
            LR_list_use.append(LR_list_pre[i])
    # ## Remove rows that contain NaN values
    # db_ligand_receptor = db_ligand_receptor.dropna(axis=0, how='any')
    ## Select ligand–receptor pairs that exist in genenames
    # Example alternative filtering approach
    #
    # # Preprocess ligands and receptors: split comma-separated strings into lists
    # db_ligand_receptor['ligand_list'] = db_ligand_receptor['ligand'].apply(lambda x: x.split(','))
    # db_ligand_receptor['receptor_list'] = db_ligand_receptor['receptor'].apply(lambda x: x.split(','))
    #
    # Check whether all ligands and receptors are in genenames_set
    # def is_valid_lr(ligand_list, receptor_list):
    #     return all(ligand in genenames_set for ligand in ligand_list) and all(receptor in genenames_set for receptor in receptor_list)
    #
    # Vectorized filtering example
    # LR_choose = db_ligand_receptor[
    #     db_ligand_receptor.apply(lambda row: is_valid_lr(row['ligand_list'], row['receptor_list']), axis=1)
    # ][['ligand_list', 'receptor_list']].values.tolist()
    #
    # Convert ligand_list and receptor_list to tuples for deduplication
    # LR_choose_unique = list(set(tuple(map(tuple, pair)) for pair in LR_choose))
    ##
    print("Number of ligand-receptor pairs from CellChatDB: " + str(len(LR_list_use)))
    return LR_list_use, db_ligand_receptor_choose

def Ligand_Receptor_gene_extraction_scSeqComm(ligand_receptor_filedir, adata):
    genenames = adata.var_names
    adata_copy = copy.deepcopy(adata)
    ## Check whether adata.X is sparse or dense
    if sparse.issparse(adata_copy.X):
        adata_copy.X = adata_copy.X.todense()
    sum_1 = np.sum(np.array(adata_copy.X), axis=0)
    ## Check whether sum_1 is sparse or dense
    if sparse.issparse(adata_copy.X):
        genenames = genenames[np.sum(np.array(adata_copy.X.todense()), axis=0) > 0]
    else:
        genenames = genenames[np.sum(np.array(adata_copy.X), axis=0) > 0]
    ## Load the ligand–receptor pairs from the database
    db_ligand_receptor = pd.read_csv(ligand_receptor_filedir, header=0)
    ## Remove rows that contain NaN values
    db_ligand_receptor = db_ligand_receptor.dropna(axis=0, how='any')
    # Keep ligand-receptor pairs present in the gene set
    # Example alternative filtering approach
    genenames_set = set(genenames)  # Convert gene names to a set for fast lookup

    # Preprocess ligands and receptors: split comma-separated strings into lists
    db_ligand_receptor['ligand_list'] = db_ligand_receptor['ligand'].apply(lambda x: x.split(','))
    db_ligand_receptor['receptor_list'] = db_ligand_receptor['receptor'].apply(lambda x: x.split(','))

    # Check whether all ligands and receptors are present in genenames_set
    def is_valid_lr(ligand_list, receptor_list):
        return all(ligand in genenames_set for ligand in ligand_list) and all(receptor in genenames_set for receptor in receptor_list)

    LR_choose = db_ligand_receptor[
        db_ligand_receptor.apply(lambda row: is_valid_lr(row['ligand_list'], row['receptor_list']), axis=1)
    ][['ligand_list', 'receptor_list']].values.tolist()

    LR_choose_unique = list(set(tuple(map(tuple, pair)) for pair in LR_choose))
    ##
    print("Number of ligand-receptor pairs from scSeqComm: " + str(len(LR_choose_unique)))
    return LR_choose_unique

def scatter_nanmean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
    """
    scatter_nanmean: Similar to scatter_mean, but ignores NaN values and computes the nanmean.
    """
    # mask: valid values = 1, NaN = 0
    mask = (~torch.isnan(src)).float()
    src_no_nan = torch.nan_to_num(src, nan=0.0)

    # Numerator: sum
    num = scatter_add(src_no_nan, index, dim=dim, dim_size=dim_size)
    # Denominator: number of valid elements
    den = scatter_add(mask, index, dim=dim, dim_size=dim_size)

    # nanmean
    out = num / den.clamp(min=1)
    out[den == 0] = float("nan")  # If all elements in a group are NaN, set the result to NaN
    return out


from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_ROOT / "resources"

def _normalize_species(species: str) -> str:
    species = species.strip().lower()
    alias_map = {
        "human": "human",
        "homo_sapiens": "human",
        "mouse": "mouse",
        "mus_musculus": "mouse",
    }
    if species not in alias_map:
        raise ValueError(
            f"Unsupported species: {species}. Choose from ['human', 'mouse']."
        )
    return alias_map[species]

def get_default_cellchat_db(species: str = "human") -> Path:
    species = _normalize_species(species)
    filename_map = {
        "human": "Human_LR_pairs_Cellchatdb.csv",
        "mouse": "Mouse_LR_pairs_Cellchatdb.csv",
    }
    path = RESOURCE_DIR / filename_map[species]
    if not path.exists():
        raise FileNotFoundError(f"Default CellChat DB not found: {path}")
    return path

def get_default_scseqcomm_db(species: str = "human") -> Path:
    species = _normalize_species(species)
    filename_map = {
        "human": "Human_LR_pairs_scSeqComm.csv",
        "mouse": "Mouse_LR_pairs_scSeqComm.csv",
    }
    path = RESOURCE_DIR / filename_map[species]
    if not path.exists():
        raise FileNotFoundError(f"Default scSeqComm DB not found: {path}")
    return path