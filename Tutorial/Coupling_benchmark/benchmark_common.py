"""Shared loading, curated-program scoring and table helpers.

Used by the coupling/directionality notebook. Its aggregation and LR-name
handling remain in the notebook alongside the analysis controls.
"""

import os
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, spearmanr, t
from sklearn.decomposition import NMF

def build_nmflr_factors(SpiderNet_data_pyg_list, n_components, random_state=0, max_iter=1000):
    Factor_LR_list = []
    for slice_index in range(len(SpiderNet_data_pyg_list)):
        print(f"Running NMF-LR for slice {slice_index + 1}/{len(SpiderNet_data_pyg_list)}")
        cellpair_LRpair_neigh_cur = SpiderNet_data_pyg_list[slice_index]["cellpair_LRpair_neigh"]

        nmf_LR = NMF(
            n_components=n_components,
            init="nndsvda",
            random_state=random_state,
            max_iter=max_iter,
        )
        factor_nmf_LR = nmf_LR.fit_transform(cellpair_LRpair_neigh_cur)

        factor_nmf_LR_colmax = np.max(factor_nmf_LR, axis=0)
        factor_nmf_LR_colmax[factor_nmf_LR_colmax == 0] = 1.0
        factor_nmf_LR = factor_nmf_LR / factor_nmf_LR_colmax

        Factor_LR_list.append(factor_nmf_LR.astype(np.float32, copy=False))
    return Factor_LR_list


def edge_matrix_from_square(square_mat, rows, cols):
    if sp.issparse(square_mat):
        square_mat = square_mat.tocsr()
        return square_mat[rows, cols].A1.astype(np.float32, copy=False)
    return square_mat[rows, cols].astype(np.float32, copy=False)


def one_sided_mw_greater(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if (x.size == 0) or (y.size == 0):
        return np.nan
    return mannwhitneyu(x, y, alternative="greater").pvalue


def get_commot_file_for_slice(adata_sub, cfg):
    if cfg["dataset_name"] == "HGSOC":
        sample_cur = np.unique(adata_sub.obs[cfg["sample_obs_key"]])[0]
        return os.path.join(cfg["COMMOT_path_main"], f"{sample_cur}_COMMOT_cellchat.h5ad")
    elif cfg["dataset_name"] == "AgingMousebrain":
        age_cur = np.unique(adata_sub.obs[cfg["sample_obs_key"]])[0]
        return os.path.join(cfg["COMMOT_path_main"], f"aging_coronal_age{age_cur}_commot.h5ad")
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset_name']}")


def load_commot_outputs(adata_list, SpiderNet_data_pyg_list, cfg):
    COMMOT_score_list = []

    for slice_index, adata_sub in enumerate(adata_list):
        print(f"Loading COMMOT for slice {slice_index + 1}/{len(adata_list)}")
        commot_path = get_commot_file_for_slice(adata_sub, cfg)
        COMMOT_adata = sc.read_h5ad(commot_path)

        edge_index_cur = SpiderNet_data_pyg_list[slice_index]["edge_index"]
        edge_np = edge_index_cur.detach().cpu().numpy()
        rows = edge_np[:, 0].astype(np.int64, copy=False)
        cols = edge_np[:, 1].astype(np.int64, copy=False)
        E = rows.shape[0]

        keys = list(COMMOT_adata.obsp.keys())
        total_key = "commot-cellchat-total-total"
        pathway_keys = [
            k for k in keys
            if k.startswith("commot-cellchat-") and len(k.split("-")) == 3 and k != total_key
        ]

        # Load pathway-level COMMOT scores for each edge
        pathway_array = np.zeros((E, len(pathway_keys)), dtype=np.float32)
        for j, key in enumerate(pathway_keys):
            pathway_array[:, j] = edge_matrix_from_square(COMMOT_adata.obsp[key], rows, cols)
        COMMOT_score_list.append(pathway_array)

    return COMMOT_score_list


def load_sccchain_outputs(adata_list, SpiderNet_data_pyg_list, cfg):
    txt_path = os.path.join(cfg["ScCChain_path_main"], "h5ad_files.txt")
    adata_list_ScCChain = pd.read_csv(txt_path, header=None)[0].tolist()

    n_obs_ref = [adata_sub.n_obs for adata_sub in adata_list]
    n_obs_scc = []
    for p in adata_list_ScCChain:
        ad = sc.read_h5ad(p)
        n_obs_scc.append(ad.n_obs)

    index_map = []
    for n in n_obs_ref:
        matched = np.where(np.array(n_obs_scc) == n)[0]
        if len(matched) == 0:
            raise ValueError(f"Cannot match ScCChain result by n_obs={n}")
        index_map.append(matched[0])

    ScCChain_CPscore_list = []
    for slice_index in range(len(adata_list)):
        print(f"Loading ScCChain for slice {slice_index + 1}/{len(adata_list)}")
        matched_idx = index_map[slice_index]

        src_path = adata_list_ScCChain[matched_idx]
        stem = os.path.splitext(os.path.basename(src_path))[0]
        score_path = os.path.join(cfg["ScCChain_path_main"], f"{stem}_ScCChain_edge_program_scores.csv")
        score_df = pd.read_csv(score_path, index_col=None)

        edge_index_cur = SpiderNet_data_pyg_list[slice_index]["edge_index"]
        edge_np = edge_index_cur.detach().cpu().numpy()
        rows = edge_np[:, 0].astype(np.int64, copy=False)
        cols = edge_np[:, 1].astype(np.int64, copy=False)
        E = rows.shape[0]

        key_use = score_df.columns.tolist()[2:]
        cellpair_CPscore_array = np.zeros((E, len(key_use)), dtype=np.float32)

        for j, cp_key in enumerate(key_use):
            sub_df = score_df[["sender_index", "receiver_index", cp_key]].copy()
            sender_index_cur = sub_df["sender_index"].values.astype(np.int64) - 1
            receiver_index_cur = sub_df["receiver_index"].values.astype(np.int64) - 1

            A = np.zeros((adata_list[slice_index].n_obs, adata_list[slice_index].n_obs), dtype=np.float32)
            A[sender_index_cur, receiver_index_cur] = sub_df[cp_key].values.astype(np.float32)
            cellpair_CPscore_array[:, j] = edge_matrix_from_square(A, rows, cols)

        ScCChain_CPscore_list.append(cellpair_CPscore_array)

    return ScCChain_CPscore_list


def load_spacia_outputs(adata_list, SpiderNet_data_pyg_list, cfg):
    if not cfg["include_spacia"]:
        return None

    spacia_dir = os.path.join(cfg["Spacia_path_main"], "spacia_outputs")
    adata_list_Spacia = sorted([f for f in os.listdir(spacia_dir) if f.endswith(".h5ad")])

    n_obs_ref = [adata_sub.n_obs for adata_sub in adata_list]
    n_obs_spacia = []
    for fname in adata_list_Spacia:
        ad = sc.read_h5ad(os.path.join(spacia_dir, fname), backed="r")
        n_obs_spacia.append(ad.n_obs)
        del ad

    index_map = []
    for n in n_obs_ref:
        matched = np.where(np.array(n_obs_spacia) == n)[0]
        if len(matched) == 0:
            raise ValueError(f"Cannot match Spacia result by n_obs={n}")
        index_map.append(matched[0])

    Spacia_CP_list = []
    for slice_index in range(len(adata_list)):
        print(f"Loading Spacia for slice {slice_index + 1}/{len(adata_list)}")
        fname = adata_list_Spacia[index_map[slice_index]]
        spacia_path = os.path.join(spacia_dir, fname)
        Spacia_adata = sc.read_h5ad(spacia_path)

        edge_index_cur = SpiderNet_data_pyg_list[slice_index]["edge_index"]
        edge_np = edge_index_cur.detach().cpu().numpy()
        rows = edge_np[:, 0].astype(np.int64, copy=False)
        cols = edge_np[:, 1].astype(np.int64, copy=False)
        E = rows.shape[0]

        n_programs = Spacia_adata.obsp["interaction_scores"].shape[2]
        cellpair_CP_array = np.zeros((E, n_programs), dtype=np.float32)

        for j in range(n_programs):
            A = Spacia_adata.obsp["interaction_scores"][:, :, j]
            cellpair_CP_array[:, j] = edge_matrix_from_square(A, rows, cols)

        Spacia_CP_list.append(cellpair_CP_array)

    return Spacia_CP_list


def dedup_gene_sets(gene_sets_by_lr):
    gene_sets_by_setid = OrderedDict()
    repkey_by_setid = {}
    setid_by_lr = {}
    lrs_by_setid = defaultdict(list)

    canon_to_setid = {}
    set_counter = 0

    for lr, genes in gene_sets_by_lr.items():
        genes_can = sorted({g for g in genes if g is not None and g != ""})
        canon = ";".join(genes_can)

        if canon not in canon_to_setid:
            set_counter += 1
            set_id = f"set{set_counter:06d}"
            canon_to_setid[canon] = set_id
            gene_sets_by_setid[set_id] = genes_can
            repkey_by_setid[set_id] = lr

        set_id = canon_to_setid[canon]
        setid_by_lr[lr] = set_id
        lrs_by_setid[set_id].append(lr)

    return gene_sets_by_setid, repkey_by_setid, setid_by_lr, dict(lrs_by_setid)


def cell_by_pair_mean_genesets(
    adata_sub,
    genesets_by_LR_unique,
    use_layer=None,
    dtype=np.float32,
    fill_value=np.nan,
    return_dataframe=True,
):
    X = adata_sub.X if use_layer is None else adata_sub.layers[use_layer]
    if not sp.issparse(X):
        X = np.asarray(X)

    var_names = np.asarray(adata_sub.var_names)
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    pair_names = list(genesets_by_LR_unique.keys())
    n_pairs = len(pair_names)

    rows, cols, data = [], [], []
    n_found = np.zeros(n_pairs, dtype=np.int32)

    for j, lr in enumerate(pair_names):
        genes = genesets_by_LR_unique[lr]
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(idx) == 0:
            continue
        idx = np.unique(idx)
        n_found[j] = len(idx)
        w = 1.0 / float(len(idx))
        rows.extend(idx.tolist())
        cols.extend([j] * len(idx))
        data.extend([w] * len(idx))

    W = sp.csr_matrix((data, (rows, cols)), shape=(adata_sub.n_vars, n_pairs), dtype=dtype)

    if sp.issparse(X):
        scores = (X @ W).toarray().astype(dtype, copy=False)
    else:
        scores = (X @ W.toarray()).astype(dtype, copy=False)

    zero_mask = n_found == 0
    if np.any(zero_mask):
        scores[:, zero_mask] = fill_value

    if return_dataframe:
        scores = pd.DataFrame(scores, index=adata_sub.obs_names, columns=pair_names)

    return scores, pair_names, n_found


def compute_rowmax_spearman_df(X_df, Y_array, slice_index, prefix="Slice_"):
    X = np.asarray(X_df, dtype=float)
    Y = np.asarray(Y_array, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X and Y must both be 2D, got X.shape={X.shape}, Y.shape={Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same n_cells, got {X.shape} vs {Y.shape}")

    if X.shape[1] == 0:
        return pd.DataFrame(index=[f"{prefix}{slice_index + 1}"])
    if Y.shape[1] == 0:
        return pd.DataFrame(np.nan, index=[f"{prefix}{slice_index + 1}"], columns=X_df.columns)

    X = np.nan_to_num(X, nan=0.0)
    Y = np.nan_to_num(Y, nan=0.0)

    corr_all, _ = spearmanr(X, Y, axis=0)
    corr_xy = corr_all[:X.shape[1], X.shape[1]:]
    rowmax = np.nanmax(corr_xy, axis=1)

    return pd.DataFrame(
        [rowmax],
        index=[f"{prefix}{slice_index + 1}"],
        columns=X_df.columns,
    )


def corr_frames_to_long(frame_dict, feature_type):
    recs = []
    for method_name, df in frame_dict.items():
        tmp = df.copy()
        tmp["Slice"] = tmp.index.astype(str)
        tmp = tmp.melt(id_vars=["Slice"], var_name="RepresentativeLR", value_name="Correlation")
        tmp["Method"] = method_name
        tmp["FeatureType"] = feature_type
        recs.append(tmp)

    out = pd.concat(recs, axis=0, ignore_index=True)
    out["Correlation"] = pd.to_numeric(out["Correlation"], errors="coerce")
    return out


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    m = float(np.mean(x))
    if n == 1:
        return m, m, m, 1

    sem = np.std(x, ddof=1) / np.sqrt(n)
    t_crit = t.ppf(0.975, df=n - 1)
    half_width = t_crit * sem
    return m, float(m - half_width), float(m + half_width), int(n)


def make_benchmark1_tables(target_corr_frames, regulator_corr_frames):
    long_df = pd.concat(
        [
            corr_frames_to_long(target_corr_frames, "Receiver targets"),
            corr_frames_to_long(regulator_corr_frames, "Sender regulators"),
        ],
        axis=0,
        ignore_index=True,
    )

    method_order = [
        m for m in ["SpiderNet", "NMF-LR", "COMMOT", "ScCChain", "Spacia"]
        if m in long_df["Method"].unique()
    ]

    plot_summary_records = []
    stats_records = []

    for feature_type in ["Receiver targets", "Sender regulators"]:
        sub_ft = long_df[long_df["FeatureType"] == feature_type].copy()

        rep_lr_order = (
            sub_ft[sub_ft["Method"] == "SpiderNet"]
            .groupby("RepresentativeLR")["Correlation"]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        for rep_lr in rep_lr_order:
            spider_vals = sub_ft[
                (sub_ft["RepresentativeLR"] == rep_lr) &
                (sub_ft["Method"] == "SpiderNet")
            ]["Correlation"].dropna().to_numpy()

            for method in method_order:
                vals = sub_ft[
                    (sub_ft["RepresentativeLR"] == rep_lr) &
                    (sub_ft["Method"] == method)
                ]["Correlation"].dropna().to_numpy()

                mean_v, ci_low, ci_high, n = mean_ci95(vals)
                median_v = float(np.median(vals)) if vals.size > 0 else np.nan

                plot_summary_records.append({
                    "FeatureType": feature_type,
                    "RepresentativeLR": rep_lr,
                    "Method": method,
                    "mean": mean_v,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "median": median_v,
                    "n": n,
                })

                if method != "SpiderNet":
                    stats_records.append({
                        "FeatureType": feature_type,
                        "RepresentativeLR": rep_lr,
                        "Comparison": f"SpiderNet > {method}",
                        "Method": method,
                        "p_one_sided_mw": one_sided_mw_greater(spider_vals, vals),
                        "n_SpiderNet": int(spider_vals.size),
                        "n_other": int(vals.size),
                        "median_SpiderNet": float(np.median(spider_vals)) if spider_vals.size > 0 else np.nan,
                        "median_other": median_v,
                    })

    plot_summary_df = pd.DataFrame(plot_summary_records)
    stats_df = pd.DataFrame(stats_records)

    return long_df, plot_summary_df, stats_df, method_order
