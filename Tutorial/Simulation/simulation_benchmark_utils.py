from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# Shared configuration (aligned with Simulation_Benchmark_Pipeline)
# ------------------------------------------------------------
SETTING_LIST = [
    "Dropout1", "Dropout2", "Dropout3",
    "Sd_use1", "Sd_use2", "Sd_use3",
]


DATA_ROOT_CANDIDATES = [
    Path(r"D:/SpiderNet/Data/Simulation"),
    Path(r"D:/SpiderNet/Simulation/Data"),
]
RESULT_ROOT_CANDIDATES = [
    Path(r"D:/SpiderNet/Results/Simulation"),
]
SPACIA_ANALYSIS_ROOT_CANDIDATES = [
    Path(r"E:/project_SpiderNet/Simulation/Spacia_analysis"),
    Path(r"D:/SpiderNet/Results/Simulation/Spacia_analysis"),
]


def _resolve_first_existing_path(candidates):
    for path in candidates:
        path = Path(path)
        if path.exists():
            return path
    return Path(candidates[0])


def resolve_data_root() -> Path:
    return _resolve_first_existing_path(DATA_ROOT_CANDIDATES)


def resolve_result_root() -> Path:
    return _resolve_first_existing_path(RESULT_ROOT_CANDIDATES)


def resolve_spacia_analysis_root() -> Path:
    return _resolve_first_existing_path(SPACIA_ANALYSIS_ROOT_CANDIDATES)


DATA_ROOT = resolve_data_root()
RESULT_ROOT = resolve_result_root()
SPACIA_ANALYSIS_ROOT = resolve_spacia_analysis_root()

SPIDERNET_RESULT_ROOT = RESULT_ROOT / "SpiderNet"
COMMOT_RESULT_ROOT = RESULT_ROOT / "COMMOT"
NMF_LR_RESULT_ROOT = RESULT_ROOT / "NMF_LR"
SCCCHAIN_RESULT_ROOT = RESULT_ROOT / "ScCChain"
SPACIA_RESULT_ROOT = RESULT_ROOT / "Spacia"

SCCCHAIN_ANALYSIS_ROOT = RESULT_ROOT / "ScCChain_analysis"
MERGED_RESULT_ROOT = RESULT_ROOT / "Merged_Benchmark"

METHOD_ORDER = ["SpiderNet", "COMMOT", "Spacia", "NMF-LR", "ScCChain"]
METHOD_DISPLAY_NAMES = {
    "SpiderNet": "SpiderNet",
    "COMMOT": "COMMOT",
    "NMF_LR": "NMF-LR",
    "NMF-LR": "NMF-LR",
    "ScCChain": "ScCChain",
    "Spacia": "Spacia",
}

LOCAL_RESULT_SUBDIRS = {
    "SpiderNet": "SpiderNet_Result_Mode_cell_class",
    "COMMOT": "COMMOT_Result",
    "NMF-LR": "NMF_LR_Result",
}


def resolve_spacia_pipeline_metrics_path(setting_name: str, experiment_idx: int) -> Path:
    exp_folder = experiment_name_from_index(experiment_idx)
    sample_name = f"{setting_name}_{exp_folder}"
    return SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / f"{sample_name}_metrics.csv"


def _coerce_edge_score_df(df):
    df = df.copy()
    required_cols = ["sender_index", "receiver_index", "MI-1", "MI-2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required edge-score columns: {missing}")

    for col in ["sender_index", "receiver_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["MI-1"] = pd.to_numeric(df["MI-1"], errors="coerce")
    df["MI-2"] = pd.to_numeric(df["MI-2"], errors="coerce")
    df = df.dropna(subset=["sender_index", "receiver_index"]).copy()
    df["sender_index"] = df["sender_index"].astype(np.int64)
    df["receiver_index"] = df["receiver_index"].astype(np.int64)
    df[["MI-1", "MI-2"]] = df[["MI-1", "MI-2"]].fillna(0.0)
    return df


def _align_edge_scores_by_pairs(df, reference_pairs):
    df = _coerce_edge_score_df(df)
    df = df.drop_duplicates(subset=["sender_index", "receiver_index"], keep="first").copy()
    merged = reference_pairs.merge(
        df[["sender_index", "receiver_index", "MI-1", "MI-2"]],
        on=["sender_index", "receiver_index"],
        how="left",
    )
    if merged[["MI-1", "MI-2"]].isna().any().any():
        return None
    merged[["MI-1", "MI-2"]] = merged[["MI-1", "MI-2"]].astype(float)
    return merged


def align_edge_scores_to_reference(edge_scores_df, edge_index, method_name="method", verbose=False):
    """
    Standardize an edge-score table to the reference simulation edge order.

    This prevents spatial plots from being distorted when a CSV stores
    sender/receiver indices in a different convention than the benchmark
    pipeline, while preserving the score columns used for benchmarking.
    """
    reference_pairs = pd.DataFrame(edge_index, columns=["sender_index", "receiver_index"]).copy()
    raw_df = _coerce_edge_score_df(edge_scores_df)

    strategies = [("direct_pair_match", raw_df)]

    if raw_df[["sender_index", "receiver_index"]].min().min() >= 1:
        shifted = raw_df.copy()
        shifted["sender_index"] = shifted["sender_index"] - 1
        shifted["receiver_index"] = shifted["receiver_index"] - 1
        strategies.append(("one_based_to_zero_based", shifted))

    swapped = raw_df.rename(
        columns={"sender_index": "receiver_index", "receiver_index": "sender_index"}
    )[["sender_index", "receiver_index", "MI-1", "MI-2"]].copy()
    strategies.append(("swapped_direction", swapped))

    if raw_df[["sender_index", "receiver_index"]].min().min() >= 1:
        swapped_shifted = swapped.copy()
        swapped_shifted["sender_index"] = swapped_shifted["sender_index"] - 1
        swapped_shifted["receiver_index"] = swapped_shifted["receiver_index"] - 1
        strategies.append(("swapped_direction_one_based", swapped_shifted))

    for strategy_name, candidate in strategies:
        aligned = _align_edge_scores_by_pairs(candidate, reference_pairs)
        if aligned is not None:
            aligned.attrs["alignment_strategy"] = strategy_name
            aligned.attrs["alignment_note"] = (
                f"{method_name}: edge indices aligned to reference via {strategy_name}."
            )
            if verbose:
                print(aligned.attrs["alignment_note"])
            return aligned

    if raw_df.shape[0] == reference_pairs.shape[0]:
        aligned = reference_pairs.copy()
        aligned[["MI-1", "MI-2"]] = raw_df[["MI-1", "MI-2"]].to_numpy(dtype=float, copy=False)
        aligned.attrs["alignment_strategy"] = "row_order_fallback"
        aligned.attrs["alignment_note"] = (
            f"{method_name}: sender/receiver columns did not match the reference edge list, "
            f"so the notebook falls back to the benchmark row order."
        )
        if verbose:
            print(aligned.attrs["alignment_note"])
        return aligned

    raise ValueError(
        f"{method_name}: could not align edge scores to the reference graph. "
        f"CSV rows = {raw_df.shape[0]}, reference edges = {reference_pairs.shape[0]}."
    )


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setting_name_from_index(setting_idx: int) -> str:
    return SETTING_LIST[setting_idx]


def experiment_name_from_index(experiment_idx: int) -> str:
    return f"Experiment_{experiment_idx}"


def get_local_result_dir(method_name: str, setting_name: str, experiment_idx: int) -> Path:
    exp_folder = experiment_name_from_index(experiment_idx)
    root_map = {
        "SpiderNet": SPIDERNET_RESULT_ROOT,
        "COMMOT": COMMOT_RESULT_ROOT,
        "NMF-LR": NMF_LR_RESULT_ROOT,
    }
    if method_name not in root_map:
        raise KeyError(f"Unsupported local-result method: {method_name}")
    return root_map[method_name] / setting_name / exp_folder / LOCAL_RESULT_SUBDIRS[method_name]


def spatial_neighborindex_generation(cell_spatial, num_neighbor_available):
    spatialnn_use = NearestNeighbors(n_neighbors=num_neighbor_available + 1, algorithm="auto")
    spatialnn_use.fit(cell_spatial)
    _, spatial_neighborindex = spatialnn_use.kneighbors(cell_spatial)

    for i in range(spatial_neighborindex.shape[0]):
        if spatial_neighborindex[i, 0] != i:
            spatial_neighborindex[i, :] = np.hstack((i, np.setdiff1d(spatial_neighborindex[i, :], i)))

    spatial_neighborindex_further = spatial_neighborindex[:, 1:]
    centralindex = spatial_neighborindex[:, 0]
    edge_index = np.vstack(
        (
            np.repeat(centralindex, spatial_neighborindex_further.shape[1]),
            spatial_neighborindex_further.flatten(),
        )
    )
    return spatial_neighborindex, edge_index.T


def load_simulation_inputs(setting_idx: int, experiment_idx: int, num_neigh: int = 10):
    setting_name = setting_name_from_index(setting_idx)
    exp_folder = experiment_name_from_index(experiment_idx)
    data_root = resolve_data_root()
    data_dir = data_root / setting_name / exp_folder

    exp_nor_pd = pd.read_csv(data_dir / "gene_exp.csv", index_col=0)
    cellmeta_data = pd.read_csv(data_dir / "cell_metadf.csv", index_col=0)
    genemeta_data = pd.read_csv(data_dir / "gene_metadf.csv", index_col=0)
    edgemeta_data = pd.read_csv(data_dir / "edge_metadf.csv", index_col=0)
    spatial_location = pd.read_csv(data_dir / "spatial_location.csv", index_col=0).values

    _, edge_index = spatial_neighborindex_generation(spatial_location, num_neighbor_available=num_neigh)
    edge_index = edge_index.astype(np.int64)

    ligand_output = list(genemeta_data.index[np.where(genemeta_data["Gene_type"] == "Ligand")[0]])
    receptor_output = list(genemeta_data.index[np.where(genemeta_data["Gene_type"] == "Receptor")[0]])
    lr_list = np.vstack((ligand_output, receptor_output)).T

    gene_names = np.asarray(exp_nor_pd.columns)
    exp_mat = exp_nor_pd.values.astype(np.float32)
    cellpair_lr = np.zeros((edge_index.shape[0], lr_list.shape[0]), dtype=np.float32)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    sender_idx = edge_index[:, 0]
    receiver_idx = edge_index[:, 1]
    for lr_idx, (lig, rec) in enumerate(lr_list):
        lig_idx = gene_to_idx[lig]
        rec_idx = gene_to_idx[rec]
        lig_exp = exp_mat[:, lig_idx]
        rec_exp = exp_mat[:, rec_idx]
        cellpair_lr[:, lr_idx] = np.sqrt(lig_exp[sender_idx] * rec_exp[receiver_idx])

    return {
        "setting_name": setting_name,
        "experiment_idx": experiment_idx,
        "exp_folder": exp_folder,
        "data_dir": data_dir,
        "data_root": data_root,
        "expression": exp_mat,
        "gene_names": gene_names,
        "cell_names": np.asarray(exp_nor_pd.index),
        "cell_types": np.asarray(cellmeta_data["Celltype"]),
        "spatial_location": spatial_location.astype(np.float32),
        "edge_index": edge_index,
        "cellpair_lr": cellpair_lr,
        "edgemeta_data": edgemeta_data,
        "genemeta_data": genemeta_data,
    }


# ---------------------------
# Standardized score helpers
# ---------------------------

def safe_column_max_normalize(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        max_val = np.nanmax(arr) if arr.size > 0 else np.nan
        if (not np.isfinite(max_val)) or max_val <= 0:
            return np.zeros_like(arr, dtype=float)
        out = arr / max_val
        out[~np.isfinite(out)] = 0.0
        return out

    out = arr.copy()
    if out.size == 0:
        return out
    col_max = np.nanmax(out, axis=0)
    col_max[(~np.isfinite(col_max)) | (col_max <= 0)] = 1.0
    out = out / col_max
    out[~np.isfinite(out)] = 0.0
    return out


def safe_row_normalize(arr):
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


def get_best_component_scores(factor_matrix, edge_MI1, edge_MI2, mi_index):
    factor_matrix = np.asarray(factor_matrix, dtype=float)
    if factor_matrix.ndim == 1:
        return safe_column_max_normalize(factor_matrix)
    if factor_matrix.shape[1] == 0:
        return np.zeros(factor_matrix.shape[0], dtype=float)

    ref_edges = edge_MI1 if mi_index == 0 else edge_MI2
    best_idx = int(np.argmax(np.mean(factor_matrix[ref_edges, :], axis=0)))
    scores = factor_matrix[:, best_idx]
    return safe_column_max_normalize(scores)


def safe_binary_metrics(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)

    if y_true.size == 0 or np.unique(y_true).shape[0] < 2:
        return np.nan, np.nan

    try:
        auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
    except Exception:
        auroc = np.nan

    try:
        auprc = sklearn.metrics.average_precision_score(y_true, y_score)
    except Exception:
        auprc = np.nan

    return auroc, auprc


def edge_factors_to_df(factor_matrix, edge_index, edgemeta_data):
    factor_matrix = np.asarray(factor_matrix, dtype=float)
    edge_MI1 = np.where(edgemeta_data["MetaItype"].values == "MI-1")[0]
    edge_MI2 = np.where(edgemeta_data["MetaItype"].values == "MI-2")[0]
    mi1_scores = get_best_component_scores(factor_matrix, edge_MI1=edge_MI1, edge_MI2=edge_MI2, mi_index=0)
    mi2_scores = get_best_component_scores(factor_matrix, edge_MI1=edge_MI1, edge_MI2=edge_MI2, mi_index=1)
    return pd.DataFrame(
        {
            "sender_index": edge_index[:, 0],
            "receiver_index": edge_index[:, 1],
            "MI-1": mi1_scores,
            "MI-2": mi2_scores,
        }
    )


def export_method_metrics(method_name, factor_matrix, edgemeta_data, edge_index, output_dir):
    output_dir = ensure_dir(output_dir)

    edge_MI1 = np.where(edgemeta_data["MetaItype"].values == "MI-1")[0]
    edge_MI2 = np.where(edgemeta_data["MetaItype"].values == "MI-2")[0]
    y_true_mi1 = (edgemeta_data["MetaItype"].values == "MI-1").astype(np.int32)
    y_true_mi2 = (edgemeta_data["MetaItype"].values == "MI-2").astype(np.int32)

    mi1_scores = get_best_component_scores(factor_matrix, edge_MI1=edge_MI1, edge_MI2=edge_MI2, mi_index=0)
    mi2_scores = get_best_component_scores(factor_matrix, edge_MI1=edge_MI1, edge_MI2=edge_MI2, mi_index=1)

    auroc_mi1, auprc_mi1 = safe_binary_metrics(y_true_mi1, mi1_scores)
    auroc_mi2, auprc_mi2 = safe_binary_metrics(y_true_mi2, mi2_scores)

    roc_df = pd.DataFrame([{"Method": method_name, "MI-1": auroc_mi1, "MI-2": auroc_mi2}])
    prc_df = pd.DataFrame([{"Method": method_name, "MI-1": auprc_mi1, "MI-2": auprc_mi2}])
    macro_df = pd.DataFrame(
        [
            {
                "Method": method_name,
                "Macro_AUROC": np.nanmean([auroc_mi1, auroc_mi2]),
                "Macro_AUPRC": np.nanmean([auprc_mi1, auprc_mi2]),
            }
        ]
    )
    edge_scores_df = pd.DataFrame(
        {
            "sender_index": edge_index[:, 0],
            "receiver_index": edge_index[:, 1],
            "MI-1": mi1_scores,
            "MI-2": mi2_scores,
        }
    )

    roc_df.to_csv(output_dir / "ROC_AUROC_all.csv", index=False)
    prc_df.to_csv(output_dir / "PUC_AUPRC_all.csv", index=False)
    macro_df.to_csv(output_dir / "Benchmark_MacroMetrics.csv", index=False)
    edge_scores_df.to_csv(output_dir / "EdgeProgramScores.csv", index=False)

    return {
        "roc": roc_df,
        "prc": prc_df,
        "macro": macro_df,
        "edge_scores": edge_scores_df,
    }


def export_loading_summary(method_name, output_dir, lr_loading=None, sender_loading=None, receiver_loading=None):
    output_dir = ensure_dir(output_dir)

    lr_ratio = np.nan
    sender_ratio = np.nan
    receiver_ratio = np.nan

    if lr_loading is not None:
        loading_lr = safe_row_normalize(np.asarray(lr_loading, dtype=float))
        n_lr = loading_lr.shape[1]
        lr_split = max(1, n_lr // 2)
        lr_idx_mi1 = np.arange(0, lr_split)
        lr_idx_mi2 = np.arange(lr_split, n_lr)
        lr_ratio = np.nanmean(
            [
                np.nansum(loading_lr[0, lr_idx_mi1]) if loading_lr.shape[0] >= 1 else np.nan,
                np.nansum(loading_lr[1, lr_idx_mi2]) if loading_lr.shape[0] >= 2 else np.nan,
            ]
        )

    if sender_loading is not None:
        loading_sender = safe_row_normalize(np.asarray(sender_loading, dtype=float))
        n_gene = loading_sender.shape[1]
        reg_mid = min(50, n_gene)
        reg_end = min(60, n_gene)
        sender_idx_mi1 = np.arange(min(40, n_gene), reg_mid)
        sender_idx_mi2 = np.arange(reg_mid, reg_end)
        sender_ratio = np.nanmean(
            [
                np.nansum(loading_sender[0, sender_idx_mi1])
                if (loading_sender.shape[0] >= 1 and sender_idx_mi1.size > 0)
                else np.nan,
                np.nansum(loading_sender[1, sender_idx_mi2])
                if (loading_sender.shape[0] >= 2 and sender_idx_mi2.size > 0)
                else np.nan,
            ]
        )

    if receiver_loading is not None:
        loading_receiver = safe_row_normalize(np.asarray(receiver_loading, dtype=float))
        n_gene = loading_receiver.shape[1]
        tar_start = min(60, n_gene)
        tar_mid = min(70, n_gene)
        tar_end = min(80, n_gene)
        receiver_idx_mi1 = np.arange(tar_start, tar_mid)
        receiver_idx_mi2 = np.arange(tar_mid, tar_end)
        receiver_ratio = np.nanmean(
            [
                np.nansum(loading_receiver[0, receiver_idx_mi1])
                if (loading_receiver.shape[0] >= 1 and receiver_idx_mi1.size > 0)
                else np.nan,
                np.nansum(loading_receiver[1, receiver_idx_mi2])
                if (loading_receiver.shape[0] >= 2 and receiver_idx_mi2.size > 0)
                else np.nan,
            ]
        )

    df = pd.DataFrame(
        [
            {
                "Method": method_name,
                "LR_loading_ratio": lr_ratio,
                "Sender_loading_ratio": sender_ratio,
                "Receiver_loading_ratio": receiver_ratio,
            }
        ]
    )
    df.to_csv(output_dir / "LoadingRank_Ratio_Summary.csv", index=False)
    return df


# ---------------------------
# Method-specific score loaders
# ---------------------------

def compute_nmf_lr_scores(cellpair_lr, n_components=2):
    nmf_lr = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=10000)
    factor = nmf_lr.fit_transform(np.asarray(cellpair_lr, dtype=float))
    loading = nmf_lr.components_
    factor = safe_column_max_normalize(factor)
    return factor, loading


def compute_commot_scores(expression, gene_names, cell_names, spatial_pos, edge_index_np):
    import commot as ct

    adata_commot = sc.AnnData(
        X=np.asarray(expression, dtype=np.float32),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    adata_commot.obsm["spatial"] = np.asarray(spatial_pos, dtype=np.float32)

    lr = np.array(
        [
            ["L1", "R1", "MI1_pathway"],
            ["L2", "R2", "MI1_pathway"],
            ["L3", "R3", "MI1_pathway"],
            ["L4", "R4", "MI1_pathway"],
            ["L5", "R5", "MI1_pathway"],
            ["L6", "R6", "MI1_pathway"],
            ["L7", "R7", "MI1_pathway"],
            ["L8", "R8", "MI1_pathway"],
            ["L9", "R9", "MI1_pathway"],
            ["L10", "R10", "MI2_pathway"],
            ["L11", "R11", "MI2_pathway"],
            ["L12", "R12", "MI2_pathway"],
            ["L13", "R13", "MI2_pathway"],
            ["L14", "R14", "MI2_pathway"],
            ["L15", "R15", "MI2_pathway"],
            ["L16", "R16", "MI2_pathway"],
            ["L17", "R17", "MI2_pathway"],
            ["L18", "R18", "MI2_pathway"],
            ["L19", "R19", "MI2_pathway"],
            ["L20", "R20", "MI2_pathway"],
        ],
        dtype=str,
    )
    df_ligrec = pd.DataFrame(lr)

    spatial_coords = np.asarray(spatial_pos, dtype=np.float32)
    n = spatial_coords.shape[0]
    k = 10
    if n <= k:
        raise ValueError("Not enough cells to build the COMMOT distance threshold.")

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(spatial_coords)
    dists, _ = nn.kneighbors(spatial_coords, return_distance=True)
    dists_k = dists[:, 1:]
    dist_vec = dists_k.mean(axis=1).astype(np.float32)
    dist_median = float(np.median(dist_vec[~np.isnan(dist_vec)]))

    ct.tl.spatial_communication(
        adata_commot,
        database_name="cellchat",
        df_ligrec=df_ligrec,
        dis_thr=dist_median,
        heteromeric=True,
        pathway_sum=True,
        cot_nitermax=2000,
    )

    factor = np.vstack(
        [
            adata_commot.obsp["commot-cellchat-MI1_pathway"].toarray()[edge_index_np[:, 0], edge_index_np[:, 1]],
            adata_commot.obsp["commot-cellchat-MI2_pathway"].toarray()[edge_index_np[:, 0], edge_index_np[:, 1]],
        ]
    ).T
    return safe_column_max_normalize(factor)


# ---------------------------
# External method readers
# ---------------------------

def read_external_macro_metrics(sample_metrics_csv, method_name, setting_name, experiment_idx):
    sample_metrics_csv = Path(sample_metrics_csv)
    if not sample_metrics_csv.is_file():
        return None
    df = pd.read_csv(sample_metrics_csv)
    if df.shape[0] == 0:
        return None
    row = df.iloc[0]
    return pd.DataFrame(
        [
            {
                "Method": method_name,
                "Macro_AUROC": row.get("Macro_AUROC", np.nan),
                "Macro_AUPRC": row.get("Macro_AUPRC", np.nan),
                "Setting": setting_name,
                "Experiment": experiment_idx,
            }
        ]
    )


def _glob_first(patterns):
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            return Path(matches[0])
    return None


def find_sccchain_csv(sample_result_root):
    sample_result_root = Path(sample_result_root)
    pattern = sample_result_root / "*_ScCChain_edge_program_scores.csv"
    files = sorted(sample_result_root.glob("*_ScCChain_edge_program_scores.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No ScCChain result file found in: {sample_result_root}")
    return files[0]


def load_sccchain_scores_to_reference_edges(sc_csv_path, edge_index):
    sc_df = pd.read_csv(sc_csv_path)
    sender_col = sc_df.columns[0]
    receiver_col = sc_df.columns[1]
    program_cols = sc_df.columns.tolist()[2:]

    sc_df = sc_df.rename(columns={sender_col: "sender_index", receiver_col: "receiver_index"}).copy()
    sc_df["sender_index"] = pd.to_numeric(sc_df["sender_index"], errors="coerce").astype("Int64")
    sc_df["receiver_index"] = pd.to_numeric(sc_df["receiver_index"], errors="coerce").astype("Int64")
    sc_df = sc_df.dropna(subset=["sender_index", "receiver_index"]).copy()
    sc_df["sender_index"] = sc_df["sender_index"].astype(np.int64) - 1
    sc_df["receiver_index"] = sc_df["receiver_index"].astype(np.int64) - 1

    edge_index_df = pd.DataFrame(edge_index, columns=["sender_index", "receiver_index"])
    edge_mi = pd.MultiIndex.from_frame(edge_index_df[["sender_index", "receiver_index"]])

    sc_df = sc_df.drop_duplicates(subset=["sender_index", "receiver_index"], keep="first").copy()
    sc_df_indexed = sc_df.set_index(["sender_index", "receiver_index"])[program_cols]
    aligned_scores = sc_df_indexed.reindex(edge_mi).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    return safe_column_max_normalize(aligned_scores)


# Backward-compatible alias for the earlier typo used in the notebook.
load_scchain_scores_to_reference_edges = load_sccchain_scores_to_reference_edges


def load_spacia_scores(spacia_h5ad_path, edge_index):
    adata = sc.read_h5ad(spacia_h5ad_path)

    interaction_scores = None
    if "interaction_scores" in getattr(adata, "obsp", {}):
        interaction_scores = adata.obsp["interaction_scores"]
    elif "interaction_scores" in getattr(adata, "uns", {}):
        interaction_scores = adata.uns["interaction_scores"]
    else:
        raise KeyError(
            f"'interaction_scores' was not found in either adata.obsp or adata.uns for: {spacia_h5ad_path}"
        )

    interaction_scores = np.asarray(interaction_scores)
    if interaction_scores.ndim != 3:
        raise ValueError(
            f"Expected Spacia interaction scores to be a 3D array, but got shape {interaction_scores.shape} "
            f"from: {spacia_h5ad_path}"
        )

    factor_dim1 = np.mean(interaction_scores[:, :, 0:10], axis=2)
    factor_dim2 = np.mean(interaction_scores[:, :, 10:20], axis=2)
    factor_dim1_long = factor_dim1[edge_index[:, 0], edge_index[:, 1]]
    factor_dim2_long = factor_dim2[edge_index[:, 0], edge_index[:, 1]]
    factor = np.vstack([factor_dim1_long, factor_dim2_long]).T
    return safe_column_max_normalize(factor)


def read_edge_scores_csv(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required_cols = ["sender_index", "receiver_index", "MI-1", "MI-2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return _coerce_edge_score_df(df)


def resolve_sccchain_score_csv(setting_name: str, experiment_idx: int) -> Path:
    exp_folder = experiment_name_from_index(experiment_idx)
    candidates = [
        SCCCHAIN_ANALYSIS_ROOT / setting_name / exp_folder,
        SCCCHAIN_RESULT_ROOT / setting_name / exp_folder,
    ]
    for base_dir in candidates:
        if base_dir.is_dir():
            try:
                return find_sccchain_csv(base_dir)
            except FileNotFoundError:
                pass

    patterns = [
        SCCCHAIN_ANALYSIS_ROOT / setting_name / exp_folder / "*_ScCChain_edge_program_scores.csv",
        SCCCHAIN_RESULT_ROOT / setting_name / exp_folder / "*_ScCChain_edge_program_scores.csv",
    ]
    out = _glob_first(patterns)
    if out is None:
        raise FileNotFoundError(
            f"Could not find ScCChain edge-program scores for {setting_name} / {exp_folder} "
            f"in either {SCCCHAIN_ANALYSIS_ROOT} or {SCCCHAIN_RESULT_ROOT}."
        )
    return out


def resolve_spacia_edge_scores_csv(setting_name: str, experiment_idx: int) -> Path | None:
    exp_folder = experiment_name_from_index(experiment_idx)
    sample_name = f"{setting_name}_{exp_folder}"

    direct_patterns = [
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / "EdgeProgramScores.csv",
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / f"{sample_name}_EdgeProgramScores.csv",
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / f"{sample_name}_edge_program_scores.csv",
        SPACIA_RESULT_ROOT / setting_name / exp_folder / "EdgeProgramScores.csv",
        SPACIA_RESULT_ROOT / setting_name / exp_folder / f"{sample_name}_EdgeProgramScores.csv",
        SPACIA_RESULT_ROOT / setting_name / exp_folder / f"{sample_name}_edge_program_scores.csv",
    ]
    for path in direct_patterns:
        if Path(path).is_file():
            return Path(path)

    glob_patterns = [
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / "*EdgeProgramScores*.csv",
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / "*edge_program_scores*.csv",
        SPACIA_RESULT_ROOT / setting_name / exp_folder / "*EdgeProgramScores*.csv",
        SPACIA_RESULT_ROOT / setting_name / exp_folder / "*edge_program_scores*.csv",
    ]
    out = _glob_first(glob_patterns)
    if out is not None:
        return out

    search_roots = [
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder,
        SPACIA_RESULT_ROOT / setting_name / exp_folder,
        SPACIA_RESULT_ROOT / f"{setting_name}_export",
        SPACIA_ANALYSIS_ROOT / f"{setting_name}_export",
        SPACIA_ANALYSIS_ROOT,
        SPACIA_RESULT_ROOT,
    ]
    recursive_name_filters = [
        f"*{sample_name}*EdgeProgramScores*.csv",
        f"*{sample_name}*edge_program_scores*.csv",
        f"*Experiment_{experiment_idx}*EdgeProgramScores*.csv",
        f"*Experiment_{experiment_idx}*edge_program_scores*.csv",
        "*Spacia*EdgeProgramScores*.csv",
        "*spacia*edge_program_scores*.csv",
    ]
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for pat in recursive_name_filters:
            matches = sorted(root.rglob(pat))
            if matches:
                return matches[0]
    return None


def resolve_spacia_h5ad_path(setting_name: str, experiment_idx: int) -> Path:
    exp_folder = experiment_name_from_index(experiment_idx)
    sample_name = f"{setting_name}_{exp_folder}"

    patterns = [
        SPACIA_RESULT_ROOT / f"{setting_name}_export" / f"Experiment_{experiment_idx}_compiled_spacia_output.h5ad",
        SPACIA_RESULT_ROOT / f"{setting_name}_export" / f"{sample_name}_compiled_spacia_output.h5ad",
        SPACIA_RESULT_ROOT / setting_name / exp_folder / "*compiled_spacia_output.h5ad",
        SPACIA_RESULT_ROOT / setting_name / exp_folder / "*.h5ad",
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / f"{sample_name}_compiled_spacia_output.h5ad",
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / "*compiled_spacia_output.h5ad",
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder / "*.h5ad",
        SPACIA_ANALYSIS_ROOT / f"{setting_name}_export" / f"Experiment_{experiment_idx}_compiled_spacia_output.h5ad",
        SPACIA_ANALYSIS_ROOT / f"{setting_name}_export" / f"{sample_name}_compiled_spacia_output.h5ad",
    ]
    out = _glob_first(patterns)
    if out is not None:
        return out

    search_roots = [
        SPACIA_ANALYSIS_ROOT / setting_name / exp_folder,
        SPACIA_RESULT_ROOT / setting_name / exp_folder,
        SPACIA_RESULT_ROOT / f"{setting_name}_export",
        SPACIA_ANALYSIS_ROOT / f"{setting_name}_export",
        SPACIA_ANALYSIS_ROOT,
        SPACIA_RESULT_ROOT,
    ]
    recursive_name_filters = [
        f"*{sample_name}*compiled*spacia*.h5ad",
        f"*{sample_name}*.h5ad",
        f"*Experiment_{experiment_idx}*compiled*spacia*.h5ad",
        f"*Experiment_{experiment_idx}*.h5ad",
        "*compiled_spacia_output*.h5ad",
        "*spacia*.h5ad",
    ]
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for pat in recursive_name_filters:
            matches = sorted(root.rglob(pat))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Could not find Spacia compiled output for {setting_name} / {exp_folder}. "
        f"Checked both the direct export-style paths used by the benchmark pipeline and broader recursive "
        f"searches under {SPACIA_RESULT_ROOT} and {SPACIA_ANALYSIS_ROOT}."
    )



def load_method_edge_scores_for_sample(setting_idx: int, experiment_idx: int, methods_to_show=None):
    payload = load_simulation_inputs(setting_idx, experiment_idx)
    setting_name = payload["setting_name"]
    edge_index = payload["edge_index"]
    edgemeta_data = payload["edgemeta_data"]

    if methods_to_show is None:
        methods_to_show = METHOD_ORDER.copy()

    method_edge_scores = {}

    local_csv_map = {
        "SpiderNet": get_local_result_dir("SpiderNet", setting_name, experiment_idx) / "EdgeProgramScores.csv",
        "COMMOT": get_local_result_dir("COMMOT", setting_name, experiment_idx) / "EdgeProgramScores.csv",
        "NMF-LR": get_local_result_dir("NMF-LR", setting_name, experiment_idx) / "EdgeProgramScores.csv",
    }
    for method_name, csv_path in local_csv_map.items():
        if method_name in methods_to_show and csv_path.is_file():
            local_df = read_edge_scores_csv(csv_path)
            if method_name == "SpiderNet":
                local_df = align_edge_scores_to_reference(
                    local_df,
                    edge_index=edge_index,
                    method_name=method_name,
                )
            else:
                local_df.attrs["alignment_strategy"] = "legacy_raw_csv"
                local_df.attrs["alignment_note"] = (
                    f"{method_name}: using the original notebook behavior "
                    f"(raw edge-score CSV without additional re-alignment)."
                )
            method_edge_scores[method_name] = local_df

    if "ScCChain" in methods_to_show:
        try:
            sc_csv = resolve_sccchain_score_csv(setting_name, experiment_idx)
            factor_sc = load_sccchain_scores_to_reference_edges(sc_csv, edge_index)
            sc_df = edge_factors_to_df(factor_sc, edge_index, edgemeta_data)
            sc_df.attrs["alignment_strategy"] = "legacy_reference_order"
            sc_df.attrs["alignment_note"] = (
                "ScCChain: using the original notebook behavior "
                "(scores reconstructed directly on the reference edge order)."
            )
            method_edge_scores["ScCChain"] = sc_df
        except Exception as e:
            print(f"[Warning] ScCChain could not be loaded: {e}")

    if "Spacia" in methods_to_show:
        try:
            spacia_edge_csv = resolve_spacia_edge_scores_csv(setting_name, experiment_idx)
            if spacia_edge_csv is not None:
                print(f"[Spacia] Using precomputed edge scores: {spacia_edge_csv}")
                sp_df = read_edge_scores_csv(spacia_edge_csv)
                sp_df.attrs["alignment_strategy"] = "legacy_raw_csv"
                sp_df.attrs["alignment_note"] = (
                    "Spacia: using the original notebook behavior "
                    "(precomputed edge-score CSV without additional re-alignment)."
                )
                method_edge_scores["Spacia"] = sp_df
            else:
                spacia_h5ad = resolve_spacia_h5ad_path(setting_name, experiment_idx)
                print(f"[Spacia] Using compiled output: {spacia_h5ad}")
                factor_sp = load_spacia_scores(spacia_h5ad, edge_index)
                sp_df = edge_factors_to_df(factor_sp, edge_index, edgemeta_data)
                sp_df.attrs["alignment_strategy"] = "legacy_reference_order"
                sp_df.attrs["alignment_note"] = (
                    "Spacia: using the original notebook behavior "
                    "(scores reconstructed directly on the reference edge order)."
                )
                method_edge_scores["Spacia"] = sp_df
        except Exception as e:
            print(f"[Warning] Spacia could not be loaded: {e}")

    return payload, method_edge_scores
