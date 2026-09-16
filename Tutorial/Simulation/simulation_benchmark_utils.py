from __future__ import annotations

import os
from pathlib import Path
import hashlib
import json
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd
import scanpy as sc
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
    return Path(os.environ["SIMULATION_DATA_ROOT"]) if "SIMULATION_DATA_ROOT" in os.environ else _resolve_first_existing_path(DATA_ROOT_CANDIDATES)


def resolve_result_root() -> Path:
    return Path(os.environ["SIMULATION_RESULT_ROOT"]) if "SIMULATION_RESULT_ROOT" in os.environ else _resolve_first_existing_path(RESULT_ROOT_CANDIDATES)


def resolve_spacia_analysis_root() -> Path:
    return Path(os.environ["SIMULATION_SPACIA_ROOT"]) if "SIMULATION_SPACIA_ROOT" in os.environ else _resolve_first_existing_path(SPACIA_ANALYSIS_ROOT_CANDIDATES)


DATA_ROOT = resolve_data_root()
RESULT_ROOT = resolve_result_root()
SPACIA_ANALYSIS_ROOT = resolve_spacia_analysis_root()

SPIDERNET_RESULT_ROOT = RESULT_ROOT / "SpiderNet"
COMMOT_RESULT_ROOT = RESULT_ROOT / "COMMOT"
NMF_LR_RESULT_ROOT = RESULT_ROOT / "NMF_LR"
SCCCHAIN_RESULT_ROOT = RESULT_ROOT / "ScCChain"
SPACIA_RESULT_ROOT = RESULT_ROOT / "Spacia"

SCCCHAIN_ANALYSIS_ROOT = RESULT_ROOT / "ScCChain_analysis"
OUTPUT_ROOT = Path(os.environ.get("SIMULATION_OUTPUT_ROOT", str(Path(__file__).resolve().parent / "output")))
MERGED_RESULT_ROOT = OUTPUT_ROOT / "Merged_Benchmark"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def insitu_cache_provenance(payload, parameters):
    """Record exact inputs, analysis source, options and numerical environment."""
    source_dir = Path(__file__).resolve().parent
    inputs = {name: _sha256(Path(payload["data_dir"]) / name) for name in (
        "gene_exp.csv", "cell_metadf.csv", "gene_metadf.csv", "edge_metadf.csv",
        "spatial_location.csv", "adata_simulation.h5ad",
    )}
    inputs["SpiderNet/EdgeProgramScores.csv"] = _sha256(
        get_local_result_dir("SpiderNet", payload["setting_name"], payload["experiment_idx"])
        / "EdgeProgramScores.csv"
    )
    if parameters["load_compatible_spacia"]:
        sample = f'{payload["setting_name"]}_Experiment_{payload["experiment_idx"]}'
        folder = SPACIA_ANALYSIS_ROOT / payload["setting_name"] / f'Experiment_{payload["experiment_idx"]}'
        for suffix in ("reference_edge_index.csv", "Spacia_edge_scores.npy"):
            path = folder / f"{sample}_{suffix}"
            inputs[f"Spacia/{suffix}"] = _sha256(path) if path.is_file() else None
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "commot", "scanpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "format": 1, "parameters": parameters, "inputs_sha256": inputs,
        "source_sha256": {name: _sha256(source_dir / name) for name in (
            "simulation_benchmark_utils.py", "Simulation_Benchmark_InSitu_Comparison.ipynb",
            "ScCChain_runner.jl",
        )},
        "versions": versions,
    }


def save_insitu_cache(cache_path, payload, parameters, scores):
    """Persist full precision scores only after the requested computations succeed."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_pairs = np.asarray(payload["edge_index"], dtype=np.int64)
    for method, table in scores.items():
        if not np.array_equal(table[["sender_index", "receiver_index"]].to_numpy(), expected_pairs):
            raise ValueError(f"Cannot cache {method}: its reference graph differs.")
    np.savez_compressed(cache_path, edge_index=expected_pairs,
                        **{name: table[["MI-1", "MI-2"]].to_numpy(dtype=float)
                           for name, table in scores.items()})
    provenance = insitu_cache_provenance(payload, parameters)
    provenance["methods"] = list(scores)
    provenance["score_sources"] = {name: dict(table.attrs) for name, table in scores.items()}
    provenance["scores_sha256"] = _sha256(cache_path)
    cache_path.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def load_insitu_cache(cache_path, payload, parameters):
    """Fail on absent/stale cache; plot-only never substitutes other method scores."""
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing verified in-situ cache: {cache_path}. Run --stage insitu first.")
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    current = insitu_cache_provenance(payload, parameters)
    changed = [key for key, value in current.items() if saved.get(key) != value]
    if saved.get("scores_sha256") != _sha256(cache_path):
        changed.append("scores_sha256")
    if changed:
        raise ValueError(f"In-situ cache is stale ({', '.join(changed)}); run --stage insitu explicitly.")
    scores = {}
    with np.load(cache_path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["edge_index"], payload["edge_index"]):
            raise ValueError("Cached edge order differs from the selected data.")
        for name in saved["methods"]:
            values = arrays[name]
            if values.shape != (len(payload["edge_index"]), 2):
                raise ValueError(f"Invalid cached score shape for {name}: {values.shape}")
            table = pd.DataFrame(payload["edge_index"], columns=["sender_index", "receiver_index"])
            table[["MI-1", "MI-2"]] = values
            table.attrs.update(saved["score_sources"].get(name, {}))
            scores[name] = table
    print(f"Plot-only: loaded verified scores from {cache_path}")
    return scores

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


def read_edge_scores_csv(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required_cols = ["sender_index", "receiver_index", "MI-1", "MI-2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return _coerce_edge_score_df(df)
