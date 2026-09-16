"""Benchmark within-cell-type communication-state concordance in two studies.

This local module reproduces the cell-similarity stage (Benchmark 2) of the
CCC coupling benchmark for the HGSOC CosMx SMI and ageing mouse brain MERFISH
studies. For each study, it loads precomputed SpiderNet and comparator outputs, aggregates
directed edge features into concatenated sender/receiver cell profiles, and
compares communication-derived with expression-derived cell-cell similarity
within each cell type. Per-slice concordance scores and summary statistics are
saved together with a two-row figure containing one study per row.

Major inputs are the processed AnnData/graph objects and precomputed outputs
from SpiderNet, NMF-LR, COMMOT, scCChain and, for HGSOC, Spacia. The benchmark
does not train or alter any CCC model. Tables and figures are saved under
``output/cell_similarity`` next to this module. Use ``--plot-only`` to redraw
the figure from the bundled per-study long tables. Data and results locations
default to the shared ``benchmark_config`` settings.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import NMF
import torch

from benchmark_config import DATA_ROOT, PROJECT_ROOT, RESULTS_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "cell_similarity"

STUDY_ORDER = ["HGSOC", "AgingMousebrain"]
CANONICAL_METHOD_ORDER = ["SpiderNet", "NMF-LR", "COMMOT", "ScCChain", "Spacia"]
MIN_CELLS_PER_CELLTYPE = 30
NMF_RANDOM_STATE = 0
NMF_MAX_ITER = 1000

METHOD_COLORS = {
    "SpiderNet": {"edge": "#9F3B38", "fill": "#E1B6A7"},
    "NMF-LR": {"edge": "#82CCE2", "fill": "#D4ECF1"},
    "COMMOT": {"edge": "#519384", "fill": "#B9CEC7"},
    "ScCChain": {"edge": "#636491", "fill": "#A6A2B9"},
    "Spacia": {"edge": "#FED881", "fill": "#FFF2D2"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory containing AgingBrain, HGSOC and Database inputs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory containing precomputed method outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark tables and the combined figure.",
    )
    parser.add_argument(
        "--recompute-nmflr",
        action="store_true",
        help="Recompute NMF-LR factors instead of using an existing cache.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the combined figure from existing per-study long tables.",
    )
    return parser.parse_args()


def build_study_config(data_root: Path, results_root: Path) -> dict[str, dict]:
    return {
        "AgingMousebrain": {
            "dataset_name": "AgingMousebrain",
            "display_name": "Ageing mouse brain (MERFISH)",
            "version": "V1",
            "dim_envir": 30,
            "cellclass_name": "celltype",
            "processed_data_dir": results_root / "AgingBrain" / "ProcessedData",
            "result_dir": (
                results_root
                / "AgingBrain"
                / "V1"
                / "SpiderNet_Result_dim30"
            ),
            "commot_dir": results_root / "AgingBrain" / "COMMOT",
            "sccchain_dir": results_root / "AgingBrain" / "ScCChain",
            "spacia_dir": None,
            "include_spacia": False,
            "sample_obs_key": "age",
            "data_dir": data_root / "AgingBrain",
        },
        "HGSOC": {
            "dataset_name": "HGSOC",
            "display_name": "HGSOC (CosMx SMI)",
            "version": "V1",
            "dim_envir": 15,
            "cellclass_name": "cell.types",
            "processed_data_dir": results_root / "HGSOC" / "ProcessedData",
            "result_dir": (
                results_root
                / "HGSOC"
                / "V1"
                / "SpiderNet_Result_dim15"
            ),
            "commot_dir": results_root / "HGSOC" / "COMMOT",
            "sccchain_dir": results_root / "HGSOC" / "ScCChain",
            "spacia_dir": results_root / "HGSOC" / "Spacia" / "spacia_outputs",
            "include_spacia": True,
            "sample_obs_key": "samples",
            "data_dir": data_root / "HGSOC",
        },
    }


def add_spidernet_package_to_path() -> None:
    # Retain the source analysis package by default for numerical continuity.
    override = os.environ.get("SPIDERNET_PACKAGE_ROOT")
    package_parent = Path(override).resolve() if override else PROJECT_ROOT / "SpiderNet"
    if override and not (package_parent / "SpiderNet" / "__init__.py").is_file():
        raise FileNotFoundError(f"No SpiderNet package under {package_parent}")
    if not package_parent.exists():
        raise FileNotFoundError(f"SpiderNet package directory not found: {package_parent}")
    package_parent_text = str(package_parent)
    if package_parent_text not in sys.path:
        sys.path.insert(0, package_parent_text)


def validate_input_paths(cfg: dict) -> None:
    required = [
        cfg["processed_data_dir"] / "adata_list.pkl",
        cfg["processed_data_dir"] / "SpiderNet_data_pyg_list.pkl",
        cfg["result_dir"] / "Factor_envir_list.pkl",
        cfg["commot_dir"],
        cfg["sccchain_dir"] / "h5ad_files.txt",
    ]
    if cfg["include_spacia"]:
        required.append(cfg["spacia_dir"])
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs for "
            f"{cfg['dataset_name']}: " + ", ".join(str(path) for path in missing)
        )


def set_plot_style() -> None:
    plt.close("all")
    plt.style.use("default")
    rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def to_dense_2d(matrix):
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def edge_matrix_from_square(square_matrix, rows, cols) -> np.ndarray:
    if sp.issparse(square_matrix):
        square_matrix = square_matrix.tocsr()
        return square_matrix[rows, cols].A1.astype(np.float32, copy=False)
    return square_matrix[rows, cols].astype(np.float32, copy=False)


def build_nmflr_factors(
    graph_list,
    n_components: int,
    random_state: int = NMF_RANDOM_STATE,
    max_iter: int = NMF_MAX_ITER,
) -> list[np.ndarray]:
    factor_list = []
    for slice_index, graph in enumerate(graph_list):
        print(f"Running NMF-LR for slice {slice_index + 1}/{len(graph_list)}")
        edge_lr_matrix = graph["cellpair_LRpair_neigh"]
        model = NMF(
            n_components=n_components,
            init="nndsvda",
            random_state=random_state,
            max_iter=max_iter,
        )
        factors = model.fit_transform(edge_lr_matrix)
        column_maxima = np.max(factors, axis=0)
        column_maxima[column_maxima == 0] = 1.0
        factors = factors / column_maxima
        factor_list.append(factors.astype(np.float32, copy=False))
    return factor_list


def load_or_build_nmflr_factors(
    cfg: dict,
    graph_list,
    recompute: bool,
) -> list[np.ndarray]:
    factor_path = cfg["result_dir"] / "Factor_LR_list.pkl"
    if factor_path.exists() and not recompute:
        print(f"Loading existing NMF-LR factors: {factor_path}")
        return pd.read_pickle(factor_path)

    factors = build_nmflr_factors(graph_list, n_components=cfg["dim_envir"])
    with factor_path.open("wb") as handle:
        pickle.dump(factors, handle)
    print(f"Saved NMF-LR factors: {factor_path}")
    return factors


def commot_file_for_slice(adata_sub, cfg: dict) -> Path:
    sample_value = np.unique(adata_sub.obs[cfg["sample_obs_key"]])[0]
    if cfg["dataset_name"] == "HGSOC":
        return cfg["commot_dir"] / f"{sample_value}_COMMOT_cellchat.h5ad"
    if cfg["dataset_name"] == "AgingMousebrain":
        return cfg["commot_dir"] / f"aging_coronal_age{sample_value}_commot.h5ad"
    raise ValueError(f"Unsupported study: {cfg['dataset_name']}")


def load_commot_outputs(adata_list, graph_list, cfg: dict) -> list[np.ndarray]:
    score_list = []
    for slice_index, adata_sub in enumerate(adata_list):
        print(f"Loading COMMOT for slice {slice_index + 1}/{len(adata_list)}")
        commot_adata = sc.read_h5ad(commot_file_for_slice(adata_sub, cfg))
        edge_index = graph_list[slice_index]["edge_index"].detach().cpu().numpy()
        rows = edge_index[:, 0].astype(np.int64, copy=False)
        cols = edge_index[:, 1].astype(np.int64, copy=False)
        keys = list(commot_adata.obsp.keys())
        total_key = "commot-cellchat-total-total"
        pathway_keys = [
            key
            for key in keys
            if key.startswith("commot-cellchat-")
            and len(key.split("-")) == 3
            and key != total_key
        ]
        pathway_array = np.zeros((len(rows), len(pathway_keys)), dtype=np.float32)
        for pathway_index, key in enumerate(pathway_keys):
            pathway_array[:, pathway_index] = edge_matrix_from_square(
                commot_adata.obsp[key], rows, cols
            )
        score_list.append(pathway_array)
        del commot_adata
    return score_list


def load_sccchain_outputs(adata_list, graph_list, cfg: dict) -> list[np.ndarray]:
    file_list_path = cfg["sccchain_dir"] / "h5ad_files.txt"
    sccchain_adata_paths = pd.read_csv(file_list_path, header=None)[0].tolist()
    reference_sizes = [adata_sub.n_obs for adata_sub in adata_list]
    comparator_sizes = []
    for adata_path in sccchain_adata_paths:
        comparator_adata = sc.read_h5ad(adata_path)
        comparator_sizes.append(comparator_adata.n_obs)
        del comparator_adata

    index_map = []
    for n_obs in reference_sizes:
        matched = np.where(np.asarray(comparator_sizes) == n_obs)[0]
        if len(matched) == 0:
            raise ValueError(f"Cannot match scCChain result by n_obs={n_obs}")
        index_map.append(matched[0])

    score_list = []
    for slice_index, adata_sub in enumerate(adata_list):
        print(f"Loading scCChain for slice {slice_index + 1}/{len(adata_list)}")
        source_path = Path(sccchain_adata_paths[index_map[slice_index]])
        score_path = (
            cfg["sccchain_dir"]
            / f"{source_path.stem}_ScCChain_edge_program_scores.csv"
        )
        score_frame = pd.read_csv(score_path, index_col=None)
        edge_index = graph_list[slice_index]["edge_index"].detach().cpu().numpy()
        rows = edge_index[:, 0].astype(np.int64, copy=False)
        cols = edge_index[:, 1].astype(np.int64, copy=False)
        program_keys = score_frame.columns.tolist()[2:]
        edge_program_array = np.zeros(
            (len(rows), len(program_keys)), dtype=np.float32
        )

        for program_index, program_key in enumerate(program_keys):
            sub_frame = score_frame[["sender_index", "receiver_index", program_key]]
            sender_indices = sub_frame["sender_index"].values.astype(np.int64) - 1
            receiver_indices = sub_frame["receiver_index"].values.astype(np.int64) - 1
            square_matrix = np.zeros(
                (adata_sub.n_obs, adata_sub.n_obs), dtype=np.float32
            )
            square_matrix[sender_indices, receiver_indices] = sub_frame[
                program_key
            ].values.astype(np.float32)
            edge_program_array[:, program_index] = edge_matrix_from_square(
                square_matrix, rows, cols
            )
        score_list.append(edge_program_array)
    return score_list


def load_spacia_outputs(adata_list, graph_list, cfg: dict) -> list[np.ndarray] | None:
    if not cfg["include_spacia"]:
        return None

    filenames = sorted(path.name for path in cfg["spacia_dir"].glob("*.h5ad"))
    reference_sizes = [adata_sub.n_obs for adata_sub in adata_list]
    comparator_sizes = []
    for filename in filenames:
        comparator_adata = sc.read_h5ad(
            cfg["spacia_dir"] / filename, backed="r"
        )
        comparator_sizes.append(comparator_adata.n_obs)
        comparator_adata.file.close()

    index_map = []
    for n_obs in reference_sizes:
        matched = np.where(np.asarray(comparator_sizes) == n_obs)[0]
        if len(matched) == 0:
            raise ValueError(f"Cannot match Spacia result by n_obs={n_obs}")
        index_map.append(matched[0])

    score_list = []
    for slice_index in range(len(adata_list)):
        print(f"Loading Spacia for slice {slice_index + 1}/{len(adata_list)}")
        spacia_adata = sc.read_h5ad(cfg["spacia_dir"] / filenames[index_map[slice_index]])
        edge_index = graph_list[slice_index]["edge_index"].detach().cpu().numpy()
        rows = edge_index[:, 0].astype(np.int64, copy=False)
        cols = edge_index[:, 1].astype(np.int64, copy=False)
        n_programs = spacia_adata.obsp["interaction_scores"].shape[2]
        edge_program_array = np.zeros((len(rows), n_programs), dtype=np.float32)
        for program_index in range(n_programs):
            square_matrix = spacia_adata.obsp["interaction_scores"][:, :, program_index]
            edge_program_array[:, program_index] = edge_matrix_from_square(
                square_matrix, rows, cols
            )
        score_list.append(edge_program_array)
        del spacia_adata
    return score_list


def aggregate_edge_features(
    edge_features: np.ndarray,
    edge_index_column: torch.Tensor,
    num_cells: int,
    device: torch.device,
) -> np.ndarray:
    from SpiderNet.utils import scatter_nanmean

    aggregated = scatter_nanmean(
        torch.as_tensor(edge_features, dtype=torch.float32, device=device),
        edge_index_column.to(torch.int64).to(device),
        dim=0,
        dim_size=num_cells,
    )
    return aggregated.to("cpu").numpy()


def aggregate_sender_receiver(
    edge_features: np.ndarray,
    edge_index: torch.Tensor,
    num_cells: int,
    device: torch.device,
) -> np.ndarray:
    sender_features = aggregate_edge_features(
        edge_features, edge_index[:, 0], num_cells, device
    )
    receiver_features = aggregate_edge_features(
        edge_features, edge_index[:, 1], num_cells, device
    )
    return np.hstack([sender_features, receiver_features])


def build_method_cell_features(
    adata_list,
    graph_list,
    factor_envir_list,
    factor_lr_list,
    commot_score_list,
    sccchain_score_list,
    spacia_score_list,
    include_spacia: bool,
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    method_cell_features = {
        "SpiderNet": [],
        "NMF-LR": [],
        "COMMOT": [],
        "ScCChain": [],
    }
    if include_spacia:
        method_cell_features["Spacia"] = []

    for slice_index in range(len(adata_list)):
        print(f"Preparing cell features for slice {slice_index + 1}/{len(adata_list)}")
        edge_index = graph_list[slice_index]["edge_index"]
        num_cells = graph_list[slice_index].x.shape[0]
        edge_features_by_method = {
            "SpiderNet": np.asarray(factor_envir_list[slice_index], dtype=float),
            "NMF-LR": np.asarray(factor_lr_list[slice_index], dtype=float),
            "COMMOT": np.asarray(commot_score_list[slice_index], dtype=float),
            "ScCChain": np.asarray(sccchain_score_list[slice_index], dtype=float),
        }
        if include_spacia:
            edge_features_by_method["Spacia"] = np.asarray(
                spacia_score_list[slice_index], dtype=float
            )

        for method, edge_features in edge_features_by_method.items():
            method_cell_features[method].append(
                aggregate_sender_receiver(
                    edge_features, edge_index, num_cells, device
                )
            )
    return method_cell_features


def upper_triangular_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def compute_similarity_preservation_table(
    adata_list,
    method_cell_features: dict[str, list[np.ndarray]],
    cellclass_name: str,
    min_cells_per_celltype: int = MIN_CELLS_PER_CELLTYPE,
) -> pd.DataFrame:
    records = []
    for slice_index, adata_sub in enumerate(adata_list):
        print(f"Computing concordance for slice {slice_index + 1}/{len(adata_list)}")
        expression = to_dense_2d(adata_sub.X)
        for celltype in np.unique(adata_sub.obs[cellclass_name]).tolist():
            celltype_mask = (adata_sub.obs[cellclass_name] == celltype).values
            n_cells = int(np.sum(celltype_mask))
            if n_cells < min_cells_per_celltype:
                continue

            expression_block = expression[celltype_mask]
            expression_similarity = np.corrcoef(expression_block)
            expression_similarity = np.nan_to_num(expression_similarity, nan=0.0)
            expression_values = upper_triangular_values(expression_similarity)
            record = {
                "Slice": f"Slice_{slice_index + 1}",
                "CellType": celltype,
                "n_cells": n_cells,
            }

            for method, feature_list in method_cell_features.items():
                feature_block = np.asarray(
                    feature_list[slice_index][celltype_mask, :], dtype=float
                )
                feature_block = np.nan_to_num(feature_block, nan=0.0)
                feature_similarity = np.corrcoef(feature_block)
                feature_similarity = np.nan_to_num(feature_similarity, nan=0.0)
                feature_values = upper_triangular_values(feature_similarity)
                correlation, _ = spearmanr(expression_values, feature_values)
                record[method] = correlation
            records.append(record)
        del expression
    return pd.DataFrame(records)


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan}
    if values.size == 1:
        return {
            "n": 1,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd": np.nan,
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
    }


def one_sided_mannwhitney_greater(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    return float(mannwhitneyu(x, y, alternative="greater").pvalue)


def make_long_and_summary_tables(
    correlation_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    method_order = [
        method
        for method in CANONICAL_METHOD_ORDER
        if method in correlation_frame.columns
    ]
    long_frame = correlation_frame.melt(
        id_vars=["Slice", "CellType", "n_cells"],
        value_vars=method_order,
        var_name="Method",
        value_name="Correlation",
    )
    long_frame["Correlation"] = pd.to_numeric(
        long_frame["Correlation"], errors="coerce"
    )
    spider_median = (
        long_frame[long_frame["Method"] == "SpiderNet"]
        .groupby("CellType")["Correlation"]
        .median()
        .sort_values(ascending=False)
    )
    cell_order = spider_median.index.tolist()
    for celltype in sorted(long_frame["CellType"].unique()):
        if celltype not in cell_order:
            cell_order.append(celltype)

    summary_records = []
    for celltype in cell_order:
        subset = long_frame[long_frame["CellType"] == celltype]
        spider_values = subset[subset["Method"] == "SpiderNet"][
            "Correlation"
        ].to_numpy()
        for method in method_order:
            values = subset[subset["Method"] == method]["Correlation"].to_numpy()
            summary = summarize_values(values)
            p_value = (
                np.nan
                if method == "SpiderNet"
                else one_sided_mannwhitney_greater(spider_values, values)
            )
            summary_records.append(
                {
                    "CellType": celltype,
                    "Method": method,
                    "n": summary["n"],
                    "mean": summary["mean"],
                    "median": summary["median"],
                    "sd": summary["sd"],
                    "p_SpiderNet_greater": p_value,
                }
            )
    return long_frame, pd.DataFrame(summary_records), method_order, cell_order


def run_study(
    cfg: dict,
    output_dir: Path,
    recompute_nmflr: bool,
    device: torch.device,
) -> pd.DataFrame:
    validate_input_paths(cfg)
    study_output_dir = output_dir / cfg["dataset_name"]
    study_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {cfg['dataset_name']} ===")
    print(f"Output directory: {study_output_dir}")

    adata_list = pd.read_pickle(cfg["processed_data_dir"] / "adata_list.pkl")
    graph_list = pd.read_pickle(
        cfg["processed_data_dir"] / "SpiderNet_data_pyg_list.pkl"
    )
    factor_envir_list = pd.read_pickle(cfg["result_dir"] / "Factor_envir_list.pkl")
    factor_lr_list = load_or_build_nmflr_factors(
        cfg, graph_list, recompute_nmflr
    )
    print(f"Loaded {len(adata_list)} tissue slices")

    commot_scores = load_commot_outputs(adata_list, graph_list, cfg)
    sccchain_scores = load_sccchain_outputs(adata_list, graph_list, cfg)
    spacia_scores = load_spacia_outputs(adata_list, graph_list, cfg)
    method_cell_features = build_method_cell_features(
        adata_list=adata_list,
        graph_list=graph_list,
        factor_envir_list=factor_envir_list,
        factor_lr_list=factor_lr_list,
        commot_score_list=commot_scores,
        sccchain_score_list=sccchain_scores,
        spacia_score_list=spacia_scores,
        include_spacia=cfg["include_spacia"],
        device=device,
    )
    correlation_frame = compute_similarity_preservation_table(
        adata_list=adata_list,
        method_cell_features=method_cell_features,
        cellclass_name=cfg["cellclass_name"],
    )
    long_frame, summary_frame, _, _ = make_long_and_summary_tables(correlation_frame)

    correlation_frame.to_csv(
        study_output_dir / "cell_similarity_concordance_raw.csv", index=False
    )
    long_frame.insert(0, "Study", cfg["dataset_name"])
    long_frame.to_csv(
        study_output_dir / "cell_similarity_concordance_long.csv", index=False
    )
    summary_frame.insert(0, "Study", cfg["dataset_name"])
    summary_frame.to_csv(
        study_output_dir / "cell_similarity_concordance_summary.csv", index=False
    )
    print(f"Completed {cfg['dataset_name']}: {len(correlation_frame)} slice/cell-type rows")
    return long_frame


def load_existing_long_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    records = {}
    for study in STUDY_ORDER:
        path = output_dir / study / "cell_similarity_concordance_long.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing plot-only input: {path}")
        records[study] = pd.read_csv(path)
    return records


def plot_two_study_figure(
    long_tables: dict[str, pd.DataFrame],
    study_config: dict[str, dict],
    output_dir: Path,
) -> tuple[Path, Path, pd.DataFrame]:
    set_plot_style()
    cell_orders = {}
    summaries = []
    for study in STUDY_ORDER:
        long_frame = long_tables[study]
        spider_median = (
            long_frame[long_frame["Method"] == "SpiderNet"]
            .groupby("CellType")["Correlation"]
            .median()
            .sort_values(ascending=False)
        )
        cell_order = spider_median.index.tolist()
        for celltype in sorted(long_frame["CellType"].dropna().unique()):
            if celltype not in cell_order:
                cell_order.append(celltype)
        cell_orders[study] = cell_order

        for celltype in cell_order:
            for method in CANONICAL_METHOD_ORDER:
                values = long_frame.loc[
                    long_frame["CellType"].eq(celltype)
                    & long_frame["Method"].eq(method),
                    "Correlation",
                ].to_numpy(dtype=float)
                summary = summarize_values(values)
                summaries.append(
                    {
                        "Study": study,
                        "CellType": celltype,
                        "Method": method,
                        "mean": summary["mean"],
                        "sd": summary["sd"],
                        "n": summary["n"],
                    }
                )
    plot_summary = pd.DataFrame(summaries)
    plot_summary = plot_summary[plot_summary["n"] > 0].copy()
    plot_summary.to_csv(
        output_dir / "cell_similarity_concordance_two_studies_plot_summary.csv",
        index=False,
    )

    max_celltypes = max(len(order) for order in cell_orders.values())
    figure_width = max(8.0, 0.75 * max_celltypes)
    fig, axes = plt.subplots(
        len(STUDY_ORDER),
        1,
        figsize=(figure_width, 3.0 * len(STUDY_ORDER)),
        constrained_layout=True,
        squeeze=False,
    )
    methods_present = set()
    for row_index, study in enumerate(STUDY_ORDER):
        ax = axes[row_index, 0]
        long_frame = long_tables[study]
        cell_order = cell_orders[study]
        method_order = [
            method
            for method in CANONICAL_METHOD_ORDER
            if method in set(long_frame["Method"].dropna())
        ]
        methods_present.update(method_order)
        base_positions = np.arange(len(cell_order))
        offsets = np.linspace(-0.30, 0.30, len(method_order))

        for offset, method in zip(offsets, method_order):
            subset = (
                plot_summary[
                    plot_summary["Study"].eq(study)
                    & plot_summary["Method"].eq(method)
                ]
                .set_index("CellType")
                .reindex(cell_order)
            )
            ax.errorbar(
                base_positions + offset,
                subset["mean"].to_numpy(),
                yerr=subset["sd"].to_numpy(),
                fmt="o",
                linestyle="none",
                markersize=3.5,
                linewidth=1.0,
                capsize=2,
                color=METHOD_COLORS[method]["edge"],
                markerfacecolor=METHOD_COLORS[method]["fill"],
                markeredgecolor=METHOD_COLORS[method]["edge"],
            )

        ax.set_xticks(base_positions)
        ax.set_xticklabels(cell_order, rotation=45, ha="right")
        ax.set_ylabel("Spearman correlation")
        ax.set_title(study_config[study]["display_name"])
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    legend_order = [
        method for method in CANONICAL_METHOD_ORDER if method in methods_present
    ]
    handles = [
        mpatches.Patch(
            facecolor=METHOD_COLORS[method]["fill"],
            edgecolor=METHOD_COLORS[method]["edge"],
            label=method,
        )
        for method in legend_order
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
    )
    output_stem = output_dir / "cell_similarity_concordance_two_studies"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, plot_summary


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_spidernet_package_to_path()
    study_config = build_study_config(args.data_root, args.results_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Aggregation device: {device}")

    if args.plot_only:
        long_tables = load_existing_long_tables(args.output_dir)
    else:
        long_tables = {}
        for study in STUDY_ORDER:
            long_tables[study] = run_study(
                cfg=study_config[study],
                output_dir=args.output_dir,
                recompute_nmflr=args.recompute_nmflr,
                device=device,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    combined_long = pd.concat(
        [long_tables[study] for study in STUDY_ORDER], ignore_index=True
    )
    combined_long.to_csv(
        args.output_dir / "cell_similarity_concordance_two_studies_long.csv",
        index=False,
    )
    png_path, pdf_path, _ = plot_two_study_figure(
        long_tables=long_tables,
        study_config=study_config,
        output_dir=args.output_dir,
    )
    print(f"Saved combined PNG: {png_path}")
    print(f"Saved combined PDF: {pdf_path}")


if __name__ == "__main__":
    main()
