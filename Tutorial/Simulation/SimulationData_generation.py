
#!/usr/bin/env python3
"""
Generate simulation datasets for the SpiderNet benchmarking pipeline.

This script creates six benchmark settings that match the downstream pipeline:
    - Dropout1, Dropout2, Dropout3
    - Sd_use1, Sd_use2, Sd_use3

For each setting and experiment replicate, it generates:
    - simulated gene expression
    - spatial coordinates
    - cell metadata
    - edge metadata with ground-truth meta-interaction labels
    - an AnnData object saved as .h5ad

The output directory structure is:

    DATA_ROOT/
        Dropout1/
            Experiment_0/
                gene_exp.csv
                spatial_location.csv
                gene_metadf.csv
                cell_metadf.csv
                edge_metadf.csv
                cell_neigh_metaIprop.csv
                adata_simulation.h5ad
                ...
        Dropout2/
        ...
        Sd_use3/

Optional diagnostic figures can also be saved for each experiment.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

@dataclass
class SimulationConfig:
    data_root: str = r"D:/SpiderNet/Data/Simulation"
    num_experiments: int = 30

    # Tissue layout
    num_cells: int = 2000
    num_cell_types: int = 3
    tissue_size: Tuple[float, float] = (1000.0, 1000.0)
    inner_circle_radius_ratio: float = 1 / 3
    min_distance: float = 10.0
    num_neighbors: int = 10

    # Meta-interaction design
    num_meta_interactions: int = 2

    # Gene panel design
    num_marker_intrinsic: int = 5
    num_lr_pairs_per_mi: int = 10
    num_upregulated_genes_per_mi_side: int = 10

    # Expression scale
    explevel_intrinsic_marker: float = 3.0
    explevel_lr: float = 3.0
    explevel_upgene: float = 3.0

    # Noise and setting defaults
    base_dropout_ratio: float = 0.0
    base_sd_use: float = 0.4
    dropout_ratio_list: Tuple[float, float, float] = (0.1, 0.3, 0.5)
    sd_use_list: Tuple[float, float, float] = (0.4, 0.7, 1.0)

    # Optional outputs
    save_diagnostic_figures: bool = True


# ============================================================
# Helper functions
# ============================================================

def spatial_neighborindex_generation(
    cell_spatial: np.ndarray,
    num_neighbor_available: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build directed k-nearest-neighbor edges.

    Each cell is connected to its k nearest neighbors.
    The returned edge index has shape (E, 2) with columns:
        [sender_index, receiver_index]
    """
    spatialnn_use = NearestNeighbors(
        n_neighbors=num_neighbor_available + 1,
        algorithm="auto",
    )
    spatialnn_use.fit(cell_spatial)
    _, spatial_neighborindex = spatialnn_use.kneighbors(cell_spatial)

    for i in range(spatial_neighborindex.shape[0]):
        if spatial_neighborindex[i, 0] != i:
            spatial_neighborindex[i, :] = np.hstack(
                (i, np.setdiff1d(spatial_neighborindex[i, :], i))
            )

    spatial_neighborindex_further = spatial_neighborindex[:, 1:]
    centralindex = spatial_neighborindex[:, 0]
    edge_index = np.vstack(
        (
            np.repeat(centralindex, spatial_neighborindex_further.shape[1]),
            spatial_neighborindex_further.flatten(),
        )
    )
    return spatial_neighborindex, edge_index.T


def build_settings_table(cfg: SimulationConfig) -> pd.DataFrame:
    """
    Create the six benchmark settings used by the downstream pipeline.
    """
    settings: List[List[float | str]] = []

    for i, dropout in enumerate(cfg.dropout_ratio_list, start=1):
        settings.append(
            [dropout, cfg.base_sd_use, f"Dropout{i}"]
        )

    for i, sd_use in enumerate(cfg.sd_use_list, start=1):
        settings.append(
            [cfg.base_dropout_ratio, sd_use, f"Sd_use{i}"]
        )

    settings_df = pd.DataFrame(
        settings,
        columns=["Dropout_ratio", "sd_use", "Dominant_Setting"],
    )
    return settings_df


def sample_spatial_locations(
    rng: np.random.Generator,
    num_cells: int,
    tissue_size: Tuple[float, float],
    min_distance: float,
) -> np.ndarray:
    """
    Sample cell coordinates with a simple minimum-distance constraint.
    """
    x_min, x_max = -tissue_size[0] / 2, tissue_size[0] / 2
    y_min, y_max = -tissue_size[1] / 2, tissue_size[1] / 2

    spatial_location: List[np.ndarray] = []

    first_point = np.array(
        [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
        dtype=float,
    )
    spatial_location.append(first_point)

    def is_too_close(new_point: np.ndarray, existing_points: Iterable[np.ndarray]) -> bool:
        return any(np.linalg.norm(new_point - point) < min_distance for point in existing_points)

    for _ in range(1, num_cells):
        while True:
            new_point = np.array(
                [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
                dtype=float,
            )
            if not is_too_close(new_point, spatial_location):
                spatial_location.append(new_point)
                break

    return np.asarray(spatial_location, dtype=float)


def build_gene_metadata(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Build the gene metadata table for intrinsic markers, ligands, receptors,
    sender-upregulated genes, and receiver-upregulated genes.
    """
    records: List[Dict[str, object]] = []

    # Ligands and receptors
    for mi_idx in range(cfg.num_meta_interactions):
        mi_name = f"MetaItype_{mi_idx + 1}"
        for lr_idx in range(cfg.num_lr_pairs_per_mi):
            records.append(
                {
                    "Gene_name": f"L{mi_idx * cfg.num_lr_pairs_per_mi + lr_idx + 1}",
                    "Gene_type": "Ligand",
                    "Associated_celltype": "Non_marker",
                    "Associated_metaItype": mi_name,
                }
            )
            records.append(
                {
                    "Gene_name": f"R{mi_idx * cfg.num_lr_pairs_per_mi + lr_idx + 1}",
                    "Gene_type": "Receptor",
                    "Associated_celltype": "Non_marker",
                    "Associated_metaItype": mi_name,
                }
            )

    # Sender- and receiver-upregulated genes
    sender_counter = 1
    receiver_counter = 1
    for mi_idx in range(cfg.num_meta_interactions):
        mi_name = f"MetaItype_{mi_idx + 1}"

        for _ in range(cfg.num_upregulated_genes_per_mi_side):
            records.append(
                {
                    "Gene_name": f"SG{sender_counter}",
                    "Gene_type": "Upregulated_gene_Sender",
                    "Associated_celltype": "Non_marker",
                    "Associated_metaItype": mi_name,
                }
            )
            sender_counter += 1

        for _ in range(cfg.num_upregulated_genes_per_mi_side):
            records.append(
                {
                    "Gene_name": f"RG{receiver_counter}",
                    "Gene_type": "Upregulated_gene_Receiver",
                    "Associated_celltype": "Non_marker",
                    "Associated_metaItype": mi_name,
                }
            )
            receiver_counter += 1

    gene_metadf = pd.DataFrame.from_records(records)

    # Match each ligand with its paired receptor and vice versa
    gene_metadf["Associated_Ligand_or_Receptor"] = np.nan
    lr_mask = gene_metadf["Gene_type"].isin(["Ligand", "Receptor"])
    gene_name_lr = gene_metadf.loc[lr_mask, "Gene_name"].tolist()
    paired_names: List[str] = []
    for gene_name in gene_name_lr:
        if gene_name.startswith("L"):
            paired_names.append(gene_name.replace("L", "R", 1))
        else:
            paired_names.append(gene_name.replace("R", "L", 1))
    gene_metadf.loc[lr_mask, "Associated_Ligand_or_Receptor"] = paired_names

    # Randomly assign intrinsic marker genes from the non-LR / non-LR-paired pool
    eligible_mask = ~gene_metadf["Gene_type"].isin(["Ligand", "Receptor"])
    eligible_indices = gene_metadf.index[eligible_mask].to_numpy()

    chosen_all: List[int] = []
    for celltype_idx in range(cfg.num_cell_types):
        available = np.setdiff1d(eligible_indices, np.array(chosen_all, dtype=int))
        chosen = rng.choice(
            available,
            size=cfg.num_marker_intrinsic,
            replace=False,
        )
        gene_metadf.loc[chosen, "Associated_celltype"] = f"Celltype_{celltype_idx}"
        chosen_all.extend(chosen.tolist())

    gene_metadf["Gene_ID"] = np.arange(gene_metadf.shape[0], dtype=int)
    gene_metadf.index = np.arange(gene_metadf.shape[0], dtype=int)
    return gene_metadf


def assign_edge_types(
    spatial_location: np.ndarray,
    inner_circle_radius: float,
    num_neighbors: int,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build edges and assign ground-truth meta-interaction labels.

    MI-1: sender outside the inner circle, receiver inside
    MI-2: sender inside the inner circle, receiver outside
    """
    x = spatial_location[:, 0]
    y = spatial_location[:, 1]

    inner_cell = (np.sqrt(x ** 2 + y ** 2) < inner_circle_radius).astype(int)

    # Slightly expand outer cells to create a clearer spatial separation
    scale_factor = 1.02
    spatial_location = spatial_location.copy()
    spatial_location[inner_cell == 0, 0] *= scale_factor
    spatial_location[inner_cell == 0, 1] *= scale_factor

    _, edge_index = spatial_neighborindex_generation(spatial_location, num_neighbors)
    edge_index_df = pd.DataFrame(edge_index, columns=["Sender", "Receiver"])
    edge_index_df["MI_type"] = "non-interaction"

    sender_idx = edge_index_df["Sender"].to_numpy(dtype=int)
    receiver_idx = edge_index_df["Receiver"].to_numpy(dtype=int)

    mi1_mask = (inner_cell[sender_idx] == 0) & (inner_cell[receiver_idx] == 1)
    mi2_mask = (inner_cell[sender_idx] == 1) & (inner_cell[receiver_idx] == 0)

    edge_index_df.loc[mi1_mask, "MI_type"] = "MI-1"
    edge_index_df.loc[mi2_mask, "MI_type"] = "MI-2"

    return spatial_location, edge_index_df, inner_cell, edge_index


def apply_expression_dropout(
    rng: np.random.Generator,
    gene_exp: np.ndarray,
    dropout_ratio: float,
) -> np.ndarray:
    """
    Apply element-wise dropout to the final expression matrix.
    """
    keep_mask = rng.binomial(1, 1 - dropout_ratio, gene_exp.shape)
    return gene_exp * keep_mask


def compute_cell_mi_summary(
    edge_index_df: pd.DataFrame,
    num_cells: int,
    num_meta_interactions: int,
    num_neighbors: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Compute per-cell neighborhood MI proportions, sending activity,
    and receiving activity.
    """
    edge_index_all = edge_index_df[["Sender", "Receiver"]].to_numpy(dtype=int)
    mi_types = edge_index_df["MI_type"].to_numpy(dtype=object)

    mi_numeric = np.where(mi_types == "non-interaction", 0, mi_types)
    mi_numeric = np.where(mi_numeric == "MI-1", 1, mi_numeric)
    mi_numeric = np.where(mi_numeric == "MI-2", 2, mi_numeric).astype(int)

    cell_neigh_metaIprop = np.zeros((num_cells, num_meta_interactions), dtype=float)
    cell_sending_MI = np.zeros((num_cells, num_meta_interactions), dtype=float)
    cell_receiving_MI = np.zeros((num_cells, num_meta_interactions), dtype=float)

    for cell_index in range(num_cells):
        outgoing = mi_numeric[edge_index_all[:, 0] == cell_index]
        incoming = mi_numeric[edge_index_all[:, 1] == cell_index]

        outgoing_counts = np.bincount(outgoing, minlength=num_meta_interactions + 1)[1 : num_meta_interactions + 1]
        incoming_counts = np.bincount(incoming, minlength=num_meta_interactions + 1)[1 : num_meta_interactions + 1]

        if outgoing.shape[0] > 0:
            cell_neigh_metaIprop[cell_index, :] = outgoing_counts / outgoing.shape[0]

        cell_sending_MI[cell_index, :] = outgoing_counts / num_neighbors
        cell_receiving_MI[cell_index, :] = incoming_counts / num_neighbors

    columns = [f"MI-{i + 1}" for i in range(num_meta_interactions)]
    cell_index_labels = [f"Cell_{i}" for i in range(num_cells)]

    cell_neigh_metaIprop_df = pd.DataFrame(
        cell_neigh_metaIprop,
        columns=columns,
        index=cell_index_labels,
    )
    cell_sending_MI_df = pd.DataFrame(
        cell_sending_MI,
        columns=columns,
        index=cell_index_labels,
    )
    cell_receiving_MI_df = pd.DataFrame(
        cell_receiving_MI,
        columns=columns,
        index=cell_index_labels,
    )

    return cell_neigh_metaIprop_df, cell_sending_MI_df, cell_receiving_MI_df, mi_numeric


def simulate_intrinsic_expression(
    cfg: SimulationConfig,
    rng: np.random.Generator,
    gene_metadf: pd.DataFrame,
    celltype_cell: np.ndarray,
    sd_use: float,
) -> np.ndarray:
    """
    Simulate cell-intrinsic marker expression.
    """
    num_cells = celltype_cell.shape[0]
    num_genes = gene_metadf.shape[0]
    gene_exp_intrinsic = np.zeros((num_cells, num_genes), dtype=float)

    for cell_index in range(num_cells):
        celltype_index = int(celltype_cell[cell_index])

        background = rng.normal(
            cfg.explevel_intrinsic_marker / 5,
            sd_use,
            num_genes,
        )
        gene_exp_intrinsic[cell_index, :] = background

        marker_gene_ids = gene_metadf.loc[
            gene_metadf["Associated_celltype"] == f"Celltype_{celltype_index}",
            "Gene_ID",
        ].to_numpy(dtype=int)

        marker_expression = rng.normal(
            cfg.explevel_intrinsic_marker,
            sd_use,
            marker_gene_ids.shape[0],
        )
        gene_exp_intrinsic[cell_index, marker_gene_ids] = marker_expression

    gene_exp_intrinsic[gene_exp_intrinsic < 0] = 0
    return gene_exp_intrinsic


def simulate_lr_expression(
    cfg: SimulationConfig,
    rng: np.random.Generator,
    gene_metadf: pd.DataFrame,
    cell_sending_MI: pd.DataFrame,
    cell_receiving_MI: pd.DataFrame,
    sd_use: float,
) -> np.ndarray:
    """
    Simulate ligand-receptor expression associated with each meta-interaction.
    """
    num_cells = cell_sending_MI.shape[0]
    num_genes = gene_metadf.shape[0]
    gene_exp_lr = rng.normal(0, sd_use, (num_cells, num_genes))

    for mi_idx in range(cfg.num_meta_interactions):
        mi_label = f"MetaItype_{mi_idx + 1}"
        mi_column = f"MI-{mi_idx + 1}"

        sending_gene_ids = gene_metadf.loc[
            (gene_metadf["Gene_type"] == "Ligand")
            & (gene_metadf["Associated_metaItype"] == mi_label),
            "Gene_ID",
        ].to_numpy(dtype=int)
        receiving_gene_ids = gene_metadf.loc[
            (gene_metadf["Gene_type"] == "Receptor")
            & (gene_metadf["Associated_metaItype"] == mi_label),
            "Gene_ID",
        ].to_numpy(dtype=int)

        gene_exp_lr[:, sending_gene_ids] += np.repeat(
            cfg.explevel_lr * cell_sending_MI[mi_column].to_numpy().reshape(-1, 1),
            repeats=len(sending_gene_ids),
            axis=1,
        )
        gene_exp_lr[:, receiving_gene_ids] += np.repeat(
            cfg.explevel_lr * cell_receiving_MI[mi_column].to_numpy().reshape(-1, 1),
            repeats=len(receiving_gene_ids),
            axis=1,
        )

    keep_gene_ids = gene_metadf.loc[
        gene_metadf["Gene_type"].isin(["Ligand", "Receptor"]),
        "Gene_ID",
    ].to_numpy(dtype=int)
    other_gene_ids = np.setdiff1d(np.arange(num_genes), keep_gene_ids)

    gene_exp_lr[gene_exp_lr < 0] = 0
    gene_exp_lr[:, other_gene_ids] = 0
    return gene_exp_lr


def simulate_upregulated_expression(
    cfg: SimulationConfig,
    rng: np.random.Generator,
    gene_metadf: pd.DataFrame,
    cell_sending_MI: pd.DataFrame,
    cell_receiving_MI: pd.DataFrame,
    sd_use: float,
) -> np.ndarray:
    """
    Simulate sender- and receiver-upregulated gene programs.
    """
    num_cells = cell_sending_MI.shape[0]
    num_genes = gene_metadf.shape[0]
    gene_exp_upgene = rng.normal(0, sd_use, (num_cells, num_genes))

    for mi_idx in range(cfg.num_meta_interactions):
        mi_label = f"MetaItype_{mi_idx + 1}"
        mi_column = f"MI-{mi_idx + 1}"

        sending_gene_ids = gene_metadf.loc[
            (gene_metadf["Gene_type"] == "Upregulated_gene_Sender")
            & (gene_metadf["Associated_metaItype"] == mi_label),
            "Gene_ID",
        ].to_numpy(dtype=int)
        receiving_gene_ids = gene_metadf.loc[
            (gene_metadf["Gene_type"] == "Upregulated_gene_Receiver")
            & (gene_metadf["Associated_metaItype"] == mi_label),
            "Gene_ID",
        ].to_numpy(dtype=int)

        gene_exp_upgene[:, sending_gene_ids] += np.repeat(
            cfg.explevel_upgene * cell_sending_MI[mi_column].to_numpy().reshape(-1, 1),
            repeats=len(sending_gene_ids),
            axis=1,
        )
        gene_exp_upgene[:, receiving_gene_ids] += np.repeat(
            cfg.explevel_upgene * cell_receiving_MI[mi_column].to_numpy().reshape(-1, 1),
            repeats=len(receiving_gene_ids),
            axis=1,
        )

    keep_gene_ids = gene_metadf.loc[
        gene_metadf["Gene_type"].isin(["Upregulated_gene_Sender", "Upregulated_gene_Receiver"]),
        "Gene_ID",
    ].to_numpy(dtype=int)
    other_gene_ids = np.setdiff1d(np.arange(num_genes), keep_gene_ids)

    gene_exp_upgene[gene_exp_upgene < 0] = 0
    gene_exp_upgene[:, other_gene_ids] = 0
    return gene_exp_upgene


# ============================================================
# Optional diagnostic plots
# ============================================================

def _save_figure(out_path: Path) -> None:
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_celltype_map(
    out_dir: Path,
    spatial_location: np.ndarray,
    celltype_cell: np.ndarray,
    cfg: SimulationConfig,
) -> None:
    cmap = ListedColormap(["#E59693", "#BBD5E7", "#C0E7DA"])
    norm = BoundaryNorm(np.arange(cfg.num_cell_types + 1) - 0.5, cfg.num_cell_types)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        spatial_location[:, 0],
        spatial_location[:, 1],
        c=celltype_cell,
        cmap=cmap,
        norm=norm,
        s=1.0,
    )
    plt.colorbar(ticks=np.arange(cfg.num_cell_types), label="Cell Type")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Spatial distribution of simulated cell types")
    _save_figure(out_dir / "Spatial_location_celltype.pdf")


def plot_inner_area_map(
    out_dir: Path,
    spatial_location: np.ndarray,
    inner_cell: np.ndarray,
) -> None:
    cmap_area = ListedColormap(["#54BAB9", "#E9DAC1"])
    norm_area = BoundaryNorm(np.arange(3) - 0.5, 2)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        spatial_location[:, 0],
        spatial_location[:, 1],
        c=inner_cell,
        cmap=cmap_area,
        norm=norm_area,
        s=1.0,
    )
    plt.colorbar(ticks=np.arange(2), label="Inner region flag")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Inner versus outer spatial regions")
    _save_figure(out_dir / "Spatial_location_innerarea.pdf")


def plot_edge_map(
    out_dir: Path,
    spatial_location: np.ndarray,
    celltype_cell: np.ndarray,
    edge_index_df: pd.DataFrame,
    mode: str = "all",
) -> None:
    cmap_cell = ListedColormap(["#E59693", "#BBD5E7", "#C0E7DA"])
    norm = BoundaryNorm(np.arange(4) - 0.5, 3)
    cmap_edge = ListedColormap(["#FBA518", "#ED2086", "grey"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        spatial_location[:, 0],
        spatial_location[:, 1],
        c=celltype_cell,
        cmap=cmap_cell,
        norm=norm,
        s=1.5,
    )

    if mode == "all":
        valid_types = {"MI-1", "MI-2"}
        suffix = "Spatial_location_celltype_edge.pdf"
    elif mode == "MI-1":
        valid_types = {"MI-1"}
        suffix = "Spatial_location_celltype_edge_MI1.pdf"
    elif mode == "MI-2":
        valid_types = {"MI-2"}
        suffix = "Spatial_location_celltype_edge_MI2.pdf"
    else:
        raise ValueError(f"Unknown edge plot mode: {mode}")

    for _, row in edge_index_df.loc[edge_index_df["MI_type"].isin(valid_types)].iterrows():
        sender = int(row["Sender"])
        receiver = int(row["Receiver"])
        mi_type = row["MI_type"]
        color = cmap_edge.colors[0] if mi_type == "MI-1" else cmap_edge.colors[1]

        ax.annotate(
            "",
            xy=(spatial_location[receiver, 0], spatial_location[receiver, 1]),
            xytext=(spatial_location[sender, 0], spatial_location[sender, 1]),
            arrowprops=dict(
                arrowstyle="->,head_length=0.06,head_width=0.06",
                color=color,
                lw=0.7,
            ),
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    sm = plt.cm.ScalarMappable(cmap=cmap_edge, norm=plt.Normalize(vmin=0, vmax=2))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, ticks=np.arange(3), label="Meta-interaction type")
    _save_figure(out_dir / suffix)


def plot_intrinsic_expression_summary(
    out_dir: Path,
    gene_exp_intrinsic: np.ndarray,
    celltype_cell: np.ndarray,
    gene_metadf: pd.DataFrame,
    cfg: SimulationConfig,
) -> None:
    gene_exp_intrinsic_celltype = np.zeros(
        (cfg.num_cell_types, gene_exp_intrinsic.shape[1]),
        dtype=float,
    )
    for celltype_idx in range(cfg.num_cell_types):
        gene_exp_intrinsic_celltype[celltype_idx, :] = np.mean(
            gene_exp_intrinsic[celltype_cell == celltype_idx, :],
            axis=0,
        )

    fig, ax = plt.subplots()
    im = ax.imshow(gene_exp_intrinsic_celltype, aspect="auto")
    plt.colorbar(im)
    ax.set_xticks([])
    ax.set_yticks(np.arange(cfg.num_cell_types))
    ax.set_yticklabels([f"Celltype_{name}" for name in ["A", "B", "C"][: cfg.num_cell_types]])
    ax.set_xlabel("")
    ax.set_ylabel("Cell Type")
    ax.set_title("")
    ax.set_xticks(np.arange(-0.5, gene_exp_intrinsic_celltype.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, gene_exp_intrinsic_celltype.shape[0]), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    _save_figure(out_dir / "Gene_expression_celltype.pdf")

    chosen = gene_exp_intrinsic_celltype[:, 0 : (cfg.num_cell_types * cfg.num_marker_intrinsic)]
    chosen = chosen[[2, 1, 0], :]
    num_cell_types, num_genes_use = chosen.shape

    cell_types = np.repeat(np.arange(num_cell_types), num_genes_use)
    genes = np.tile(np.arange(num_genes_use), num_cell_types)
    gene_expression = chosen.flatten()

    grey_to_red = mcolors.LinearSegmentedColormap.from_list(
        "grey_to_red",
        ["#D3D3D3", "#FF0000"],
    )

    plt.figure(figsize=(5.5, 3))
    scatter = plt.scatter(genes, cell_types, c=gene_expression, cmap=grey_to_red, s=200)
    plt.colorbar(scatter, label="")
    plt.ylabel("Cell types", fontsize=18)
    plt.xlabel("Genes", fontsize=18)

    x_labels = gene_metadf["Gene_name"].iloc[0 : (cfg.num_cell_types * cfg.num_marker_intrinsic)].tolist()
    x_colors = ["blue"] * cfg.num_marker_intrinsic + ["green"] * cfg.num_marker_intrinsic + ["red"] * cfg.num_marker_intrinsic
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize=15, rotation=30)
    for tick, color in zip(plt.gca().get_xticklabels(), x_colors):
        tick.set_color(color)

    y_positions = np.arange(num_cell_types)
    y_labels = [f"CT-{i}" for i in ["C", "B", "A"][:num_cell_types]]
    y_colors = ["blue", "green", "red"][:num_cell_types]
    plt.yticks(ticks=y_positions, labels=y_labels)
    for tick, color in zip(plt.gca().get_yticklabels(), y_colors):
        tick.set_color(color)
        tick.set_fontsize(15)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_figure(out_dir / "Gene_expression_celltype_choose.pdf")


def plot_nmf_boxplots(
    out_dir: Path,
    pair_matrix: np.ndarray,
    mi_numeric: np.ndarray,
    num_meta_interactions: int,
    prefix: str,
) -> None:
    nmf_model = NMF(n_components=2, init="nndsvda", random_state=0)
    pair_nmf = nmf_model.fit_transform(pair_matrix)

    for comp_idx in range(2):
        plt.figure()
        plt.boxplot(
            [pair_nmf[mi_numeric == (i + 1), comp_idx] for i in range(num_meta_interactions)],
            showfliers=False,
        )
        plt.xticks(
            np.arange(1, num_meta_interactions + 1),
            [f"MetaItype_{i + 1}" for i in range(num_meta_interactions)],
        )
        plt.xlabel("MetaI Type")
        plt.ylabel(f"NMF component {comp_idx + 1}")
        plt.title(f"NMF component {comp_idx + 1} of cell-pair features")
        _save_figure(out_dir / f"{prefix}_component{comp_idx + 1}_cellpair.pdf")


# ============================================================
# Core simulation for one experiment
# ============================================================

def run_single_experiment(
    cfg: SimulationConfig,
    setting_name: str,
    dropout_ratio: float,
    sd_use: float,
    experiment_index: int,
) -> None:
    """
    Generate one simulation replicate and save all outputs.
    """
    seed = hash((setting_name, experiment_index)) % (2**32)
    rng = np.random.default_rng(seed)

    out_dir = Path(cfg.data_root) / setting_name / f"Experiment_{experiment_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{setting_name} | Experiment_{experiment_index}] "
        f"dropout={dropout_ratio:.3f}, sd_use={sd_use:.3f}"
    )

    # 1. Build metadata and spatial layout
    gene_metadf = build_gene_metadata(cfg, rng)
    spatial_location = sample_spatial_locations(
        rng=rng,
        num_cells=cfg.num_cells,
        tissue_size=cfg.tissue_size,
        min_distance=cfg.min_distance,
    )
    celltype_cell = rng.integers(0, cfg.num_cell_types, size=cfg.num_cells)

    inner_circle_radius = cfg.tissue_size[0] * cfg.inner_circle_radius_ratio
    spatial_location, edge_index_df, inner_cell, edge_index = assign_edge_types(
        spatial_location=spatial_location,
        inner_circle_radius=inner_circle_radius,
        num_neighbors=cfg.num_neighbors,
    )

    # 2. Compute per-cell MI summaries
    (
        cell_neigh_metaIprop_df,
        cell_sending_MI_df,
        cell_receiving_MI_df,
        mi_numeric,
    ) = compute_cell_mi_summary(
        edge_index_df=edge_index_df,
        num_cells=cfg.num_cells,
        num_meta_interactions=cfg.num_meta_interactions,
        num_neighbors=cfg.num_neighbors,
    )

    # 3. Simulate expression layers
    gene_exp_intrinsic = simulate_intrinsic_expression(
        cfg=cfg,
        rng=rng,
        gene_metadf=gene_metadf,
        celltype_cell=celltype_cell,
        sd_use=sd_use,
    )
    gene_exp_lr = simulate_lr_expression(
        cfg=cfg,
        rng=rng,
        gene_metadf=gene_metadf,
        cell_sending_MI=cell_sending_MI_df,
        cell_receiving_MI=cell_receiving_MI_df,
        sd_use=sd_use,
    )
    gene_exp_upgene = simulate_upregulated_expression(
        cfg=cfg,
        rng=rng,
        gene_metadf=gene_metadf,
        cell_sending_MI=cell_sending_MI_df,
        cell_receiving_MI=cell_receiving_MI_df,
        sd_use=sd_use,
    )

    gene_exp = gene_exp_intrinsic + gene_exp_lr + gene_exp_upgene
    gene_exp = apply_expression_dropout(
        rng=rng,
        gene_exp=gene_exp,
        dropout_ratio=dropout_ratio,
    )

    # 4. Save required tables
    gene_exp_pd = pd.DataFrame(
        gene_exp,
        columns=gene_metadf["Gene_name"].tolist(),
        index=[f"Cell_{i}" for i in range(cfg.num_cells)],
    )
    spatial_location_pd = pd.DataFrame(
        spatial_location,
        columns=["X", "Y"],
        index=gene_exp_pd.index,
    )
    cell_metadf = cell_neigh_metaIprop_df.copy()
    cell_metadf["Celltype"] = celltype_cell

    edge_metadf = pd.DataFrame(
        {
            "Sender": edge_index_df["Sender"].to_numpy(dtype=int),
            "Receiver": edge_index_df["Receiver"].to_numpy(dtype=int),
            "MetaItype": edge_index_df["MI_type"].to_numpy(dtype=object),
        },
        index=[f"Edge_{i}" for i in range(edge_index_df.shape[0])],
    )

    gene_exp_pd.to_csv(out_dir / "gene_exp.csv", index=True)
    spatial_location_pd.to_csv(out_dir / "spatial_location.csv", index=True)
    gene_metadf.to_csv(out_dir / "gene_metadf.csv", index=False)
    cell_metadf.to_csv(out_dir / "cell_metadf.csv", index=True)
    edge_metadf.to_csv(out_dir / "edge_metadf.csv", index=True)
    cell_neigh_metaIprop_df.to_csv(out_dir / "cell_neigh_metaIprop.csv", index=True)

    # 5. Save AnnData object
    adata = sc.AnnData(X=gene_exp_pd)
    adata.obs = cell_metadf.copy()
    adata.obs.index = spatial_location_pd.index
    adata.var = gene_metadf.copy()
    adata.var.index = gene_metadf["Gene_name"].astype(str).to_numpy()
    adata.obsm["spatial"] = spatial_location_pd.to_numpy(dtype=float)
    adata.uns["edge_metadf"] = edge_metadf
    adata.write(out_dir / "adata_simulation.h5ad")

    # 6. Optional diagnostic figures
    if cfg.save_diagnostic_figures:
        plot_celltype_map(out_dir, spatial_location, celltype_cell, cfg)
        plot_inner_area_map(out_dir, spatial_location, inner_cell)
        plot_edge_map(out_dir, spatial_location, celltype_cell, edge_index_df, mode="all")
        plot_edge_map(out_dir, spatial_location, celltype_cell, edge_index_df, mode="MI-1")
        plot_edge_map(out_dir, spatial_location, celltype_cell, edge_index_df, mode="MI-2")
        plot_intrinsic_expression_summary(
            out_dir,
            gene_exp_intrinsic,
            celltype_cell,
            gene_metadf,
            cfg,
        )

        # Ligand-receptor pair diagnostic
        lr_sender_mask = gene_metadf["Gene_type"] == "Ligand"
        lr_receiver_mask = gene_metadf["Gene_type"] == "Receptor"
        edge_pairs = edge_index_df[["Sender", "Receiver"]].to_numpy(dtype=int)

        cellpair_lr = np.zeros((edge_pairs.shape[0], int((lr_sender_mask.sum() + lr_receiver_mask.sum()) / 2)))
        for edge_idx, (sender, receiver) in enumerate(edge_pairs):
            sender_exp = gene_exp_lr[sender, lr_sender_mask.to_numpy()]
            receiver_exp = gene_exp_lr[receiver, lr_receiver_mask.to_numpy()]
            cellpair_lr[edge_idx, :] = np.sqrt(sender_exp * receiver_exp)

        plot_nmf_boxplots(
            out_dir=out_dir,
            pair_matrix=cellpair_lr,
            mi_numeric=mi_numeric,
            num_meta_interactions=cfg.num_meta_interactions,
            prefix="NMF",
        )

        # Upregulated gene diagnostic
        up_sender_mask = gene_metadf["Gene_type"] == "Upregulated_gene_Sender"
        up_receiver_mask = gene_metadf["Gene_type"] == "Upregulated_gene_Receiver"
        cellpair_upgene = np.zeros((edge_pairs.shape[0], int((up_sender_mask.sum() + up_receiver_mask.sum()) / 2)))
        for edge_idx, (sender, receiver) in enumerate(edge_pairs):
            sender_exp = gene_exp_upgene[sender, up_sender_mask.to_numpy()]
            receiver_exp = gene_exp_upgene[receiver, up_receiver_mask.to_numpy()]
            cellpair_upgene[edge_idx, :] = np.sqrt(sender_exp * receiver_exp)

        plot_nmf_boxplots(
            out_dir=out_dir,
            pair_matrix=cellpair_upgene,
            mi_numeric=mi_numeric,
            num_meta_interactions=cfg.num_meta_interactions,
            prefix="NMF_upgenes",
        )


# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    cfg = SimulationConfig()
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)

    settings_df = build_settings_table(cfg)

    print("Generating simulation datasets for the following settings:")
    print(settings_df.to_string(index=False))
    print("")

    for _, row in settings_df.iterrows():
        setting_name = str(row["Dominant_Setting"])
        dropout_ratio = float(row["Dropout_ratio"])
        sd_use = float(row["sd_use"])

        for experiment_index in range(cfg.num_experiments):
            run_single_experiment(
                cfg=cfg,
                setting_name=setting_name,
                dropout_ratio=dropout_ratio,
                sd_use=sd_use,
                experiment_index=experiment_index,
            )

    print("Simulation data generation completed.")


if __name__ == "__main__":
    main()
