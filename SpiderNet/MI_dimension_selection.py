import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

DEFAULT_TARGET_N_CLIQUES = 10
DEFAULT_MIN_SIZE_FLOOR = 2

def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)

def compute_lr_spearman_correlation(processed) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack edge-level LR coexpression across batches and compute the
    LR-by-LR Spearman correlation matrix.
    """
    lr_coexp_all = []
    for batch_data in processed.spidernet_data:
        lr_coexp_all.append(_as_numpy(batch_data["cellpair_LRpair_neigh"]).astype(np.float32, copy=False))

    if len(lr_coexp_all) == 0:
        raise ValueError("processed.spidernet_data is empty; cannot perform MI dimension selection.")

    lr_coexp_all = np.concatenate(lr_coexp_all, axis=0)
    if lr_coexp_all.ndim != 2:
        raise ValueError(
            f"Expected a 2D LR coexpression matrix after stacking, got shape {lr_coexp_all.shape}."
        )

    n_lr = lr_coexp_all.shape[1]
    if n_lr == 0:
        raise ValueError("No LR pairs were found in the processed object.")

    if n_lr == 1:
        corr = np.ones((1, 1), dtype=np.float32)
    else:
        corr = stats.spearmanr(lr_coexp_all, axis=0).correlation
        corr = np.asarray(corr, dtype=np.float32)
        if corr.ndim == 0:
            corr = np.ones((1, 1), dtype=np.float32)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return lr_coexp_all, corr

def auto_selection_parameters(
    n_lr: int,
    lr_spearcor_threshold: float | None = None,
    min_clique_size: int | None = None,
    jaccard_thr: float | None = None,
) -> dict[str, float | int]:
    if n_lr > 100:
        params = {
            "lr_spearcor_threshold": 0.30,
            "min_clique_size": 4,
            "jaccard_thr": 0.50,
        }
    else:
        params = {
            "lr_spearcor_threshold": 0.20,
            "min_clique_size": 3,
            "jaccard_thr": 0.70,
        }

    if lr_spearcor_threshold is not None:
        params["lr_spearcor_threshold"] = float(lr_spearcor_threshold)
    if min_clique_size is not None:
        params["min_clique_size"] = int(min_clique_size)
    if jaccard_thr is not None:
        params["jaccard_thr"] = float(jaccard_thr)
    return params

def corr_to_graph(corr: np.ndarray, thr: float = 0.3, use_abs: bool = False) -> nx.Graph:
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got shape {corr.shape}")

    graph = nx.Graph()
    graph.add_nodes_from(range(corr.shape[0]))

    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            value = abs(corr[i, j]) if use_abs else corr[i, j]
            if np.isfinite(value) and value > thr:
                graph.add_edge(i, j, weight=float(corr[i, j]))
    return graph

def filter_cliques_by_min_size(cliques: list[list[int]], min_size: int) -> list[list[int]]:
    return [list(c) for c in cliques if len(c) >= min_size]

def find_maximal_cliques_with_relaxation(
    corr: np.ndarray,
    lr_spearcor_threshold: float,
    min_clique_size: int,
    target_n_cliques: int = DEFAULT_TARGET_N_CLIQUES,
    min_size_floor: int = DEFAULT_MIN_SIZE_FLOOR,
) -> dict[str, Any]:
    graph = corr_to_graph(corr, thr=lr_spearcor_threshold, use_abs=False)
    maximal_cliques_all = list(nx.find_cliques(graph))

    cur_min = int(min_clique_size)
    maximal_cliques = filter_cliques_by_min_size(maximal_cliques_all, cur_min)
    while len(maximal_cliques) < target_n_cliques and cur_min > min_size_floor:
        cur_min -= 1
        maximal_cliques = filter_cliques_by_min_size(maximal_cliques_all, cur_min)

    maximal_cliques = sorted(maximal_cliques, key=len, reverse=True)
    return {
        "graph": graph,
        "maximal_cliques_all": maximal_cliques_all,
        "maximal_cliques": maximal_cliques,
        "effective_min_clique_size": cur_min,
    }

def merge_cliques_by_jaccard(
    maximal_cliques: list[list[int]],
    jaccard_thr: float = 0.5,
) -> tuple[list[list[int]], list[list[int]]]:
    clq_sets = [set(c) for c in maximal_cliques]
    m = len(clq_sets)
    if m == 0:
        return [], []

    parent = list(range(m))
    rank = [0] * m

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    inv: dict[int, list[int]] = defaultdict(list)
    for i, subset in enumerate(clq_sets):
        for value in subset:
            inv[value].append(i)

    candidate_pairs: set[tuple[int, int]] = set()
    for ids in inv.values():
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                candidate_pairs.add((ids[a], ids[b]))

    for i, j in candidate_pairs:
        A, B = clq_sets[i], clq_sets[j]
        if len(A) < len(B):
            inter = sum((x in B) for x in A)
        else:
            inter = sum((x in A) for x in B)
        if inter == 0:
            continue
        union_size = len(A) + len(B) - inter
        jac = inter / union_size
        if jac >= jaccard_thr:
            union(i, j)

    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(m):
        comp[find(i)].append(i)

    groups = []
    merged_subsets = []
    for _, idxs in sorted(comp.items(), key=lambda item: min(item[1])):
        groups.append(sorted(idxs))
        merged = set()
        for idx in idxs:
            merged |= clq_sets[idx]
        merged_subsets.append(sorted(merged))
    return merged_subsets, groups

def build_disjoint_subset_order(
    corr: np.ndarray,
    merged_subsets: list[list[int]],
) -> dict[str, Any]:
    corr = np.asarray(corr, dtype=float)
    p = corr.shape[0]

    if p == 1:
        global_order = [0]
    else:
        distance = 1.0 - corr
        np.fill_diagonal(distance, 0.0)
        distance = np.clip(distance, 0.0, 2.0)
        linkage_matrix = linkage(squareform(distance, checks=False), method="average")
        global_order = list(leaves_list(linkage_matrix))

    membership: dict[int, list[int]] = {}
    subset_sizes = [len(s) for s in merged_subsets]
    for sid, subset in enumerate(merged_subsets):
        for idx in subset:
            membership.setdefault(idx, []).append(sid)

    assigned_to: dict[int, int] = {}
    for idx, sids in membership.items():
        best = sorted(sids, key=lambda sid: (-subset_sizes[sid], sid))[0]
        assigned_to[idx] = best

    clusters: dict[int, list[int]] = {sid: [] for sid in range(len(merged_subsets))}
    for idx, sid in assigned_to.items():
        clusters[sid].append(idx)
    clusters = {sid: sorted(idxs) for sid, idxs in clusters.items() if len(idxs) > 0}

    ordered_indices = []
    block_spans = []
    pos = 0
    for sid in sorted(clusters.keys()):
        idxs = clusters[sid]
        ordered_indices.extend(idxs)
        block_spans.append((pos, len(idxs), sid))
        pos += len(idxs)

    assigned_set = set(ordered_indices)
    unassigned = [i for i in global_order if i not in assigned_set]
    ordered_indices.extend(unassigned)

    return {
        "ordered_indices": ordered_indices,
        "block_spans": block_spans,
        "unassigned": unassigned,
        "clusters": clusters,
    }

def plot_lr_spearcorr_heatmap(
    corr: np.ndarray,
    merged_subsets: list[list[int]],
    output_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> dict[str, Any]:
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    order_info = build_disjoint_subset_order(corr, merged_subsets)
    ordered_indices = order_info["ordered_indices"]
    block_spans = order_info["block_spans"]
    corr_ord = corr[np.ix_(ordered_indices, ordered_indices)]

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        corr_ord,
        cmap="vlag",
        center=0,
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    ax.set_title(
        f"Spearman correlation between LR pairs ({len(merged_subsets)} filtered components)",
        pad=12,
    )
    ax.set_xlabel("LR pairs")
    ax.set_ylabel("LR pairs")

    for start, size, _ in block_spans:
        rect = Rectangle((start, start), size, size, fill=False, edgecolor="yellow", linewidth=3.0)
        ax.add_patch(rect)
        end = start + size
        ax.axhline(end, color="yellow", linewidth=0.8, alpha=0.6)
        ax.axvline(end, color="yellow", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        **order_info,
        "corr_ord": corr_ord,
        "output_path": str(output_path) if output_path is not None else None,
    }

def run_mi_dimension_selection(
    processed,
    output_dir: str | Path,
    lr_list: list[Any] | None = None,
    lr_spearcor_threshold: float | None = None,
    min_clique_size: int | None = None,
    jaccard_thr: float | None = None,
    target_n_cliques: int = DEFAULT_TARGET_N_CLIQUES,
    min_size_floor: int = DEFAULT_MIN_SIZE_FLOOR,
    show: bool = True,
    save_prefix: str = "mi_dimension_selection",
) -> dict[str, Any]:
    """
    Heuristic MI dimension selection based on the correlation structure among
    edge-level LR coexpression features.

    Returns a dictionary with the recommended MI dimension and diagnostic
    statistics. The recommendation is the number of merged LR correlation
    subsets using the same selection logic as the original Part0 workflow.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, corr = compute_lr_spearman_correlation(processed)
    n_lr = int(corr.shape[0])
    params = auto_selection_parameters(
        n_lr=n_lr,
        lr_spearcor_threshold=lr_spearcor_threshold,
        min_clique_size=min_clique_size,
        jaccard_thr=jaccard_thr,
    )

    clique_info = find_maximal_cliques_with_relaxation(
        corr=corr,
        lr_spearcor_threshold=float(params["lr_spearcor_threshold"]),
        min_clique_size=int(params["min_clique_size"]),
        target_n_cliques=target_n_cliques,
        min_size_floor=min_size_floor,
    )

    merged_subsets, clique_groups = merge_cliques_by_jaccard(
        clique_info["maximal_cliques"],
        jaccard_thr=float(params["jaccard_thr"]),
    )

    if len(merged_subsets) == 0:
        merged_subsets = [list(range(n_lr))]
        clique_groups = [list(range(len(clique_info["maximal_cliques"]))) if clique_info["maximal_cliques"] else []]
        warning_message = (
            "No merged LR subset was found under the current thresholds. "
            "Falling back to a single MI dimension recommendation."
        )
    else:
        warning_message = None

    heatmap_path = output_dir / f"{save_prefix}_LR_spearcorr_heatmap_by_merged_subsets.pdf"
    heatmap_info = plot_lr_spearcorr_heatmap(
        corr=corr,
        merged_subsets=merged_subsets,
        output_path=heatmap_path,
        show=show,
    )

    subset_sizes = [len(s) for s in merged_subsets]
    recommended_dim = int(len(merged_subsets))

    summary = {
        "recommended_dim_envir": recommended_dim,
        "num_lr_pairs": n_lr,
        "num_graph_edges": int(clique_info["graph"].number_of_edges()),
        "num_maximal_cliques_all": int(len(clique_info["maximal_cliques_all"])),
        "num_maximal_cliques_filtered": int(len(clique_info["maximal_cliques"])),
        "num_merged_subsets": int(len(merged_subsets)),
        "subset_sizes": subset_sizes,
        "num_unassigned_lr_pairs": int(len(heatmap_info["unassigned"])),
        "lr_spearcor_threshold": float(params["lr_spearcor_threshold"]),
        "min_clique_size_default": int(params["min_clique_size"]),
        "effective_min_clique_size": int(clique_info["effective_min_clique_size"]),
        "jaccard_thr": float(params["jaccard_thr"]),
        "target_n_cliques": int(target_n_cliques),
        "min_size_floor": int(min_size_floor),
        "warning": warning_message,
    }

    np.save(output_dir / f"{save_prefix}_LR_spearcorr.npy", corr)
    with open(output_dir / f"{save_prefix}_merged_subsets.pkl", "wb") as handle:
        pickle.dump(merged_subsets, handle)
    with open(output_dir / f"{save_prefix}_clique_groups.pkl", "wb") as handle:
        pickle.dump(clique_groups, handle)
    with open(output_dir / f"{save_prefix}_maximal_cliques.pkl", "wb") as handle:
        pickle.dump(clique_info["maximal_cliques"], handle)
    with open(output_dir / f"{save_prefix}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    subset_df = pd.DataFrame(
        {
            "subset_id": [f"subset_{i + 1}" for i in range(len(merged_subsets))],
            "size": subset_sizes,
            "lr_indices": [list(map(int, subset)) for subset in merged_subsets],
            "n_original_cliques": [len(group) for group in clique_groups],
        }
    )
    subset_df.to_csv(output_dir / f"{save_prefix}_merged_subset_summary.csv", index=False)

    if lr_list is not None:
        lr_names = []
        for lr in lr_list:
            try:
                ligand = "+".join(lr[0])
                receptor = "+".join(lr[1])
                lr_names.append(f"{ligand} -> {receptor}")
            except Exception:
                lr_names.append(str(lr))
        assignment_rows = []
        for subset_id, subset in enumerate(merged_subsets, start=1):
            for lr_idx in subset:
                assignment_rows.append(
                    {
                        "subset_id": f"subset_{subset_id}",
                        "lr_index": int(lr_idx),
                        "lr_name": lr_names[lr_idx],
                    }
                )
        pd.DataFrame(assignment_rows).to_csv(
            output_dir / f"{save_prefix}_lr_assignments.csv", index=False
        )

    return {
        **summary,
        "merged_subsets": merged_subsets,
        "clique_groups": clique_groups,
        "maximal_cliques": clique_info["maximal_cliques"],
        "lr_spearcorr": corr,
        "heatmap_path": str(heatmap_path),
        "summary_path": str(output_dir / f"{save_prefix}_summary.json"),
        "subset_summary_path": str(output_dir / f"{save_prefix}_merged_subset_summary.csv"),
    }
