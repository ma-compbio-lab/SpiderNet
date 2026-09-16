"""Graph perturbation, model fitting, MI alignment and LR component analysis.

Scientific functions are preserved from the original notebook. Paths are supplied
by run_benchmarks.py; fitting and scratch caches stay in the original Results tree.
"""
from pathlib import Path
import os
import sys
import copy
import gc
import hashlib
import io
import json
import pickle
import random
import time
from types import SimpleNamespace

WORKSPACE = Path(os.environ.get("SPIDERNET_WORKSPACE", "D:/SpiderNet")).resolve()
PACKAGE_ROOT = Path(os.environ.get("SPIDERNET_PACKAGE_ROOT", str(WORKSPACE / "SpiderNet_proj/SpiderNet_Project/SpiderNet"))).resolve()
RESULTS_ROOT = Path(os.environ.get("SPIDERNET_RESULTS_ROOT", str(WORKSPACE / "Results"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("SPIDERNET_ROBUSTNESS_WORK_DIR", str(RESULTS_ROOT / "MI_robustness_S31_S32"))).resolve()
LOCAL_OUTPUT_ROOT = Path(os.environ.get("SPIDERNET_ROBUSTNESS_OUTPUT_DIR", str(Path(__file__).parent / "output"))).resolve()
TABLES = LOCAL_OUTPUT_ROOT / "tables"
FIGURES = LOCAL_OUTPUT_ROOT / "figures"
CACHE = OUTPUT_ROOT / "cache"
RUN_MODE = "full"
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ["NUMBA_CACHE_DIR"] = str(CACHE / "numba")
os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import torch
from SpiderNet.api import build_model, run_training, normalize_outputs
from SpiderNet.config import TrainingConfig
from SpiderNet.io import ProcessedData
from SpiderNet import MI_dimension_selection as mids
from SpiderNet.utils import spatial_neighborindex_generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATUS = []
PAIR_RESULTS = {}
STABILITY_TABLES = {}
LR_PANELS = {}


MAX_EPOCH = 20000
N_JOBS = 5
BASE_SEED = 202400
RANDOM_SEEDS = list(range(202401, 202411))
SUBSAMPLE_SEEDS = list(range(202501, 202511))
MISANNOTATION_RATES = (0.10, 0.20, 0.30)
MISANNOTATION_SEEDS = {0.10: 202610, 0.20: 202620, 0.30: 202630}
SUBSAMPLE_FRACTION = 0.70
FIX_HIDDEN_WIDTH_WITHIN_DATASET = False
REUSE_LEGACY_STABILITY = True
LEGACY_STABILITY_ROOT = RESULTS_ROOT / "HGSOC/MI_stability/HGSOC_MI_stability_dim15_V1"
RANK_CHUNK_ROWS = 100000
INFERENCE_EDGE_CHUNK = 100000
USE_PUBLISHED_Y_LIMITS_WHEN_POSSIBLE = True  # always expand to include every finite point

DATASETS = {
    "HGSOC": {
        "title": "HGSOC (CosMx)", "processed_dir": RESULTS_ROOT / "HGSOC/ProcessedData",
        "reference_dir": RESULTS_ROOT / "HGSOC/V1/SpiderNet_Result_dim15",
        "reference_dim": 15, "dims": (12, 15, 18), "reference_k": 10,
    },
    "PerturbFISH": {
        "title": "Melanoma (Perturb-FISH)", "processed_dir": RESULTS_ROOT / "PerturbFISH/ProcessedData",
        # Explicit V1 path: the mutable run_dirs.json currently points to a different V3 analysis.
        "reference_dir": RESULTS_ROOT / "PerturbFISH/V1/SpiderNet_Result_dim23",
        "reference_dim": 23, "dims": (20, 23, 26), "reference_k": 10,
    },
    "AgingBrain": {
        "title": "Ageing mouse brain (MERFISH)", "processed_dir": RESULTS_ROOT / "AgingBrain/ProcessedData",
    },
    "Pancancer": {
        "title": "Immuno-oncology (MERSCOPE)", "processed_dir": RESULTS_ROOT / "Pancancer/ProcessedData_entire",
    },
}


def jsonable(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value

def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def file_stamp(path):
    path = Path(path)
    s = path.stat()
    return {"path": str(path.resolve()), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}

def record(panel, item, state, detail=""):
    row = {"panel": panel, "item": item, "status": state, "detail": str(detail)}
    STATUS.append(row)
    print(f"[{panel}] {item}: {state} {detail}", flush=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def torch_load_cpu(path_or_buffer):
    return torch.load(path_or_buffer, map_location="cpu")

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch_load_cpu(io.BytesIO(b))
        return super().find_class(module, name)

def read_pickle_cpu(path):
    with Path(path).open("rb") as handle:
        return CPUUnpickler(handle).load()

def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def edges_numpy(graph):
    edge = as_numpy(graph.edge_index)
    # SpiderNet uses E x 2 (not standard PyG's 2 x E).
    if edge.ndim != 2 or edge.shape[1] != 2:
        raise ValueError(f"Expected SpiderNet E x 2 edges, got {edge.shape}")
    return edge.astype(np.int64, copy=False)

def load_context(dataset):
    cfg = DATASETS[dataset]
    d = Path(cfg["processed_dir"])
    graph_path = d / "SpiderNet_data_pyg_list.pkl"
    if graph_path.exists():
        graphs = read_pickle_cpu(graph_path)
    else:
        graph_path = d / "SpiderNet_data_pyg_list.pt"
        graphs = torch_load_cpu(graph_path)
    genes = np.asarray(read_pickle_cpu(d / "genenames_train.pkl")).astype(str)
    lr = read_pickle_cpu(d / "LR_list.pkl")
    samples = list(map(str, read_pickle_cpu(d / "batch_cell_unique.pkl")))
    if len(samples) != len(graphs):
        raise ValueError("Sample order does not match the graph list.")
    for sample, g in zip(samples, graphs):
        edge = edges_numpy(g)
        if g.x.shape[1] != len(genes) or g.cellpair_LRpair_neigh.shape != (len(edge), len(lr)):
            raise ValueError(f"Expression/LR dimensions disagree in slice {sample}.")
        if len(g.cellnames) != g.x.shape[0] or len(set(map(str, g.cellnames))) != g.x.shape[0]:
            raise ValueError(f"Invalid cell IDs in slice {sample}.")
        if not np.array_equal(np.asarray(g.genenames).astype(str), genes):
            raise ValueError(f"Gene ordering disagrees in slice {sample}.")
        degree = np.bincount(edge[:, 0], minlength=g.x.shape[0])
        if np.any(degree != cfg["reference_k"]) or np.any(edge[:, 0] == edge[:, 1]):
            raise ValueError(f"Expected directed K={cfg['reference_k']} graph without self edges: {sample}")
    source = {
        "graphs": file_stamp(graph_path), "genes": file_stamp(d / "genenames_train.pkl"),
        "lr": file_stamp(d / "LR_list.pkl"), "sample_ids": samples,
    }
    context = SimpleNamespace(
        dataset=dataset, cfg=cfg, graphs=graphs, genes=genes, lr=lr, samples=samples,
        source=source, fingerprint=digest(source),
        counts=np.array([len(edges_numpy(g)) for g in graphs], dtype=np.int64),
    )
    context.n_cells = int(sum(g.x.shape[0] for g in graphs))
    print(f"Loaded {dataset}: {len(graphs)} slices, {context.n_cells:,} cells, {sum(context.counts):,} edges")
    return context

CODE_HASH = digest({
    p: hashlib.sha256((PACKAGE_ROOT / "SpiderNet" / p).read_bytes()).hexdigest()
    for p in ["api.py", "model.py", "MI_dimension_selection.py"]
})
PROTOCOL_VERSION = "S31_S32_spearman_alignment_v1"


def processed_view(context, graphs):
    # Separate Data stores: api.run_training moves its graphs to GPU in place.
    return ProcessedData(
        spidernet_data=[copy.copy(g) for g in graphs],
        genenames_train=context.genes.copy(), lr_list=context.lr,
    )

def knn_view(context, k):
    if k == context.cfg["reference_k"]:
        return list(context.graphs)
    if not 0 < k < context.cfg["reference_k"]:
        raise ValueError("This controlled sweep supports K <= the original graph K only.")
    out = []
    rows = []
    for sid, g in zip(context.samples, context.graphs):
        edge = edges_numpy(g)
        pos = as_numpy(g.pos).astype(np.float64)
        dist2 = np.sum((pos[edge[:, 0]] - pos[edge[:, 1]]) ** 2, axis=1)
        order = np.lexsort((np.arange(len(edge)), dist2, edge[:, 0]))
        # Every original sender has exactly reference_k neighbors (checked on load).
        chosen = order.reshape(g.x.shape[0], context.cfg["reference_k"])[:, :k].reshape(-1)
        h = copy.copy(g)
        h.edge_index = torch.from_numpy(edge[chosen].copy())
        h.cellpair_LRpair_neigh = g.cellpair_LRpair_neigh[torch.as_tensor(chosen)].clone()
        out.append(h)
        rows.append({"slice_id": sid, "k": k, "n_cells": g.x.shape[0], "n_edges": len(chosen)})
    pd.DataFrame(rows).to_csv(TABLES / f"{context.dataset}_K{k}_graph_summary.csv", index=False)
    return out

def misannotate(context, rate, seed, audit_path=None):
    sizes = np.array([g.x.shape[0] for g in context.graphs])
    offsets = np.r_[0, np.cumsum(sizes)]
    original = np.concatenate([as_numpy(g.cell_class_onehot).argmax(axis=1) for g in context.graphs])
    classes = np.asarray(context.graphs[0].cell_class_unique).astype(str)
    for g in context.graphs:
        if not np.array_equal(np.asarray(g.cell_class_unique).astype(str), classes):
            raise ValueError("One-hot cell-type vocabulary/order differs across slices.")
    if len(classes) < 2:
        raise ValueError("At least two cell types are required to misannotate labels.")
    rng = np.random.default_rng(seed)
    n_changed = int(np.floor(rate * len(original) + 0.5))
    selected = np.sort(rng.choice(len(original), size=n_changed, replace=False))
    changed = original.copy()
    # A nonzero offset modulo C samples uniformly from all incorrect labels.
    changed[selected] = (original[selected] + rng.integers(1, len(classes), len(selected))) % len(classes)
    assert np.count_nonzero(original != changed) == n_changed
    graphs = []
    audit = []
    for i, g in enumerate(context.graphs):
        labels = changed[offsets[i]:offsets[i + 1]]
        h = copy.copy(g)
        h.cell_class_onehot = torch.nn.functional.one_hot(torch.as_tensor(labels, dtype=torch.long), len(classes)).to(g.cell_class_onehot.dtype)
        h.cell_class = classes[labels]
        graphs.append(h)
        local = selected[(selected >= offsets[i]) & (selected < offsets[i + 1])] - offsets[i]
        audit.append(pd.DataFrame({
            "slice_id": context.samples[i], "cell_id": np.asarray(g.cellnames).astype(str)[local],
            "old_label": classes[original[offsets[i] + local]], "new_label": classes[labels[local]],
            "rate": rate, "label_seed": seed,
        }))
    audit = pd.concat(audit, ignore_index=True)
    if audit_path is not None:
        audit.to_csv(audit_path, index=False)
    return graphs, audit

FACTOR_FILE = "Factor_envir_use.npy"
LOADING_FILES = {"LR loading": "loading_LR_use", "Regulatory gene": "loading_sender_use", "Target gene": "loading_receiver_use"}

def load_outputs(run_dir, context, graphs, slice_indices, dim):
    run_dir = Path(run_dir)
    factors = np.load(run_dir / FACTOR_FILE, mmap_mode="r")
    counts = np.array([len(edges_numpy(g)) for g in graphs], dtype=np.int64)
    if factors.shape != (int(counts.sum()), dim):
        raise ValueError(f"{run_dir}: expected factor shape {(int(counts.sum()), dim)}, got {factors.shape}")
    offsets = np.r_[0, np.cumsum(counts)]
    loads = {}
    for metric, stem in LOADING_FILES.items():
        if (run_dir / f"{stem}.npy").exists():
            arr = np.load(run_dir / f"{stem}.npy")
        else:
            df = pd.read_csv(run_dir / f"{stem}.csv", index_col=0)
            if metric != "LR loading":
                if set(df.columns) != set(context.genes):
                    raise ValueError(f"Gene identities differ in {stem}")
                df = df.loc[:, context.genes]
            arr = df.to_numpy()
        expected = (dim, len(context.lr) if metric == "LR loading" else len(context.genes))
        if arr.shape != expected:
            raise ValueError(f"Loading shape mismatch: {stem}: {arr.shape} vs {expected}")
        loads[metric] = arr
    result = SimpleNamespace(
        run_dir=run_dir, graphs=graphs, slice_indices=list(slice_indices), factors=factors,
        blocks=[factors[offsets[i]:offsets[i + 1]] for i in range(len(graphs))],
        loadings=loads, dim=dim, context=context,
    )
    result.stamp = digest([file_stamp(run_dir / FACTOR_FILE)] + [
        file_stamp(run_dir / (stem + (".npy" if (run_dir / f"{stem}.npy").exists() else ".csv")))
        for stem in LOADING_FILES.values()
    ])
    return result

def infer_chunked(model, graphs, device=DEVICE, chunk_size=INFERENCE_EDGE_CHUNK):
    model = model.to(device).eval()
    factor_list = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        for g in graphs:
            exp = g.x.to(device)
            sender = model.enc_factor_envir_pre_sender(exp)
            receiver = model.enc_factor_envir_pre_receiver(exp)
            edge = edges_numpy(g)
            chunks = []
            for start in range(0, len(edge), chunk_size):
                e = torch.as_tensor(edge[start:start + chunk_size], device=device)
                joined = torch.cat((sender[e[:, 0]], receiver[e[:, 1]]), dim=1)
                f = model.Sigmoid(model.enc_factor_envir(joined))
                chunks.append(f.float().cpu().numpy())
            factor_list.append(np.concatenate(chunks, axis=0))
            del exp, sender, receiver, chunks
        outputs = {
            "factor_envir_list": factor_list, "factor_envir": np.vstack(factor_list),
            "loading_lr": model.Relu(model.loading_LR_ori).float().cpu().numpy(),
            "loading_sender": model.Relu(model.loading_sender_ori).float().cpu().numpy(),
            "loading_receiver": model.Relu(model.loading_receiver_ori).float().cpu().numpy(),
        }
    return normalize_outputs(outputs)

def save_outputs(run_dir, results):
    mapping = {FACTOR_FILE: "factor_envir", "loading_LR_use.npy": "loading_lr",
               "loading_sender_use.npy": "loading_sender", "loading_receiver_use.npy": "loading_receiver"}
    for name, key in mapping.items():
        np.save(Path(run_dir) / name, results[key])

def reference_output(context):
    cfg = context.cfg
    return load_outputs(cfg["reference_dir"], context, list(context.graphs), range(len(context.graphs)), cfg["reference_dim"])

def run_or_load(context, experiment, graphs, dim, seed, k=None, slice_indices=None, extra=None):
    if slice_indices is None:
        slice_indices = list(range(len(context.graphs)))
    hidden = (64 if context.cfg["reference_dim"] <= 20 else 128) if FIX_HIDDEN_WIDTH_WITHIN_DATASET else (64 if dim <= 20 else 128)
    spec = {
        "protocol": PROTOCOL_VERSION, "dataset": context.dataset, "experiment": experiment,
        "source_fingerprint": context.fingerprint, "dim": dim, "k": k,
        "seed": seed, "slice_indices": list(slice_indices), "max_epoch": MAX_EPOCH,
        "hidden_channels": hidden, "n_jobs": N_JOBS, "extra": extra or {}, "code_hash": CODE_HASH,
    }
    key = digest(spec)
    run_dir = OUTPUT_ROOT / "runs" / context.dataset / experiment / f"M{dim}_K{k}_seed{seed}_{key[:12]}"
    if (run_dir / "complete.json").exists():
        if read_json(run_dir / "run_manifest.json") != jsonable(spec):
            raise ValueError(f"Cached run specification differs: {run_dir}")
        return load_outputs(run_dir, context, graphs, slice_indices, dim)
    if RUN_MODE != "full":
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_manifest.json", spec)
    print(f"Training/recovering {context.dataset} {experiment} M={dim}, K={k}, seed={seed}", flush=True)
    processed = processed_view(context, graphs)
    cfg = TrainingConfig(dim_envir=dim, n_jobs=N_JOBS, max_epoch=MAX_EPOCH, version="S31_S32")
    set_seed(seed)
    model = build_model(processed, cfg, device=DEVICE, hidden_channels=hidden)
    t0 = time.time()
    try:
        model = run_training(model, processed, cfg, model_dir=run_dir / "Model", device=DEVICE)
        results = infer_chunked(model, graphs)
        save_outputs(run_dir, results)
        write_json(run_dir / "complete.json", {"spec_hash": key, "elapsed_min": (time.time() - t0) / 60})
    finally:
        for g in processed.spidernet_data:
            g.cpu()
        del model, processed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return load_outputs(run_dir, context, graphs, slice_indices, dim)

def legacy_output(context, experiment, repeat_id, seed, graphs, indices):
    if not REUSE_LEGACY_STABILITY or MAX_EPOCH != 20000:
        return None
    dirname = "experiment_1_random_seed" if experiment == "Random seed" else "experiment_2_70pct_slices"
    d = LEGACY_STABILITY_ROOT / dirname / f"repeat_{repeat_id:02d}_seed_{seed}"
    if not (d / "experiment_metadata.json").exists() or not (d / FACTOR_FILE).exists():
        return None
    ref_meta = read_json(LEGACY_STABILITY_ROOT / "reference_summary.json")
    if Path(ref_meta["reference_run_dir"]).resolve() != Path(context.cfg["reference_dir"]).resolve():
        raise ValueError("Legacy stability fits used a different reference model.")
    meta = read_json(d / "experiment_metadata.json")
    if meta["seed"] != seed or list(meta["slice_indices"]) != list(indices):
        raise ValueError(f"Legacy seed/slice indices disagree: {d}")
    if ref_meta["n_slices"] != len(context.graphs) or ref_meta["dim_envir"] != 15:
        raise ValueError("Legacy stability dataset dimensions differ.")
    return load_outputs(d, context, graphs, indices, 15)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan
    x = x[good] - np.mean(x[good])
    y = y[good] - np.mean(y[good])
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / den, -1, 1)) if den > 0 else np.nan

def normalized_columns(matrix, path, method="spearman"):
    n, m = matrix.shape
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(n, m), fortran_order=True)
    valid = np.zeros(m, dtype=bool)
    for j in range(m):
        col = np.array(matrix[:, j], dtype=np.float64)
        if not np.all(np.isfinite(col)):
            arr[:, j] = 0
            continue
        if method == "spearman":
            col = stats.rankdata(col, method="average")
        col -= np.mean(col)
        norm = np.linalg.norm(col)
        valid[j] = norm > 0
        arr[:, j] = col / norm if norm > 0 else 0
    arr.flush()
    return arr, valid

def cross_spearman(a, b, cache_dir):
    if a.shape[0] != b.shape[0] or a.shape[0] < 3:
        raise ValueError("Spearman profiles need >=3 common observations.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    za, va = normalized_columns(a, cache_dir / "rank_a.npy")
    zb, vb = normalized_columns(b, cache_dir / "rank_b.npy")
    corr = np.zeros((a.shape[1], b.shape[1]), dtype=np.float64)
    for start in range(0, a.shape[0], RANK_CHUNK_ROWS):
        corr += np.asarray(za[start:start + RANK_CHUNK_ROWS]).T @ np.asarray(zb[start:start + RANK_CHUNK_ROWS])
    corr = np.clip(corr, -1, 1)
    corr[~va, :] = np.nan
    corr[:, ~vb] = np.nan
    del za, zb
    # These are task-owned scratch files, recreated on each comparison.
    (cache_dir / "rank_a.npy").unlink()
    (cache_dir / "rank_b.npy").unlink()
    return corr

def greedy_pairs(corr):
    score = np.asarray(corr, dtype=float).copy()
    pairs = []
    for _ in range(min(score.shape)):
        if not np.any(np.isfinite(score)):
            raise ValueError("Cannot complete one-to-one MI alignment: remaining correlations undefined.")
        i, j = np.unravel_index(np.nanargmax(score), score.shape)
        pairs.append((int(i), int(j)))
        score[i, :] = np.nan
        score[:, j] = np.nan
    return pairs

def match_edges(a, b):
    if a.context.fingerprint != b.context.fingerprint:
        raise ValueError("Comparison uses different processed cells, genes, or LR features.")
    apos = {s: i for i, s in enumerate(a.slice_indices)}
    bpos = {s: i for i, s in enumerate(b.slice_indices)}
    plan = []
    for sid in sorted(set(apos) & set(bpos)):
        ia, ib = apos[sid], bpos[sid]
        ga, gb = a.graphs[ia], b.graphs[ib]
        if not np.array_equal(np.asarray(ga.cellnames).astype(str), np.asarray(gb.cellnames).astype(str)):
            raise ValueError(f"Cell order differs in slice {sid}.")
        ea, eb = edges_numpy(ga), edges_numpy(gb)
        n = ga.x.shape[0]
        ka = ea[:, 0] * n + ea[:, 1]
        kb = eb[:, 0] * n + eb[:, 1]
        if len(np.unique(ka)) != len(ka) or len(np.unique(kb)) != len(kb):
            raise ValueError(f"Duplicate directed edges in slice {sid}.")
        if np.array_equal(ea, eb):
            xa = xb = np.arange(len(ea))
        else:
            _, xa, xb = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
        if not len(xa):
            raise ValueError(f"No common edges in slice {sid}.")
        plan.append((sid, ia, ib, xa, xb))
    if not plan:
        raise ValueError("No shared slices.")
    return plan

def compare_outputs(a, b, tag):
    # Matrix rows = model a MIs, columns = model b MIs.
    out = TABLES / tag
    out.mkdir(parents=True, exist_ok=True)
    signature = digest({"protocol": PROTOCOL_VERSION, "a": a.stamp, "b": b.stamp,
                        "source": a.context.fingerprint, "a_slices": a.slice_indices, "b_slices": b.slice_indices})
    meta_path = out / "comparison.json"
    if meta_path.exists() and read_json(meta_path).get("signature") == signature:
        corr = pd.read_csv(out / "spearman_matrix.csv", index_col=0).to_numpy()
        pairs = [(int(r.row_mi) - 1, int(r.col_mi) - 1) for r in pd.read_csv(out / "alignment.csv").itertuples()]
        metrics = pd.read_csv(out / "pearson_by_MI.csv")
        return {"corr": corr, "pairs": pairs, "metrics": metrics, "tag": tag}
    plan = match_edges(a, b)
    n = sum(len(t[3]) for t in plan)
    scratch = CACHE / tag
    scratch.mkdir(parents=True, exist_ok=True)
    pa = np.lib.format.open_memmap(scratch / "profile_a.npy", mode="w+", dtype=np.float32, shape=(n, a.dim))
    pb = np.lib.format.open_memmap(scratch / "profile_b.npy", mode="w+", dtype=np.float32, shape=(n, b.dim))
    offset = 0
    summary = []
    for sid, ia, ib, xa, xb in plan:
        end = offset + len(xa)
        pa[offset:end] = a.blocks[ia][xa]
        pb[offset:end] = b.blocks[ib][xb]
        summary.append({"slice_index": sid, "slice_id": a.context.samples[sid], "n_common_edges": len(xa)})
        # Exact index pairs are saved per slice to make comparisons auditable without huge text CSVs.
        np.savez_compressed(out / f"common_edge_indices_slice_{sid:03d}.npz", row_edge=xa, col_edge=xb)
        offset = end
    pa.flush(); pb.flush()
    print(f"Comparing {tag}: {n:,} common directed edges ({a.dim} x {b.dim} MIs)", flush=True)
    corr = cross_spearman(pa, pb, scratch)
    pairs = greedy_pairs(corr)
    rows = []
    for i, j in pairs:
        for metric in ["MI strength", *LOADING_FILES]:
            x, y = (pa[:, i], pb[:, j]) if metric == "MI strength" else (a.loadings[metric][i], b.loadings[metric][j])
            rows.append({"row_mi": i + 1, "col_mi": j + 1, "metric": metric,
                         "corr": safe_pearson(x, y), "alignment_spearman": corr[i, j],
                         "n_common_edges": n, "n_slices_used": len(plan)})
    metrics = pd.DataFrame(rows)
    pd.DataFrame(corr, index=[f"MI-{i+1}" for i in range(a.dim)], columns=[f"MI-{i+1}" for i in range(b.dim)]).to_csv(out / "spearman_matrix.csv")
    pd.DataFrame([{"row_mi": i + 1, "col_mi": j + 1, "spearman": corr[i, j]} for i, j in pairs]).to_csv(out / "alignment.csv", index=False)
    metrics.to_csv(out / "pearson_by_MI.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "common_edges_by_slice.csv", index=False)
    write_json(meta_path, {"signature": signature, "row_model": str(a.run_dir), "column_model": str(b.run_dir),
                          "alignment": "greedy_one_to_one_signed_spearman", "n_common_edges": n,
                          "undefined_spearman_entries": int(np.isnan(corr).sum()), "undefined_pearson_entries": int(metrics['corr'].isna().sum())})
    del pa, pb
    (scratch / "profile_a.npy").unlink()
    (scratch / "profile_b.npy").unlink()
    return {"corr": corr, "pairs": pairs, "metrics": metrics, "tag": tag}


def lr_label(pair):
    return "+".join(map(str, pair[0])) + " -> " + "+".join(map(str, pair[1]))

def stream_lr_matrix(processed_dir, out_dir):
    import anndata as ad
    from scipy import sparse
    d = Path(processed_dir)
    summary = read_json(d / "bundle_summary.json")
    cfg = summary["config_snapshot"]
    lr = read_pickle_cpu(d / "LR_list.pkl")
    samples = list(map(str, read_pickle_cpu(d / "batch_cell_unique.pkl")))
    source_stamp = digest([file_stamp(d / "adata_all.h5ad"), file_stamp(d / "LR_list.pkl"), cfg])
    coexp_path = out_dir / "lr_coexpression.npy"
    meta_path = out_dir / "lr_coexpression_complete.json"
    if meta_path.exists() and read_json(meta_path).get("source_stamp") == source_stamp and coexp_path.exists():
        return np.load(coexp_path, mmap_mode="r"), lr
    backed = ad.read_h5ad(d / "adata_all.h5ad", backed="r")
    try:
        group_col = cfg["sample_group_col"]
        # GroupBy.indices avoids repeatedly scanning all cells for each tissue slice.
        labels = np.asarray(backed.obs[group_col]).astype(str)
        indices = pd.Series(np.arange(len(labels))).groupby(labels, sort=False).apply(np.asarray).to_dict()
        if set(samples) != set(indices):
            raise ValueError("Processed AnnData sample IDs do not match the saved graph sample IDs.")
        k = int(cfg["num_neighbors"])
        both = bool(cfg.get("if_bothdirections", False))
        if both:
            raise ValueError("Streaming fallback currently expects the directed, non-symmetrized graph used in this paper.")
        spatial = np.asarray(backed.obsm["spatial"])
        all_lr_genes = sorted({str(g) for pair in lr for side in pair[:2] for g in side})
        missing = set(all_lr_genes) - set(backed.var_names)
        if missing:
            raise ValueError(f"Processed expression lacks retained LR genes: {sorted(missing)}")
        loc = {g: i for i, g in enumerate(all_lr_genes)}
        var_idx = np.array([backed.var_names.get_loc(g) for g in all_lr_genes])
        n_edges = sum(len(indices[s]) * k for s in samples)
        matrix = np.lib.format.open_memmap(coexp_path, mode="w+", dtype=np.float32, shape=(n_edges, len(lr)))
        offset = 0
        for sid in samples:
            rows = np.sort(np.asarray(indices[sid], dtype=int))
            if len(rows) <= k:
                raise ValueError(f"Not enough cells for K={k} in slice {sid}")
            # Backed sparse indexing supports this two-axis view in the installed AnnData version.
            x = backed[rows, var_idx].X
            x = x.toarray() if sparse.issparse(x) else np.asarray(x)
            _, edges = spatial_neighborindex_generation(spatial[rows], k)
            block = np.empty((len(edges), len(lr)), dtype=np.float32)
            for j, pair in enumerate(lr):
                lig = x[:, [loc[str(g)] for g in pair[0]]]
                rec = x[:, [loc[str(g)] for g in pair[1]]]
                lig = np.prod(lig, axis=1) ** (1.0 / lig.shape[1]) if lig.shape[1] > 1 else lig[:, 0]
                rec = np.prod(rec, axis=1) ** (1.0 / rec.shape[1]) if rec.shape[1] > 1 else rec[:, 0]
                block[:, j] = np.sqrt(lig[edges[:, 0]] * rec[edges[:, 1]])
            matrix[offset:offset + len(edges)] = block
            offset += len(edges)
            print(f"S31a LR features: {sid}, {offset:,}/{n_edges:,} edges", flush=True)
        assert offset == n_edges
        matrix.flush()
        write_json(meta_path, {"source_stamp": source_stamp, "n_edges": n_edges, "n_lr": len(lr)})
        return matrix, lr
    finally:
        backed.file.close()

def prepare_lr_panel(dataset):
    d = DATASETS[dataset]["processed_dir"]
    dest = TABLES / f"S31a_{dataset}"
    dest.mkdir(parents=True, exist_ok=True)
    original_corr = d / "mi_dimension_selection_LR_spearcorr.npy"
    original_subsets = d / "mi_dimension_selection_merged_subsets.pkl"
    recompute_signature = digest({"adata": file_stamp(d / "adata_all.h5ad"),
                                  "lr": file_stamp(d / "LR_list.pkl"),
                                  "bundle": file_stamp(d / "bundle_summary.json"), "code_hash": CODE_HASH})
    if original_corr.exists() and original_subsets.exists():
        corr = np.load(original_corr)
        subsets = read_pickle_cpu(original_subsets)
        summary = read_json(d / "mi_dimension_selection_summary.json")
        provenance = {"mode": "existing_dimension_selection", "corr": file_stamp(original_corr), "subsets": file_stamp(original_subsets)}
    elif ((dest / "complete.json").exists()
          and read_json(dest / "complete.json").get("source_signature") == recompute_signature):
        corr = np.load(dest / "lr_spearman.npy")
        subsets = read_json(dest / "merged_subsets.json")
        summary = read_json(dest / "selection_summary.json")
        provenance = read_json(dest / "complete.json")
    elif RUN_MODE == "full":
        scratch = CACHE / f"S31a_{dataset}"
        scratch.mkdir(parents=True, exist_ok=True)
        matrix, _ = stream_lr_matrix(d, scratch)
        rank_path = scratch / "lr_ranks.npy"
        ranks, valid = normalized_columns(matrix, rank_path)
        corr = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
        for start in range(0, len(matrix), RANK_CHUNK_ROWS):
            block = np.asarray(ranks[start:start + RANK_CHUNK_ROWS])
            corr += block.T @ block
        corr = np.clip(corr, -1, 1)
        corr[~valid, :] = 0
        corr[:, ~valid] = 0
        np.fill_diagonal(corr, 1)  # same convention as the original selection utility
        del ranks, matrix, block
        rank_path.unlink()
        params = mids.auto_selection_parameters(len(corr))
        cliques = mids.find_maximal_cliques_with_relaxation(
            corr, params["lr_spearcor_threshold"], params["min_clique_size"])
        subsets, _ = mids.merge_cliques_by_jaccard(cliques["maximal_cliques"], params["jaccard_thr"])
        if not subsets:
            subsets = [list(range(len(corr)))]
        summary = {**params, "recommended_dim_envir": len(subsets), "num_lr_pairs": len(corr),
                   "effective_min_clique_size": cliques["effective_min_clique_size"],
                   "constant_lr_columns": int((~valid).sum())}
        provenance = {"mode": "recomputed_from_processed_AnnData", "adata": file_stamp(d / "adata_all.h5ad"),
                      "source_signature": recompute_signature}
    else:
        record("S31a", dataset, "missing", "No saved LR component analysis; RUN_MODE='full' computes it.")
        return None
    order = mids.build_disjoint_subset_order(corr, subsets)
    lr = read_pickle_cpu(d / "LR_list.pkl")
    if corr.shape != (len(lr), len(lr)):
        raise ValueError(f"Cached LR correlation and retained LR names disagree: {dataset}")
    np.save(dest / "lr_spearman.npy", corr)
    write_json(dest / "merged_subsets.json", subsets)
    write_json(dest / "selection_summary.json", summary)
    write_json(dest / "complete.json", provenance)
    pd.DataFrame({"display_position": np.arange(len(corr)) + 1,
                  "lr_index": np.array(order["ordered_indices"]) + 1,
                  "lr_pair": [lr_label(lr[i]) for i in order["ordered_indices"]]}).to_csv(dest / "LR_display_order.csv", index=False)
    record("S31a", dataset, "ready", f"{len(subsets)} current-data LR components")
    return {"corr": corr, "order": order, "summary": summary}


def label_rank_sum_tests(df):
    rows = []
    for left, right in zip(MISANNOTATION_RATES[:-1], MISANNOTATION_RATES[1:]):
        x = df.loc[np.isclose(df.rate, left), "corr"].to_numpy()
        y = df.loc[np.isclose(df.rate, right), "corr"].to_numpy()
        z, p = stats.ranksums(x, y, alternative="two-sided")
        rows.append({"left_rate": left, "right_rate": right, "n_left": len(x), "n_right": len(y),
                     "test": "two-sided Wilcoxon rank-sum (scipy.stats.ranksums)", "statistic": z, "pvalue": p})
    return pd.DataFrame(rows)

def stability_repeat_summary(df):
    if df["corr"].isna().any():
        raise ValueError("Undefined stability correlations; inspect tables before plotting.")
    out = (df.groupby(["experiment", "repeat_id", "seed", "metric"], observed=True)
           .agg(corr=("corr", "mean"), n_mi=("corr", "size"), n_slices_used=("n_slices_used", "first"))
           .reset_index())
    if not (out.n_mi == 15).all():
        raise ValueError("Every repeat/metric must summarize all 15 aligned MIs.")
    return out

def dimension_sweep(context, reference):
    runs = {context.cfg["reference_dim"]: reference}
    for dim in context.cfg["dims"]:
        if dim not in runs:
            runs[dim] = run_or_load(context, "dimension", list(context.graphs), dim, BASE_SEED, k=context.cfg["reference_k"])
    for a, b in zip(context.cfg["dims"][:-1], context.cfg["dims"][1:]):
        tag = f"S31b_{context.dataset}_M{a}_vs_M{b}"
        if runs[a] is None or runs[b] is None:
            record("S31b", tag, "missing", "Set RUN_MODE='full' to train the missing dimensionality.")
        else:
            PAIR_RESULTS[tag] = compare_outputs(runs[a], runs[b], tag)
            record("S31b", tag, "ready")

def k_sweep(context, reference):
    runs = {10: reference}
    for k in (5, 8):
        graphs = knn_view(context, k)
        runs[k] = run_or_load(context, "neighborhood", graphs, 15, BASE_SEED, k=k,
                              extra={"graph_rule": "nearest_K_within_original_directed_K10_fixed_LR"})
    for a, b in ((5, 8), (8, 10)):
        tag = f"S31c_HGSOC_K{a}_vs_K{b}"
        if runs[a] is None or runs[b] is None:
            record("S31c", tag, "missing", "Set RUN_MODE='full' to train the missing neighborhood.")
        else:
            PAIR_RESULTS[tag] = compare_outputs(runs[a], runs[b], tag)
            record("S31c", tag, "ready")

def label_experiment(context, reference):
    tables = []
    for rate in MISANNOTATION_RATES:
        pct = int(round(rate*100))
        graphs, audit = misannotate(context, rate, MISANNOTATION_SEEDS[rate], TABLES / f"S32a_labels_changed_{pct}pct.csv")
        current = run_or_load(context, "misannotation", graphs, 15, BASE_SEED, k=context.cfg["reference_k"],
                              extra={"rate": rate, "label_seed": MISANNOTATION_SEEDS[rate], "n_changed": len(audit)})
        if current is None:
            record("S32a", f"{pct}% labels", "missing", "Requires retraining with perturbed one-hot labels.")
            continue
        comparison = compare_outputs(current, reference, f"S32a_HGSOC_labels_{pct}pct")
        df = comparison["metrics"].query("metric == 'MI strength'").copy()
        df["rate"] = rate
        df["label_seed"] = MISANNOTATION_SEEDS[rate]
        df["model_seed"] = BASE_SEED
        tables.append(df)
        record("S32a", f"{pct}% labels", "ready", f"{len(audit):,} cells changed")
    if len(tables) != len(MISANNOTATION_RATES):
        return None
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(TABLES / "S32a_label_misannotation_Pearson_by_MI.csv", index=False)
    label_rank_sum_tests(result).to_csv(TABLES / "S32a_adjacent_rate_rank_sum_tests.csv", index=False)
    return result

def stability_experiment(context, reference, experiment):
    panel = "S32b" if experiment == "Random seed" else "S32c"
    seeds = RANDOM_SEEDS if panel == "S32b" else SUBSAMPLE_SEEDS
    n = len(context.graphs)
    n_take = int(np.floor(n * SUBSAMPLE_FRACTION + 0.5))
    if n != 48 or n_take != 34 or len(seeds) != 10:
        raise ValueError("Paper protocol requires 48 total slices, 34 sampled slices and ten repeats.")
    tables = []
    for repeat_id, seed in enumerate(seeds, 1):
        indices = list(range(n)) if panel == "S32b" else sorted(np.random.default_rng(seed).choice(n, size=n_take, replace=False).tolist())
        graphs = [context.graphs[i] for i in indices]
        write_json(TABLES / f"{panel}_repeat{repeat_id:02d}_slices.json", {
            "seed": seed, "slice_indices": indices, "slice_ids": [context.samples[i] for i in indices],
            "n_slices": len(indices), "fraction": len(indices)/n,
        })
        current = legacy_output(context, experiment, repeat_id, seed, graphs, indices)
        source = "legacy saved factors/loadings" if current is not None else "new run"
        if current is None:
            current = run_or_load(context, "random_seed" if panel == "S32b" else "subsampling",
                                  graphs, 15, seed, k=context.cfg["reference_k"], slice_indices=indices,
                                  extra={"experiment": experiment, "repeat_id": repeat_id})
        if current is None:
            record(panel, f"repeat {repeat_id}", "missing", "No compatible cached fit.")
            continue
        comparison = compare_outputs(current, reference, f"{panel}_HGSOC_repeat{repeat_id:02d}_seed{seed}")
        df = comparison["metrics"].copy()
        df["experiment"] = experiment
        df["repeat_id"] = repeat_id
        df["seed"] = seed
        df["source_run"] = str(current.run_dir)
        tables.append(df)
        record(panel, f"repeat {repeat_id}", "ready", source)
    if len(tables) != 10:
        return None
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(TABLES / f"{panel}_Pearson_by_MI_all_repeats.csv", index=False)
    summary = stability_repeat_summary(result)
    summary.to_csv(TABLES / f"{panel}_Pearson_repeat_means.csv", index=False)
    return result
