"""Isolated old/new SpiderNet training benchmark; original sources are never edited.

Run --help for controls. Inputs must be trusted local SpiderNet PyG .pkl/.pt files.
Every measured fit has its own process, the same prepared data/weights/RNG, and
its own checkpoint directory. CUDA allocator peaks exclude verification passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback


HERE = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def nvidia_info():
    result = {}
    commands = {
        "gpus": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
        "compute_processes": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
    }
    for key, command in commands.items():
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
            result[key] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = str(exc)
    return result


class RssSampler:
    """Sample current process + children; this is sampled RSS, not exact peak."""
    def __init__(self, interval=0.01):
        import psutil
        self.process = psutil.Process()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        import psutil
        rss = self.process.memory_info().rss
        try:
            children = self.process.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        self.peak = max(self.peak, rss)
        self.samples += 1
        return rss

    def run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.start_rss = self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.sample()
        self.stop_event.set()
        self.thread.join()


def os_lifecycle_peak_rss():
    import psutil
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return int(info.peak_wset)
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def load_data(path):
    import pickle
    from torch_geometric.data import Data
    path = Path(path).resolve()
    if path.is_dir():
        candidates = [path / "SpiderNet_data_pyg_list.pkl", path / "SpiderNet_data_pyg_list.pt"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError("Directory must contain SpiderNet_data_pyg_list.pkl or SpiderNet_data_pyg_list.pt")
    if path.suffix.lower() in (".pkl", ".pickle"):
        with path.open("rb") as stream:
            data = pickle.load(stream)
    else:
        data = torch_load(path)
    if isinstance(data, dict):
        for key in ("SpiderNet_data_pyg_list", "spidernet_data", "data"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("Expected a non-empty list of SpiderNet PyG batches")
    data = [Data(**batch) if isinstance(batch, dict) else batch for batch in data]
    return data, str(path)


def validate_data(data, factor_mode):
    import torch
    genes, lr_pairs = data[0].x.shape[1], data[0]["cellpair_LRpair_neigh"].shape[1]
    for index, batch in enumerate(data):
        if batch.x.ndim != 2 or batch.x.shape[1] != genes:
            raise ValueError(f"Batch {index}: inconsistent expression shape")
        edge = batch["edge_index"]
        if edge.ndim != 2 or edge.shape[1] != 2:
            raise ValueError(f"Batch {index}: SpiderNet expects edge_index shaped [edges, 2], not conventional PyG [2, edges]")
        if not edge.numel() or int(edge.min()) < 0 or int(edge.max()) >= batch.x.shape[0]:
            raise ValueError(f"Batch {index}: empty or invalid edge index")
        target = batch["cellpair_LRpair_neigh"]
        if tuple(target.shape) != (edge.shape[0], lr_pairs):
            raise ValueError(f"Batch {index}: inconsistent LR target shape")
        if factor_mode != "NMF" and batch[factor_mode + "_onehot"].shape[0] != batch.x.shape[0]:
            raise ValueError(f"Batch {index}: inconsistent intrinsic factor shape")
        if "num_cells" not in batch:
            batch["num_cells"] = batch.x.shape[0]
        for name in ("x", "cellpair_LRpair_neigh"):
            value = batch[name]
            rows_per_chunk = max(1, (8 * 1024 * 1024) // max(1, value.shape[1] * value.element_size()))
            for start in range(0, value.shape[0], rows_per_chunk):
                if not torch.isfinite(value[start:start + rows_per_chunk]).all():
                    raise ValueError(f"Batch {index}: non-finite values in {name}")


def synthetic_data(config):
    import torch
    from torch_geometric.data import Data
    data = []
    for _ in range(config["synthetic_batches"]):
        nodes = config["synthetic_nodes"]
        labels = torch.arange(nodes) % config["synthetic_classes"]
        data.append(Data(
            x=torch.rand(nodes, config["synthetic_genes"]),
            edge_index=torch.randint(nodes, (config["synthetic_edges"], 2)),
            cell_class_onehot=torch.nn.functional.one_hot(labels, config["synthetic_classes"]).float(),
            cellpair_LRpair_neigh=torch.rand(config["synthetic_edges"], config["synthetic_lr_pairs"]),
            num_cells=nodes,
        ))
    return data


def synthetic_initialization(data, model_args):
    import numpy as np
    d, k, g, lr = (model_args[name] for name in ("dim_envir", "dim_intri", "num_gene", "num_LR"))
    return {
        "factor_intrinsic_init": [batch.cell_class_onehot.cpu().numpy().copy() for batch in data],
        "loading_intrinsic_init": np.random.uniform(0.01, 0.1, (k, g)).astype("float32"),
        "factor_GP_minibatchNMF": [np.random.uniform(0, 1, (batch.edge_index.shape[0], d)).astype("float32") for batch in data],
        "loading_receiver_init": np.random.uniform(0.01, 0.1, (d, g)).astype("float32"),
        "loading_sender_init": np.random.uniform(0.01, 0.1, (d, g)).astype("float32"),
        "loading_LR_init": np.random.uniform(0.01, 0.1, (d, lr)).astype("float32"),
    }


def prepare(config, run_dir):
    import torch
    from initialization_cache import capture_rng_state, restore_rng_state, initialize_with_cache
    from spidernet_optimized import load_baseline_module
    baseline = load_baseline_module(config.get("baseline_source"))
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False. Use the configured SpiderNet GPU environment, or explicitly pass --device cpu.")
    seed_all(config["seed"])
    if config["synthetic"]:
        data, input_path = synthetic_data(config), "SYNTHETIC (not biological validation)"
    else:
        data, input_path = load_data(config["data"])
    validate_data(data, config["factor_mode"])
    dim_envir = config["dim_envir"]
    dim_intri = config["dim_intri"] if config["factor_mode"] == "NMF" else data[0][config["factor_mode"] + "_onehot"].shape[1]
    model_args = dict(num_gene=data[0].x.shape[1], num_LR=data[0].cellpair_LRpair_neigh.shape[1],
                      hidden_channels=config["hidden_channels"] or (128 if dim_envir > 20 else 64),
                      Factor_mode=config["factor_mode"], dim_intri=dim_intri, dim_envir=dim_envir)
    # Match api.build_model -> api.run_training ordering: model exists before Initial_model.
    model = baseline.SpiderNet_model(**model_args).to(config["device"])
    initial_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    data = [batch.to(config["device"]) for batch in data]
    if config["synthetic"]:
        initial = synthetic_initialization(data, model_args)
        init_metadata = {"synthetic": True, "note": "Generated fixed targets; original NMF/regression and initialization-cache speed were not benchmarked."}
    else:
        init_config = dict(dim_envir=dim_envir, Factor_mode=config["factor_mode"],
                           dim_intri=dim_intri if config["factor_mode"] == "NMF" else None,
                           n_jobs=config["n_jobs"], Initial_regression="Linear",
                           enhance_init_with_gene_coexp=(model_args["num_gene"] < 300),
                           threshold_crosscorr=0.1, numtop_crosscorr=10, spearcorr_use_rowmax_threshold=0.2)
        if config["enhance_init"] != "auto":
            init_config["enhance_init_with_gene_coexp"] = config["enhance_init"] == "yes"
        pre_init_rng = capture_rng_state()
        cache_dir = Path(config["init_cache"] or (HERE / "initialization_cache_data"))
        initial, init_metadata = initialize_with_cache(baseline, data, config=init_config,
                                                      cache_dir=cache_dir, enabled=not config["no_init_cache"])
        post_init_rng = capture_rng_state()
        if not config["no_init_cache"]:
            restore_rng_state(pre_init_rng)
            _, replay_metadata = initialize_with_cache(baseline, data, config=init_config, cache_dir=cache_dir, enabled=True)
            init_metadata["cache_replay"] = replay_metadata
            restore_rng_state(post_init_rng)
    rng = capture_rng_state()
    bundle = dict(data=[batch.cpu() for batch in data], initial_dict=initial,
                  initial_model_state=initial_model_state, model_args=model_args, rng=rng)
    torch.save(bundle, run_dir / "prepared_bundle.pt")
    import numpy as np
    import torch_scatter
    import torch_geometric
    metadata = {
        "input": input_path, "synthetic": config["synthetic"], "initialization": init_metadata,
        "batch_count": len(data), "node_rows": sum(batch.x.shape[0] for batch in data),
        "edges": sum(batch.edge_index.shape[0] for batch in data), "model_args": model_args,
        "baseline_source": str(Path(baseline.__file__).resolve()), "baseline_sha256": sha256(baseline.__file__),
        "optimized_sha256": sha256(HERE / "spidernet_optimized.py"),
        "harness_sha256": sha256(__file__), "initialization_cache_sha256": sha256(HERE / "initialization_cache.py"),
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                    "torch": torch.__version__, "numpy": np.__version__, "torch_scatter": torch_scatter.__version__,
                    "torch_geometric": torch_geometric.__version__, "cuda_runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
    }
    if config["device"].startswith("cuda"):
        metadata["runtime"]["gpu"] = torch.cuda.get_device_name(config["device"])
        metadata["runtime"]["gpu_total_bytes"] = torch.cuda.get_device_properties(config["device"]).total_memory
    write_json(run_dir / "preparation.json", metadata)


def recording_baseline_class(baseline):
    """Worker-local wrappers, used only for unmeasured correctness replay."""
    history = []
    scaler_original = baseline.GradScaler
    scheduler_original = baseline.ReduceLROnPlateau
    optimizer_ref = []

    class RecordingScaler(scaler_original):
        def update(self, *args, **kwargs):
            result = super().update(*args, **kwargs)
            if history:
                history[-1]["scale"] = self.get_scale()
                if optimizer_ref:
                    history[-1]["lr"] = optimizer_ref[-1].param_groups[0]["lr"]
            return result

    class RecordingScheduler(scheduler_original):
        def __init__(self, optimizer, *args, **kwargs):
            optimizer_ref.append(optimizer)
            super().__init__(optimizer, *args, **kwargs)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            if history:
                history[-1]["lr"] = self.optimizer.param_groups[0]["lr"]
            return result

    baseline.GradScaler = RecordingScaler
    baseline.ReduceLROnPlateau = RecordingScheduler

    class RecordingModel(baseline.SpiderNet_model):
        def compute_loss_in_batches(self, *args, **kwargs):
            result = super().compute_loss_in_batches(*args, **kwargs)
            warmup, exp, lr = result
            history.append(dict(epoch=len(history), loss_warmup=warmup, loss_exp=exp,
                                loss_LR=lr, loss_total=warmup + exp + lr))
            return result

    return RecordingModel, history


def output_snapshot(model, data, max_elements):
    """Bounded deterministic samples from every returned tensor of every batch."""
    import torch
    snapshot = {}
    with torch.no_grad():
        model.eval()
        for batch_index, batch in enumerate(data):
            outputs = model(batch)
            for index, value in enumerate(outputs):
                flat = value.detach().flatten()
                stride = max(1, math.ceil(flat.numel() / max_elements))
                snapshot[f"batch{batch_index}/output{index}"] = flat[::stride].cpu().clone()
    return snapshot


def train_worker(config, run_dir, variant, repetition, verify):
    import gc
    import torch
    import spidernet_optimized
    from initialization_cache import restore_rng_state
    from spidernet_optimized import load_baseline_module, make_optimized_model_class, OptimizationOptions
    baseline = load_baseline_module(config.get("baseline_source"))
    if config["device"] == "cpu":
        # Original CUDA AMP is enabled even on CPU tensors when a CUDA GPU is
        # present. Disable it symmetrically for the explicit CPU-only benchmark.
        scaler_original = torch.cuda.amp.GradScaler
        autocast_original = torch.cuda.amp.autocast

        class CpuScaler(scaler_original):
            def __init__(self, *args, **kwargs):
                kwargs["enabled"] = False
                super().__init__(*args, **kwargs)

        class CpuAutocast(autocast_original):
            def __init__(self, *args, **kwargs):
                kwargs["enabled"] = False
                super().__init__(*args, **kwargs)

        baseline.GradScaler = spidernet_optimized.GradScaler = CpuScaler
        baseline.autocast = spidernet_optimized.autocast = CpuAutocast
    bundle = torch_load(run_dir / "prepared_bundle.pt")
    data = [batch.to(config["device"]) for batch in bundle["data"]]
    history = None
    if variant == "optimized":
        model_class = make_optimized_model_class(baseline)
    elif verify:
        model_class, history = recording_baseline_class(baseline)
    else:
        model_class = baseline.SpiderNet_model
    model = model_class(**bundle["model_args"]).to(config["device"])
    model.load_state_dict(bundle["initial_model_state"])
    # The SAME original forward/backward prewarm for each variant, no optimizer step.
    # Restore weights + RNG afterwards. No model-specific caches are prewarmed.
    for _ in range(config["prewarm"]):
        outputs = baseline.SpiderNet_model.forward(model, data[0])
        (outputs[0].float().square().mean() + outputs[1].float().square().mean()).backward()
        del outputs
        model.zero_grad(set_to_none=True)
    model.load_state_dict(bundle["initial_model_state"])
    restore_rng_state(bundle["rng"])
    synchronize(config["device"])
    gc.collect()
    if config["device"].startswith("cuda"):
        torch.cuda.empty_cache()
    prefix = "verify" if verify else f"repeat{repetition:02d}"
    worker_dir = run_dir / f"{prefix}_{variant}"
    worker_dir.mkdir(exist_ok=False)
    kwargs = dict(SpiderNet_data_pyg_list=data, device=config["device"], optim_type=config["optimizer"],
                  lr=config["lr"], weight_decay=config["weight_decay"], LR_loss_weight=config["lr_loss_weight"],
                  warmup=config["warmup"], max_epoch=min(config["verify_epochs"], config["epochs"]) if verify else config["epochs"],
                  loss_fn=config["loss_fn"], Initial_dict=bundle["initial_dict"],
                  file_savepath_model_main=str(worker_dir / "checkpoints"))
    if variant == "optimized":
        kwargs["options"] = OptimizationOptions(
            defer_metrics=not config["no_defer_metrics"], warmup_factor_only=not config["no_warmup_pruning"],
            cache_targets=not config["no_target_cache"], cache_degrees=not config["no_degree_cache"],
            target_cache_max_bytes=int(config["target_cache_mib"] * 1024 * 1024))
        kwargs["record_history"] = verify
    memory = {}
    if config["device"].startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config["device"])
        memory["gpu_allocated_before_fit_bytes"] = torch.cuda.memory_allocated(config["device"])
        memory["gpu_reserved_before_fit_bytes"] = torch.cuda.memory_reserved(config["device"])
    with RssSampler(config["rss_interval_ms"] / 1000) as sampler:
        started = time.perf_counter()
        model.fit(**kwargs)
        synchronize(config["device"])
        elapsed = time.perf_counter() - started
    memory.update(cpu_rss_start_bytes=sampler.start_rss, cpu_peak_rss_sampled_bytes=sampler.peak,
                  cpu_rss_samples=sampler.samples, cpu_os_lifecycle_peak_rss_bytes=os_lifecycle_peak_rss())
    if config["device"].startswith("cuda"):
        memory["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(config["device"])
        memory["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(config["device"])
    # Everything below is explicitly outside timing and recorded peak-memory window.
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(state, worker_dir / "final_state.pt")
    snapshots = output_snapshot(model, data, config["output_sample_elements"])
    torch.save(snapshots, worker_dir / "output_samples.pt")
    result = dict(variant=variant, repetition=repetition, verification_only=verify, epochs=kwargs["max_epoch"],
                  fit_seconds=elapsed, seconds_per_epoch=elapsed / kwargs["max_epoch"], memory=memory,
                  cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
                  cpu_amp_disabled=config["device"] == "cpu", prewarm_passes=config["prewarm"],
                  optimization_stats=getattr(model, "optimization_stats", {}),
                  checkpoint_count=len(list((worker_dir / "checkpoints").glob("*.pth"))))
    if verify:
        result["training_history"] = model.training_history if variant == "optimized" else history
    write_json(worker_dir / "result.json", result)


def compare_tensors(left, right, rtol, atol):
    import torch
    if set(left) != set(right):
        return {"allclose": False, "reason": "Different tensor keys"}
    max_abs, square_diff, square_ref, count, bad = 0.0, 0.0, 0.0, 0, []
    for name in left:
        a, b = left[name].double(), right[name].double()
        if a.shape != b.shape:
            return {"allclose": False, "reason": f"Different shape for {name}"}
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return {"allclose": False, "reason": f"Non-finite tensor {name}"}
        delta = b - a
        max_abs = max(max_abs, delta.abs().max().item() if delta.numel() else 0)
        square_diff += delta.square().sum().item()
        square_ref += a.square().sum().item()
        count += delta.numel()
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            bad.append(name)
    return dict(allclose=not bad, max_absolute_difference=max_abs,
                rmse=math.sqrt(square_diff / max(count, 1)),
                relative_l2=math.sqrt(square_diff / max(square_ref, 1e-300)),
                compared_elements=count, failed_tensor_count=len(bad), failed_tensor_names=bad[:30])


def compare_worker(config, run_dir):
    comparisons = []
    def compare(label, left_dir, right_dir):
        comparisons.append(dict(label=label, left=left_dir.name, right=right_dir.name,
            state=compare_tensors(torch_load(left_dir / "final_state.pt"), torch_load(right_dir / "final_state.pt"), config["rtol"], config["atol"]),
            sampled_outputs=compare_tensors(torch_load(left_dir / "output_samples.pt"), torch_load(right_dir / "output_samples.pt"), config["rtol"], config["atol"])))
    for repetition in range(config["repeats"]):
        compare("baseline_vs_optimized", run_dir / f"repeat{repetition:02d}_baseline", run_dir / f"repeat{repetition:02d}_optimized")
        if repetition:
            compare("baseline_repeat_variability", run_dir / "repeat00_baseline", run_dir / f"repeat{repetition:02d}_baseline")
    if config["verify_epochs"]:
        compare("verification_replay", run_dir / "verify_baseline", run_dir / "verify_optimized")
        left = json.loads((run_dir / "verify_baseline" / "result.json").read_text(encoding="utf-8"))["training_history"]
        right = json.loads((run_dir / "verify_optimized" / "result.json").read_text(encoding="utf-8"))["training_history"]
        history_metrics = {}
        for name in ("loss_warmup", "loss_exp", "loss_LR", "loss_total", "lr", "scale"):
            pairs = [(float(a[name]), float(b[name])) for a, b in zip(left, right) if name in a and name in b]
            history_metrics[name] = dict(compared_epochs=len(pairs),
                max_absolute_difference=max((abs(a-b) for a, b in pairs), default=0),
                allclose=(len(pairs) == len(left) == len(right) and all(math.isclose(a,b,rel_tol=config["rtol"],abs_tol=config["atol"]) for a,b in pairs)))
    else:
        history_metrics = None
    state_output_pass = all(item["state"]["allclose"] and item["sampled_outputs"]["allclose"]
                            for item in comparisons)
    history_pass = (all(item["allclose"] for item in history_metrics.values())
                    if history_metrics is not None else None)
    write_json(run_dir / "correctness.json", {"rtol": config["rtol"], "atol": config["atol"],
        "scope": "All state tensors; bounded evenly spaced samples of all 8 forward outputs for every batch. No claim of full biological equivalence.",
        "all_checks_passed": state_output_pass and history_pass is not False,
        "state_and_sampled_outputs_passed": state_output_pass,
        "verification_history_passed": history_pass,
        "comparisons": comparisons, "verification_history": history_metrics})


def child_environment(config, variant=None):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(config["seed"])
    env["PYTHONUNBUFFERED"] = "1"
    if variant == "baseline" and config["baseline_launch_blocking"] != "inherit":
        env["CUDA_LAUNCH_BLOCKING"] = config["baseline_launch_blocking"]
    if variant == "optimized" and not config["keep_launch_blocking"]:
        env.pop("CUDA_LAUNCH_BLOCKING", None)
    return env


def run_child(config, run_dir, stage, variant=None, repetition=0, verify=False):
    command = [sys.executable, str(Path(__file__).resolve()), "--_stage", stage, "--_config", str(run_dir / "config.json")]
    if variant:
        command += ["--_variant", variant, "--_repetition", str(repetition)]
    if verify:
        command += ["--_verify"]
    label = stage if not variant else ("verify" if verify else f"repeat{repetition:02d}") + "_" + variant
    print(f"[{label}] starting (log: {run_dir / (label + '.log')})", flush=True)
    started = time.perf_counter()
    with (run_dir / (label + ".log")).open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=child_environment(config, variant))
    if completed.returncode:
        tail = (run_dir / (label + ".log")).read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"{label} failed (exit {completed.returncode}). Log tail:\n{tail}")
    print(f"[{label}] finished, child lifecycle {time.perf_counter() - started:.2f} s", flush=True)


def summarize(config, run_dir, environment):
    rows, aggregates = [], {}
    for repetition in range(config["repeats"]):
        for variant in ("baseline", "optimized"):
            result = json.loads((run_dir / f"repeat{repetition:02d}_{variant}" / "result.json").read_text(encoding="utf-8"))
            row = {name: result[name] for name in ("variant", "repetition", "epochs", "fit_seconds", "seconds_per_epoch", "cuda_launch_blocking", "checkpoint_count")}
            row.update(result["memory"])
            rows.append(row)
    for variant in ("baseline", "optimized"):
        chosen = [row for row in rows if row["variant"] == variant]
        times = [row["fit_seconds"] for row in chosen]
        aggregates[variant] = dict(median_fit_seconds=statistics.median(times), min_fit_seconds=min(times), max_fit_seconds=max(times),
            fit_seconds_stdev=statistics.stdev(times) if len(times) > 1 else None,
            max_cpu_peak_rss_sampled_bytes=max(row["cpu_peak_rss_sampled_bytes"] for row in chosen),
            max_gpu_peak_allocated_bytes=max((row.get("gpu_peak_allocated_bytes", 0) for row in chosen), default=0),
            max_gpu_peak_reserved_bytes=max((row.get("gpu_peak_reserved_bytes", 0) for row in chosen), default=0))
    old, new = (aggregates[name]["median_fit_seconds"] for name in ("baseline", "optimized"))
    ratio = old / new
    correctness = json.loads((run_dir / "correctness.json").read_text(encoding="utf-8"))
    report = dict(config=config, preparation=json.loads((run_dir / "preparation.json").read_text(encoding="utf-8")),
                  environment=environment, runs=rows, aggregates=aggregates,
                  speedup_ratio=ratio, elapsed_time_reduction_percent=100*(1-new/old), correctness=correctness,
                  measurement_notes=[
                      "fit time includes warmup, main epochs, fit setup (including optimized caches), optimizer/scheduler and symmetric checkpoint writes; excludes input loading, initialization, common prewarm and result validation.",
                      "CPU RSS: process plus descendants sampled every configured interval; short peaks can be missed. Lifecycle OS peak is process-only and includes imports/loading/prewarm through fit.",
                      "GPU peaks: PyTorch max_memory_allocated and max_memory_reserved, absolute allocator peaks during fit. They exclude CUDA context/driver allocations and post-fit verification.",
                      "Fresh child per variant/repeat; AB then BA order alternates. Repeats=1 cannot estimate normal run-to-run variability.",
                      "Default baseline inherits CUDA_LAUNCH_BLOCKING, optimized removes it before importing torch. If inherited=1, overall speedup includes optimization 6; rerun baseline blocking=0 to isolate code-only gains.",
                      "Initialization cache is measured separately during real-data preparation; synthetic mode does not measure optimization 7. No initialization gain is included in fit speedup.",
                      "Fixed seed and shared weights/data/initialization/RNG do not eliminate GPU atomic scatter nondeterminism. allclose failures are reported, not silently accepted.",
                  ])
    write_json(run_dir / "report.json", report)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (run_dir / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# SpiderNet training benchmark", "", f"Input: `{report['preparation']['input']}`", "",
             "| Variant | Median fit (s) | Sampled CPU peak (MiB) | GPU allocated peak (MiB) | GPU reserved peak (MiB) |",
             "|---|---:|---:|---:|---:|"]
    for variant in ("baseline", "optimized"):
        value = aggregates[variant]
        lines.append(f"| {variant} | {value['median_fit_seconds']:.3f} | {value['max_cpu_peak_rss_sampled_bytes']/2**20:.1f} | {value['max_gpu_peak_allocated_bytes']/2**20:.1f} | {value['max_gpu_peak_reserved_bytes']/2**20:.1f} |")
    lines += ["", f"Speed: {ratio:.4f}x; fit elapsed-time reduction: {100*(1-new/old):.2f}%.", "",
              "Correctness (configured allclose tolerances, not a biological-equivalence guarantee):"]
    for comparison in correctness["comparisons"]:
        lines.append(f"- {comparison['label']} {comparison['left']} vs {comparison['right']}: state allclose={comparison['state']['allclose']}, sampled outputs allclose={comparison['sampled_outputs']['allclose']}.")
    history_metrics = correctness.get("verification_history")
    if history_metrics is not None:
        lines += ["", "Verification history (separate replay, excluded from performance timing):", ""]
        for name, value in history_metrics.items():
            lines.append(f"- {name}: allclose={value['allclose']}, max absolute difference={value['max_absolute_difference']:.10g}, compared epochs={value['compared_epochs']}.")
    else:
        lines += ["", "Verification history: disabled; no loss/LR/scaler trajectory check was performed."]
    lines += ["", "Measurement scope:", ""] + ["- " + note for note in report["measurement_notes"]]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:11]), flush=True)
    failed = (any(not item["state"]["allclose"] or not item["sampled_outputs"]["allclose"] for item in correctness["comparisons"])
              or (history_metrics is not None and any(not value["allclose"] for value in history_metrics.values())))
    print(f"Correctness tolerances: rtol={config['rtol']}, atol={config['atol']}; {'FAILURES PRESENT — inspect correctness.json' if failed else 'all executed state/output/history checks passed'}", flush=True)
    print(f"Reports: {run_dir / 'report.md'}\nRaw data: {run_dir / 'report.json'}\nCSV: {run_dir / 'runs.csv'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data", help="Trusted processed-data directory or PyG batch-list .pkl/.pt")
    source.add_argument("--synthetic", action="store_true", help="Quick generated workload; not a real-data quality or initialization benchmark")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=None, help="Default: floor(epochs * 0.1), matching original API")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--verify-epochs", type=int, default=50, help="Separate unmeasured history replay, capped at epochs; 0 disables")
    p.add_argument("--dim-envir", type=int, default=15, help="Use the same environmental dimension as your real training")
    p.add_argument("--hidden-channels", type=int, default=None, help="Original rule: 128 if dim_envir > 20, else 64")
    p.add_argument("--factor-mode", choices=("cell_class", "NMF"), default="cell_class")
    p.add_argument("--dim-intri", type=int, default=10, help="Used only for NMF mode")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--optimizer", choices=("adam", "SGD"), default="adam")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lr-loss-weight", type=float, default=1.0)
    p.add_argument("--loss-fn", choices=("mse", "poisson"), default="mse")
    p.add_argument("--n-jobs", type=int, default=5, help="Original initializer CPU regression parallelism")
    p.add_argument("--enhance-init", choices=("auto", "yes", "no"), default="auto", help="Original API auto enables coexpression only when genes < 300")
    p.add_argument("--init-cache", help="Reusable initialization-cache directory; default: alongside benchmark script")
    p.add_argument("--no-init-cache", action="store_true")
    p.add_argument("--target-cache-mib", type=float, default=256)
    p.add_argument("--no-target-cache", action="store_true")
    p.add_argument("--no-degree-cache", action="store_true")
    p.add_argument("--no-defer-metrics", action="store_true")
    p.add_argument("--no-warmup-pruning", action="store_true")
    p.add_argument("--baseline-launch-blocking", choices=("inherit", "0", "1"), default="inherit")
    p.add_argument("--keep-launch-blocking", action="store_true", help="Keep optimized child CUDA_LAUNCH_BLOCKING too (disables optimization 6)")
    p.add_argument("--prewarm", type=int, default=2, help="Common original forward/backward passes before timed fit; no parameter updates")
    p.add_argument("--rss-interval-ms", type=float, default=10)
    p.add_argument("--output-sample-elements", type=int, default=4096, help="At most this many sampled values per output tensor per batch")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--baseline-source", help="Optional explicit original model.py path")
    p.add_argument("--output", help="New result directory; must not already exist")
    for name, default in (("batches", 4), ("nodes", 512), ("edges", 4096), ("genes", 128), ("lr-pairs", 64), ("classes", 8)):
        p.add_argument("--synthetic-" + name, type=int, default=default)
    p.add_argument("--_stage", choices=("prepare", "train", "compare"), help=argparse.SUPPRESS)
    p.add_argument("--_config", help=argparse.SUPPRESS)
    p.add_argument("--_variant", choices=("baseline", "optimized"), help=argparse.SUPPRESS)
    p.add_argument("--_repetition", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--_verify", action="store_true", help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args._stage:
        config_path = Path(args._config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_dir = config_path.parent
        if args._stage == "prepare":
            prepare(config, run_dir)
        elif args._stage == "train":
            train_worker(config, run_dir, args._variant, args._repetition, args._verify)
        else:
            compare_worker(config, run_dir)
        return
    if not args.data and not args.synthetic:
        p.error("Pass --data PATH or --synthetic")
    if args.epochs < 1 or args.repeats < 1 or args.verify_epochs < 0 or args.prewarm < 0:
        p.error("epochs/repeats must be positive; verify-epochs/prewarm must be nonnegative")
    if args.warmup is None:
        args.warmup = int(args.epochs * 0.1)
    if not 0 <= args.warmup <= args.epochs:
        p.error("warmup must be between 0 and epochs")
    if args.rss_interval_ms <= 0 or args.output_sample_elements < 1 or args.target_cache_mib < 0 or min(args.rtol, args.atol) < 0:
        p.error("Invalid sampling interval, output sample count, cache budget, or tolerance")
    if args.synthetic and args.factor_mode == "NMF":
        p.error("Synthetic mode currently uses cell_class only; use real --data for NMF mode")
    for name in ("synthetic_batches", "synthetic_nodes", "synthetic_edges", "synthetic_genes", "synthetic_lr_pairs", "synthetic_classes", "dim_envir", "dim_intri"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    run_dir = Path(args.output).resolve() if args.output else HERE / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    config["output"] = str(run_dir)
    if config["data"]:
        config["data"] = str(Path(config["data"]).resolve())
    write_json(run_dir / "config.json", config)
    environment = dict(before=nvidia_info(), inherited_cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"))
    print(f"Output: {run_dir}\nPython: {sys.executable}", flush=True)
    print(f"CUDA_LAUNCH_BLOCKING baseline={child_environment(config, 'baseline').get('CUDA_LAUNCH_BLOCKING', '<unset>')}; optimized={child_environment(config, 'optimized').get('CUDA_LAUNCH_BLOCKING', '<unset>')}", flush=True)
    print("GPU inventory / existing compute processes:\n" + json.dumps(environment["before"], ensure_ascii=False, indent=2), flush=True)
    print("Other GPU jobs and display workloads may affect timing/VRAM. Existing processes are not stopped.", flush=True)
    run_child(config, run_dir, "prepare")
    for repetition in range(config["repeats"]):
        order = ("baseline", "optimized") if repetition % 2 == 0 else ("optimized", "baseline")
        for variant in order:
            run_child(config, run_dir, "train", variant, repetition)
    if config["verify_epochs"]:
        if min(config["verify_epochs"], config["epochs"]) <= config["warmup"] + 10:
            print("Verification replay does not cover the later alternating encoder update phase; increase --verify-epochs above warmup+10 for that check.", flush=True)
        for variant in ("baseline", "optimized"):
            run_child(config, run_dir, "train", variant, verify=True)
    run_child(config, run_dir, "compare")
    environment["after"] = nvidia_info()
    summarize(config, run_dir, environment)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
