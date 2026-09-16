"""Optional, content-addressed cache for the unmodified SpiderNet initializer.

Only load cache directories written by you: torch cache files use Python pickle.
The key includes input values, initializer source, configuration, runtime, and
incoming RNG states. A hit restores the RNG states left by the original call,
so later model construction/training sees the same random stream. Fingerprinting
and disk I/O are included in ``elapsed_s``; reuse is not a free first-run speedup.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import time
import warnings

import numpy as np
import torch


_SCHEMA = 1
_CHUNK_BYTES = 8 * 1024 * 1024


def capture_rng_state() -> dict:
    """Snapshot the Python, NumPy, CPU Torch, and available CUDA RNGs."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Restore a snapshot made by :func:`capture_rng_state`."""
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["torch_cuda"]) != expected_cuda:
        raise ValueError("RNG snapshot CUDA device count differs from this process")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if expected_cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _tag(digest, value) -> None:
    encoded = _json_bytes(value)
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _hash_value(digest, value) -> None:
    """Hash values without torch/pickle storage IDs or a full-size byte copy."""
    if torch.is_tensor(value):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Initialization fingerprint requires ordinary dense tensors")
        _tag(digest, ["tensor", str(value.dtype), list(value.shape), list(value.stride())])
        array = value.detach()
        if array.ndim == 0:
            array = array.reshape(1)
        per_row = max(1, array[0].numel() * array.element_size()) if len(array) else 1
        rows = max(1, _CHUNK_BYTES // per_row)
        for start in range(0, len(array), rows):
            # CPU copies are bounded by row chunks, even for GPU input datasets.
            chunk = array[start:start + rows].to("cpu").contiguous()
            raw = chunk.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be initialization cache inputs")
        _tag(digest, ["numpy", value.dtype.str, list(value.shape)])
        array = value.reshape(1) if value.ndim == 0 else value
        per_row = max(1, array[0].size * array.itemsize) if len(array) else 1
        rows = max(1, _CHUNK_BYTES // per_row)
        for start in range(0, len(array), rows):
            digest.update(memoryview(np.ascontiguousarray(array[start:start + rows])).cast("B"))
    elif isinstance(value, dict):
        _tag(digest, ["dict", sorted(value)])
        for key in sorted(value):
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        _tag(digest, [type(value).__name__, len(value)])
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    elif value is None or isinstance(value, (str, int, float, bool)):
        _tag(digest, [type(value).__name__, value])
    else:
        raise TypeError(f"Unsupported initialization fingerprint value: {type(value).__name__}")


def _value_digest(value) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _runtime_info(device: torch.device) -> dict:
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "torch-scatter", "joblib", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            cuda_devices.append({"name": prop.name, "major": prop.major, "minor": prop.minor,
                                 "total_memory": prop.total_memory})
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "versions": versions, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cuda_devices": cuda_devices,
        "device": str(device), "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING")},
    }


def _fingerprint(baseline_module, data, config: dict, incoming_rng: dict, device: torch.device) -> dict:
    path = getattr(baseline_module, "__file__", None)
    if path and Path(path).is_file():
        source = Path(path).read_bytes()
    else:
        source = inspect.getsource(baseline_module.Initial_model).encode("utf-8")
    attributes = ["x", "edge_index", "cellpair_LRpair_neigh", "num_cells"]
    if config["Factor_mode"] != "NMF":
        attributes.append(config["Factor_mode"] + "_onehot")
    batches = []
    for batch in data:
        fields = {}
        for name in attributes:
            value = batch[name]
            fields[name] = {"sha256": _value_digest(value),
                            "shape": list(value.shape) if hasattr(value, "shape") else None,
                            "dtype": str(value.dtype) if hasattr(value, "dtype") else type(value).__name__}
        batches.append(fields)
    return {"schema": _SCHEMA, "source_sha256": hashlib.sha256(source).hexdigest(),
            "config_sha256": _value_digest(config), "config": config,
            "batches": batches, "runtime": _runtime_info(device),
            "incoming_rng_sha256": _value_digest(incoming_rng)}


def _run_original(baseline_module, data, config: dict, device: torch.device):
    sentinel = object()
    original_device = getattr(baseline_module, "device", sentinel)
    # Initial_model uses its module global device in two scatter_mean operations.
    baseline_module.device = str(device)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = baseline_module.Initial_model(data, **config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result, time.perf_counter() - start
    finally:
        if original_device is sentinel:
            del baseline_module.device
        else:
            baseline_module.device = original_device


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def initialize_with_cache(baseline_module, data, *, config: dict, cache_dir: Path | None,
                          enabled: bool = True) -> tuple[dict, dict]:
    """Run exact ``Initial_model`` or reuse a matching local initialization.

    ``data`` must already be on the intended training device. ``config`` accepts
    the original Initial_model keyword arguments; omitted defaults are resolved
    from that function's signature. With caching disabled, no input hashing or
    disk I/O is performed. On cache corruption a warning is emitted and the exact
    original initializer is rerun. This function does not alter original files.
    """
    start = time.perf_counter()
    if not data:
        raise ValueError("Initialization requires at least one data batch")
    signature = inspect.signature(baseline_module.Initial_model)
    bound = signature.bind(data, **config)
    bound.apply_defaults()
    first_parameter = next(iter(signature.parameters))
    config = {name: value for name, value in bound.arguments.items() if name != first_parameter}
    device = torch.device(data[0]["x"].device)
    if not enabled or cache_dir is None:
        result, elapsed = _run_original(baseline_module, data, config, device)
        return result, {"hit": False, "enabled": False, "key": None, "path": None,
                        "fingerprint": None, "compute_s": elapsed,
                        "elapsed_s": time.perf_counter() - start}

    incoming_rng = capture_rng_state()
    fingerprint = _fingerprint(baseline_module, data, config, incoming_rng, device)
    key = hashlib.sha256(_json_bytes(fingerprint)).hexdigest()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (key + ".pt")
    metadata = {"hit": False, "enabled": True, "key": key, "path": str(path), "fingerprint": fingerprint}
    if path.is_file():
        try:
            # Trust boundary: caller-owned, locally generated cache directory.
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["schema"] != _SCHEMA or payload["key"] != key or payload["fingerprint"] != fingerprint:
                raise ValueError("Cache metadata does not match the requested initialization")
            if not isinstance(payload["initial_dict"], dict):
                raise ValueError("Cache initialization is not a dictionary")
            if _value_digest(payload["initial_dict"]) != payload["output_sha256"]:
                raise ValueError("Cached initialization content failed its checksum")
            if _value_digest(payload["post_rng"]) != payload["post_rng_sha256"]:
                raise ValueError("Cached RNG state failed its checksum")
            restore_rng_state(payload["post_rng"])
            metadata.update(hit=True, compute_s=0.0, original_compute_s=payload["compute_s"],
                            elapsed_s=time.perf_counter() - start)
            return payload["initial_dict"], metadata
        except Exception as exc:
            restore_rng_state(incoming_rng)
            warnings.warn(f"Ignoring unusable initialization cache {path}: {exc}", RuntimeWarning)

    # Fingerprinting and local I/O must not consume the initializer's RNG stream.
    restore_rng_state(incoming_rng)
    result, compute_s = _run_original(baseline_module, data, config, device)
    post_rng = capture_rng_state()
    payload = {"schema": _SCHEMA, "key": key, "fingerprint": fingerprint,
               "initial_dict": result, "output_sha256": _value_digest(result),
               "post_rng": post_rng, "post_rng_sha256": _value_digest(post_rng), "compute_s": compute_s}
    _atomic_save(payload, path)
    metadata.update(compute_s=compute_s, original_compute_s=compute_s,
                    elapsed_s=time.perf_counter() - start)
    return result, metadata
