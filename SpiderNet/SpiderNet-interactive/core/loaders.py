"""Process-wide loaders for SpiderNet datasets.

Loads `adata_list`, `SpiderNet_data_pyg_list`, and `Factor_envir_list` once
per dataset and keeps them in memory. These are large (multi-GB) so per-request
loading is not viable.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from core.datasets import DatasetPaths


# name -> {"adata_list", "pyg_list", "factor_envir_list"}
_BUNDLE_CACHE: dict[str, dict[str, Any]] = {}
_BUNDLE_LOCK = Lock()


def _load_pyg_list(processed_dir: Path) -> Any:
    """Load SpiderNet_data_pyg_list, supporting both .pkl (legacy) and .pt
    (new fallback for very large processed objects).

    Prefers SpiderNet's own helper when available, since it handles
    PyTorch version quirks; otherwise falls back to a plain read."""
    try:
        from SpiderNet.io import load_spidernet_pyg_list
        return load_spidernet_pyg_list(processed_dir)
    except ImportError:
        pass

    pkl = processed_dir / "SpiderNet_data_pyg_list.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    pt = processed_dir / "SpiderNet_data_pyg_list.pt"
    if pt.exists():
        import torch
        try:
            return torch.load(pt, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(pt, map_location="cpu")
    raise FileNotFoundError(
        f"Missing SpiderNet PyG data file. Expected one of: {pkl} or {pt}"
    )


def get_core_bundle(ds: DatasetPaths, progress=None) -> dict[str, Any]:
    """Return the cached (adata_list, pyg_list, factor_envir_list) bundle.

    Loads from disk on first call; subsequent calls return the same dict.
    """
    with _BUNDLE_LOCK:
        cached = _BUNDLE_CACHE.get(ds.name)
        if cached is not None:
            if progress:
                progress("Reusing spatial data already loaded in memory")
            return cached
        if progress:
            progress("Reading expression data (adata_list.pkl)")
        adata_list = pd.read_pickle(ds.processed_dir / "adata_list.pkl")
        if progress:
            progress("Reading spatial graphs (large file; this step can take several minutes)")
        pyg_list = _load_pyg_list(ds.processed_dir)
        if progress:
            progress("Reading trained MI factors (Factor_envir_list.pkl)")
        bundle = {
            "adata_list": adata_list,
            "pyg_list": pyg_list,
            "factor_envir_list": pd.read_pickle(ds.run_dir / "Factor_envir_list.pkl"),
        }
        _BUNDLE_CACHE[ds.name] = bundle
        return bundle


def edge_index_array(pyg_obj: Any) -> np.ndarray:
    """Return edge_index as a (n_edges, 2) numpy array, regardless of source."""
    import torch  # local import — torch is a heavy dep
    ei = pyg_obj["edge_index"] if isinstance(pyg_obj, dict) or hasattr(pyg_obj, "__getitem__") else getattr(pyg_obj, "edge_index")
    if torch.is_tensor(ei):
        ei = ei.cpu().numpy()
    return np.asarray(ei)
