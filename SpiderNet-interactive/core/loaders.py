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


def get_core_bundle(ds: DatasetPaths) -> dict[str, Any]:
    """Return the cached (adata_list, pyg_list, factor_envir_list) bundle.

    Loads from disk on first call; subsequent calls return the same dict.
    """
    with _BUNDLE_LOCK:
        cached = _BUNDLE_CACHE.get(ds.name)
        if cached is not None:
            return cached
        bundle = {
            "adata_list": pd.read_pickle(ds.processed_dir / "adata_list.pkl"),
            "pyg_list": pd.read_pickle(ds.processed_dir / "SpiderNet_data_pyg_list.pkl"),
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
