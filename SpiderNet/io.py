"""Input/output helpers for SpiderNet processed data.

This module intentionally keeps the public API compatible with older notebooks
that expect ``SpiderNet_data_pyg_list.pkl``, while also supporting the newer
``SpiderNet_data_pyg_list.pt`` fallback used when pickling the PyG list is too
large or unstable on some systems.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ProcessedData:
    adata_all: Any | None = None
    spidernet_data: Any | None = None
    metadata_sample: Any | None = None
    lr_list: Any | None = None
    batch_cell_unique: Any | None = None
    genenames_train: Any | None = None
    lr_list_cellchatdb: Any | None = None
    lr_meta_cellchatdb: Any | None = None
    batch_cell: Any | None = None
    adata_list: Any | None = None


def _require_scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for loading .h5ad files. Install SpiderNet with "
            "the '[full]' extra or install scanpy separately."
        ) from exc
    return sc


def _torch_load_cpu(path: str | Path) -> Any:
    """Load a torch-saved object onto CPU, compatible with old/new PyTorch."""
    import torch

    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose weights_only.
        return torch.load(path, map_location="cpu")


def get_spidernet_pyg_list_path(
    processed_dir_or_path: str | Path,
    *,
    prefer: str = "pkl",
    must_exist: bool = True,
) -> Path:
    """Resolve the SpiderNet PyG-list file, supporting both .pkl and .pt.

    Parameters
    ----------
    processed_dir_or_path
        Either the processed-data directory or a path to
        ``SpiderNet_data_pyg_list.pkl`` / ``SpiderNet_data_pyg_list.pt``.
    prefer
        Which extension to prefer when both files exist. Must be ``"pkl"`` or
        ``"pt"``. The default keeps old behavior.
    must_exist
        If True, raise FileNotFoundError when neither file exists. If False,
        return the preferred candidate path even when no candidate exists.

    Notes
    -----
    Passing an old hard-coded ``.../SpiderNet_data_pyg_list.pkl`` path is safe:
    if the .pkl file is missing but the companion .pt file exists, this function
    returns the .pt file automatically.
    """
    if prefer not in {"pkl", "pt"}:
        raise ValueError("prefer must be either 'pkl' or 'pt'.")

    path = Path(processed_dir_or_path)

    if path.is_dir() or path.suffix == "":
        pkl_path = path / "SpiderNet_data_pyg_list.pkl"
        pt_path = path / "SpiderNet_data_pyg_list.pt"
    else:
        if path.name not in {"SpiderNet_data_pyg_list.pkl", "SpiderNet_data_pyg_list.pt"}:
            if must_exist and not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")
            return path
        pkl_path = path.with_suffix(".pkl")
        pt_path = path.with_suffix(".pt")

    candidates = [pkl_path, pt_path] if prefer == "pkl" else [pt_path, pkl_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if must_exist:
        raise FileNotFoundError(
            "Missing SpiderNet PyG data file. Expected one of: "
            f"{pkl_path} or {pt_path}"
        )

    return candidates[0]


def spidernet_pyg_list_exists(processed_dir_or_path: str | Path) -> bool:
    """Return True if either SpiderNet_data_pyg_list.pkl or .pt exists."""
    try:
        get_spidernet_pyg_list_path(processed_dir_or_path, must_exist=True)
        return True
    except FileNotFoundError:
        return False


def load_spidernet_pyg_list(processed_dir_or_path: str | Path) -> Any:
    """Load SpiderNet_data_pyg_list from .pkl if present, otherwise from .pt."""
    path = get_spidernet_pyg_list_path(processed_dir_or_path, must_exist=True)

    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    if path.suffix == ".pt":
        return _torch_load_cpu(path)

    raise ValueError(f"Unsupported SpiderNet PyG data format: {path}")


def _load_if_exists(path: Path, loader, **kwargs) -> Any | None:
    return loader(path, **kwargs) if path.exists() else None


def load_processed_data(processed_dir: str | Path) -> ProcessedData:
    """Load a processed SpiderNet directory.

    This is backward-compatible with directories containing
    ``SpiderNet_data_pyg_list.pkl`` and also works when only
    ``SpiderNet_data_pyg_list.pt`` is present.
    """
    processed_dir = Path(processed_dir)
    sc = _require_scanpy()

    spidernet_path = get_spidernet_pyg_list_path(processed_dir, must_exist=False)
    spidernet_data = load_spidernet_pyg_list(spidernet_path) if spidernet_path.exists() else None

    return ProcessedData(
        adata_all=_load_if_exists(
            processed_dir / "adata_all.h5ad",
            sc.read_h5ad,
            backed="r",
        ),
        spidernet_data=spidernet_data,
        metadata_sample=_load_if_exists(
            processed_dir / "metadata_sample.csv",
            pd.read_csv,
        ),
        lr_list=_load_if_exists(
            processed_dir / "LR_list.pkl",
            pd.read_pickle,
        ),
        batch_cell_unique=_load_if_exists(
            processed_dir / "batch_cell_unique.pkl",
            pd.read_pickle,
        ),
        genenames_train=_load_if_exists(
            processed_dir / "genenames_train.pkl",
            pd.read_pickle,
        ),
        lr_list_cellchatdb=_load_if_exists(
            processed_dir / "LR_list_cellchatdb.pkl",
            pd.read_pickle,
        ),
        lr_meta_cellchatdb=_load_if_exists(
            processed_dir / "LR_meta_cellchatdb.pkl",
            pd.read_pickle,
        ),
        batch_cell=_load_if_exists(
            processed_dir / "batch_cell.pkl",
            pd.read_pickle,
        ),
        adata_list=_load_if_exists(
            processed_dir / "adata_list.pkl",
            pd.read_pickle,
        ),
    )


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
