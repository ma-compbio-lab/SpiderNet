import pickle

from dataclasses import dataclass
<<<<<<< HEAD
from typing import Any

=======
from pathlib import Path
from typing import Any

import pandas as pd


>>>>>>> 21dee28 (Update SpiderNet package and tutorials)
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

<<<<<<< HEAD
=======

>>>>>>> 21dee28 (Update SpiderNet package and tutorials)
def _require_scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for loading .h5ad files. Install SpiderNet with the '[full]' extra or install scanpy separately."
        ) from exc
    return sc

<<<<<<< HEAD
from pathlib import Path
import pandas as pd
=======

def _torch_load_cpu(path: str | Path):
    import torch

    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only argument.
        return torch.load(path, map_location="cpu")


def get_spidernet_pyg_list_path(
    processed_dir_or_path: str | Path,
    *,
    prefer: str = "pkl",
    must_exist: bool = True,
) -> Path:
    """Return the existing SpiderNet PyG file path, supporting both .pkl and .pt.

    If a directory is provided, this checks:
      1. SpiderNet_data_pyg_list.pkl
      2. SpiderNet_data_pyg_list.pt

    If a file path is provided and that exact file is missing, this also checks the
    companion file with the other extension. This preserves old calls that pass
    ".../SpiderNet_data_pyg_list.pkl" while allowing the new .pt fallback.
    """
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
    try:
        get_spidernet_pyg_list_path(processed_dir_or_path, must_exist=True)
        return True
    except FileNotFoundError:
        return False


def load_spidernet_pyg_list(processed_dir_or_path: str | Path):
    """Load SpiderNet_data_pyg_list from .pkl if present, otherwise from .pt.

    The returned object is the same Python object structure as before: a list of
    PyTorch Geometric Data/dict-like objects. Loading .pt uses map_location="cpu"
    so downstream code can move the objects to GPU explicitly as before.
    """
    path = get_spidernet_pyg_list_path(processed_dir_or_path, must_exist=True)

    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    if path.suffix == ".pt":
        return _torch_load_cpu(path)

    raise ValueError(f"Unsupported SpiderNet PyG data format: {path}")

>>>>>>> 21dee28 (Update SpiderNet package and tutorials)

def load_processed_data(processed_dir: str | Path) -> ProcessedData:
    processed_dir = Path(processed_dir)
    sc = _require_scanpy()

    def _load_if_exists(path: Path, loader, **kwargs):
        return loader(path, **kwargs) if path.exists() else None

<<<<<<< HEAD
=======
    spidernet_path = get_spidernet_pyg_list_path(processed_dir, must_exist=False)
    spidernet_data = (
        load_spidernet_pyg_list(spidernet_path)
        if spidernet_path.exists()
        else None
    )

>>>>>>> 21dee28 (Update SpiderNet package and tutorials)
    return ProcessedData(
        adata_all=_load_if_exists(
            processed_dir / "adata_all.h5ad",
            sc.read_h5ad,
            backed="r",
        ),
<<<<<<< HEAD
        spidernet_data=_load_if_exists(
            processed_dir / "SpiderNet_data_pyg_list.pkl",
            pd.read_pickle,
        ),
=======
        spidernet_data=spidernet_data,
>>>>>>> 21dee28 (Update SpiderNet package and tutorials)
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

<<<<<<< HEAD
=======

>>>>>>> 21dee28 (Update SpiderNet package and tutorials)
def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
