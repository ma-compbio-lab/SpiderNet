import pickle

from dataclasses import dataclass
from typing import Any

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
            "scanpy is required for loading .h5ad files. Install SpiderNet with the '[full]' extra or install scanpy separately."
        ) from exc
    return sc

from pathlib import Path
import pandas as pd

def load_processed_data(processed_dir: str | Path) -> ProcessedData:
    processed_dir = Path(processed_dir)
    sc = _require_scanpy()

    def _load_if_exists(path: Path, loader, **kwargs):
        return loader(path, **kwargs) if path.exists() else None

    return ProcessedData(
        adata_all=_load_if_exists(
            processed_dir / "adata_all.h5ad",
            sc.read_h5ad,
            backed="r",
        ),
        spidernet_data=_load_if_exists(
            processed_dir / "SpiderNet_data_pyg_list.pkl",
            pd.read_pickle,
        ),
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
