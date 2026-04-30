"""
Dataset discovery for the V2 *_UI layout.

A dataset is any directory of the form `<Name>_UI/` that contains:
  - ProcessedData/                     (adata_list.pkl, SpiderNet_data_pyg_list.pkl, ...)
  - <VERSION>/SpiderNet_Result_dim*/   (Factor_envir_list.pkl, loading_*.npy, Model/)
  - <Name>_modeltraining_setup.json    (DIM_ENVIR, VERSION, SPECIES)
  - config.json                        (CELL_TYPE_COL, SAMPLE_ID_COL, SPATIAL_KEY, ...)
  - run_dirs.json                      (run_dir, model_dir — Windows paths inside; ignored)

The path values inside the JSON files are Windows paths from the original
training environment and must NOT be trusted; we resolve everything from the
location of the JSON files on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetPaths:
    name: str                        # e.g. "AgingBrain", "HGSOC"
    root: Path                       # <Name>_UI/
    processed_dir: Path              # <Name>_UI/ProcessedData/
    run_dir: Path                    # <Name>_UI/<VERSION>/SpiderNet_Result_dim*/
    model_dir: Path                  # <run_dir>/Model/
    setup: dict                      # parsed <Name>_modeltraining_setup.json
    config: dict                     # parsed config.json
    dim_envir: int                   # parsed from run_dir folder name
    version: str                     # e.g. "V1"

    def __repr__(self) -> str:
        return f"DatasetPaths(name={self.name!r}, dim_envir={self.dim_envir}, version={self.version!r})"


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dataset(ui_dir: Path) -> Optional[DatasetPaths]:
    """Resolve one *_UI directory. Returns None if it doesn't look like a dataset."""
    if not ui_dir.is_dir() or not ui_dir.name.endswith("_UI"):
        return None

    name = ui_dir.name[: -len("_UI")]
    setup_path = ui_dir / f"{name}_modeltraining_setup.json"
    config_path = ui_dir / "config.json"
    if not setup_path.exists() or not config_path.exists():
        return None

    setup = _read_json(setup_path)
    config = _read_json(config_path)
    version = setup.get("VERSION") or config.get("VERSION") or "V1"

    version_dir = ui_dir / version
    if not version_dir.is_dir():
        return None

    run_dirs = sorted(version_dir.glob("SpiderNet_Result_dim*"))
    if not run_dirs:
        return None
    run_dir = run_dirs[0]

    try:
        dim_envir = int(run_dir.name.split("dim", 1)[1])
    except (ValueError, IndexError):
        dim_envir = int(setup.get("DIM_ENVIR") or config.get("DIM_ENVIR") or 0)

    processed_dir = ui_dir / "ProcessedData"
    model_dir = run_dir / "Model"

    if not processed_dir.is_dir():
        return None

    return DatasetPaths(
        name=name,
        root=ui_dir,
        processed_dir=processed_dir,
        run_dir=run_dir,
        model_dir=model_dir,
        setup=setup,
        config=config,
        dim_envir=dim_envir,
        version=version,
    )


def discover_datasets(search_root: Path) -> dict[str, DatasetPaths]:
    """Scan one parent directory for *_UI dataset folders."""
    datasets: dict[str, DatasetPaths] = {}
    if not search_root.is_dir():
        return datasets
    for child in sorted(search_root.iterdir()):
        ds = _resolve_dataset(child)
        if ds is not None:
            datasets[ds.name] = ds
    return datasets
