"""Filesystem locations shared by the real-tissue benchmarks.

Set SPIDERNET_DATA_ROOT and SPIDERNET_RESULTS_ROOT before starting Python to
use another data installation. Scientific parameters remain in the notebooks.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
_workspace = next(
    (p for p in BENCHMARK_DIR.parents if (p / "Data").is_dir() and (p / "Results").is_dir()),
    PROJECT_ROOT,
)
DATA_ROOT = Path(os.environ.get("SPIDERNET_DATA_ROOT", _workspace / "Data")).resolve()
RESULTS_ROOT = Path(os.environ.get("SPIDERNET_RESULTS_ROOT", _workspace / "Results")).resolve()


def add_spidernet_to_path():
    """Use the installed analysis package, or an explicitly selected source tree.

    SPIDERNET_PACKAGE_ROOT names the directory containing the SpiderNet package.
    Keep the installed version by default so checkpoint inference uses the same
    implementation as the analysis environment.
    """
    override = os.environ.get("SPIDERNET_PACKAGE_ROOT")
    if override:
        root = Path(override).resolve()
        if not (root / "SpiderNet" / "__init__.py").is_file():
            raise FileNotFoundError(f"No SpiderNet package under {root}")
        sys.path.insert(0, str(root))
        return
    spec = importlib.util.find_spec("SpiderNet")
    if spec is not None and spec.origin is not None:
        return
    for root in (PROJECT_ROOT, PROJECT_ROOT / "SpiderNet"):
        if (root / "SpiderNet" / "__init__.py").is_file():
            sys.path.insert(0, str(root))
            return
    raise ModuleNotFoundError(
        "Install the SpiderNet version used for the saved results, or set "
        "SPIDERNET_PACKAGE_ROOT to its source directory."
    )


DATASET_CONFIG = {
    "AgingMousebrain": {
        "dataset_name": "AgingMousebrain",
        "display_name": "aging mouse brain",
        "version": "V1",
        "dim_envir": 30,
        "cellclass_name": "celltype",
        "data_path_main": (DATA_ROOT / "AgingBrain").as_posix() + "/",
        "results_path_main": (RESULTS_ROOT / "AgingBrain").as_posix() + "/",
        "COMMOT_path_main": str(RESULTS_ROOT / "AgingBrain" / "COMMOT"),
        "ScCChain_path_main": str(RESULTS_ROOT / "AgingBrain" / "ScCChain"),
        "Spacia_path_main": None,
        "include_spacia": False,
        "targets_json_name": "targets_by_LR.json",
        "omnipath_regulatory_csv": str(DATA_ROOT / "Database" / "OmnipathR" / "interactions_regulatory_mouse.csv"),
        "min_upstream_regulators": 1,
        "sample_obs_key": "age",
    },
    "HGSOC": {
        "dataset_name": "HGSOC",
        "display_name": "HGSC / HGSOC",
        "version": "V1",
        "dim_envir": 15,
        "cellclass_name": "cell.types",
        "data_path_main": (DATA_ROOT / "HGSOC").as_posix() + "/",
        "results_path_main": (RESULTS_ROOT / "HGSOC").as_posix() + "/",
        "COMMOT_path_main": str(RESULTS_ROOT / "HGSOC" / "COMMOT"),
        "ScCChain_path_main": str(RESULTS_ROOT / "HGSOC" / "ScCChain"),
        "Spacia_path_main": str(RESULTS_ROOT / "HGSOC" / "Spacia"),
        "include_spacia": True,
        "targets_json_name": "targets_by_LR.json",
        "omnipath_regulatory_csv": str(DATA_ROOT / "Database" / "OmnipathR" / "interactions_regulatory_human.csv"),
        "min_upstream_regulators": 5,
        "sample_obs_key": "samples",
    },
}
