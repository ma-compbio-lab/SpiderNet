"""Paths for the HGSOC analysis; model/data bundles remain upstream inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib
import shutil

HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("SPIDERNET_ROOT", "D:/SpiderNet"))
DATA_ROOT = Path(os.environ.get("HGSOC_DATA_ROOT", WORKSPACE / "Data/HGSOC"))
RESULTS_ROOT = Path(os.environ.get("HGSOC_RESULTS_ROOT", WORKSPACE / "Results/HGSOC"))
OUTPUT = Path(os.environ.get("HGSOC_OUTPUT", HERE / "output")).resolve()

_setup_path = RESULTS_ROOT / "HGSOC_modeltraining_setup.json"
training_setup = json.loads(_setup_path.read_text(encoding="utf-8")) if _setup_path.is_file() else {}
# Preserve the original setup-file precedence unless the caller explicitly relocates it.
if "HGSOC_RESULTS_ROOT" not in os.environ:
    RESULTS_ROOT = Path(training_setup.get("OUTPUT_ROOT", RESULTS_ROOT))
PROCESSED_DATA_DIR = Path(os.environ.get(
    "HGSOC_PROCESSED_DATA", training_setup.get("PROCESSED_DATA_DIR", RESULTS_ROOT / "ProcessedData")))
_manifest_path = RESULTS_ROOT / "run_dirs.json"
_manifest = json.loads(_manifest_path.read_text(encoding="utf-8")) if _manifest_path.is_file() else {}
RUN = Path(os.environ.get("HGSOC_RUN_DIR", _manifest.get("run_dir", RESULTS_ROOT / "V1/SpiderNet_Result_dim15")))
MODEL_DIR = (RUN / "Model" if "HGSOC_RUN_DIR" in os.environ else
             Path(_manifest.get("model_dir", RUN / "Model")))
LOCAL_RUN = OUTPUT / RUN.parent.name / RUN.name
run_dirs = {"run_dir": str(LOCAL_RUN), "model_dir": str(MODEL_DIR)}
OUTPUT_ROOT = RESULTS_ROOT


def input_path(path):
    """Read a new local result first, otherwise the matching saved upstream file."""
    if not isinstance(path, (str, os.PathLike)):
        return path
    candidate = Path(path)
    if candidate.exists():
        return path
    try:
        resolved = RUN / candidate.resolve().relative_to(LOCAL_RUN.resolve())
    except ValueError:
        return path
    return str(resolved) if isinstance(path, str) else resolved


def output_path(path):
    """Relocate a result path without changing its established relative filename."""
    candidate = Path(path)
    try:
        return LOCAL_RUN / candidate.resolve().relative_to(RUN.resolve())
    except ValueError:
        return candidate


def saved(name):
    return input_path(LOCAL_RUN / name)


def input_sources():
    return {key: str(path.resolve()) for key, path in {
        "run": RUN, "processed_data": PROCESSED_DATA_DIR,
        "data": DATA_ROOT, "model": MODEL_DIR,
    }.items()}


def check_output_identity():
    record = OUTPUT / "executed" / "input_sources.json"
    if record.exists() and json.loads(record.read_text(encoding="utf-8")) != input_sources():
        raise ValueError("This output folder belongs to different upstream inputs. "
                         "Set HGSOC_OUTPUT to a new empty folder to avoid mixing results.")


def ensure_output():
    check_output_identity()
    LOCAL_RUN.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "executed").mkdir(parents=True, exist_ok=True)
    record = OUTPUT / "executed" / "input_sources.json"
    if not record.exists():
        record.write_text(json.dumps(input_sources(), indent=2), encoding="utf-8")


def retain_existing(names, reason):
    """Keep unrebuildable old figures as explicitly recorded copies, not redraws."""
    record_path = OUTPUT / "executed" / "retained_artifacts.json"
    ensure_output()
    records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    for name in names:
        origin, destination = RUN / name, LOCAL_RUN / name
        if origin.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            records[name] = {"source": str(origin), "reason": reason,
                             "sha256": hashlib.sha256(origin.read_bytes()).hexdigest()}
            print(f"RETAINED existing artifact (not regenerated): {name}")
    record_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
