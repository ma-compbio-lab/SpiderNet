"""Paths only: scientific settings and cache policies stay in the notebooks."""
from pathlib import Path
import os
import hashlib

HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("SPIDERNET_WORKSPACE_ROOT", next(
    (p for p in HERE.parents if (p / "Data").is_dir() and (p / "Results").is_dir()),
    "D:/SpiderNet",
))).expanduser().resolve()
DATA_DIR = Path(os.environ.get("SPIDERNET_AGINGBRAIN_DATA_ROOT", WORKSPACE / "Data/AgingBrain"))
RESULTS_ROOT = Path(os.environ.get("SPIDERNET_AGINGBRAIN_RESULTS_ROOT", WORKSPACE / "Results/AgingBrain"))
OUTPUT = Path(os.environ.get("SPIDERNET_AGINGBRAIN_OUTPUT", HERE / "output")).resolve()
RUN_SUFFIX = Path("V1/SpiderNet_Result_dim30")
RUNS = {
    "coronal": RESULTS_ROOT / RUN_SUFFIX,
    "sagittal": RESULTS_ROOT / "AgingBrain_Sagittal" / RUN_SUFFIX,
    "hippocampus": RESULTS_ROOT / "AgingBrain_Hippocampus" / RUN_SUFFIX,
}


def local_path(path):
    """Translate analysis-result paths; raw and processed bundles stay upstream."""
    path = Path(path)
    for label, upstream in RUNS.items():
        try:
            return OUTPUT / label / path.relative_to(upstream)
        except ValueError:
            pass
    if path.parent == RESULTS_ROOT:
        return OUTPUT / "coronal" / path.name
    return path


def input_path(path):
    """Use a newly generated local file, otherwise the original upstream file.

    Directories are never substituted: processed-bundle loaders require the
    complete upstream bundle, not a directory containing only analysis exports.
    """
    candidate = local_path(path)
    return candidate if candidate.is_file() else Path(path)


def output_path(path):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def saved(label, relative):
    return input_path(RUNS[label] / relative)


def same_file_content(first, second):
    """Guard reuse across two independently loaded copies of the same graph bundle."""
    first, second = Path(first), Path(second)
    if first.resolve() == second.resolve():
        return True
    if first.stat().st_size != second.stat().st_size:
        return False
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.digest()
    return digest(first) == digest(second)
