"""Redraw fixed-model LR perturbation panels from saved cell-level changes."""

import ast
import json
from pathlib import Path
import shutil


NOTEBOOK = Path(__file__).with_name("Pancancer_MI4_Fibroblast_to_Tumor_LRKO.ipynb")
RESULT_SUBDIR = "MI4_fibroblast_to_tumor_LRKO_CancerSEA"


def _source(tag):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if tag in c.get("metadata", {}).get("tags", [])]
    if len(cells) != 1:
        raise ValueError(f"Expected one notebook cell tagged {tag!r}.")
    return "".join(cells[0]["source"])


def _settings():
    namespace = {}
    for node in ast.parse(_source("lrko-settings")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"PC_LRKO_MI", "PC_LRKO_MODULE_SCORE_METHOD"}
            for t in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["PC_LRKO_MI"], namespace["PC_LRKO_MODULE_SCORE_METHOD"]


def input_path(results_dir, output_dir=None):
    mi, method = _settings()
    filename = f"{mi}_Fibroblast_to_tumor_LRKO_CancerSEA_delta_long_{method}.csv"
    if output_dir is not None and (Path(output_dir) / "lrko" / filename).is_file():
        return Path(output_dir) / "lrko" / filename
    return Path(results_dir) / RESULT_SUBDIR / filename


def check(results_dir):
    source = input_path(results_dir)
    return [] if source.is_file() else [f"LR perturbation cell-level changes: {source}"]


def run(results_dir: Path, output_dir: Path) -> list[Path]:
    """Use the original plotting cell, without rerunning inference or tests.

    Only the four columns consumed by the figure are read. Every row is retained;
    neither KDE input nor percentile limits are sampled or approximated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd

    source = input_path(results_dir, output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"LR perturbation cell-level changes: {source}")
    output_dir = Path(output_dir) / "lrko"
    output_dir.mkdir(parents=True, exist_ok=True)
    mi, method = _settings()
    table = pd.read_csv(source, usecols=["CancerType", "GO_program", "Perturbation", "delta_module_score"])
    namespace = {
        "np": np, "pd": pd, "PC_LRKO_MI": mi,
        "PC_LRKO_MODULE_SCORE_METHOD": method,
        "pc_mi4_lrko_outdir": output_dir,
        "pc_mi4_lrko_cancersea_delta_long": table,
    }
    # The original full table is already saved by analysis. Plot-only avoids
    # exporting a second multi-gigabyte copy of the same cell-level records.
    tree = ast.parse(_source("lrko-plot"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ((isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == "plot_df" and node.value.func.attr == "to_csv")
             or (isinstance(node.value.func, ast.Name) and node.value.func.id == "print"
                 and any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and a.value.startswith("Saved LR-KO plotting table:") for a in node.value.args)))
    )]
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    outputs = [Path(str(namespace["fig_prefix"]) + suffix) for suffix in (".pdf", ".png")]
    for suffix in (
        f"delta_summary_by_cancertype_{method}.csv",
        f"KOTop_vs_KORandom_by_cancertype_{method}.csv",
        f"KOTop_vs_KORandom_per_panel_two_sided_tests_{method}.csv",
    ):
        path = source.parent / f"{mi}_Fibroblast_to_tumor_LRKO_CancerSEA_{suffix}"
        if path.is_file():
            target = output_dir / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
    (output_dir / "plot_status.json").write_text(json.dumps({
        "source": str(source), "rows": len(table), "subsampling": False,
        "regenerated_figures": [str(p) for p in outputs],
        "statistics": "Existing statistical summaries copied; tests were not rerun.",
    }, indent=2) + "\n", encoding="utf-8")
    return outputs
