"""Check, execute, or replot HGSOC analyses in the invoking environment."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from workflow_paths import (HERE, DATA_ROOT, RESULTS_ROOT, PROCESSED_DATA_DIR,
                            OUTPUT, RUN, LOCAL_RUN, MODEL_DIR, saved, ensure_output, check_output_identity)

DEFAULT_STAGES = ["model-summary", "malignant", "baseline", "cascade", "ccc", "perturbation"]
STAGES = ["preprocess", "train", *DEFAULT_STAGES, "r-plots"]
RMD = "MI_CAF_functionalstate_visualization_V2_SMD_annotation.Rmd"


def bootstrap():
    return '''import os, json, re, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from workflow_paths import *
ensure_output()
run_dirs = dict(run_dirs)
run_dir = Path(run_dirs["run_dir"])
match = re.search(r"Result_dim(\\d+)", str(run_dir))
if match is None:
    raise ValueError(f"Cannot parse MI dimension from {run_dir}")
dim_envir = int(match.group(1))
DIM_ENVIR = dim_envir
cellclass_choose = "Malignant"
plt.rcdefaults()
'''


def stage_cells(stage, plot_only):
    if stage in ("malignant", "ccc", "perturbation"):
        import hgsoc_primary as workflow
    else:
        import hgsoc_secondary as workflow
    selected = workflow.plot_cells(stage) if plot_only else workflow.analysis_cells(stage)
    if stage == "preprocess" and not plot_only:
        selected += ['''# Publish the original preprocessing diagnostics without changing the bundle.
import shutil
preprocess_output = OUTPUT / "preprocess"
preprocess_output.mkdir(parents=True, exist_ok=True)
for original in PROCESSED_DATA_DIR.iterdir():
    if original.is_file() and original.suffix.lower() in (".csv", ".pdf", ".png", ".json") and (
        original.name.startswith("mi_dimension_selection_") or original.name == "LR_pairs_correlation_heatmap_clustered.png"
    ):
        shutil.copy2(original, preprocess_output / original.name)
''']
    return [bootstrap(), *selected]


def required_inputs(stage, plot_only):
    """Required data products; read fallback uses the same precedence as execution."""
    cancersea = DATA_ROOT / "CancerSEA_OV/functional_geneset_list_df.csv"
    kegg = saved("TCGA_OV_KEGG_gene_signature/C5_four_KEGG_pathway_reference_genes_long.csv")
    program_inputs = []
    for program in ("PD1_PDL1", "ECM_receptor"):
        for mi in ("MI10", "MI6", "MI13"):
            choices = [saved(f"Malignant_COI5_{program}_vs_Receiving{mi}_celllevel_scatter.csv"),
                       saved(f"Malignant_COI5_{program}_vs_Receiving_{mi}_scatter.csv")]
            program_inputs.append(next((p for p in choices if p.is_file()), choices[0]))
    if stage == "r-plots":
        return [saved(f"Malignant_COI5_{state}_vs_{direction}_{mi}_scatter.csv")
                for state, direction in [("CAF_score", "Sending"), ("Metastasis", "Receiving"),
                    ("Hypoxia", "Receiving"), ("Angiogenesis", "Receiving"), ("Inflammation", "Receiving")]
                for mi in ["MI10", "MI6", "MI13"]] + program_inputs
    if plot_only:
        if stage == "malignant":
            correlation = saved("Corr_MI_Receiver_TumorFunctionalState_Malignant5.csv")
            extra = [correlation] if correlation.is_file() else [PROCESSED_DATA_DIR / "adata_all.h5ad", cancersea]
            if not correlation.is_file() and cancersea.is_file():
                genes = None
                if importlib.util.find_spec("h5py") and (PROCESSED_DATA_DIR / "adata_all.h5ad").is_file():
                    import h5py
                    with h5py.File(PROCESSED_DATA_DIR / "adata_all.h5ad", "r") as handle:
                        var = handle["var"]
                        genes = set(var[var.attrs["_index"]].asstr()[:])
                with cancersea.open(encoding="utf-8-sig") as stream:
                    states = sorted({row["GeneSet"] for row in csv.DictReader(stream)
                                     if genes is None or row["Gene"] in genes})
                extra += [saved(f"Malignant_COI5_{state}_vs_Receiving_{mi}_scatter.csv")
                          for state in states for mi in ("MI10", "MI6", "MI13")]
            return [saved(n) for n in ["adata_choose_Malignant.h5ad", "Mean_MI_Intensity_cluster.csv",
                "FunctionalState_scores_by_MI_louvain_long.csv",
                "Malignant_KEGG_module_score_MI_UMAP_continuous_celllevel_module_scores_with_coordinates.csv",
                "Malignant_KEGG_module_score_MI_UMAP_continuous_MI_UMAP_continuous_score_summary.csv"]] + required_inputs("r-plots", True) + extra + [
                    saved(f"Malignant_COI5_{program}_vs_Receiving{mi}_celllevel_scatter.csv")
                    for program in ("PD1_PDL1", "ECM_receptor") for mi in ("MI10", "MI6", "MI13")]
        if stage == "ccc":
            return [saved("CCC_baseline_feature_SMD_comparison/" + n) for n in [
                "HGSOC_CCC_method_selected_feature_program_SMD_summary.csv",
                "HGSOC_CCC_method_selected_feature_program_score_long.csv"]]
        if stage == "perturbation":
            return [saved("adata_choose_Malignant.h5ad")] + [saved(n) for n in [
                "LFC_perturb_all_MI-10_Fibroblast_to_Malignant_C5_C5.csv",
                "LFC_perturb_all_with_KEGG_MI-10_Fibroblast_to_Malignant_C5_C5.csv"]]
        if stage == "baseline":
            return [saved(n) for n in ["adata_choose2_Malignant.h5ad", "adata_choose3_Malignant.h5ad"]]
        if stage == "cascade":
            stem = "UpstreamGated_MI12_MI10_Monocyte-Fibroblast-Malignant"
            return [saved(stem + suffix) for suffix in ["_single_unit_feature_records.csv", "_sample_level_feature_scores.csv"]]
        if stage == "model-summary":
            return [saved("MI_component_concordance/MI_component_concordance.csv")]
        raise ValueError(f"--plot-only does not run {stage}; select an analysis stage.")
    if stage == "preprocess":
        adata_dir = DATA_ROOT / "adata" if (DATA_ROOT / "adata").is_dir() else DATA_ROOT
        raw_files = sorted(adata_dir.glob("*.h5ad"))
        return [DATA_ROOT / "HGSOC_metadata.csv", *(raw_files or [adata_dir / "MISSING_RAW_H5AD"])]
    bundle = [PROCESSED_DATA_DIR / n for n in ["adata_all.h5ad", "adata_list.pkl",
        "SpiderNet_data_pyg_list.pkl", "LR_list.pkl", "LR_list_cellchatdb.pkl", "LR_meta_cellchatdb.pkl",
        "batch_cell_unique.pkl", "batch_cell.pkl", "genenames_train.pkl", "cellclass_unique.pkl", "metadata_sample.csv"]]
    if stage == "train":
        return bundle
    common = bundle + [RESULTS_ROOT / "run_dirs.json", saved("Factor_envir_list.pkl")]
    if stage == "model-summary":
        return common + [saved(n) for n in ["Factor_envir_use.npy", "loading_LR_use.npy", "loading_sender_use.npy", "loading_receiver_use.npy"]]
    if stage == "malignant":
        return common + [cancersea, DATA_ROOT / "Malignant_TIL_upgenes.csv", kegg] + [saved(n) for n in [
            "LR_list_merge.pkl", "Avg_MI_cellclass_pair_merge_use.pkl", "loading_LR_use.npy"]]
    if stage == "baseline":
        return common + [cancersea, kegg, saved("adata_choose_Malignant.h5ad"),
            saved("HGSOC_malignant_cell_KEGG_module_scores_SpiderNet_MI_louvain.csv"),
            DATA_ROOT / "Banksy/Banksy_Umap_Malignant.csv", DATA_ROOT / "Banksy/Banksy_cluster_Malignant.csv"]
    if stage == "cascade":
        return common + [cancersea, saved("adata_choose_Malignant.h5ad"), saved("LR_meta_incellchatdb.csv")]
    if stage == "ccc":
        return common + [cancersea, kegg, saved("adata_choose_Malignant.h5ad"),
                         RESULTS_ROOT / "COMMOT", RESULTS_ROOT / "ScCChain/h5ad_files.txt", RESULTS_ROOT / "Spacia"]
    if stage == "perturbation":
        checkpoints = sorted(MODEL_DIR.glob("model_epoch*.pth"), key=lambda p: int(re.findall(r"epoch(\d+)", p.name)[0]))
        checkpoint = MODEL_DIR / "model_epoch19999.pth"
        if not checkpoint.exists() and checkpoints:
            checkpoint = checkpoints[-1]
        return common + [cancersea, kegg, saved("adata_choose_Malignant.h5ad"),
                         saved("loading_LR_use.csv"),
                         MODEL_DIR / "SpiderNet_model_config.json", checkpoint]
    raise ValueError(stage)


def rscript():
    configured = os.environ.get("RSCRIPT")
    if configured:
        return configured
    discovered = shutil.which("Rscript")
    if discovered:
        return discovered
    candidates = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"))
    return str(candidates[-1]) if candidates else None


def check(stages, plot_only):
    failures = []
    try:
        check_output_identity()
    except ValueError as exc:
        failures.append(str(exc))
    modules = ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "nbformat", "nbclient", "ipykernel"]
    if any(s in ("model-summary", "malignant", "baseline", "perturbation") for s in stages) or not plot_only:
        modules += ["scanpy", "SpiderNet"]
    if not plot_only and any(s != "r-plots" for s in stages):
        modules += ["torch", "torch_geometric", "torch_scatter", "gseapy"]
    for module in dict.fromkeys(modules):
        if importlib.util.find_spec(module) is None:
            failures.append(f"Python dependency: {module} ({sys.executable})")
    for stage in stages:
        paths = required_inputs(stage, plot_only)
        missing = [str(path) for path in paths if not path.exists()]
        failures += [f"{stage}: {path}" for path in missing]
        if stage == "r-plots":
            executable = rscript()
            if not executable:
                failures.append("r-plots: Rscript (set RSCRIPT)")
            else:
                probe = subprocess.run([executable, "-e", 'pkgs <- c("ggplot2", "dplyr", "knitr", "tibble", "rmarkdown"); stopifnot(all(vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)))'], capture_output=True, text=True)
                if probe.returncode:
                    failures.append("r-plots: " + probe.stderr.strip())
        else:
            try:
                for i, code in enumerate(stage_cells(stage, plot_only)):
                    compile(code, f"{stage}:cell{i}", "exec")
            except Exception as exc:
                failures.append(f"{stage}: source assembly: {exc}")
        print(f"{stage}: {len(paths)-len(missing)}/{len(paths)} required inputs present")
    for notebook in HERE.glob("*.ipynb"):
        data = json.loads(notebook.read_text(encoding="utf-8"))
        for index, cell in enumerate(data["cells"]):
            if cell["cell_type"] == "code":
                try:
                    compile("".join(cell["source"]), f"{notebook.name}:{index}", "exec")
                except SyntaxError as exc:
                    failures.append(str(exc))
    print(f"Python: {sys.executable}\nUpstream run: {RUN}\nOutput: {LOCAL_RUN}")
    if failures:
        print("Missing inputs or validation failures:\n- " + "\n- ".join(failures))
    else:
        print("Input/source checks passed. This is not a numerical reproduction check.")
    return failures


def execute_notebook(stage, plot_only, stamp):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(c) for c in stage_cells(stage, plot_only)])
    mode = "plot" if plot_only else "analysis"
    notebook.metadata["hgsoc_run"] = {"stage": stage, "mode": mode, "python": sys.executable,
                                     "upstream_run": str(RUN), "output": str(LOCAL_RUN)}
    path = OUTPUT / "executed" / f"{stamp}_{stage}_{mode}.ipynb"
    km = KernelManager()
    # Never use a kernelspec that silently selects a different analysis environment.
    km._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                 display_name="Current HGSOC environment", language="python")
    client = NotebookClient(notebook, km=km, timeout=None, resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute(cwd=str(HERE), cleanup_kc=True)
    finally:
        nbformat.write(notebook, path)
    return path


def execute_r(stamp):
    destination = OUTPUT / "executed" / f"{stamp}_{RMD}"
    shutil.copy2(HERE / RMD, destination)
    env = os.environ.copy()
    env.update(HGSOC_RUN_DIR=str(RUN), HGSOC_LOCAL_RUN=str(LOCAL_RUN), HGSOC_R_OUTPUT=str(OUTPUT / "r-plots"))
    pandoc = Path("C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools")
    if "RSTUDIO_PANDOC" not in env and (pandoc / "pandoc.exe").exists():
        env["RSTUDIO_PANDOC"] = str(pandoc)
    # Paths are passed as process arguments, never interpolated as R source code.
    script = 'args <- commandArgs(TRUE); rmarkdown::render(args[1], output_dir=args[2], quiet=TRUE, envir=new.env())'
    subprocess.run([rscript(), "-e", script, str(destination), str(destination.parent)], env=env, check=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check selected inputs, dependencies and source syntax; do not run analysis.")
    parser.add_argument("--plot-only", action="store_true", help="Redraw supported components from saved scores/embeddings; never fit models.")
    parser.add_argument("--stage", action="append", choices=STAGES, help="Select stages in execution order; repeat for multiple stages.")
    args = parser.parse_args()
    stages = args.stage or DEFAULT_STAGES
    if args.plot_only and any(s in ("preprocess", "train") for s in stages):
        parser.error("--plot-only requires analysis stages, not preprocess/train")
    if check(stages, args.plot_only):
        return 1
    if args.check:
        return 0
    ensure_output()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {"started": stamp, "python": sys.executable, "plot_only": args.plot_only,
                "upstream_run": str(RUN), "output": str(LOCAL_RUN), "stages": [],
                "processed_data": str(PROCESSED_DATA_DIR), "data": str(DATA_ROOT), "model": str(MODEL_DIR),
                "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in HERE.iterdir() if p.suffix in (".py", ".ipynb", ".Rmd")}}
    manifest_path = OUTPUT / "executed" / f"{stamp}_run.json"
    for stage in stages:
        print(f"Running {stage} ({'saved-result plotting' if args.plot_only else 'analysis'})", flush=True)
        record = {"stage": stage, "status": "running"}
        manifest["stages"].append(record)
        try:
            path = execute_r(stamp) if stage == "r-plots" else execute_notebook(stage, args.plot_only, stamp)
            record.update(status="completed", executed_copy=str(path))
            if path.suffix == ".ipynb":
                executed = json.loads(path.read_text(encoding="utf-8"))
                notes = [line for cell in executed["cells"] for output in cell.get("outputs", [])
                         for line in "".join(output.get("text", [])).splitlines()
                         if line.startswith("SKIP ") or "was not regenerated:" in line or line.startswith("Cascade in situ:")]
                if notes:
                    record.update(status="completed_with_limits", notes=notes)
                    for note in notes:
                        print(note, flush=True)
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Results: {LOCAL_RUN}\nExecution record: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
