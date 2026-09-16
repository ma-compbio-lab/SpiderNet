"""Run this folder's coupling, cell-similarity and spatial benchmark stages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark_config import BENCHMARK_DIR, DATA_ROOT, RESULTS_ROOT, DATASET_CONFIG


STUDIES = ["AgingMousebrain", "HGSOC"]
COUPLING_NOTEBOOK = BENCHMARK_DIR / "CCC_Coupling_benchmark_with_directionality.ipynb"
SPATIAL_NOTEBOOK = BENCHMARK_DIR / "Spatial specificity.ipynb"
SIMILARITY_SCRIPT = BENCHMARK_DIR / "ccc_cell_similarity.py"
SIMILARITY_OUTPUT_DIR = BENCHMARK_DIR / "output" / "cell_similarity"


def execute_notebook(path):
    """Execute in a fresh kernel using this Python environment; retain a run copy."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    client = NotebookClient(notebook, km=manager, timeout=None)
    outdir = BENCHMARK_DIR / "output" / "executed"
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = os.environ.get("SPIDERNET_SINGLE_DATASET", "both_studies")
    output = outdir / f"{path.stem}_{suffix}.ipynb"
    print(f"Executing {path.name} with {sys.executable}", flush=True)
    try:
        client.execute(cwd=str(path.parent), cleanup_kc=True)
    finally:
        nbformat.write(notebook, output)
        print(f"Execution record: {output}", flush=True)


def result_dir(cfg):
    return Path(cfg["results_path_main"]) / cfg["version"] / f"SpiderNet_Result_dim{cfg['dim_envir']}"


def processed_dir(cfg):
    root = Path(cfg["results_path_main"])
    candidates = [root / "ProcessedData", root / cfg["version"] / "ProcessedData"]
    return next((p for p in candidates if p.is_dir()), candidates[0])


def check_inputs(stages, plot_only):
    """Check the on-disk dependency chain before starting costly computations."""
    required = [BENCHMARK_DIR / "benchmark_common.py", BENCHMARK_DIR / "benchmark_config.py"]
    for study in STUDIES:
        cfg = DATASET_CONFIG[study]
        run = result_dir(cfg)
        if "coupling" in stages:
            required.append(COUPLING_NOTEBOOK)
            if plot_only:
                outdir = run / "Unified_CCC_benchmarking" / "Directionality_analyses"
                required += [outdir / name for name in (
                    "Analysis2_Benchmark1_with_reverse_direction_long.csv",
                    "Analysis1_MI_reverse_direction_Pearson_by_slice.csv",
                )]
            else:
                required += [processed_dir(cfg) / name for name in (
                    "adata_list.pkl", "SpiderNet_data_pyg_list.pkl", "LR_list.pkl"
                )]
                required += [run / "Factor_envir_list.pkl", run / cfg["targets_json_name"],
                             Path(cfg["omnipath_regulatory_csv"])]
        if "similarity" in stages:
            required.append(SIMILARITY_SCRIPT)
            if plot_only:
                required.append(SIMILARITY_OUTPUT_DIR / study / "cell_similarity_concordance_long.csv")
            else:
                # The standalone similarity calculation uses the dataset-level bundle.
                similarity_processed = Path(cfg["results_path_main"]) / "ProcessedData"
                required += [similarity_processed / name for name in (
                    "adata_list.pkl", "SpiderNet_data_pyg_list.pkl"
                )]
                required.append(run / "Factor_envir_list.pkl")
        if not plot_only and ("coupling" in stages or "similarity" in stages):
            required.append(Path(cfg["COMMOT_path_main"]))
            manifest = Path(cfg["ScCChain_path_main"]) / "h5ad_files.txt"
            required.append(manifest)
            if manifest.is_file():
                for line in manifest.read_text(encoding="utf-8-sig").splitlines():
                    if line.strip():
                        source = Path(line.strip().strip('"'))
                        required += [source, manifest.parent / f"{source.stem}_ScCChain_edge_program_scores.csv"]
            if cfg["include_spacia"]:
                required.append(Path(cfg["Spacia_path_main"]) / "spacia_outputs")
        if "spatial" in stages:
            required.append(SPATIAL_NOTEBOOK)
            dirs_path = Path(cfg["results_path_main"]) / "run_dirs.json"
            required.append(dirs_path)
            if dirs_path.is_file():
                dirs = json.loads(dirs_path.read_text(encoding="utf-8"))
                spatial_run = Path(dirs["run_dir"])
                if plot_only:
                    required.append(spatial_run / "Spatial_false_positive_null_benchmark" /
                        "Experiment1_topMIpair_true_neighbor_vs_far_edges" /
                        f"{study}_Exp1_topMIpair_true_neighbor_vs_far_summary_mean.csv")
                else:
                    model = Path(dirs.get("model_dir", spatial_run / "Model"))
                    required.append(model / "SpiderNet_model_config.json")
                    checkpoints = list(model.glob("model_epoch*.pth"))
                    if not checkpoints:
                        required.append(model / "model_epoch*.pth")
                    required += [processed_dir(cfg) / name for name in (
                        "adata_list.pkl", "SpiderNet_data_pyg_list.pkl", "LR_list.pkl",
                        "genenames_train.pkl",
                    )]
    missing = sorted({str(p) for p in required if not p.exists()})
    if missing:
        raise FileNotFoundError("Required inputs are missing:\n" + "\n".join(missing))
    print(f"Input path check passed ({len(set(required))} paths).", flush=True)
    if not plot_only:
        print("Comparator matrix contents and slice alignment are validated when loaded.", flush=True)


def plot_coupling():
    """Use the original plotting functions and saved per-slice values."""
    import ast
    from ccc_directionality_two_studies import plot_two_study_displays
    # Read the palette from the notebook's literal controls without executing analysis.
    notebook = json.loads(COUPLING_NOTEBOOK.read_text(encoding="utf-8"))
    palette = None
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "METHOD_COLORS" for t in node.targets
                ):
                    palette = ast.literal_eval(node.value)
    if palette is None:
        raise ValueError("METHOD_COLORS was not found in the coupling notebook.")
    outdir = BENCHMARK_DIR / "output" / "CCC_Coupling_benchmark_with_directionality"
    for output in plot_two_study_displays(DATASET_CONFIG, STUDIES, palette, outdir):
        print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "coupling", "similarity", "spatial"], default="all")
    parser.add_argument("--plot-only", action="store_true", help="Reuse saved plot-input tables; no model fitting or inference.")
    parser.add_argument("--check", action="store_true", help="Check input paths without running analyses.")
    parser.add_argument("--reuse-nmflr", action="store_true", help="Reuse archived NMF-LR factors known to match the notebook's settings.")
    parser.add_argument("--execute-notebook", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute_notebook:
        execute_notebook(args.execute_notebook)
        return
    stages = ["coupling", "similarity", "spatial"] if args.stage == "all" else [args.stage]
    check_inputs(stages, args.plot_only)
    if args.check:
        return
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["SPIDERNET_RECOMPUTE_NMFLR"] = "0" if args.reuse_nmflr else "1"
    if "coupling" in stages:
        if args.plot_only:
            plot_coupling()
        else:
            # The runner always processes both studies; the notebook also supports
            # interactive single-study work through its environment control.
            os.environ.pop("SPIDERNET_SINGLE_DATASET", None)
            execute_notebook(COUPLING_NOTEBOOK)
    if "similarity" in stages:
        command = [sys.executable, str(SIMILARITY_SCRIPT), "--data-root", str(DATA_ROOT),
                   "--results-root", str(RESULTS_ROOT), "--output-dir", str(SIMILARITY_OUTPUT_DIR)]
        if args.plot_only:
            command.append("--plot-only")
        elif not args.reuse_nmflr and "coupling" not in stages:
            command.append("--recompute-nmflr")
        subprocess.run(command, check=True)
    if "spatial" in stages:
        os.environ["SPIDERNET_PLOT_ONLY"] = "1" if args.plot_only else "0"
        execute_notebook(SPATIAL_NOTEBOOK)


if __name__ == "__main__":
    main()
