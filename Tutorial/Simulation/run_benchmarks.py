#!/usr/bin/env python3
"""Run simulation analysis from existing upstream data and method outputs."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

HERE = Path(__file__).resolve().parent
SETTINGS = ["Dropout1", "Dropout2", "Dropout3", "Sd_use1", "Sd_use2", "Sd_use3"]
NOTEBOOKS = {"summary": "Simulation_Benchmark_Pipeline.ipynb",
             "insitu": "Simulation_Benchmark_InSitu_Comparison.ipynb"}
MERGED_FILES = ["Benchmark_MacroMetrics_all_five_methods.csv", "AUROC_macro_all_five_methods.csv",
                "AUPRC_macro_all_five_methods.csv", "LoadingRank_Ratio_Summary_all_available_methods.csv",
                "LoadingRank_Ratio_Summary_SpiderNet.csv", "LoadingMeanRankRatio_by_MI.csv",
                "LoadingMeanRankRatio_definition.json", "LoadingMeanRankRatio_all80_by_MI.csv",
                "LoadingMeanRankRatio_all80_definition.json"]
DATA_FILES = ["gene_exp.csv", "cell_metadf.csv", "gene_metadf.csv", "edge_metadf.csv", "spatial_location.csv"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the requested mode without running analysis")
    parser.add_argument("--plot-only", action="store_true", help="Use saved merged tables and verified in-situ scores; never fit")
    parser.add_argument("--stage", choices=["all", "summary", "sample", "insitu", "train", "generate"], default="all")
    parser.add_argument("--setting", type=int, choices=range(6), help="Representative sample setting (default 0); training/generation subset")
    parser.add_argument("--experiment", type=int, choices=range(30), help="Representative replicate (default 18); training/generation subset")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("SIMULATION_DATA_ROOT", "D:/SpiderNet/Data/Simulation")))
    parser.add_argument("--result-root", type=Path, default=Path(os.environ.get("SIMULATION_RESULT_ROOT", "D:/SpiderNet/Results/Simulation")))
    parser.add_argument("--spacia-root", type=Path, help="Override saved Spacia analysis directory")
    parser.add_argument("--output-root", type=Path, default=HERE / "output")
    args = parser.parse_args()
    if args.plot_only and args.stage in ("generate", "train"):
        parser.error("--plot-only cannot be combined with --stage generate/train")
    for name in ("data_root", "result_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def configure(args):
    for key in ("data", "result", "output"):
        os.environ[f"SIMULATION_{key.upper()}_ROOT"] = str(getattr(args, f"{key}_root"))
    if args.spacia_root is not None:
        os.environ["SIMULATION_SPACIA_ROOT"] = str(args.spacia_root.resolve())
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Respect the active environment's installed SpiderNet package. The shared
    # repository package remains an allowed upstream dependency when not installed.
    package_parent = HERE.parents[1] / "SpiderNet"
    if importlib.util.find_spec("SpiderNet") is None and (package_parent / "SpiderNet" / "api.py").is_file():
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(package_parent), os.environ.get("PYTHONPATH", "")]))


def representative(args):
    return (0 if args.setting is None else args.setting,
            18 if args.experiment is None else args.experiment)


def grid(args):
    return [(s, e) for s in (range(6) if args.setting is None else [args.setting])
            for e in (range(30) if args.experiment is None else [args.experiment])]


def selected_stages(args):
    return ["summary", "sample", "insitu"] if args.stage == "all" else [args.stage]


def check_inputs(args):
    """Preflight includes every replicate; missing data never yields a partial chart silently."""
    stages = selected_stages(args)
    required, errors = [], []
    modules = {"numpy", "pandas", "matplotlib", "scipy"}
    if set(stages) & {"summary", "insitu"}:
        modules |= {"nbformat", "nbclient", "ipykernel"}
    if set(stages) & {"sample", "insitu", "train", "generate"}:
        modules |= {"scanpy", "sklearn"}
    if set(stages) & {"sample", "train"}:
        modules |= {"torch"}
    if "train" in stages:
        modules |= {"torch_geometric", "torch_scatter"}
    if "insitu" in stages and not args.plot_only:
        modules |= {"commot"}
    for module in sorted(modules):
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"Python import {module}: {exc}")
    for stage in set(stages) & set(NOTEBOOKS):
        required.append(HERE / NOTEBOOKS[stage])
    s, e = representative(args)
    sample = args.data_root / SETTINGS[s] / f"Experiment_{e}"
    model = args.result_root / "SpiderNet" / SETTINGS[s] / f"Experiment_{e}" / "SpiderNet_Result_Mode_cell_class"
    if "summary" in stages:
        if args.plot_only:
            required += [args.output_root / "Merged_Benchmark" / name for name in MERGED_FILES]
        else:
            for setting in SETTINGS:
                for exp in range(30):
                    suffix = Path(setting) / f"Experiment_{exp}"
                    for method, leaf in [("SpiderNet", "SpiderNet_Result_Mode_cell_class"),
                                         ("COMMOT", "COMMOT_Result"), ("NMF_LR", "NMF_LR_Result")]:
                        required.append(args.result_root / method / suffix / leaf / "Benchmark_MacroMetrics.csv")
                    required.append(args.result_root / "SpiderNet" / suffix / "SpiderNet_Result_Mode_cell_class" / "LoadingRank_Ratio_Summary.csv")
                    required += [args.data_root / suffix / name for name in ("gene_metadf.csv", "edge_metadf.csv")]
                    saved_sample = args.result_root / "SpiderNet" / suffix
                    required.append(saved_sample / "ProcessedData" / "LR_list.pkl")
                    required += [saved_sample / "SpiderNet_Result_Mode_cell_class" / name for name in (
                        "loading_LR_use.npy", "loading_sender_use.csv", "loading_receiver_use.csv",
                        "Factor_envir_use.npy", "EdgeProgramScores.csv")]
                    required.append(args.result_root / "ScCChain_analysis" / suffix / f"{setting}_Experiment_{exp}_metrics.csv")
                    spacia_root = Path(os.environ.get("SIMULATION_SPACIA_ROOT", str(args.result_root / "Spacia_analysis")))
                    required.append(spacia_root / suffix / f"{setting}_Experiment_{exp}_metrics.csv")
    if set(stages) & {"sample", "insitu"}:
        required += [sample / name for name in DATA_FILES]
    if "sample" in stages:
        required += [model / name for name in ("loading_LR_use.npy", "loading_sender_use.npy",
                                               "loading_receiver_use.npy", "loading_intrinsic_use.csv")]
    if "insitu" in stages:
        required += [sample / "adata_simulation.h5ad", model / "EdgeProgramScores.csv", HERE / "ScCChain_runner.jl"]
        if args.plot_only:
            required += [args.output_root / "SpiderNet" / SETTINGS[s] / f"Experiment_{e}" /
                         "SpiderNet_Result_Mode_cell_class" / f"insitu_scores.{ext}" for ext in ("npz", "json")]
        else:
            julia = shutil.which("julia")
            if julia is None:
                errors.append("Julia executable is missing from PATH")
            else:
                try:
                    checked = subprocess.run([julia, "--startup-file=no", "-e", "using ScCChain, CSV, DataFrames, Statistics, Printf, Dates"],
                                             capture_output=True, text=True, timeout=55)
                    if checked.returncode:
                        errors.append("Julia packages: " + checked.stderr[-2000:])
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"Julia package check: {exc}")
    if "train" in stages:
        required += [args.data_root / SETTINGS[si] / f"Experiment_{ei}" / name
                     for si, ei in grid(args) for name in DATA_FILES + ["cell_neigh_metaIprop.csv"]]
        try:
            tested = subprocess.run([sys.executable, "-c", "from SpiderNet.api import build_model, run_training, infer_meta_interactions, normalize_outputs, export_results; from SpiderNet.config import TrainingConfig; from SpiderNet.io import load_processed_data"],
                                    capture_output=True, text=True, timeout=55)
            if tested.returncode:
                errors.append("SpiderNet package API: " + tested.stderr[-2000:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    for path in required:
        if not path.is_file():
            errors.append(f"Missing file: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"Empty file: {path}")
    if not errors and "insitu" in stages:
        try:
            import simulation_benchmark_utils as sim
            payload = sim.load_simulation_inputs(s, e)
            scores = sim.read_edge_scores_csv(model / "EdgeProgramScores.csv")
            aligned = sim.align_edge_scores_to_reference(scores, payload["edge_index"], "SpiderNet")
            if aligned.attrs["alignment_strategy"] != "direct_pair_match":
                errors.append("SpiderNet scores do not directly match the current simulation graph")
            if args.plot_only:
                cache = args.output_root / "SpiderNet" / SETTINGS[s] / f"Experiment_{e}" / "SpiderNet_Result_Mode_cell_class" / "insitu_scores.npz"
                sim.load_insitu_cache(cache, payload, dict(setting=s, experiment=e, run_nmf_lr=True,
                    run_commot=True, run_sccchain=True, load_compatible_spacia=False, methods=sim.METHOD_ORDER.copy()))
        except Exception as exc:
            errors.append(str(exc))
    print(f"Environment: {sys.executable}")
    print(f"Mode: {'plot-only' if args.plot_only else 'analysis'}; stages: {', '.join(stages)}")
    print(f"Checked {len(required)} required files; {len(errors)} error(s).")
    for error in errors:
        print(error)
    if "insitu" in stages:
        print("In-situ policy: Spacia remains disabled; full mode recomputes NMF-LR/COMMOT/ScCChain.")
    return errors


def execute_notebook(stage, args):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    source = HERE / NOTEBOOKS[stage]
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    s, e = representative(args)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        f"PLOT_ONLY = {args.plot_only!r}\nSETTING_INDEX = {s}\nEXPERIMENT_INDEX = {e}\n"))
    output = args.output_root / "executed"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = output / f"{source.stem}_{'plot_only' if args.plot_only else 'analysis'}_{stamp}.ipynb"
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=None,
                            resources={"metadata": {"path": str(HERE)}})
    print(f"Executing {source.name}; copy: {destination}", flush=True)
    try:
        client.execute(cwd=str(HERE), env=os.environ.copy())
    finally:
        nbformat.write(notebook, destination)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


def run(args):
    configure(args)
    errors = check_inputs(args)
    if errors:
        return 1
    if args.check:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    for stage in selected_stages(args):
        if stage in NOTEBOOKS:
            execute_notebook(stage, args)
        elif stage == "sample":
            from simulation_sample_plots import reproduce_sample_figures
            reproduce_sample_figures(*representative(args))
        elif stage == "train":
            print("Training selected replicates; original API/checkpoint behavior is retained.", flush=True)
            for s, e in grid(args):
                subprocess.run([sys.executable, str(HERE / "SpiderNet_Simulation.py"),
                                "--setting", str(s), "--experiment", str(e)], check=True, cwd=HERE)
                relative = Path("SpiderNet") / SETTINGS[s] / f"Experiment_{e}" / "SpiderNet_Result_Mode_cell_class"
                destination = args.output_root / relative
                destination.mkdir(parents=True, exist_ok=True)
                for path in (args.result_root / relative).iterdir():
                    if path.is_file() and path.suffix.lower() in (".csv", ".pdf", ".png", ".json"):
                        shutil.copy2(path, destination / path.name)
        elif stage == "generate":
            from SimulationData_generation import SimulationConfig, build_settings_table, run_single_experiment
            cfg = SimulationConfig(data_root=str(args.data_root))
            settings = build_settings_table(cfg)
            print("Regenerating selected data with the original hash-based seed; existing files are overwritten.", flush=True)
            for s, e in grid(args):
                row = settings.iloc[s]
                run_single_experiment(cfg, str(row["Dominant_Setting"]),
                                      float(row["Dropout_ratio"]), float(row["sd_use"]), e)
    print(f"Complete. Results: {args.output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(arguments()))
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
