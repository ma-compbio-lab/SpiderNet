"""Check inputs, regenerate figures, or run component-ablation analyses."""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

STUDIES = {"hgsoc": "HGSOC", "aging": "AgingMousebrain"}


def check_inputs(stages, plot_only=False):
    from ablation_settings import (
        DATASET_CONFIG, OUTPUT_ROOT, WORKSPACE_ROOT, VERSION_ORDER, VERSION_SPECS,
        RETRAIN_ABLATIONS, RUN_TRAINING_IF_MISSING, infer_full_run_dir,
        get_training_hparams, cache_matches_current_hparams, coupling_output_dir,
    )
    errors = []
    print("Python:", sys.executable)
    print("Workspace:", WORKSPACE_ROOT)
    print("Output:", OUTPUT_ROOT)
    print("Mode:", "saved-result plotting" if plot_only else "analysis from upstream data/model results")
    for module_name in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            module = importlib.import_module(module_name)
            print(f"  {module_name}: {module.__version__}")
        except Exception as exc:
            errors.append(f"Cannot import {module_name}: {exc}")

    paths = []
    needs_training = []
    for stage in stages:
        if stage in STUDIES:
            dataset = STUDIES[stage]
            if plot_only:
                paths.append(coupling_output_dir(dataset) / "Benchmark1_long.csv")
                continue
            cfg = DATASET_CONFIG[dataset]
            run_dir = infer_full_run_dir(cfg)
            paths.extend(cfg["processed_data_dir"] / name for name in (
                "adata_list.pkl", "SpiderNet_data_pyg_list.pkl", "LR_list.pkl"))
            paths.extend((run_dir / "Factor_envir_list.pkl", run_dir / cfg["targets_json_name"],
                          cfg["omnipath_regulatory_csv"]))
            print(f"{dataset}: full-model factors from {run_dir}")
            for version in VERSION_ORDER[1:]:
                model_dir = run_dir / "Component_ablation" / version
                valid = ((model_dir / "Factor_envir_list.pkl").is_file()
                         and not RETRAIN_ABLATIONS
                         and cache_matches_current_hparams(model_dir, get_training_hparams(dataset),
                                                           spec=VERSION_SPECS[version]))
                if valid:
                    print(f"  {version}: reuse compatible cached factors")
                elif RUN_TRAINING_IF_MISSING:
                    print(f"  {version}: TRAIN (cache missing/incompatible or retraining configured)")
                    needs_training.append((dataset, version, model_dir))
                else:
                    errors.append(f"No valid factor cache and training disabled: {model_dir}")
        else:
            if errors:
                continue
            import perturbation_prediction as prediction
            if plot_only:
                paths.append(prediction.OUTPUT_DIR /
                             f"PerturbFISH_component_ablation_spearman_{prediction.FIGURE_VALUE_SOURCE}.csv")
            else:
                paths.extend([prediction.observed_path, prediction.canonical_matrix_path,
                              *prediction.prediction_paths.values()])

    for path in paths:
        if not path.is_file():
            errors.append(f"Missing input: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"Empty input: {path}")
    print(f"Checked {len(paths)} required input files.")

    if not plot_only and any(s in STUDIES for s in stages):
        try:
            import ablation_training as training
            print("Analysis device:", training.DEVICE)
            print("Initial_model module:", training.Initial_model.__module__)
            # The original reinit_ablation branch is deliberately preserved.
            import inspect
            if (training.MiniBatchNMF is not None and
                    "n_init" not in inspect.signature(training.MiniBatchNMF).parameters):
                print("NOTE: this sklearn cannot create a fresh No intrinsic NMF initializer with n_init=1. "
                      "Compatible factor/initializer caches are still reusable; no fallback is substituted.")
                for dataset, version, model_dir in needs_training:
                    if version == "no_intrinsic_reinit":
                        print(f"  Inspect the existing initialization cache before training: {model_dir.parent}")
        except Exception as exc:
            errors.append(f"Cannot import the full analysis environment: {type(exc).__name__}: {exc}")

    if not errors and plot_only:
        try:
            import pandas as pd
            for stage in stages:
                if stage in STUDIES:
                    path = coupling_output_dir(STUDIES[stage]) / "Benchmark1_long.csv"
                    table = pd.read_csv(path)
                    required = {"Version_ID", "ModelVersion", "RepresentativeLR", "FeatureType", "Correlation"}
                    if not required.issubset(table.columns) or table.empty:
                        errors.append(f"Invalid coupling table: {path}")
                    elif set(table["Version_ID"]) != set(VERSION_ORDER):
                        errors.append(f"Expected the original four model versions: {path}")
                else:
                    import perturbation_prediction as prediction
                    path = prediction.OUTPUT_DIR / f"PerturbFISH_component_ablation_spearman_{prediction.FIGURE_VALUE_SOURCE}.csv"
                    table = pd.read_csv(path, index_col=0).reindex(
                        index=prediction.PERTURBATION_ORDER, columns=prediction.METHOD_ORDER)
                    if table.isna().any().any():
                        errors.append(f"Incomplete perturbation figure matrix: {path}")
        except Exception as exc:
            errors.append(f"Cannot read saved plot inputs: {exc}")

    if errors:
        for error in errors:
            print("ERROR:", error, file=sys.stderr)
        print("Use the SpiderNet analysis environment; set SPIDERNET_WORKSPACE_DIR if inputs live elsewhere.",
              file=sys.stderr)
        return False
    print("Input/import checks passed. Large pickle contents and full numerical reproduction are not checked here.")
    return True


def run_stage(stage, plot_only):
    if stage in STUDIES:
        import regulatory_coupling as coupling
        (coupling.plot_only if plot_only else coupling.run)(STUDIES[stage])
    else:
        import perturbation_prediction as prediction
        (prediction.plot_only if plot_only else prediction.run)()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check inputs/imports without running analyses")
    parser.add_argument("--plot-only", action="store_true", help="read local output tables; do not train or recompute correlations")
    parser.add_argument("--stage", choices=["all", *STUDIES, "perturbfish"], default="all")
    args = parser.parse_args()
    stages = [*STUDIES, "perturbfish"] if args.stage == "all" else [args.stage]
    # Select the noninteractive backend before any pyplot import.
    import os
    os.environ["MPLBACKEND"] = "Agg"
    if not check_inputs(stages, args.plot_only):
        return 1
    if args.check:
        return 0
    for stage in stages:
        print(f"\nRunning {stage} ({'plot-only' if args.plot_only else 'full analysis'})", flush=True)
        # Fresh processes preserve each original notebook's plotting defaults and
        # release large dataset/model objects before loading the next study.
        subprocess.run([sys.executable, "-u", "-c",
                        f"from run_benchmarks import run_stage; run_stage({stage!r}, {args.plot_only!r})"],
                       cwd=Path(__file__).resolve().parent, check=True)
    print("Completed selected stages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
