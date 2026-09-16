"""Check inputs, render saved results, or execute the robustness analysis."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys


STAGES = {
    "lr": "S31a", "dimension": "S31b", "neighborhood": "S31c",
    "labels": "S32a", "seeds": "S32b", "subsampling": "S32c",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check selected inputs without running the analysis.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true", help="Render only from output/tables; never load training data or fit models.")
    mode.add_argument("--cached", action="store_true", help="Run analysis with existing fits; report missing fits without training.")
    parser.add_argument("--stage", nargs="+", choices=["all", *STAGES], default=["all"])
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("SPIDERNET_WORKSPACE", "D:/SpiderNet")))
    parser.add_argument("--results-root", type=Path, help="Shared processed data and reference/legacy models; default WORKSPACE/Results.")
    parser.add_argument("--package-root", type=Path, default=Path(os.environ["SPIDERNET_PACKAGE_ROOT"]) if os.environ.get("SPIDERNET_PACKAGE_ROOT") else None, help="Directory containing the SpiderNet Python package.")
    parser.add_argument("--work-dir", type=Path, help="Existing fitting/scratch cache; default RESULTS_ROOT/MI_robustness_S31_S32.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    return parser.parse_args()


def import_analysis(args):
    workspace = args.workspace.resolve()
    results = (args.results_root or workspace / "Results").resolve()
    package = (args.package_root or workspace / "SpiderNet_proj/SpiderNet_Project/SpiderNet").resolve()
    paths = {
        "SPIDERNET_WORKSPACE": workspace, "SPIDERNET_RESULTS_ROOT": results,
        "SPIDERNET_PACKAGE_ROOT": package,
        "SPIDERNET_ROBUSTNESS_WORK_DIR": (args.work_dir or results / "MI_robustness_S31_S32").resolve(),
        "SPIDERNET_ROBUSTNESS_OUTPUT_DIR": args.output_dir.resolve(),
    }
    os.environ.update({k: str(v) for k, v in paths.items()})
    os.environ["MPLBACKEND"] = "Agg"
    missing = [name for name in ["numpy", "pandas", "scipy", "matplotlib", "torch", "torch_geometric", "anndata"]
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python modules: {', '.join(missing)}. Activate SpiderNet_env. Interpreter: {sys.executable}")
    for name in ["api.py", "model.py", "MI_dimension_selection.py"]:
        if not (package / "SpiderNet" / name).is_file():
            raise FileNotFoundError(f"Missing package source: {package / 'SpiderNet' / name}")
    import robustness_analysis as analysis
    return analysis


def selected_panels(args):
    return set(STAGES.values()) if "all" in args.stage else {STAGES[s] for s in args.stage}


def pair_tags(panels):
    tags = []
    if "S31b" in panels:
        tags.extend(["S31b_HGSOC_M12_vs_M15", "S31b_HGSOC_M15_vs_M18",
                     "S31b_PerturbFISH_M20_vs_M23", "S31b_PerturbFISH_M23_vs_M26"])
    if "S31c" in panels:
        tags.extend(["S31c_HGSOC_K5_vs_K8", "S31c_HGSOC_K8_vs_K10"])
    return tags


def saved_requirements(a, panels):
    paths = []
    if "S31a" in panels:
        for dataset in a.DATASETS:
            paths.extend(a.TABLES / f"S31a_{dataset}" / name for name in
                         ["lr_spearman.npy", "merged_subsets.json", "selection_summary.json", "complete.json", "LR_display_order.csv"])
    for tag in pair_tags(panels):
        paths.extend(a.TABLES / tag / name for name in ["spearman_matrix.csv", "alignment.csv", "pearson_by_MI.csv", "comparison.json"])
    for panel, filename in [("S32a", "S32a_label_misannotation_Pearson_by_MI.csv"),
                            ("S32b", "S32b_Pearson_by_MI_all_repeats.csv"),
                            ("S32c", "S32c_Pearson_by_MI_all_repeats.csv")]:
        if panel in panels:
            paths.append(a.TABLES / filename)
    return paths


def analysis_requirements(a, panels):
    paths = []
    if "S31a" in panels:
        for cfg in a.DATASETS.values():
            d = cfg["processed_dir"]
            paths.extend(d / name for name in ["adata_all.h5ad", "LR_list.pkl", "bundle_summary.json"])
            if (d / "mi_dimension_selection_LR_spearcorr.npy").is_file() and (d / "mi_dimension_selection_merged_subsets.pkl").is_file():
                paths.append(d / "mi_dimension_selection_summary.json")
            else:
                paths.append(d / "batch_cell_unique.pkl")
    datasets = []
    if panels - {"S31a"}:
        datasets.append("HGSOC")
    if "S31b" in panels:
        datasets.append("PerturbFISH")
    for dataset in datasets:
        cfg = a.DATASETS[dataset]
        d = cfg["processed_dir"]
        graph = d / "SpiderNet_data_pyg_list.pkl"
        paths.append(graph if graph.is_file() else d / "SpiderNet_data_pyg_list.pt")
        paths.extend(d / name for name in ["genenames_train.pkl", "LR_list.pkl", "batch_cell_unique.pkl"])
        ref = cfg["reference_dir"]
        paths.append(ref / a.FACTOR_FILE)
        for stem in a.LOADING_FILES.values():
            paths.append(ref / (stem + (".npy" if (ref / f"{stem}.npy").is_file() else ".csv")))
    return list(dict.fromkeys(paths))


def check_inputs(a, panels, plot_only):
    required = saved_requirements(a, panels) if plot_only else analysis_requirements(a, panels)
    missing = [p for p in required if not p.is_file()]
    print(f"Interpreter: {sys.executable}\nPackage: {a.PACKAGE_ROOT}\nOutput: {a.LOCAL_OUTPUT_ROOT}")
    print(f"{'Plot' if plot_only else 'Analysis'} inputs: {len(required) - len(missing)}/{len(required)} present")
    for path in missing:
        print(f"MISSING: {path}")
    if not plot_only:
        plot_missing = [p for p in saved_requirements(a, panels) if not p.is_file()]
        print(f"Saved plotting inputs: {'ready' if not plot_missing else str(len(plot_missing)) + ' missing; run full analysis first'}")
        print(f"Fitting cache: {a.OUTPUT_ROOT / 'runs'}")
        print(f"Legacy stability fits: {a.LEGACY_STABILITY_ROOT}")
        print("Full mode validates cached signatures at runtime and trains missing fits (20,000 epochs).")
        print("--cached never trains. Input checks do not load large graphs or certify fit compatibility.")
    if missing:
        raise FileNotFoundError("Required inputs are missing; see the paths above.")


def load_saved(a, panels):
    if "S31a" in panels:
        for dataset in a.DATASETS:
            d = a.TABLES / f"S31a_{dataset}"
            corr = a.np.load(d / "lr_spearman.npy")
            subsets = a.read_json(d / "merged_subsets.json")
            order = a.mids.build_disjoint_subset_order(corr, subsets)
            saved_order = a.pd.read_csv(d / "LR_display_order.csv").lr_index.to_numpy() - 1
            if not a.np.array_equal(order["ordered_indices"], saved_order):
                raise ValueError(f"Saved LR display order disagrees with package ordering: {d}")
            a.LR_PANELS[dataset] = {"corr": corr, "order": order, "summary": a.read_json(d / "selection_summary.json")}
            a.record("S31a", dataset, "ready", "saved tables")
    for tag in pair_tags(panels):
        d = a.TABLES / tag
        a.PAIR_RESULTS[tag] = {
            "corr": a.pd.read_csv(d / "spearman_matrix.csv", index_col=0).to_numpy(),
            "pairs": [(int(r.row_mi) - 1, int(r.col_mi) - 1) for r in a.pd.read_csv(d / "alignment.csv").itertuples()],
            "metrics": a.pd.read_csv(d / "pearson_by_MI.csv"), "tag": tag,
        }
        a.record(tag.split("_")[0], tag, "ready", "saved tables")
    if "S32a" in panels:
        a.STABILITY_TABLES["S32a"] = a.pd.read_csv(a.TABLES / "S32a_label_misannotation_Pearson_by_MI.csv")
        a.record("S32a", "labels", "ready", "saved tables")
    for panel in ["S32b", "S32c"]:
        if panel in panels:
            a.STABILITY_TABLES[panel] = a.pd.read_csv(a.TABLES / f"{panel}_Pearson_by_MI_all_repeats.csv")
            a.record(panel, "repeats", "ready", "saved tables")


def analyze(a, panels):
    if "S31a" in panels:
        for dataset in a.DATASETS:
            result = a.prepare_lr_panel(dataset)
            if result is not None:
                a.LR_PANELS[dataset] = result
    if panels - {"S31a"}:
        context = a.load_context("HGSOC")
        if len(context.graphs) != 48:
            raise ValueError("Paper reproduction requires the 48-slice HGSOC dataset.")
        reference = a.reference_output(context)
        if "S31b" in panels:
            a.dimension_sweep(context, reference)
        if "S31c" in panels:
            a.k_sweep(context, reference)
        if "S32a" in panels:
            a.STABILITY_TABLES["S32a"] = a.label_experiment(context, reference)
        if "S32b" in panels:
            a.STABILITY_TABLES["S32b"] = a.stability_experiment(context, reference, "Random seed")
        if "S32c" in panels:
            a.STABILITY_TABLES["S32c"] = a.stability_experiment(context, reference, "70% slices")
        del reference, context
        a.gc.collect()
        if a.torch.cuda.is_available():
            a.torch.cuda.empty_cache()
    if "S31b" in panels:
        context = a.load_context("PerturbFISH")
        reference = a.reference_output(context)
        a.dimension_sweep(context, reference)
        del reference, context
        a.gc.collect()


def execute(args):
    a = import_analysis(args)
    panels = selected_panels(args)
    check_inputs(a, panels, args.plot_only)
    if args.check:
        return
    a.RUN_MODE = "cached" if args.cached or args.plot_only else "full"
    for path in [a.TABLES, a.FIGURES, a.CACHE / "numba", a.CACHE / "matplotlib"]:
        path.mkdir(parents=True, exist_ok=True)
    import robustness_plots as plots
    a.write_json(a.LOCAL_OUTPUT_ROOT / "manifests/current_execution.json", {
        "protocol": a.PROTOCOL_VERSION, "paper": "SpiderNet (39).pdf, A.10, Figures S31-S32",
        "mode": "plot-only" if args.plot_only else a.RUN_MODE, "panels": sorted(panels),
        "datasets": a.DATASETS, "max_epoch": a.MAX_EPOCH, "base_seed": a.BASE_SEED,
        "random_seeds": a.RANDOM_SEEDS, "subsample_seeds": a.SUBSAMPLE_SEEDS,
        "misannotation_seeds": a.MISANNOTATION_SEEDS, "package_code_hash": a.CODE_HASH,
        "python": sys.version, "torch": a.torch.__version__, "numpy": a.np.__version__,
        "scipy": a.scipy.__version__, "device": a.DEVICE,
        "work_dir": a.OUTPUT_ROOT, "output_dir": a.LOCAL_OUTPUT_ROOT,
    })
    try:
        load_saved(a, panels) if args.plot_only else analyze(a, panels)
        if panels & {"S31a", "S31b", "S31c"}:
            plots.plot_s31()
        if panels & {"S32a", "S32b", "S32c"}:
            plots.plot_s32()
        if any(r["status"] == "missing" for r in a.STATUS):
            raise RuntimeError("Requested results remain missing; use full mode to compute missing fits.")
    finally:
        a.pd.DataFrame(a.STATUS).to_csv(a.LOCAL_OUTPUT_ROOT / "panel_status.csv", index=False)
        plots.plt.close("all")
    print(f"Completed selected stages. Figures: {a.FIGURES}; tables: {a.TABLES}")


if __name__ == "__main__":
    try:
        execute(arguments())
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
