"""Run the Pancancer analyses with their existing scientific parameters."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import warnings

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {
    "preprocess": "spidernet_dataloading_MIdimselection_Pancancer.ipynb",
    "train": "Pancancer_modeltraining.ipynb",
    "spatial": "Pancancer_analysis_V2.ipynb",
    "cascade": "Pancancer_MIcascade_analysis_V3.ipynb",
    "lrko": "Pancancer_MI4_Fibroblast_to_Tumor_LRKO.ipynb",
}
R_SCRIPTS = {
    "projection": "TCGA_MI_decomposition_V2_updated_tumor_only.R",
    "survival": "Survival_analysis_KM_tertile_groups_smaller_censor_tumor_only.R",
    "icb": "ICB_analysis_V2.R",
}
DEFAULT_STAGES = ("spatial", "cascade", "lrko", "projection", "survival", "icb")
DEPENDENCIES = {"spatial": ("train",), "cascade": ("spatial",), "lrko": ("spatial",),
                "projection": ("spatial",), "survival": ("projection",), "icb": ("spatial",)}
FINAL_SUFFIXES = {".pdf", ".png", ".svg", ".csv", ".tsv"}


def native_path(path: Path) -> Path:
    """Support existing long analysis filenames on Windows."""
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def project_root() -> Path:
    if os.environ.get("SPIDERNET_ROOT"):
        return Path(os.environ["SPIDERNET_ROOT"]).expanduser().resolve()
    return next((p for p in HERE.parents if (p / "Data").is_dir() and (p / "Results").is_dir()), Path("D:/SpiderNet"))


def parser() -> argparse.ArgumentParser:
    root = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Check inputs without fitting, downloading or loading large tensors.")
    p.add_argument("--plot-only", action="store_true", help="Regenerate plots from saved analysis tables; never fit models.")
    p.add_argument("--prepare-plot-inputs", action="store_true", help="Export missing spatial/cascade/KM caches from saved analyses; verify legacy Cox results without GDC downloads.")
    p.add_argument("--stage", nargs="+", choices=[*NOTEBOOKS, *R_SCRIPTS], help="Selected stages, in the order supplied; default: all downstream stages.")
    p.add_argument("--results-dir", type=Path, default=Path(os.environ.get("SPIDERNET_PANCANCER_RESULTS", root / "Results/Pancancer/V1/SpiderNet_Result_dim11")))
    p.add_argument("--processed-dir", type=Path, default=Path(os.environ.get("SPIDERNET_PANCANCER_PROCESSED", root / "Results/Pancancer/ProcessedData_entire")))
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("SPIDERNET_PANCANCER_DATA", root / "Data/Pancancer")))
    p.add_argument("--output-dir", type=Path, default=Path(os.environ.get("SPIDERNET_PANCANCER_OUTPUT", HERE / "output")))
    p.add_argument("--rscript", default=os.environ.get("RSCRIPT"), help="Rscript executable; otherwise use PATH, R_HOME, or a single standard Windows installation.")
    p.add_argument("--r-stage", help="Optional substage passed to a single R stage.")
    p.add_argument("--cache-dir", type=Path, help="Explicit saved-cache directory for a single R plot-only stage.")
    p.add_argument("--mi", help="Explicit MI selection for the R survival plot stage; original defaults otherwise.")
    return p


def environment(args) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in {
        "SPIDERNET_ROOT": project_root(),
        "SPIDERNET_PANCANCER_RESULTS": args.results_dir,
        "SPIDERNET_PANCANCER_PROCESSED": args.processed_dir,
        "SPIDERNET_PANCANCER_DATA": args.data_dir,
        "SPIDERNET_PANCANCER_OUTPUT": args.output_dir,
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
    }.items():
        env[key] = str(value)
    return env


def resolve_rscript(requested: str | None = None) -> str:
    """Honor explicit selections; auto-detect only an unambiguous installation."""
    if requested:
        exe = shutil.which(requested)
        if exe is None and Path(requested).expanduser().is_file():
            exe = str(Path(requested).expanduser().resolve())
        if exe is None:
            raise FileNotFoundError(f"Requested Rscript does not exist: {requested}. Update --rscript or RSCRIPT.")
        return exe
    exe = shutil.which("Rscript")
    if exe:
        return exe
    if os.environ.get("R_HOME"):
        home = Path(os.environ["R_HOME"])
        for relative in ("bin/Rscript.exe", "bin/x64/Rscript.exe", "bin/Rscript"):
            candidate = home / relative
            if candidate.is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(f"R_HOME does not contain Rscript: {home}. Set --rscript PATH explicitly.")
    candidates = set()
    if os.name == "nt":
        roots = {Path(value) / "R" for key in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)")
                 if (value := os.environ.get(key))}
        if os.environ.get("LOCALAPPDATA"):
            roots.add(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "R")
        for root in roots:
            for candidate in root.glob("R-*/bin/Rscript.exe"):
                if candidate.is_file():
                    candidates.add(str(candidate.resolve()))
    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        raise FileNotFoundError("Multiple R installations found. Select one with --rscript PATH: " + "; ".join(sorted(candidates)))
    raise FileNotFoundError("Rscript not found. Use --rscript PATH or set RSCRIPT.")


def r_command(stage: str, args) -> list[str]:
    exe = resolve_rscript(args.rscript)
    cmd = [exe, str(HERE / R_SCRIPTS[stage])]
    if args.check:
        cmd.append("--check")
    if args.plot_only:
        cmd.append("--plot-only")
    if args.prepare_plot_inputs:
        cmd.append("--prepare-plot-inputs")
    if args.r_stage:
        cmd += ["--stage", args.r_stage]
    if args.cache_dir:
        cmd += ["--cache-dir", str(args.cache_dir.resolve())]
    if args.mi:
        cmd += ["--mi", args.mi]
    return cmd


def render_saved_plots(module, args, record):
    """Keep every warning in the run record, without repeating it per panel."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            return module.run(args.results_dir, args.output_dir)
        finally:
            grouped = {}
            for warning in captured:
                key = (warning.category.__name__, str(warning.message).strip())
                grouped[key] = grouped.get(key, 0) + 1
            record["warnings"] = [{"category": category, "message": message, "count": count}
                                  for (category, message), count in grouped.items()]
            for item in record["warnings"]:
                message = " ".join(item["message"].split())
                print(f"  {item['category']} ({item['count']} occurrence(s)): {message}", flush=True)


def python_check(stage: str, args) -> list[str]:
    problems = []
    packages = ["numpy", "pandas", "scipy", "matplotlib"]
    if args.plot_only and stage in ("preprocess", "train"):
        return ["This upstream stage fits/preprocesses data and has no plot-only mode. Select downstream stages."]
    if args.plot_only:
        packages += ["seaborn"]
        for package in packages:
            if importlib.util.find_spec(package) is None:
                problems.append(f"Python package missing: {package}")
        if problems:
            return problems
        module = importlib.import_module(f"plot_{stage}")
        return list(module.check(args.results_dir))
    packages += ["nbclient", "nbformat", "ipykernel", "jupyter_client", "SpiderNet", "torch", "scanpy", "seaborn"]
    if stage in ("spatial", "cascade", "lrko"):
        packages += ["gseapy"]
    for package in packages:
        if importlib.util.find_spec(package) is None:
            problems.append(f"Python package missing: {package}")
    required = []
    if stage in ("preprocess", "train"):
        if not any((args.data_dir / "adata_entire").glob("*.h5ad")):
            problems.append(f"No annotated .h5ad input: {args.data_dir / 'adata_entire'}")
    else:
        required += [args.processed_dir / f for f in ("adata_all.h5ad", "adata_list.pkl", "LR_list.pkl", "genenames_train.pkl", "batch_cell_unique.pkl")]
        if not any(args.processed_dir.glob("SpiderNet_data_pyg_list*")):
            problems.append(f"Processed graph bundle missing under {args.processed_dir}")
        if stage == "lrko":
            required += [args.results_dir / "Factor_envir_use.npy"]
            for loading in ("loading_LR_use", "loading_sender_use", "loading_receiver_use"):
                if not any((args.results_dir / (loading + suffix)).is_file() for suffix in (".npy", ".csv")):
                    problems.append(f"Missing loading table/array: {args.results_dir / loading}")
        if stage == "cascade":
            required += [args.results_dir / "Factor_envir_list.pkl", args.results_dir / "LR_meta_incellchatdb.csv"]
            if not any(args.results_dir.rglob("*_GO_KEGG_term_reference_genes_long.csv")):
                problems.append(f"Reference GO/KEGG gene export missing under {args.results_dir}; run spatial first.")
        if stage in ("spatial", "lrko") and not any((args.results_dir / "Model").glob("model_epoch*")):
            problems.append(f"No model_epoch* checkpoint: {args.results_dir / 'Model'}")
        required += [args.data_dir / "CancerSEA_marker"]
    for path in required:
        if not path.exists():
            problems.append(f"Missing input: {path}")
    if stage == "spatial":
        print("  CCC: cached baselines are reused by the notebook's existing policy; fresh COMMOT/scCChain requires COMMOT and Julia + ScCChain.")
    return problems


def execute_notebook(source: Path, output: Path, env: dict[str, str]) -> None:
    """Execute with this Python interpreter, independently of global kernelspecs."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpec

    book = nbformat.read(source, as_version=4)
    for cell in book.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = AsyncKernelManager(kernel_name="python3")
    manager._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="Pancancer current Python", language="python",
    )
    client = NotebookClient(book, km=manager, timeout=None, allow_errors=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute(cwd=str(HERE), env=env, cleanup_kc=True)
    finally:
        nbformat.write(book, output)


def final_snapshot(roots: list[Path]) -> dict[str, tuple[int, int]]:
    """Record metadata only, without reading large scientific intermediates."""
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        scan_root = native_path(root)
        for path in scan_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in FINAL_SUFFIXES:
                st = path.stat()
                found[str(root / path.relative_to(scan_root))] = (st.st_size, st.st_mtime_ns)
    return found


def collect_new_products(roots: list[Path], before: dict, output: Path, stage: str) -> list[dict]:
    """Copy newly written figures/tables; leave large analysis matrices in Results."""
    copied = []
    for name, stamp in final_snapshot(roots).items():
        if before.get(name) == stamp:
            continue
        path = Path(name)
        if path.suffix.lower() in {".csv", ".tsv"} and stamp[0] > 64 * 1024 * 1024:
            copied.append({"source": name, "status": "large intermediate retained in Results"})
            continue
        root = next(p for p in roots if path.is_relative_to(p))
        dest = output / stage / path.relative_to(root)
        native_path(dest.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(native_path(path), native_path(dest))
        copied.append({"source": name, "output": str(dest)})
    return copied


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    stages = args.stage or list(DEFAULT_STAGES)
    if args.prepare_plot_inputs:
        stages = args.stage or ["spatial", "cascade", "survival"]
        if args.plot_only or args.r_stage or args.cache_dir or args.mi:
            p.error("--prepare-plot-inputs is a separate mode; do not combine it with plot-only or R plotting options.")
        if any(stage not in ("spatial", "cascade", "survival") for stage in stages):
            p.error("Cache preparation supports spatial, cascade and survival only.")
    if len(set(stages)) != len(stages):
        p.error("Do not repeat stages.")
    if not args.check and not args.plot_only and not args.prepare_plot_inputs:
        for stage in stages:
            for upstream in DEPENDENCIES.get(stage, ()):
                if upstream in stages and stages.index(upstream) > stages.index(stage):
                    p.error(f"Run {upstream} before {stage}, or select {stage} alone to use saved upstream results.")
    if (args.r_stage or args.cache_dir or args.mi) and (len(stages) != 1 or stages[0] not in R_SCRIPTS):
        p.error("--r-stage, --cache-dir and --mi require one R --stage.")
    if args.mi and stages != ["survival"]:
        p.error("--mi selects KM exports and is only valid with --stage survival.")
    if args.cache_dir and not args.plot_only:
        p.error("--cache-dir requires --plot-only (optionally with --check).")
    for attr in ("results_dir", "processed_dir", "data_dir", "output_dir"):
        setattr(args, attr, getattr(args, attr).expanduser().resolve())
    env = environment(args)
    # Plot helpers run in this process and use the same explicit path contract.
    os.environ.update({k: v for k, v in env.items() if k.startswith("SPIDERNET_") or k == "MPLBACKEND"})
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "mode": "check" if args.check else "prepare-plot-inputs" if args.prepare_plot_inputs else "plot-only" if args.plot_only else "analysis",
              "results_dir": str(args.results_dir), "stages": []}
    failed = False
    for stage in stages:
        print(f"\n[{stage}] {report['mode']}", flush=True)
        record = {"stage": stage}
        report["stages"].append(record)
        try:
            if not args.check and not args.plot_only and not args.prepare_plot_inputs:
                blocked = [item["stage"] for item in report["stages"][:-1]
                           if item["stage"] in DEPENDENCIES.get(stage, ()) and item["status"] != "completed"]
                if blocked:
                    raise RuntimeError("Skipped because a selected upstream stage failed: " + ", ".join(blocked))
            if stage in R_SCRIPTS:
                command = r_command(stage, args)
                record["rscript"] = command[0]
                print(f"  Rscript: {command[0]}", flush=True)
                result = subprocess.run(command, cwd=HERE, env=env, check=False)
                if result.returncode:
                    failed = True
                    record.update(status="incomplete", error=f"R stage exited with status {result.returncode}; see diagnostics above.")
            elif args.prepare_plot_inputs:
                required = [args.processed_dir / name for name in
                            ("adata_all.h5ad", "batch_cell_unique.pkl", "SpiderNet_data_pyg_list.pkl")]
                if stage == "spatial":
                    required += [args.results_dir / "Factor_envir_use.npy", args.results_dir /
                                 "MI4_fibroblast_to_tumor_celllevel_CancerSEA/MI4_fibro_to_tumor_edgelevel_strength_by_cancertype_full.csv"]
                else:
                    required += [args.results_dir / name for name in (
                        "Factor_envir_list.pkl", "MI_colocal_summary_significant_by_cancertype.csv",
                        "MIcascade_colocalization_cache/MIcascade_colocalization_cache_MIthreshold0p5_nperm100.pkl")]
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError("Cache preparation inputs missing: " + "; ".join(missing))
                if args.check:
                    print("  Saved cache preparation inputs found; no data loaded or analysis run.")
                else:
                    import prepare_plot_inputs
                    prepare_plot_inputs.run([stage], args.results_dir, args.processed_dir)
            elif args.check:
                problems = python_check(stage, args)
                if problems:
                    for problem in problems:
                        print("  MISSING:", problem)
                    raise RuntimeError(f"{len(problems)} required input/dependency checks failed")
                print("  Input files and Python packages found. This is not an execution or numerical validation.")
            elif args.plot_only:
                module = importlib.import_module(f"plot_{stage}") if stage not in ("preprocess", "train") else None
                if module is None:
                    raise ValueError(f"{stage} has no plot-only mode; select downstream stages.")
                outputs = render_saved_plots(module, args, record)
                record["outputs"] = [str(x) for x in outputs]
                missing = list(module.check(args.results_dir))
                plot_status = args.output_dir / stage / "plot_status.json"
                if plot_status.is_file():
                    detail = json.loads(plot_status.read_text(encoding="utf-8"))
                    for key in ("missing", "problems", "missing_panels", "limitations"):
                        missing.extend(str(x) for x in detail.get(key, []))
                missing = list(dict.fromkeys(missing))
                if missing:
                    failed = True
                    record["missing_plot_inputs"] = missing
                    record.update(status="incomplete", error="Some figure groups lack drawing inputs; available plots were saved.")
                    print(f"  [{stage}] PARTIAL: {len(outputs)} output files written; missing inputs or figure groups:", flush=True)
                    for item in missing:
                        print("   - " + item, flush=True)
            else:
                problems = python_check(stage, args)
                if problems:
                    raise RuntimeError("; ".join(problems))
                roots = [args.results_dir] + ([args.processed_dir] if stage == "preprocess" else [])
                before = final_snapshot(roots)
                tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                executed = args.output_dir / "executed" / f"{Path(NOTEBOOKS[stage]).stem}_{tag}.ipynb"
                record["executed_notebook"] = str(executed)
                try:
                    execute_notebook(HERE / NOTEBOOKS[stage], executed, env)
                finally:
                    record["products"] = collect_new_products(roots, before, args.output_dir, stage)
            record.setdefault("status", "passed" if args.check else "completed")
        except Exception as exc:
            failed = True
            record.update(status="incomplete", error=str(exc))
            print(f"  [{stage}] INCOMPLETE: {exc}", file=sys.stderr, flush=True)
            record["traceback"] = traceback.format_exc()
            if not args.check and not isinstance(exc, FileNotFoundError):
                traceback.print_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    print("\nStage summary:", flush=True)
    for record in report["stages"]:
        print(f"  {record['stage']}: {record['status']}", flush=True)
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        status_path = args.output_dir / "run_status.json"
        status_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nRun record: {status_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
