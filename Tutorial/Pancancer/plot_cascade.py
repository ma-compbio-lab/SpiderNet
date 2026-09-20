"""Redraw cascade summaries from saved tables using the notebook's plot code."""

import ast
import json
from pathlib import Path
import warnings


NOTEBOOK = Path(__file__).with_name("Pancancer_MIcascade_analysis_V3.ipynb")
METHOD = "zscore_mean_global_per_cancertype"
FEATURE_SUBDIR = Path("UpstreamGated_MI7_to_MI4_cancerFibroCancer") / METHOD
PREFIX = f"UpstreamGated_MI7_MI4_{METHOD}"
SHORT_PREFIX = f"UpGated_MI7_MI4_{METHOD}"
COUNT_FILE = "ObservedAndPotentialCount_summary_cancercell-Fibroblast-cancercell_MI-7_MI-4.csv"


def _sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {tag: "".join(cell["source"]) for cell in notebook["cells"]
            for tag in cell.get("metadata", {}).get("tags", []) if tag.startswith("cascade-")}


def _read_csv(path, **kwargs):
    import os
    import pandas as pd
    path = Path(path).resolve()
    name = "\\\\?\\" + str(path) if os.name == "nt" else str(path)
    return pd.read_csv(name, **kwargs)


def _is_file(path):
    import os
    name = str(Path(path).resolve())
    return os.path.isfile("\\\\?\\" + name if os.name == "nt" else name)


def _spatial_inputs(results_dir):
    """Require a completed, internally consistent drawing-input inventory."""
    import os
    directory = Path(results_dir) / "MIcascade_spatial_plot_inputs"
    manifest_path = directory / "manifest.json"
    if not _is_file(manifest_path):
        raise FileNotFoundError(f"Spatial completion manifest is missing: {manifest_path}. Run --prepare-plot-inputs --stage cascade, or full cascade analysis; isolated NPZ files are not a complete export.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Spatial manifest is not an object: {manifest_path}")
    if manifest.get("version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"Spatial drawing export is incomplete or unsupported: {manifest_path}")
    if manifest.get("mi_pair") != [7, 4] or manifest.get("MI_threshold_show") != 0.3:
        raise ValueError(f"Spatial manifest does not match the selected MI pair/display threshold: {manifest_path}")
    records = manifest.get("inputs")
    if not isinstance(records, list) or manifest.get("input_count") != len(records):
        raise ValueError(f"Spatial manifest has an inconsistent input inventory: {manifest_path}")
    paths = []
    for record in records:
        name = record.get("file") if isinstance(record, dict) else None
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".npz":
            raise ValueError(f"Invalid spatial drawing-input filename in {manifest_path}")
        path = directory / name
        if not _is_file(path):
            raise FileNotFoundError(f"Spatial drawing input listed in manifest is missing: {path}")
        native_path = "\\\\?\\" + str(path.resolve()) if os.name == "nt" else path
        if os.stat(native_path).st_size != record.get("size_bytes"):
            raise ValueError(f"Spatial drawing input size differs from the completed export: {path}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError(f"Spatial manifest contains duplicate drawing inputs: {manifest_path}")
    return paths


def check(results_dir: Path) -> list[str]:
    root = Path(results_dir)
    paths = [
        root / "MI_colocal_summary_by_cancertype.csv",
        root / "MIcascade_pvalue_method.json",
        root / "Celltype_triple_prop_within_cancertype_cancercell_pivot_unionMIs.csv",
        root / "Celltype_triple_prop_within_cancertype_long.csv",
        root / "Celltype_triple_prop_selected_MI-7_MI-4_cancercell.csv",
        root / "Insitu_High_order_MI_AllCancerTypes" / COUNT_FILE,
        root / FEATURE_SUBDIR / f"{PREFIX}_sample_level_scores.csv",
        root / FEATURE_SUBDIR / f"{PREFIX}_sample_level_stats.csv",
        root / FEATURE_SUBDIR / "Cleveland_dotplots" / f"{SHORT_PREFIX}_single_unit_group_means_SMD_by_feature_cancertype.csv",
        root / FEATURE_SUBDIR / "ReferenceGO_SelectedTerms" / "selected_reference_go_sample_level_stats_allGOsets_by_allCancerTypes.csv",
    ]
    for level in ("single_unit", "sample_level"):
        paths.extend([
            root / FEATURE_SUBDIR / "SMD_heatmaps" / f"{SHORT_PREFIX}_{level}_SMD.csv",
            root / FEATURE_SUBDIR / "Cleveland_dotplots" / f"{SHORT_PREFIX}_{level}_group_means_SMD_by_feature_cancertype.csv",
            root / FEATURE_SUBDIR / "ReferenceGO_SelectedTerms" / f"selected_reference_go_{level}_stats_allGOsets_by_allCancerTypes.csv",
        ])
    missing = [f"Cascade table: {p}" for p in dict.fromkeys(paths) if not _is_file(p)]
    try:
        _spatial_inputs(root)
    except (FileNotFoundError, ValueError) as error:
        missing.append(f"Cascade spatial drawing inputs: {error}")
    return missing


def run(results_dir: Path, output_dir: Path) -> list[Path]:
    """Regenerate available panels; record missing panel inputs explicitly.

    Plotting functions and settings are executed directly from tagged notebook
    cells. Saved SMD/mean tables avoid repeating inference, permutations, and
    multi-gigabyte single-unit aggregation. Single-unit boxplots remain a full
    analysis output; the compact-table path redraws the sample-level boxplots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import re

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) / "cascade"
    output_dir.mkdir(parents=True, exist_ok=True)
    import time
    started_ns = time.time_ns()
    sources = _sources()
    ns = {"np": np, "pd": pd, "Path": Path, "os": os, "re": re,
          "display": lambda *args, **kwargs: None, "file_savepath_main": str(output_dir)}
    problems = []
    notes = []
    completed = []

    def execute(tag, start=None, stop=None):
        source = sources[tag]
        if start is not None:
            source = source[source.index(start):]
        if stop is not None:
            source = source[:source.index(stop)]
        exec(compile(source, str(NOTEBOOK) + ":" + tag, "exec"), ns)

    def attempt(label, function):
        try:
            function()
            completed.append(label)
        except (FileNotFoundError, KeyError, ValueError) as error:
            problems.append(f"{label}: {error}")
            warnings.warn(problems[-1], stacklevel=2)
        finally:
            plt.close("all")

    execute("cascade-imports")
    execute("cascade-thresholds")
    pvalue_metadata = results_dir / "MIcascade_pvalue_method.json"
    if not pvalue_metadata.is_file():
        raise ValueError("Cascade summaries need +1-corrected P-values. Run the cascade analysis to regenerate the summary tables and MIcascade_pvalue_method.json from the saved permutation counts.")
    pvalue_method = json.loads(pvalue_metadata.read_text(encoding="utf-8"))
    if pvalue_method.get("pvalue_method") != "plus1":
        raise ValueError("Historical floored cascade P-values cannot be plotted with the new cutoff; regenerate the cascade summary tables using the +1 correction.")
    summary = _read_csv(results_dir / "MI_colocal_summary_by_cancertype.csv")
    ns["cancertype_list"] = summary["CancerType"].drop_duplicates().tolist()
    ns["sampleindex_to_cancertype"] = {}
    ns["colocal_count_zscore_by_cancertype"] = {}
    ns["colocal_count_pvalue_by_cancertype"] = {}
    for cancer, data in summary.groupby("CancerType", sort=False):
        for column, name in (("zscore_countcolocal", "colocal_count_zscore_by_cancertype"),
                             ("pvalue_adjusted", "colocal_count_pvalue_by_cancertype")):
            matrix = data.pivot(index="MI_first", columns="MI_second", values=column).sort_index().sort_index(axis=1)
            matrix.index = [f"MI-{int(v)}" for v in matrix.index]
            matrix.columns = [f"MI-{int(v)}" for v in matrix.columns]
            matrix.index.name = matrix.columns.name = None
            ns[name][cancer] = matrix
    attempt("enrichment and recurrence heatmaps", lambda: execute("cascade-enrichment-plots"))

    def triplets():
        ns["prop_avg_by_cancertype_filter_merge_pivot_select"] = _read_csv(
            results_dir / "Celltype_triple_prop_within_cancertype_cancercell_pivot_unionMIs.csv", index_col=0)
        for tag in ("cascade-triplet-order", "cascade-triplet-plot-function", "cascade-triplet-heatmap"):
            execute(tag)
        selected = _read_csv(results_dir / "Celltype_triple_prop_selected_MI-7_MI-4_cancercell.csv")
        # The saved table precedes the original final two sorts; retain row order.
        ns["prop_MI11_MI2_by_cancertype_cancercell"] = selected
        execute("cascade-selected-triplet-order")
        execute("cascade-selected-triplet-stem")
    attempt("triplet composition", triplets)

    execute("cascade-spatial", stop="cascade_count_summary = []")

    def abundance():
        ns["cascade_count_summary_df"] = _read_csv(results_dir / "Insitu_High_order_MI_AllCancerTypes" / COUNT_FILE)
        execute("cascade-abundance")
    attempt("cascade abundance", abundance)

    def spatial():
        spatial_inputs = _spatial_inputs(results_dir)
        if not spatial_inputs:
            notes.append("The completed spatial export contains no positive cascade instances to draw.")
        for index, path in enumerate(spatial_inputs, 1):
            with np.load(path, allow_pickle=False) as data:
                settings = json.loads(str(data["settings"]))
                for key, value in settings["plot_settings"].items():
                    ns[key] = value
                ns["plot_high_order_MI"](
                    spatial=data["spatial"], celltypes=data["celltypes"],
                    edge_first=data["edge_first"], edge_second=data["edge_second"],
                    classes=settings["classes"], color_map=settings["color_map"],
                    save_dir=str(output_dir), **settings["arguments"])
            if index % max(1, len(spatial_inputs) // 10) == 0 or index == len(spatial_inputs):
                print(f"  Cascade spatial export: {index}/{len(spatial_inputs)} sub-slices", flush=True)
    attempt("spatial panels", spatial)

    execute("cascade-feature-settings")
    if ns["MODULE_SCORE_METHOD"] != METHOD or (ns["MI_first_of_interest"], ns["MI_second_of_interest"]) != (7, 4):
        raise ValueError("The notebook's selected MI pair or scoring method changed. Update the saved-table path mapping before plot-only execution.")
    feature_source = results_dir / FEATURE_SUBDIR
    execute("cascade-feature-boxplots", stop="# -----------------------------\n# Single-cell / edge-unit level statistics and plot")

    def sample_boxes():
        ns["sample_level_feature_df"] = _read_csv(feature_source / f"{PREFIX}_sample_level_scores.csv")
        ns["sample_level_plot_df"] = ns["sample_level_feature_df"].rename(columns={"sample_mean_score": "value"}).copy()
        ns["sample_level_stats_df"] = _read_csv(feature_source / f"{PREFIX}_sample_level_stats.csv")
        execute("cascade-feature-boxplots", start="plot_feature_by_cancertype_boxplots(\n    sample_level_plot_df,")
    attempt("sample-level feature boxplots", sample_boxes)

    def feature_smd():
        execute("cascade-feature-smd", stop='if "feature_records_upstream_gated" not in globals():')
        for level in ("single_unit", "sample_level"):
            ns[f"feature_{level}_smd_df"] = _read_csv(feature_source / "SMD_heatmaps" / f"{SHORT_PREFIX}_{level}_SMD.csv")
        execute("cascade-feature-smd", start="for _level, _smd_df in [")
    attempt("feature SMD heatmaps", feature_smd)

    def cleveland():
        execute("cascade-feature-cleveland", stop='if "feature_records_upstream_gated" not in globals():')
        execute("cascade-feature-cleveland", start="output_dir = Path(output_dir)", stop="feature_records_for_cleveland =")
        definitions = [n for n in ast.parse(sources["cascade-feature-cleveland"]).body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
        for level in ("single_unit", "sample_level"):
            ns[f"feature_{level}_cleveland_mean_df"] = _read_csv(feature_source / "Cleveland_dotplots" / f"{SHORT_PREFIX}_{level}_group_means_SMD_by_feature_cancertype.csv")
        execute("cascade-feature-cleveland", start='if "single_unit" in FEATURE_CLEVELAND_LEVELS_TO_PLOT:')
    attempt("feature mean comparisons", cleveland)

    def reference_smd():
        execute("cascade-reference-settings", stop="selected_reference_go_specs_df =")
        # These unchanged fallback functions are shared by both reference plot cells.
        definitions = [n for n in ast.parse(sources["cascade-reference-boxplots"]).body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
        ref = feature_source / "ReferenceGO_SelectedTerms"
        for level in ("single_unit", "sample_level"):
            ns[f"reference_{level}_stats_df"] = _read_csv(ref / f"selected_reference_go_{level}_stats_allGOsets_by_allCancerTypes.csv")
        execute("cascade-reference-smd")
    attempt("reference gene-set SMD heatmaps", reference_smd)
    notes.append("Additional single-unit feature/GO boxplots and GO sample-level grids remain available in full analysis. The compact-table plot-only path omits these exploratory grids; this is not a missing required paper panel.")
    for message in problems:
        warnings.warn(message, stacklevel=2)
    outputs = sorted(p for p in output_dir.rglob("*")
                     if p.suffix.lower() in {".png", ".pdf", ".svg"}
                     and os.stat("\\\\?\\" + str(p.resolve()) if os.name == "nt" else p).st_mtime_ns >= started_ns)
    (output_dir / "plot_status.json").write_text(json.dumps({
        "source": str(results_dir), "completed": completed, "limitations": problems,
        "notes": notes,
        "regenerated_figures": [str(p) for p in outputs],
    }, indent=2) + "\n", encoding="utf-8")
    return outputs
