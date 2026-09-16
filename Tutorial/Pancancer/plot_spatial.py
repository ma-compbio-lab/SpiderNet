"""Regenerate spatial-analysis figures from recorded result tables.

Plotting code is read from tagged cells in Pancancer_analysis_V2.ipynb so the
notebook and command-line workflow share colors, ordering, and parameters.
This cache-only stage does not infer edges, fit models, or resample plot tables.
Spatial maps use compact drawing inputs exported by the notebook or cache preparation.
"""

from pathlib import Path
import ast
import json
import os
import shutil
import hashlib
import textwrap

NOTEBOOK = Path(__file__).with_name("Pancancer_analysis_V2.ipynb")
PAIR = "MIpair_top10percent_gt0p1"
PROGRAM = "MI4_fibroblast_to_tumor_celllevel_CancerSEA"
CCC = "Pancancer_Fibroblast_to_tumor_CCC_method_CancerSEA_SMD_comparison/Fig6d_in_analysis_V2"
LR = "LR_bulk_spatial_validation"
DRAWING = "In_situ_meta_interaction/drawing_inputs"


def _native_path(path):
    value = str(Path(path).resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def _output_snapshot(directory):
    value = str(Path(directory).resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    root = Path(value)
    found = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".csv"}:
            stamp = path.stat()
            found[Path(directory) / path.relative_to(root)] = (stamp.st_size, stamp.st_mtime_ns)
    return found


def _drawing_source(kind, sources=None):
    sources = _notebook_sources() if sources is None else sources
    if kind == "mi4":
        source = sources["plot-insitu-mi4"]
        source = source[source.index("        norm_edge = Normalize"):source.index("    finally:\n        plt.close", source.index("        norm_edge = Normalize"))]
        source = source.replace("        drawing_cache_frames.append(drawing_cache_path)\n", "")
        return textwrap.dedent(source)
    if kind == "mi2":
        source = sources["plot-insitu-mi2"]
        end = source.index("    write_drawing_manifest(drawing_cache_dir, drawing_cache_frames, len(selected), complete=True)")
        return textwrap.dedent(source[source.index("            norm = Normalize"):end])
    raise ValueError(f"Unknown drawing-cache kind: {kind}")


def save_drawing_inputs(directory, stem, kind, arrays, settings):
    """Save exact rendered coordinates and selected edges, without full graphs."""
    import numpy as np
    import matplotlib as mpl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(arrays["labels"]).astype(str)
    label_names, label_codes = np.unique(labels, return_inverse=True)
    arrays = {**{key: value for key, value in arrays.items() if key != "labels"}, "label_names": label_names,
              "label_codes": label_codes.astype(np.min_scalar_type(max(len(label_names)-1, 0)))}
    style = {}
    for key, value in mpl.rcParams.items():
        if key in {"backend", "backend_fallback", "interactive"}:
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        style[key] = value
    metadata = {"version": 1, "kind": kind, "settings": settings, "style": style,
                "renderer_sha256": hashlib.sha256(_drawing_source(kind).encode()).hexdigest()}
    path = directory / f"{stem}.npz"
    np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)), **arrays)
    return path


def write_drawing_manifest(directory, frames, expected_count, complete):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({
        "version": 1, "complete": bool(complete), "expected_count": int(expected_count),
        "frames": [Path(frame).name for frame in frames],
    }, indent=2) + "\n", encoding="utf-8")


def _drawing_manifests(results_dir):
    root = Path(results_dir) / DRAWING
    return [root / "MI4_Mode1/manifest.json", root / "MI2_representative/manifest.json"]


def _drawing_cache_issues(manifest_path):
    manifest_path = _native_path(manifest_path)
    if not manifest_path.is_file():
        return [f"Missing drawing-input manifest: {manifest_path}; run --prepare-plot-inputs --stage spatial, or the full spatial stage."]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return [f"Unreadable drawing-input manifest {manifest_path}: {error}"]
    if not manifest.get("complete") or len(manifest.get("frames", [])) != manifest.get("expected_count"):
        return [f"Incomplete drawing-input cache: {manifest_path}; rerun --prepare-plot-inputs --stage spatial, or the full spatial stage."]
    return [f"Missing drawing input: {manifest_path.parent / name}"
            for name in manifest["frames"] if not (manifest_path.parent / name).is_file()]


def replay_drawing_cache(manifest_path, output_dir):
    """Render an exact drawing cache using the notebook's original renderer."""
    from types import SimpleNamespace
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    manifest_path, output_dir = _native_path(manifest_path), _native_path(output_dir)
    issues = _drawing_cache_issues(manifest_path)
    if issues:
        raise FileNotFoundError("\n".join(issues))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = _notebook_sources()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    total = len(manifest["frames"])
    for index, filename in enumerate(manifest["frames"], 1):
        with np.load(manifest_path.parent / filename, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {key: archive[key] for key in archive.files if key != "metadata"}
        kind = metadata["kind"]
        source = _drawing_source(kind, sources)
        if hashlib.sha256(source.encode()).hexdigest() != metadata["renderer_sha256"]:
            raise ValueError(f"Drawing renderer changed since caching {filename}; regenerate with --stage spatial.")
        labels = arrays.pop("label_names")[arrays.pop("label_codes")]
        env = {"np": np, "mpl": mpl, "plt": plt, "Path": Path,
               "Normalize": Normalize, "FancyArrowPatch": FancyArrowPatch,
               "Line2D": Line2D, "display": lambda *a, **k: None,
               **metadata["settings"], **arrays}
        if kind == "mi4":
            _code(sources, "plot-insitu-mi4", env, end="# IPython retains")
            # The header declares the unchanged exporter and required imports.
            env.update(metadata["settings"])
            env["cellclass"] = labels
            env["passed_edge_idx"] = np.arange(len(env["factor_edge"]))
            env["cmap_edge"] = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"], N=256)
            env["save_path_insituMI"] = str(output_dir)
        else:
            env["labels"] = labels
            env["row"] = SimpleNamespace(**env.pop("row_values"))
            env["passed"] = range(len(env["strengths"]))
            env["cmap"] = LinearSegmentedColormap.from_list("mi2_white_to_red", ["white", "red"], N=256)
            env["preview_dir"] = output_dir
        with mpl.rc_context(metadata["style"]):
            exec(compile(source, f"<cached {kind} drawing>", "exec"), env)
        if kind == "mi4":
            output_paths.extend([Path(env["out_png"]), Path(env["out_pdf"])])
        else:
            output_paths.extend([output_dir / f"{env['preview_stem']}.png", output_dir / f"{env['preview_stem']}.pdf"])
        plt.close("all")
        if index % max(1, total // 10) == 0 or index == total:
            print(f"  {kind.upper()} spatial export: {index}/{total} sub-slices", flush=True)
    return output_paths


def required_inputs(results_dir):
    """Return the exact cached tables needed by this plotting stage."""
    names = [
        "LR_loading_pathway.csv",
        "MI_cancerSEA_meanrank_profile_sender.csv",
        "MI_cancerSEA_meanrank_profile_receiver.csv",
        f"{LR}/LR_coexpression_bulk_proxy_subslice_by_LR.csv",
        f"{LR}/LR_coexpression_spatial_edge_mean_subslice_by_LR.csv",
        f"{PAIR}/full_metric.csv", f"{PAIR}/sel_pairs.csv",
        f"{PAIR}/stats.csv", f"{PAIR}/stats_sorted.csv",
        f"{PAIR}/tumor_count.csv", f"{PAIR}/tumor_maxnorm.csv",
        f"{PAIR}/MI4_top30_sender_receiver_pair_median_activity.csv",
        f"{PROGRAM}/MI4_fibro_to_tumor_CancerSEA_3x8_plotting_table_median_by_cancertype_max5000.csv",
        f"{PROGRAM}/MI4_fibro_to_tumor_CancerSEA_SMD_stats_median_by_cancertype.csv",
        f"{PROGRAM}/MI4_fibro_to_tumor_edgelevel_strength_by_cancertype_plotting_table_max30000.csv",
        f"{PROGRAM}/MI4_fibro_to_tumor_edgelevel_strength_by_cancertype_summary.csv",
        f"{CCC}/Pancancer_CCC_all_axis_Fibroblast_to_tumor_CancerSEA_SMD_scan.csv",
    ]
    return [_native_path(Path(results_dir) / name) for name in names]


def check(results_dir):
    """Report missing result tables or compact spatial drawing inputs."""
    missing = [str(path) for path in required_inputs(results_dir) if not path.is_file()]
    for manifest in _drawing_manifests(results_dir):
        missing.extend(_drawing_cache_issues(manifest))
    return missing


def _notebook_sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {
        tag: "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
        for tag in cell.get("metadata", {}).get("tags", [])
    }


def _code(sources, tag, env, start=None, end=None):
    source = sources[tag]
    if source.startswith("%%"):
        source = source.split("\n", 1)[1]
    if start is not None:
        source = source[source.index(start):]
    if end is not None:
        source = source[:source.index(end)]
    exec(compile(source, f"<{NOTEBOOK.name}:{tag}>", "exec"), env)


def _settings(sources, tag, names, env):
    """Evaluate named assignments directly from the notebook configuration."""
    source = sources[tag]
    if source.startswith("%%"):
        source = source.split("\n", 1)[1]
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            keys = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if keys.intersection(names):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<plot settings>", "exec"), env)
                found.update(keys)
    if set(names) - found:
        raise ValueError(f"Missing notebook settings: {sorted(set(names) - found)}")


def run(results_dir: Path, output_dir: Path):
    """Write cached-table figures and return their output paths.

    output_dir is the workflow output root; figures are written below spatial/.
    Every prerequisite is checked before output creation. Files in results_dir
    are read-only. Existing plotting tables retain their original sampling.
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    results_dir, output_root = _native_path(results_dir), _native_path(output_dir)
    if results_dir == output_root:
        raise ValueError("Use a separate output directory to protect scientific caches.")
    output_dir = output_root / "spatial"
    before_outputs = _output_snapshot(output_dir)
    required = required_inputs(results_dir)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Spatial plot-only inputs are missing; run --stage spatial first:\n" + "\n".join(missing))
    sources = _notebook_sources()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = {"__name__": "pancancer_cached_plots", "np": np, "pd": pd,
           "mpl": mpl, "plt": plt, "Path": Path,
           "display": lambda *args, **kwargs: None,
           "run_dirs": {"run_dir": output_dir},
           "release_memory": lambda *args, **kwargs: None}

    def read(name, index_col=None):
        return pd.read_csv(results_dir / name, index_col=index_col)

    with mpl.rc_context():
        # Reuse pair-summary renderers without reconstructing edge metadata.
        _code(sources, "plot-pair-summary", env, end="processed = _ensure_processed_loaded()")
        for name, filename, index in [
            ("full_metric_df", "full_metric.csv", None),
            ("selected_pair_df", "sel_pairs.csv", None),
            ("selected_stats_df", "stats.csv", None),
            ("tumor_type_tumor_involved_count_df", "tumor_count.csv", 0),
            ("tumor_type_tumor_involved_maxnorm_df", "tumor_maxnorm.csv", 0),
        ]:
            env[name] = read(f"{PAIR}/{filename}", index)
        env["mi_order_pc"] = read(f"{PAIR}/stats_sorted.csv")["MI"].tolist()
        env["mi_order"] = env["mi_order_pc"]
        _code(sources, "plot-pair-summary", env,
              start="selected_bubble_pdf, selected_bubble_png =",
              end="# Step 8. Display summary tables")

        # Preserve LR pathway ordering, then redirect only its figure destination.
        env["run_dirs"] = {"run_dir": results_dir}
        _code(sources, "prepare-pathway-plot", env)
        env["run_dirs"] = {"run_dir": output_dir}
        _code(sources, "plot-pathway", env)

        env["MI_cancerSEA_meanrank_sender_df"] = read("MI_cancerSEA_meanrank_profile_sender.csv", 0)
        env["MI_cancerSEA_meanrank_df"] = read("MI_cancerSEA_meanrank_profile_receiver.csv", 0)
        _code(sources, "plot-cancersea", env)

        # Per-pair correlations use the original aligned sub-slice matrices.
        bulk = read(f"{LR}/LR_coexpression_bulk_proxy_subslice_by_LR.csv", 0)
        spatial = read(f"{LR}/LR_coexpression_spatial_edge_mean_subslice_by_LR.csv", 0)
        env["common_samples"] = [sample for sample in bulk.index if sample in spatial.index]
        env["common_lr_pairs"] = [pair for pair in bulk.columns if pair in spatial.columns]
        env["bulk_lr_proxy_aligned"] = bulk.loc[env["common_samples"], env["common_lr_pairs"]]
        env["spatial_lr_edge_mean_aligned"] = spatial.loc[env["common_samples"], env["common_lr_pairs"]]
        env["bulk_spatial_lr_output_dir"] = output_dir / LR
        env["bulk_spatial_lr_output_dir"].mkdir(parents=True, exist_ok=True)
        _code(sources, "plot-lr-validation", env)

        # These saved tables are already sampled with the notebook's fixed seed.
        env["mi4_program_outdir"] = output_dir / PROGRAM
        env["mi4_program_outdir"].mkdir(parents=True, exist_ok=True)
        _code(sources, "prepare-program-groups", env, end="def _p_to_star")
        env["MI_OI"] = "MI4"
        _code(sources, "plot-program-scores", env, end='if "mi4_program_long_df" not in globals():')
        env["plot_df"] = read(f"{PROGRAM}/MI4_fibro_to_tumor_CancerSEA_3x8_plotting_table_median_by_cancertype_max5000.csv")
        env["mi4_program_stats_df"] = read(f"{PROGRAM}/MI4_fibro_to_tumor_CancerSEA_SMD_stats_median_by_cancertype.csv")
        _code(sources, "plot-program-scores", env, start="def _format_smd_label")

        _code(sources, "plot-edge-distribution", env, end="def _to_numpy_local")
        env["mi4_edge_plot_df"] = read(f"{PROGRAM}/MI4_fibro_to_tumor_edgelevel_strength_by_cancertype_plotting_table_max30000.csv")
        env["mi4_edge_summary_df"] = read(f"{PROGRAM}/MI4_fibro_to_tumor_edgelevel_strength_by_cancertype_summary.csv")
        present = set(env["mi4_edge_summary_df"]["CancerType"].astype(str))
        env["cancer_type_order_use"] = [ct for ct in env["CANCERTYPE_ORDER"] if ct in present]
        env["cancer_type_order_use"] += sorted(present - set(env["cancer_type_order_use"]))
        _code(sources, "plot-edge-distribution", env, start='plt.close("all")', end="# Optional cleanup")

        _code(sources, "plot-pair-medians", env, end="def _to_numpy_local")
        env["stem_out_dir"] = output_dir / PAIR
        env["mi4_top_pair_median_activity_df"] = read(f"{PAIR}/MI4_top30_sender_receiver_pair_median_activity.csv")
        if len(env["mi4_top_pair_median_activity_df"]) > env["TOP_N_PAIRS"]:
            raise ValueError("Cached median-pair table does not match TOP_N_PAIRS.")
        _code(sources, "plot-pair-medians", env, start='plt.close("all")', end="# Optional cleanup")

        # Re-select baseline axes with the notebook's current, unchanged rule.
        _settings(sources, "prepare-ccc-settings", {
            "CANCERTYPE_ORDER", "CANCERSEA_PROGRAMS", "PROGRAM_ORDER", "METHODS_TO_RUN",
            "SPIDERNET_FIXED_DIM_ONE_BASED", "SPIDERNET_FIXED_FEATURE_NAME",
            "BASELINE_FEATURE_SELECTION_MODE", "REQUIRE_ALL_PANELS_FOR_FEATURE_SELECTION",
            "INCOMING_AGG", "MODULE_SCORE_METHOD", "SPLIT_RULE", "GROUPING_SOURCE",
            "SPIDERNET_EVAL_FILTER", "BASELINE_EVAL_FILTER",
        }, env)
        env["OUT_DIR"] = output_dir / CCC
        env["OUT_DIR"].mkdir(parents=True, exist_ok=True)
        env["all_scan_df"] = read(f"{CCC}/Pancancer_CCC_all_axis_Fibroblast_to_tumor_CancerSEA_SMD_scan.csv")
        env["method_order_use"] = env["METHODS_TO_RUN"]
        for method, group in env["all_scan_df"].groupby("Method", sort=False):
            expected = {"Incoming_Agg": env["INCOMING_AGG"],
                        "module_score_method": env["MODULE_SCORE_METHOD"],
                        "Grouping_Rule": env["SPLIT_RULE"],
                        "Grouping_Source": env["GROUPING_SOURCE"],
                        "Eval_Filter": env["SPIDERNET_EVAL_FILTER"] if method == "SpiderNet" else env["BASELINE_EVAL_FILTER"]}
            for column, value in expected.items():
                if column not in group or not group[column].astype(str).eq(str(value)).all():
                    raise ValueError(f"CCC cache for {method} does not match {column}={value!r}.")
        _code(sources, "prepare-ccc-imports", env, start='plt.close("all")')
        _code(sources, "select-ccc-features", env)
        _code(sources, "plot-ccc", env)
        plt.close("all")

    for manifest in _drawing_manifests(results_dir):
        issues = _drawing_cache_issues(manifest)
        if issues:
            print("Spatial plot-only coverage: " + "; ".join(issues))
            continue
        destination = output_dir / "In_situ_meta_interaction"
        if manifest.parent.name == "MI2_representative":
            destination = destination / "MI2_representative"
        replay_drawing_cache(manifest, destination)

    # Preserve the exact source tables alongside figures, excluding the two
    # pairwise LR matrices and the all-axis scan (upstream analysis caches).
    for source in required:
        relative = source.relative_to(results_dir)
        if "LR_coexpression_" in source.name or "all_axis_" in source.name:
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return sorted(path for path, stamp in _output_snapshot(output_dir).items()
                  if before_outputs.get(path) != stamp)
