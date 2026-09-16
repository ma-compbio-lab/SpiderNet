"""Execution recipes for preprocessing, model summaries, baselines and cascades.

Scientific code is read from the retained notebooks by stable cell ID. Plot-only
recipes extract the original drawing statements and load their saved inputs.
The caller supplies the workflow_paths bootstrap before executing these cells.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {
    "preprocess": "spidernet_dataloading_MIdimselection_HGSOC.ipynb",
    "train": "HGSOC_modeltraining.ipynb",
    "model-summary": "HGSOC_modeltraining.ipynb",
    "baseline": "HGSOC_Malignantsubtype_baseline.ipynb",
    "cascade": "HGSOC_MIcascade_analysis.ipynb",
}
_IDS = {
    "preprocess": {
        1: "02d2f5fe", 3: "b2448239", 5: "3efb50b4", 7: "d218b3b1",
        9: "1f2a7ebd", 11: "f0a5e393168ccd6f", 12: "75022144",
        14: "64faef1a", 16: "c1bc1625", 18: "03b2b747",
    },
    "train": {
        3: "d7165057", 5: "a742e890", 7: "0a8c5e32", 9: "0a3e4353",
        11: "1760514c", 13: "b5d33bb6", 15: "fca4b17f", 17: "3b0930c0",
        19: "991c1d1c", 21: "69112194f860a868", 22: "928a8b2a1dd41be7",
        23: "53eadcb3235fec8b", 25: "298e6eeb665b06c1", 27: "4893f49f",
        31: "0872d3a4", 33: "86e5c8b9", 35: "3dab30a5",
        37: "cb2f6a114a6c2114", 39: "s9a-execution", 42: "s9b-helpers",
        44: "s9b-execution",
    },
    "baseline": {
        2: "611708de768741f", 4: "af4dfbf837e5e767", 6: "ad0a72606b151651",
        8: "2aebe63f045ad7c1", 12: "5e572fa6716f2d52",
        13: "afebf36c51585f1b", 14: "6d695552", 16: "4c010d7c99cdffe",
        18: "845ee9e743ef9c81", 19: "880ce5580d3c0d4a",
        20: "97f5f4be46ac70c7", 21: "97f3aed6dfb24093",
        23: "65fa054ed9d4a3db", 27: "64e08e4d504c84b3",
        28: "2c1eed0a2187d494", 29: "52f7535cc5c8fc60",
        30: "13af12b88a9cda33", 31: "a4e12ebb2ed3f4cf",
        32: "2abc3746ae78b1b9", 34: "bce88d9e9c4da8d",
    },
    "cascade": {
        2: "8b0c523a595de605", 4: "23316b3a8b83c02d",
        6: "2e71b1dc593684f0", 8: "9236ce377d909082",
        10: "3d3100499b43995b", 12: "ec7fb6d70e6f79e2",
        13: "5717117ebb8c0dad", 14: "9760b1901f7a294c",
        15: "470e43207e2e3d76", 18: "f2cf5714ca7024c1",
        20: "e3a9dde9e050c4e1", 21: "51906017ff70b42a",
        22: "944959bbecc398eb", 24: "be88a07623eec0cb",
        27: "75156c24", 28: "ce7b35d4", 30: "c2f6c2c1",
        31: "666af4a0", 32: "068bfa1a",
        33: "mi12-mi10-cascade-count-by-slice", 35: "insitu-smi-t10-f018",
        37: "dd4cfc32", 39: "a1b8ce08",
    },
}


def source(stage, original_index):
    """Read a current notebook cell; cell order and empty cells are immaterial."""
    key = "train" if stage == "model-summary" else stage
    notebook = json.loads((HERE / NOTEBOOKS[stage]).read_text(encoding="utf-8"))
    cell_id = _IDS[key][original_index]
    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise ValueError(f"Required cell {cell_id} was removed from {NOTEBOOKS[stage]}")


def _fragment(code, start=None, end=None):
    """Take an unchanged statement block, dedenting a function body if needed."""
    lines = code.splitlines(keepends=True)
    first = 0 if start is None else next(
        i for i, line in enumerate(lines) if line.lstrip().startswith(start)
    )
    last = len(lines) if end is None else next(
        i for i in range(first + 1, len(lines)) if lines[i].lstrip().startswith(end)
    )
    return textwrap.dedent("".join(lines[first:last])).strip() + "\n"


def _definitions(code, names):
    nodes = {
        node.name: node for node in ast.parse(code).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return "\n\n".join(ast.get_source_segment(code, nodes[name]) for name in names)


def _assignment(code, name):
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.get_source_segment(code, node) + "\n"
    raise ValueError(f"Original assignment was not found: {name}")


def _library_function(name, module="SpiderNet.analysis"):
    """Read the installed upstream helper, keeping its plotting implementation."""
    path = Path(importlib.util.find_spec(module).origin)
    code = path.read_text(encoding="utf-8")
    return _definitions(code, (name,))


def _when_saved(names, body, label):
    """Missing cached products are reported, never replaced by a new fit."""
    return (
        f"_plot_inputs = [saved(name) for name in {names!r}]\n"
        "_plot_missing = [str(path) for path in _plot_inputs if not path.exists()]\n"
        "if _plot_missing:\n"
        f"    print('SKIP {label}: missing saved inputs:', _plot_missing)\n"
        "else:\n" + textwrap.indent(body.strip() + "\n", "    ")
    )


def _darkgrid_style(code):
    # Verified pixel-for-pixel against the saved count and model figures: their
    # original kernels inherited darkgrid. Keep this state local to each plot.
    return (
        "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
        "with plt.rc_context():\n    sns.set_style('darkgrid')\n"
        + textwrap.indent(code, "    ")
    )


def analysis_cells(stage):
    """Full local analyses; expensive preprocessing and training are explicit stages."""
    if stage == "preprocess":
        return [source(stage, i) for i in (1, 3, 5, 7, 9, 11, 12, 14, 16, 18)]
    if stage == "train":
        return [source(stage, i) for i in (
            3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 23, 25, 27
        )]
    if stage == "model-summary":
        return [
            _model_setup(), source(stage, 11),
            *[_darkgrid_style(source(stage, i)) if i in (33, 35, 39, 44)
              else source(stage, i) for i in (31, 33, 35, 37, 39, 42, 44)],
        ]
    if stage == "baseline":
        result = []
        for i in (2, 4, 6, 8, 12, 13, 14, 16, 18, 19, 20, 21, 23,
                  27, 28, 29, 30, 31, 32, 34):
            result.append(source(stage, i))
            if i == 21:
                result.append("baseline_asw_rows = [dict(method='Scanpy', ASW=float(asw_sub), n_sub=int(n_sub), seed=0)]")
            elif i == 30:
                result.append(
                    "baseline_asw_rows.append(dict(method='Banksy', ASW=float(asw_sub), n_sub=int(n_sub), seed=0))\n"
                    "pd.DataFrame(baseline_asw_rows).to_csv(Path(run_dirs['run_dir']) / 'Baseline_sample_ASW.csv', index=False)"
                )
        return result
    if stage == "cascade":
        result = []
        for i in (2, 4, 6, 8, 10, 12, 13, 14, 15, 18, 20, 21, 22,
                  24, 27, 28, 30, 31, 32, 33, 35, 37, 39):
            result.append(_darkgrid_style(source(stage, i)) if i == 33 else source(stage, i))
            if i == 20:
                result.append(_CASCADE_SAVE_MATRICES)
        return result
    raise ValueError(f"Unsupported secondary stage: {stage}")


def _model_setup():
    # The original scientific training configuration is retained without creating
    # a model, writing run_dirs.json, or calling run_training.
    cfg = source("train", 9)
    return (
        "from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\n"
        "from SpiderNet.config import TrainingConfig\n"
        "from SpiderNet.io import load_processed_data\n"
        + "".join(_assignment(source("train", 3), name) for name in (
            "SPECIES", "DIM_ENVIR", "N_JOBS", "MAX_EPOCH", "VERSION"
        ))
        + _assignment(cfg, "train_cfg")
        + "run_dirs = {key: Path(value) for key, value in run_dirs.items()}\n"
    )


_CASCADE_SAVE_MATRICES = """
from pathlib import Path
import pickle
_cascade_plot_matrices = {
    'colocal_count_merge_sum_zscore': colocal_count_merge_sum_zscore,
    'colocal_count_merge_sum_pvalue': colocal_count_merge_sum_pvalue,
    'prop_avg_allsample_filter_merge_pivot_select': prop_avg_allsample_filter_merge_pivot_select,
}
with open(Path(run_dirs['run_dir']) / 'MI_cascade_plot_matrices.pkl', 'wb') as handle:
    pickle.dump(_cascade_plot_matrices, handle, protocol=4)
for _key, _matrix in _cascade_plot_matrices.items():
    _matrix.to_csv(Path(run_dirs['run_dir']) / ('MI_cascade_' + _key + '.csv'))
"""


def _cascade_plots():
    stage = "cascade"
    imports = """
from pathlib import Path
import os, re, math, textwrap, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu, wilcoxon
from workflow_paths import saved, retain_existing, RUN
"""
    style = _fragment(source(stage, 2), "plt.rcParams[", "cuda_available")
    settings = _assignment(source(stage, 24), "REQUIRE_CT3_MALIGNANT_C5") + source(stage, 27)
    setup = _fragment(source(stage, 28), "output_dir =")
    funcs = _definitions(source(stage, 30), (
        "_p_to_star", "_compare_two_groups", "_set_nature_style", "_wrap_label",
        "plot_feature_panels_with_samplesize",
    ))
    single_name = "f'{cascade_output_suffix}_single_unit_feature_records.csv'"
    sample_name = "f'{cascade_output_suffix}_sample_level_feature_scores.csv'"
    single = (
        f"_feature_path = saved({single_name})\n"
        "if _feature_path.exists():\n"
        "    feature_records_upstream_gated = pd.read_csv(_feature_path, float_precision='round_trip')\n"
        + textwrap.indent(_fragment(source(stage, 37), "plot_feature_panels_with_samplesize("), "    ")
        + "else:\n    print('SKIP cascade single-unit plots: missing', _feature_path)\n"
    )
    sample = (
        f"_sample_path = saved({sample_name})\n"
        "if _sample_path.exists():\n"
        "    sample_level_feature_df = pd.read_csv(_sample_path, float_precision='round_trip')\n"
        + textwrap.indent(_fragment(source(stage, 39), "plot_sample_df ="), "    ")
        + "else:\n    print('SKIP cascade sample-level plots: missing', _sample_path)\n"
    )
    count_name = "MI12high_MI10high_Monocyte-Fibroblast-Malignant_cascade_count_by_slice.csv"
    count = _when_saved([count_name],
        f"cascade_count_by_slice = pd.read_csv(saved({count_name!r}), float_precision='round_trip')\n"
        + _assignment(source(stage, 33), "CASCADE_COUNT_HIGH_THRESHOLD")
        + _darkgrid_style(_fragment(source(stage, 33), "fig_height =")), "cascade counts")
    heatmap = _when_saved(["MI_cascade_plot_matrices.pkl"],
        "with open(saved('MI_cascade_plot_matrices.pkl'), 'rb') as handle:\n"
        "    globals().update(pickle.load(handle))\n"
        + source(stage, 13) + "\n"
        + "\n".join(source(stage, i) for i in (18, 21, 22)), "cascade heatmaps")
    heatmap += """
if _plot_missing:
    retain_existing([
        'Heatmap_Zscore_Colocalization_Count_allsample.png',
        'Heatmap_Zscore_Colocalization_Count_allsample.pdf',
        'Heatmap_Average_prop_Celltype_triple_across_MI_cascades.pdf',
        'Heatmap_Average_prop_Celltype_triple_across_MI_cascades_Malignant.pdf',
    ], 'Cascade matrix inputs were not exported by the original run; permutations were not rerun.')
"""
    # In situ rendering still requires the sample's expression/spatial objects.
    insitu_note = (
        "retain_existing([str(p.relative_to(RUN)) for p in "
        "(RUN / 'MI12_MI10_cascade_insitu').rglob('*') if p.suffix.lower() in ('.png', '.pdf')], "
        "'In situ regeneration needs processed spatial data and the cascade analysis stage.')\n"
        "print('Cascade in situ: existing PNG/PDF retained; regenerate with the cascade analysis stage.')"
    )
    insitu_style = _fragment(
        _library_function("insituplot_MIcascade", "SpiderNet.visualization"),
        "plt.rcParams[", "fig, ax =",
    )
    # Keep the notebook's drawing order: count precedes the feature helper's
    # rcParams changes; the global heatmap precedes both.
    return [imports, style, settings, setup, funcs, heatmap, count,
            insitu_style, single, sample, insitu_note]


def _model_plots():
    stage = "model-summary"
    helpers = source(stage, 37)
    draw = _fragment(source(stage, 39), "sender_cmap =")
    prefix = "MI_loading_heatmaps_sender_LR_receiver_sum_normalized_combined"
    suffixes = ("sender_regulator", "LR_pair", "receiver_target")
    names = [f"{prefix}_{suffix}_matrix.csv" for suffix in suffixes]
    loader = ""
    for name, variable, block in zip(names,
            ("sender_regulator_loading_norm_sorted", "lr_loading_norm_sorted", "receiver_target_loading_norm_sorted"),
            ("sender_block_labels", "lr_block_labels", "receiver_block_labels")):
        loader += (
            f"{variable} = pd.read_csv(saved({name!r}), index_col=0, float_precision='round_trip')\n"
            f"_values = {variable}.to_numpy()\n"
            f"{block} = [str({variable}.index[i]) if v > 0 else 'All-zero' "
            "for i, v in zip(np.argmax(_values, axis=0), np.max(_values, axis=0))]\n"
        )
    loading_plot = _when_saved(names, loader + draw, "model loading heatmaps")
    concordance = _when_saved(["MI_component_concordance/MI_component_concordance.csv"],
        "CONCORDANCE_SHOW = True\n"
        "concordance_dir = Path(run_dirs['run_dir']) / 'MI_component_concordance'\n"
        "concordance_dir.mkdir(parents=True, exist_ok=True)\n"
        "concordance_table = pd.read_csv(saved('MI_component_concordance/MI_component_concordance.csv'), float_precision='round_trip')\n"
        + _fragment(source(stage, 44), "concordance_figure =", "display(concordance_summary)"),
        "model component concordance")
    return [_model_setup(), "from workflow_paths import saved", *_model_pathway_pair_plots(),
            helpers, _darkgrid_style(loading_plot), source(stage, 42), _darkgrid_style(concordance)]


def _model_pathway_pair_plots():
    lr_code = _library_function("LRLoading_enrichment")
    pair_code = _library_function("MI_Celltypepair_enrichment")
    setup = (
        "import os\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        "from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm\n"
        "from matplotlib.patches import Rectangle\n"
        "file_savepath_main = str(run_dirs['run_dir'])\nshow = True\n"
        + _fragment(lr_code, "plt.rcParams[", '"""')
    )
    lr_plot = _when_saved(["LR_loading_pathway.csv"],
        "LR_loading_pathway = pd.read_csv(saved('LR_loading_pathway.csv'), index_col=0, float_precision='round_trip')\n"
        + _fragment(lr_code, "# Visualization", "# Save results"), "LR pathway heatmap")
    call = next(n for n in ast.walk(ast.parse(source("model-summary", 35)))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "MI_Celltypepair_enrichment")
    threshold = next(ast.literal_eval(kw.value) for kw in call.keywords
                     if kw.arg == "MIlevel_agg_threshold")
    pair_body = (
        "Avg_MI_cellclass_pair_merge_use = pd.read_csv(saved('Avg_MI_cellclass_pair_merge_use.csv'), float_precision='round_trip')\n"
        "cellclass_unique = pd.read_pickle(PROCESSED_DATA_DIR / 'cellclass_unique.pkl')\n"
        "MI_use = np.empty((0, train_cfg.dim_envir))\n"
        f"MIlevel_agg_threshold = {threshold!r}\n"
        "_cached_maxima = []\n"
        "for _ct in cellclass_unique:\n"
        "    _rows = Avg_MI_cellclass_pair_merge_use.loc[\n"
        "        (Avg_MI_cellclass_pair_merge_use['Sender'] == _ct) |\n"
        "        (Avg_MI_cellclass_pair_merge_use['Receiver'] == _ct)].iloc[:, :train_cfg.dim_envir]\n"
        "    _cached_maxima.append(float(_rows.to_numpy().max()) if len(_rows) else float('nan'))\n"
        # If every cell type retains a maximum >= the configured cutoff, the
        # original helper's conditional threshold reduction could not have run.
        "if not np.all(np.asarray(_cached_maxima) >= MIlevel_agg_threshold):\n"
        "    print('SKIP pair heatmap: the saved matrix cannot establish the original effective highlighting threshold.')\n"
        "else:\n"
        + textwrap.indent(_fragment(pair_code, "heatmap_values =", "# # Heatmap:"), "    ")
        + _fragment(pair_code, "cell_types =", "# Aggregate per-cell-type")
    )
    pair_plot = _when_saved(["Avg_MI_cellclass_pair_merge_use.csv"], pair_body,
                            "MI cell-type pairs")
    return [setup, _darkgrid_style(lr_plot), _darkgrid_style(pair_plot)]


def _baseline_plots():
    stage = "baseline"
    imports = """
from pathlib import Path
import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from workflow_paths import saved
cellclass_choose = 'Malignant'
"""
    # The scoring cell sets the style used by the original embedding plots.
    result = [imports, _fragment(source(stage, 14), end="functional_states =")]
    scoring_source = source(stage, 23)
    for which, load_index, plot_indices in (
        (2, 18, (19, 20)), (3, None, (28, 29)),
    ):
        if which == 3:
            # Scanpy's KEGG section changes style before the Banksy embedding.
            result.append(_fragment(scoring_source, end="def _display_df("))
        load = (source(stage, load_index) if load_index else
                "import scanpy as sc\nadata_choose3 = sc.read_h5ad(saved('adata_choose3_Malignant.h5ad'))")
        # Both methods historically save UMAP_stage_x.pdf; retain that filename.
        body = "import scanpy as sc\n" + load + "\n" + "\n".join(source(stage, i) for i in plot_indices)
        result.append(_when_saved([f"adata_choose{which}_Malignant.h5ad"], body,
                                  f"baseline {which} embedding"))

    result.append(_definitions(scoring_source, ("_display_df", "_safe_name")))
    result.append(_assignment(scoring_source, "score_title_map"))
    plot_code = _fragment(scoring_source, "n_panels = len(score_cols)", "# Save outputs")
    # The extracted block is inside the original function and retains all plot arguments.
    for label, embedding in (("Scanpy", "X_umap"), ("Banksy", "Banksy_umap")):
        folder = f"Baseline_{label}_KEGG_module_score_continuous_UMAP"
        table = f"{folder}/{label}_celllevel_KEGG_module_scores_with_coordinates.csv"
        gene_count = f"{folder}/{label}_KEGG_module_gene_count.csv"
        body = (
            f"baseline_label = baseline_safe = {label!r}\nembedding_key = {embedding!r}\n"
            f"out_dir = Path(run_dirs['run_dir']) / {folder!r}\nout_dir.mkdir(parents=True, exist_ok=True)\n"
            f"module_score_df = pd.read_csv(saved({table!r}), float_precision='round_trip')\n"
            f"module_gene_count_df = pd.read_csv(saved({gene_count!r}))\n"
            "score_cols = module_gene_count_df['module'].tolist()\n"
            "coords = module_score_df[[f'{baseline_safe}_UMAP1', f'{baseline_safe}_UMAP2']].to_numpy(dtype=float)\n"
            + plot_code
        )
        result.append(_when_saved([table, gene_count], body, f"{label} KEGG UMAP"))

    heat = source(stage, 32)
    result.append(_fragment(heat, end='if "adata_choose3" not in globals():'))
    result.append(_assignment(heat, "expected_score_cols") + _assignment(heat, "score_label_map"))
    heat_name = "Banksy_louvain_KEGG_module_score_heatmap_sample_representation_summary.csv"
    body = (
        f"heatmap_summary_path = saved({heat_name!r})\n"
        "sample_cluster_prop_path = saved('Banksy_louvain_KEGG_module_score_heatmap_sample_representation_sample_cluster_proportions.csv')\n"
        "heatmap_summary_df = pd.read_csv(heatmap_summary_path, dtype={BANKSY_CLUSTER_COL: str}, float_precision='round_trip')\n"
        "cluster_order = _sort_cluster_values(heatmap_summary_df[BANKSY_CLUSTER_COL])\n"
        "cluster_mean_df = heatmap_summary_df.set_index(BANKSY_CLUSTER_COL)[expected_score_cols].reindex(cluster_order)\n"
        + _fragment(heat, "heatmap_mat =")
    )
    result.append(_when_saved([heat_name], body, "Banksy cluster KEGG heatmap"))

    comparison = source(stage, 34)
    result.append(_fragment(comparison, end="# Read SpiderNet exported CSV"))
    long_name = "SpiderNet_C5_vs_Banksy_C10_KEGG_module_score_boxplot_celllevel_long.csv"
    body = (
        f"comparison_long_df = pd.read_csv(saved({long_name!r}), float_precision='round_trip')\n"
        + _assignment(comparison, "BANKSY_CLUSTER_TO_COMPARE_RESOLVED")
        + _assignment(comparison, "spider_method_group_label")
        + _assignment(comparison, "banksy_method_group_label")
        + _fragment(comparison, "module_order =", 'print(f"SpiderNet score CSV used:')
    )
    result.append(_when_saved([long_name], body, "SpiderNet C5 versus Banksy C10"))
    result.append("print('ASW fitting/statistics are not rerun in plot-only; use Baseline_sample_ASW.csv or the saved executed notebook.')")
    return result


def plot_cells(stage):
    """Replot saved products without training, embedding fits or cascade permutations."""
    if stage == "cascade":
        return _cascade_plots()
    if stage == "baseline":
        return _baseline_plots()
    if stage == "model-summary":
        return _model_plots()
    if stage == "preprocess":
        return ["""
import pickle
import numpy as np
from workflow_paths import PROCESSED_DATA_DIR, OUTPUT
from SpiderNet.MI_dimension_selection import plot_lr_spearcorr_heatmap
_corr_path = PROCESSED_DATA_DIR / 'mi_dimension_selection_LR_spearcorr.npy'
_subsets_path = PROCESSED_DATA_DIR / 'mi_dimension_selection_merged_subsets.pkl'
if not _corr_path.exists() or not _subsets_path.exists():
    raise FileNotFoundError(f'Preprocess plot-only needs {_corr_path} and {_subsets_path}')
with open(_subsets_path, 'rb') as handle:
    _merged_subsets = pickle.load(handle)
plot_lr_spearcorr_heatmap(
    corr=np.load(_corr_path), merged_subsets=_merged_subsets,
    output_path=OUTPUT / 'preprocess' / 'mi_dimension_selection_LR_spearcorr_heatmap_by_merged_subsets.pdf',
    show=True,
)
"""]
    if stage == "train":
        raise ValueError("Training has no plot-only stage; use --stage model-summary --plot-only.")
    raise ValueError(f"Unsupported secondary stage: {stage}")
