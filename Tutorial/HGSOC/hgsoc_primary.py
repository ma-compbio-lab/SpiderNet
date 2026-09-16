"""Execution selections for the malignant, perturbation and CCC analyses.

Scientific code stays in the maintained notebook. Plot selections reuse its
plotting statements with saved score tables, so fitting and rescoring are absent.
The runner supplies the workflow_paths variables and common plotting imports.
"""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("HGSOC_Malignantsubtype_analysis_V2.ipynb")


def source(index):
    """Read a cell by its original position tag, allowing empty-cell removal."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tag = f"hgsoc-{index}"
    tagged = [cell for cell in notebook["cells"]
              if tag in cell.get("metadata", {}).get("tags", [])]
    if len(tagged) == 1:
        return "".join(tagged[0]["source"])
    if not tagged and not any(
        any(str(t).startswith("hgsoc-") for t in c.get("metadata", {}).get("tags", []))
        for c in notebook["cells"]
    ):
        return "".join(notebook["cells"][index]["source"])
    raise ValueError(f"Expected exactly one {tag} cell in {NOTEBOOK.name}")


def _before(index, anchor):
    code = source(index)
    return code[:code.index(anchor)]


def _after(index, anchor):
    code = source(index)
    return code[code.index(anchor):]


def _confirmed_style(index):
    # The saved, user-reviewed triangle, CAF and CCC panels were rendered with the
    # existing Seaborn white theme. Isolate that state: running unrelated cells
    # first must not turn the grey text black or change the title alignment.
    # Each panel was compared pixel-for-pixel against its saved PNG.
    return ('with plt.rc_context():\n    sns.set_theme(style="white")\n'
            + "\n".join("    " + line for line in source(index).splitlines()) + "\n")


def analysis_cells(stage):
    """Return original analysis cells in their existing order, without training."""
    if stage == "malignant":
        indices = [2, 8, 9, 13, 15, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28,
                   29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43,
                   44, 46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60,
                   61, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 77]
    elif stage == "perturbation":
        indices = [2, 8, 9, 13, 57, 84, 86, 88, 90, 92, 93]
    elif stage == "ccc":
        indices = [2, 8, 9, 13, 57, 79, 80, 81]
    else:
        raise ValueError(f"Unknown primary analysis stage: {stage}")
    cells = []
    for index in indices:
        code = _confirmed_style(index) if index in (61, 77, 81) else source(index)
        if index == 63:
            code = code.replace(
                "    # Plot heatmap",
                '    cluster_ct_mean.to_csv(Path(run_dirs["run_dir"]) / '
                'f"MI_{direction_OI}_{MI_OI}_by_CellType_cluster_mean.csv")\n\n'
                "    # Plot heatmap", 1,
            )
        cells.append(code)
        if index == 53:
            cells.append(
                '# Preserve the unstandardized annotations for subsequent plotting.\n'
                'neglogp_matrix.to_csv(Path(run_dirs["run_dir"]) / '
                '"KEGG_enrichment_matrix_Malignant_raw.csv")\n'
            )
        elif index == 70:
            cells.append(
                'corr_tumor_state_df.to_csv(Path(run_dirs["run_dir"]) / '
                '"Corr_MI_Receiver_TumorFunctionalState_Malignant5.csv")\n'
            )
    return cells


_PLOT_SETUP = '''from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
run_dir = Path(run_dirs["run_dir"])
run_dir.mkdir(parents=True, exist_ok=True)
'''

_READ_MALIGNANT = '''import scanpy as sc
cellclass_choose = "Malignant"
adata_choose = sc.read_h5ad(input_path(run_dir / "adata_choose_Malignant.h5ad"), backed="r")
'''


def _kegg_umap_plot():
    prefix = _before(22, "def _clean_cluster_label")
    restored = '''module_score_df = pd.read_csv(input_path(Path(out_dir) / f"{OUT_PREFIX}_celllevel_module_scores_with_coordinates.csv"))
saved_plot_summary = pd.read_csv(input_path(Path(out_dir) / f"{OUT_PREFIX}_MI_UMAP_continuous_score_summary.csv"))
score_cols = saved_plot_summary["module"].tolist()
score_title_map = dict(zip(saved_plot_summary["module"], saved_plot_summary["module_label"]))
mi_umap_source = saved_plot_summary["umap_source"].iloc[0]
# The original block preserves obsm's coordinate dtype. A CSV loses it; restore
# it only when the saved provenance and every coordinate match the AnnData.
if mi_umap_source.startswith("obsm:"):
    source_coordinates = adata_choose.obsm[mi_umap_source.split(":", 1)[1]][:, :2]
    restored_coordinates = module_score_df[["MI_UMAP1", "MI_UMAP2"]].to_numpy(dtype=source_coordinates.dtype)
    if np.array_equal(restored_coordinates, source_coordinates):
        module_score_df["MI_UMAP1"] = restored_coordinates[:, 0]
        module_score_df["MI_UMAP2"] = restored_coordinates[:, 1]
'''
    # Stop before the optional AnnData write: plot-only never rewrites its input.
    plotting = _after(22, "n_panels = len(score_cols)").split(
        '# =============================\n# Optional: save updated adata_choose', 1
    )[0]
    return prefix + restored + plotting


def _functional_state_plot():
    return (
        _before(41, 'geneset_df = pd.read_csv')
        + 'df_long = pd.read_csv(input_path(run_dir / "FunctionalState_scores_by_MI_louvain_long.csv"), dtype={"MI_louvain": str})\n'
        + _after(41, 'order = sorted(df_long["MI_louvain"].unique()')
    )


def _mi_correlation_plot():
    return '''from scipy.stats import spearmanr
correlation_path = input_path(run_dir / "Corr_MI_Receiver_TumorFunctionalState_Malignant5.csv")
MI_OI_df = pd.DataFrame({"MI": ["MI10", "MI6", "MI13"], "direction": ["Receiver"] * 3})
if correlation_path.is_file():
    corr_tumor_state_df = pd.read_csv(correlation_path, index_col=0)
else:
    # Each table contains the original, already filtered MI/score pairs.
    geneset_df = pd.read_csv(DATA_ROOT / "CancerSEA_OV/functional_geneset_list_df.csv")
    adata_all_metadata = sc.read_h5ad(PROCESSED_DATA_DIR / "adata_all.h5ad", backed="r")
    states = np.unique(geneset_df.loc[geneset_df["Gene"].isin(adata_all_metadata.var_names), "GeneSet"])
    adata_all_metadata.file.close()
    corr_tumor_state_df = pd.DataFrame(index=states, columns=[f"{mi}_Receiver" for mi in MI_OI_df["MI"]])
    for state in states:
        for mi in MI_OI_df["MI"]:
            scatter = pd.read_csv(input_path(run_dir / f"Malignant_COI5_{state}_vs_Receiving_{mi}_scatter.csv"))
            corr_tumor_state_df.loc[state, f"{mi}_Receiver"] = spearmanr(scatter["MI_Level"], scatter["State_Score"])[0]
''' + source(71)


def _enrichment_plot():
    # Z-scores do not determine the original annotation values. Missing raw
    # values are an explicit limitation of old exports, never an invitation to
    # substitute z-scores as annotations or to rerun online enrichment.
    return '''raw_path = input_path(run_dir / "KEGG_enrichment_matrix_Malignant_raw.csv")
if not raw_path.is_file():
    import warnings
    from workflow_paths import retain_existing
    warnings.warn("KEGG enrichment heatmap was not regenerated: the old run saved only z-scores, not raw annotation values. The existing PDF is retained; a full malignant run exports the missing raw matrix.")
    retain_existing(["KEGG_Enrichment_Heatmap_Malignant_Clusters_Zscore.pdf"],
                    "Raw KEGG enrichment annotation matrix was not exported by the original run.")
else:
    neglogp_matrix = pd.read_csv(raw_path, index_col=0)
    neglogp_matrix_z = pd.read_csv(input_path(run_dir / "KEGG_enrichment_matrix_Malignant.csv"), index_col=0)
''' + "\n".join("    " + line for line in source(54).splitlines()) + "\n"


def _sender_type_plot():
    original = source(63)
    start = original.index("    # Find indices for current MI and direction")
    end = original.index("    # Plot heatmap", start)
    # The original aggregation after malignant subtype relabeling has different
    # cell-type semantics from the embedding cache. Only its own saved means are
    # valid inputs here; do not substitute MI_SR_agg_ct_all.npy from embedding.
    plotting = original[:start] + (
        '    cluster_ct_mean = pd.read_csv(input_path(run_dir / '
        'f"MI_{direction_OI}_{MI_OI}_by_CellType_cluster_mean.csv"), index_col=0)\n\n'
    ) + original[end:]
    return '''from matplotlib.colors import LinearSegmentedColormap
sender_type_means = [run_dir / f"MI_Receiver_{mi}_by_CellType_cluster_mean.csv"
                     for mi in ("MI10", "MI13", "MI6", "MI9", "MI12")]
missing_sender_type_means = [str(p) for p in sender_type_means if not input_path(p).is_file()]
if missing_sender_type_means:
    import warnings
    from workflow_paths import retain_existing
    warnings.warn("Sender-cell-type MI heatmap was not regenerated: the original run did not export its subtype-aware cluster means. A full malignant run now exports these matrices. Missing: " + "; ".join(missing_sender_type_means))
    retain_existing([f"Combined_MI_Levels_per_Cluster_CellType.{extension}"
                     for extension in ("png", "pdf")],
                    "Subtype-aware sender-cell-type cluster means were not exported by the original run.")
else:
''' + "\n".join("    " + line for line in plotting.splitlines()) + "\n"


def _perturbation_plots():
    setup = [_PLOT_SETUP, _READ_MALIGNANT, source(86)]
    imports = _before(92, "def compute_LFC")
    first = (
        imports
        # Cell 90 explicitly casts both model reconstructions to float32; the
        # original mean-ratio/log2 code consequently emits float32 LFC values.
        + 'LFC_perturb_all = pd.read_csv(input_path(run_dir / f"LFC_perturb_all_{MI_OI}_{sender_population}_to_{receiver_population}_{receiver_label}.csv"), dtype={"LFC": np.float32})\n'
        + _after(92, 'plt.close("all")')
    )
    second = (
        _before(93, "def compute_LFC")
        + 'LFC_perturb_all = pd.read_csv(input_path(run_dir / f"LFC_perturb_all_with_KEGG_{MI_OI}_{sender_population}_to_{receiver_population}_{receiver_label}.csv"), dtype={"LFC": np.float32})\n'
        + _after(93, 'plt.close("all")')
    )
    return setup + [first, second, "adata_choose.file.close()\n"]


def plot_cells(stage):
    """Return saved-result plotting cells; never construct graphs or fit models."""
    if stage == "ccc":
        return [_PLOT_SETUP, _confirmed_style(81)]
    if stage == "perturbation":
        return _perturbation_plots()
    if stage != "malignant":
        raise ValueError(f"Unknown primary plot stage: {stage}")
    return [
        _PLOT_SETUP, _READ_MALIGNANT, source(18), _kegg_umap_plot(),
        _functional_state_plot(), source(46), source(47), _enrichment_plot(),
        'cluster_mean_order_all = pd.read_csv(input_path(run_dir / "Mean_MI_Intensity_cluster.csv"), index_col=0)\n'
        'cluster_mean_order_sender = cluster_mean_order_all.loc[:, [c for c in cluster_mean_order_all if "Sender" in c]]\n'
        'cluster_mean_order_receiver = cluster_mean_order_all.loc[:, [c for c in cluster_mean_order_all if "Receiver" in c]]\n',
        _confirmed_style(61), _sender_type_plot(), _mi_correlation_plot(), source(72), _confirmed_style(77),
        "adata_choose.file.close()\n",
    ]
