"""Run the original PerturbFISH analysis cells with explicit inputs and outputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.environ.get("SPIDERNET_WORKSPACE_ROOT", next(
    (p for p in HERE.parents if (p / "Data").is_dir() and (p / "Results").is_dir()),
    HERE,
))).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("SPIDERNET_PERTURBFISH_DATA_ROOT", WORKSPACE_ROOT / "Data/PerturbFISH"))
UPSTREAM_ROOT = Path(os.environ.get("SPIDERNET_PERTURBFISH_OUTPUT_ROOT", WORKSPACE_ROOT / "Results/PerturbFISH"))
PROCESSED_ROOT = Path(os.environ.get("SPIDERNET_PERTURBFISH_PROCESSED_ROOT", UPSTREAM_ROOT / "ProcessedData"))
RESULTS_DIR = Path(os.environ.get("SPIDERNET_PERTURBFISH_RESULTS_DIR", UPSTREAM_ROOT / "V1/SpiderNet_Result_dim23"))
OUTPUT_DIR = Path(os.environ.get("SPIDERNET_PERTURBFISH_PLOT_DIR", HERE / "output")).resolve()
SCCCHAIN_RESULTS = Path(os.environ.get("SPIDERNET_PERTURBFISH_SCCCHAIN_ROOT", UPSTREAM_ROOT / "Version_V1/dim_envir_23/Baseline_CCC_MI17_GO_SMD_melanoma_to_Tcell/ScCChain"))

GENES = ["CHUK", "IRAK1", "TRAM1", "LBP", "IRAK4", "PELI1", "TAB2", "MAP2K2", "MAP2K6", "IRF7", "MYD88"]
CORE = "PerturbFISH_insilicospatialperturbation.ipynb"
LOO = "PerturbFISH_Leaveoneout_fullpipeline.ipynb"
TRAIN = "PerturbFISH_modeltraining.ipynb"
PREPROCESS = "spidernet_dataloading_MIdimselection_PerturbFISH.ipynb"
CCC = "Baseline_CCC_MI17_GO_SMD_melanoma_to_Tcell"
LRKO = "InSilico_LRKO_MI17_melanoma_to_Tcell_GOprograms"
DEG = "MI17_Tcell_DEG_neighboring_perturbed_melanoma"
GSEA = DEG + "/MI17_receiver_loading_GSEA"
COMBINED = "Combined_MI17_MI_targetgene_LR_panels"
SPATIAL = "Spatial_MI17_perturb_vs_control_cancer_to_Tcell_edges"
LFC = "Cached_Tcell_LFC_for_scatter_seed10042"
DEFAULT_STAGES = ["overview", "loo", "core", "enrichment", "lrko", "ccc", "go"]
PLOT_STAGES = ["overview", "loo", "core", "lrko", "ccc", "go"]
ALL_STAGES = ["preprocess", "train", "train-loo", "response", *DEFAULT_STAGES]


def saved_result(relative):
    """Prefer this workflow's saved result, otherwise use the configured upstream."""
    local = OUTPUT_DIR / relative
    return local if local.is_file() else RESULTS_DIR / relative


def cache_path(relative):
    """Keep the original exists-based cache policy; new caches are written locally."""
    source = saved_result(relative)
    return source if source.is_file() else OUTPUT_DIR / relative


def copy_inputs(files):
    records = []
    for relative in files:
        source = saved_result(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = OUTPUT_DIR / relative
        if source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        with target.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        records.append({"input": str(source.resolve()), "copy": str(target.resolve()),
                        "bytes": target.stat().st_size, "sha256": digest})
    return records


def plot_inputs(stage):
    if stage == "response":
        return [f for f in plot_inputs("core") if f.startswith(DEG + "/")]
    return {
        "loo": ["insilico_spidernet_summary_leaveoneout.csv", "insilico_linearmodel_summary_leaveoneout.csv", "celcomen_summary.csv",
                *[LFC + "/" + n for n in ["Observed_Tcell_LFC_golden.csv", "Predicted_Tcell_LFC_SpiderNet.csv", "Predicted_Tcell_LFC_LinearModel.csv"]]],
        "core": ["Part1_insilico_spatial_perturbation_summary.csv", "MI_change_perturbcancerTOTcell_df.csv", "MI_change_TcellTOTperturbcancer_df.csv",
                 DEG + "/MI17_Tcell_DEG_all_perturb_genes.csv", DEG + "/MI17_receiver_loading_sum_normalized_with_ALL_Tcell_DEG_stats.csv",
                 GSEA + "/MI17_receiver_loading_ranked_GSEA_table.csv", GSEA + "/MI17_receiver_loading_GSEA_summary.csv",
                 *[COMBINED + "/" + n for n in ["MI17_MI_strength_plot_data.csv", "MI17_VEGFA_TBX3_PLOD2_expression_plot_data.csv", "MI17_VEGFA_TBX3_PLOD2_Tcell_group_stats.csv", "LR_positive_proportion_MI17_selected.csv"]],
                 *[SPATIAL + "/MI17_" + n + "_cancer_to_Tcell_edges_randomly_plotted.csv" for n in ["perturbed", "control"]]],
        "lrko": [LRKO + "/MI17_LRKO_sender_receiver_GO_module_score_change_long.csv", LRKO + "/MI17_LRKO_GO_module_score_change_summary.csv", LRKO + "/MI17_sender_receiver_GO_program_genes.csv"],
        "ccc": [CCC + "/GO_program_SMD_axisMatch_perturb_vs_control_summary_targetPerturbEdges.csv", CCC + "/axis_matching_selected_axis_by_method.csv", CCC + "/MI17_sender_GO_gene_sets.csv", CCC + "/MI17_receiver_GO_gene_sets.csv"],
        "go": ["GO_MI17_regulator_enrichment.csv", "GO_MI17_target_enrichment.csv",
               *["GO_MI17_inputs/" + name for name in ["loading_sender_use.csv", "loading_receiver_use.csv", "selected_genes.csv", "MI_activity_max.csv", "provenance.json"]]],
    }.get(stage, [])


def get_cells(filename, tags=None):
    import nbformat
    nb = nbformat.read(HERE / filename, as_version=4)
    if tags is None:
        return [c for c in nb.cells if c.cell_type == "code"]
    by_tag = {t: c for c in nb.cells if c.cell_type == "code" for t in c.metadata.get("tags", [])}
    return [copy.deepcopy(by_tag[t]) for t in tags]


def code(source):
    import nbformat
    return nbformat.v4.new_code_cell(source)


def basic_setup():
    return code('''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from run_benchmarks import OUTPUT_DIR, RESULTS_DIR, PROCESSED_ROOT, saved_result, GENES
base_run_dir = OUTPUT_DIR
base_run_dir.mkdir(parents=True, exist_ok=True)
MIOI = MIOI_plot = MI_RECEIVER_LOADING_MIOI = MI17_KO_MIOI = "MI17"
perturb_gene_OI = perturb_gene_OI_choose = list(GENES)
''')


def plot_cells(stage):
    cells = [basic_setup()]
    if stage == "loo":
        cells.append(code('''SpiderNet_summary = pd.read_csv(base_run_dir / "insilico_spidernet_summary_leaveoneout.csv", index_col=0)
linearmodel_summary = pd.read_csv(base_run_dir / "insilico_linearmodel_summary_leaveoneout.csv", index_col=0)
INSILICO_RANDOM_SEED = 10042
'''))
        cells += get_cells(LOO, ["loo-24", "loo-25", "loo-26"])
    elif stage == "core":
        cells.append(code('''from SpiderNet.visualization import plot_gene_correlation_barplot, plot_mi_change_heatmap, apply_publication_style
summary_df = pd.read_csv(base_run_dir / "Part1_insilico_spatial_perturbation_summary.csv", index_col=0)
mi_change_pc2t = pd.read_csv(base_run_dir / "MI_change_perturbcancerTOTcell_df.csv", index_col=0)
'''))
        cells += get_cells(CORE, ["core-12", "core-14-plot", "core-15", "core-25-setup"])
        # Saved edge indices refer to the unchanged processed graph; no model inference is needed.
        cells.append(code('''import pickle
from SpiderNet.analysis import concatenate_adata_list
with open(PROCESSED_ROOT / "adata_list.pkl", "rb") as f:
    spatial_adatas = pickle.load(f)
with open(PROCESSED_ROOT / "SpiderNet_data_pyg_list.pkl", "rb") as f:
    spatial_graphs = pickle.load(f)
adata_all_spatial = concatenate_adata_list(spatial_adatas)
spatial_xy = _get_spatial_xy_from_adata(adata_all_spatial)
# The original core analysis uses the first graph's edge index.
edge_index_all_spatial = _edge_index_to_2col_spatial(spatial_graphs[0].edge_index)
celltype2 = adata_all_spatial.obs["celltype2"].astype(str).to_numpy()
cancer_mask = celltype2 == "cancer"
tcell_mask = celltype2 == "T cells"
perturb_edge_df = pd.read_csv(SPATIAL_EDGE_OUTDIR / "MI17_perturbed_cancer_to_Tcell_edges_randomly_plotted.csv")
control_edge_df = pd.read_csv(SPATIAL_EDGE_OUTDIR / "MI17_control_cancer_to_Tcell_edges_randomly_plotted.csv")
'''))
        cells += get_cells(CORE, ["core-25-plot", "core-34-setup"])
        cells.append(code('''mi_strength_plot_df = pd.read_csv(COMBINED_OUTDIR / "MI17_MI_strength_plot_data.csv")
expr_long_df = pd.read_csv(COMBINED_OUTDIR / "MI17_VEGFA_TBX3_PLOD2_expression_plot_data.csv")
target_stats_df = pd.read_csv(COMBINED_OUTDIR / "MI17_VEGFA_TBX3_PLOD2_Tcell_group_stats.csv")
target_stats_df["star"] = target_stats_df["q_fdr_bh"].apply(_p_to_star_local)
genes_in_combined = [g for g in TARGET_GENES_COMBINED if g in set(expr_long_df["Gene"])]
LR_positive_prop_combined = pd.read_csv(COMBINED_OUTDIR / "LR_positive_proportion_MI17_selected.csv")
x_perturb = _clean_values(mi_strength_plot_df.loc[mi_strength_plot_df["Group"] == "Perturb edge", "Value"].to_numpy(dtype=float))
x_control = _clean_values(mi_strength_plot_df.loc[mi_strength_plot_df["Group"] == "Control edge", "Value"].to_numpy(dtype=float))
_, p_mi_strength = mannwhitneyu(x_perturb, x_control, alternative="two-sided") if len(x_perturb) >= 2 and len(x_control) >= 2 else (np.nan, np.nan)
'''))
        cells += get_cells(CORE, ["core-34-plot"])
        cells += response_plot_cells()
    elif stage == "response":
        cells += response_plot_cells()
    elif stage == "lrko":
        cells += get_cells(CORE, ["core-50-terms"])
        cells.append(code(f'''mi17_lrko_outdir = base_run_dir / {LRKO!r}
mi17_lrko_go_change_long = pd.read_csv(mi17_lrko_outdir / "MI17_LRKO_sender_receiver_GO_module_score_change_long.csv")
'''))
        cells += get_cells(CORE, ["core-53"])
    elif stage == "ccc":
        cells += get_cells(CORE, ["core-59-terms"])
        cells.append(code(f'''BENCHMARK_OUTDIR = base_run_dir / {CCC!r}
smd_max_summary = pd.read_csv(BENCHMARK_OUTDIR / "GO_program_SMD_axisMatch_perturb_vs_control_summary_targetPerturbEdges.csv")
'''))
        cells += get_cells(CORE, ["core-67"])
    return cells


def response_plot_cells():
    cells = [code("from SpiderNet.visualization import apply_publication_style")]
    cells.append(code(f'''tcell_deg_outdir = base_run_dir / {DEG!r}
tcell_deg_outdir.mkdir(parents=True, exist_ok=True)
Tcell_DEG_LOG2FC_THRESHOLD = 0.4
Tcell_DEG_PVALUE_THRESHOLD = 0.05
tcell_deg_all_df = pd.read_csv(tcell_deg_outdir / "MI17_Tcell_DEG_all_perturb_genes.csv")
loading_plot_df = pd.read_csv(tcell_deg_outdir / "MI17_receiver_loading_sum_normalized_with_ALL_Tcell_DEG_stats.csv")
'''))
    cells += get_cells(CORE, ["core-37"])
    cells.append(code(f'''gsea_outdir = base_run_dir / {GSEA!r}
gsea_rank_df = pd.read_csv(gsea_outdir / "MI17_receiver_loading_ranked_GSEA_table.csv")
gsea_summary_df = pd.read_csv(gsea_outdir / "MI17_receiver_loading_GSEA_summary.csv")
gsaved = gsea_summary_df.iloc[0]
loading_value_col = str(gsaved["RankingMetric"])
n_total_genes = int(gsaved["N_ranked_genes"])
n_hits = int(gsaved["N_gene_set_hits"])
observed_es, nes, pval = float(gsaved["ES"]), float(gsaved["NES"]), float(gsaved["Pvalue"])
es_peak_index = int(gsaved["PeakRank"]) - 1
running_es = gsea_rank_df["RunningES"].to_numpy(dtype=float)
ranked_metric = gsea_rank_df[loading_value_col].to_numpy(dtype=float)
hit_mask = gsea_rank_df["IsHit"].astype(bool).to_numpy()
'''))
    # The null-distribution diagnostic has no saved null vector; keep it in full analysis.
    gsea_cell = get_cells(CORE, ["core-40-plot"])[0]
    # Locate the final diagnostic block, leaving the main enrichment figure untouched.
    source = gsea_cell.source
    boundary = source.rfind('\n# ---', 0, source.index('fig, ax = plt.subplots(figsize=(3.0, 2.4))'))
    gsea_cell.source = source[:boundary]
    cells.append(gsea_cell)
    return cells


def full_cells(stage):
    if stage == "preprocess":
        return get_cells(PREPROCESS)
    if stage == "train":
        return get_cells(TRAIN)
    if stage in ["loo", "train-loo"]:
        tags = ["loo-2", "loo-4", "loo-6", "loo-8", "loo-10"]
        tags += ["loo-12", "loo-14"] if stage == "train-loo" else ["loo-16", "loo-18", "loo-20", "loo-22", "loo-24", "loo-25", "loo-26"]
        return get_cells(LOO, tags)
    if stage == "core":
        return [c for c in get_cells(CORE) if int(c.metadata.tags[0].split('-')[1]) <= 47 and c.metadata.tags[0] != "core-43"]
    if stage == "enrichment":
        return [basic_setup()] + get_cells(CORE, ["core-43"])
    if stage == "lrko":
        return get_cells(CORE, ["core-2", "core-4", "core-6", "core-8", "core-20", "core-49", "core-50-terms", "core-50-compute", "core-51", "core-52", "core-53"])
    if stage == "ccc":
        return get_cells(CORE, ["core-56", "core-57", "core-58", "core-59-terms", "core-59-compute", "core-60", "core-61", "core-62", "core-64", "core-65", "core-66", "core-67"])
    raise ValueError(stage)


def combined_analysis_cells():
    """Keep the original full-notebook model/state shared through perturbation and LR-KO."""
    cells = full_cells("core")
    cells += get_cells(CORE, ["core-43", "core-49", "core-50-terms", "core-50-compute", "core-51", "core-52", "core-53"])
    cells += full_cells("ccc")
    return cells


def rscript_path():
    configured = os.environ.get("SPIDERNET_RSCRIPT") or shutil.which("Rscript")
    candidates = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob("R/R-*/bin/Rscript.exe"))
    return configured or (str(candidates[-1]) if candidates else None)


def run_go():
    from go_enrichment import validate_saved_enrichment
    validate_saved_enrichment(lambda name: OUTPUT_DIR / name, RESULTS_DIR / "Model/model_epoch19999.pth")
    # Execute the original R chunks in order. No language translation or plot reimplementation.
    text = (HERE / "PerturbFISH_GO_enrichment.Rmd").read_text(encoding="utf-8")
    chunks = re.findall(r"```\{r[^\n]*\}\s*\n(.*?)```", text, re.S)
    script = OUTPUT_DIR / "executed/PerturbFISH_GO_enrichment.R"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("grDevices::pdf(NULL)\n\n" + "\n\n".join(chunks), encoding="utf-8")
    env = os.environ.copy()
    env["SPIDERNET_PERTURBFISH_RESULTS_DIR"] = str(OUTPUT_DIR)
    env["SPIDERNET_PERTURBFISH_PLOT_DIR"] = str(OUTPUT_DIR)
    with script.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run([rscript_path(), str(script)], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("GO paired panel generated with all manuscript terms and verified checkpoint provenance.", flush=True)


def execute(cells, label):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb = nbformat.v4.new_notebook(cells=[code("import sys\nsys.path.insert(0, " + repr(str(HERE)) + ")")] + cells)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    dest = OUTPUT_DIR / "executed" / (label + ".ipynb")
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, km=km, timeout=None, resources={"metadata":{"path":str(HERE)}})
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(nb, dest)
    print("Executed:", dest, flush=True)


def check(stages, plot_only):
    missing, warnings = [], []
    def need(path):
        if not path.is_file(): missing.append(str(path))
    modules = {"nbformat", "nbclient", "ipykernel", "numpy", "pandas", "matplotlib", "seaborn", "scipy"}
    if not plot_only or "core" in stages:
        modules.update(["torch", "torch_geometric", "torch_scatter", "scanpy", "SpiderNet", "statsmodels"])
    if "overview" in stages: modules.add("anndata")
    if "response" in stages: modules.add("SpiderNet")
    if any(s in stages for s in ["enrichment", "lrko", "ccc"]) and not plot_only:
        modules.add("gseapy")
    for module in sorted(modules):
        if importlib.util.find_spec(module) is None: missing.append("Python module: " + module)
    for stage_index, stage in enumerate(stages):
        if stage == "overview": need(PROCESSED_ROOT / "adata_all.h5ad")
        if plot_only or (stage == "go" and "enrichment" not in stages[:stage_index]):
            for f in plot_inputs(stage): need(saved_result(f))
        if stage == "go" and (plot_only or "enrichment" not in stages[:stage_index]):
            from go_enrichment import validate_saved_enrichment
            try:
                validate_saved_enrichment(saved_result, RESULTS_DIR / "Model/model_epoch19999.pth")
            except (ValueError, KeyError, OSError) as error:
                missing.append(str(error))
        if stage == "core" and plot_only:
            for f in ["adata_list.pkl", "SpiderNet_data_pyg_list.pkl"]: need(PROCESSED_ROOT / f)
        if not plot_only:
            if stage == "preprocess":
                if not list((DATA_ROOT / "adata").glob("*.h5ad")):
                    missing.append(str(DATA_ROOT / "adata/*.h5ad"))
            if stage == "enrichment":
                for f in ["SpiderNet_data_pyg_list.pkl", "LR_list.pkl", "genenames_train.pkl"]:
                    need(PROCESSED_ROOT / f)
            if stage in ["train", "train-loo", "loo", "core", "lrko", "ccc"]:
                for f in ["adata_all.h5ad", "adata_list.pkl", "SpiderNet_data_pyg_list.pkl", "LR_list.pkl", "genenames_train.pkl", "batch_cell.pkl", "batch_cell_unique.pkl"]:
                    need(PROCESSED_ROOT / f)
                if stage in ["core", "lrko"]:
                    for f in ["LR_list_cellchatdb.pkl", "LR_meta_cellchatdb.pkl"]:
                        need(PROCESSED_ROOT / f)
            if stage in ["core", "lrko", "enrichment"]: need(RESULTS_DIR / "Model/model_epoch19999.pth")
            if stage in ["core", "lrko", "ccc"]:
                for f in ["loading_receiver_use.csv", "loading_sender_use.csv", "loading_LR_use.npy", "Factor_envir_use.npy"]: need(RESULTS_DIR / f)
            if stage == "ccc":
                need(RESULTS_DIR / "Factor_envir_list.pkl")
                if not (SCCCHAIN_RESULTS / "ScCChain_edge_scores_aligned.csv").is_file() and not list(SCCCHAIN_RESULTS.glob("*/*_ScCChain_edge_program_scores.csv")):
                    missing.append("ScCChain scores: " + str(SCCCHAIN_RESULTS))
                if importlib.util.find_spec("commot") is None and not cache_path(CCC + "/COMMOT/COMMOT_edge_scores.csv").is_file():
                    warnings.append("COMMOT unavailable: original notebook skips this optional method.")
                warnings.append("ScCChain fitting runner is absent; existing ScCChain Results are required upstream.")
            if stage in ["loo", "train-loo"]:
                for gene in GENES:
                    for f in ["adata_all.h5ad", "adata_list.pkl", "SpiderNet_data_pyg_list.pkl", "LR_list.pkl", "genenames_train.pkl"]:
                        need(PROCESSED_ROOT / ("Leaveoneout_" + gene) / f)
                    if stage == "loo":
                        need(RESULTS_DIR / f"Leaveoneout_{gene}/Model/model_epoch19999.pth")
                        need(RESULTS_DIR / f"Leaveoneout_{gene}/Linearbaseline/LinearRegression_model_Tcell_geneexp_prediction_leaveoneout_{gene}.joblib")
                if stage == "loo": need(saved_result("celcomen_summary.csv"))
    if "go" in stages:
        rscript = rscript_path()
        if not rscript: missing.append("Rscript (set SPIDERNET_RSCRIPT)")
        else:
            cmd = "p<-c('ggplot2','dplyr','stringr','scales','knitr'); m<-p[!vapply(p,requireNamespace,logical(1),quietly=TRUE)]; if(length(m)) {cat(paste(m,collapse=', ')); quit(status=1)}"
            result = subprocess.run([rscript, "-e", cmd], capture_output=True, text=True)
            if result.returncode: missing.append("R packages: " + result.stdout.strip())
    print("Python:", sys.executable)
    print("Processed:", PROCESSED_ROOT)
    print("Upstream results:", RESULTS_DIR)
    print("Output:", OUTPUT_DIR)
    print("Mode:", "plot-only" if plot_only else "analysis from upstream data/models")
    print("Stages:", ", ".join(stages))
    for warning in dict.fromkeys(warnings): print("NOTE:", warning)
    for item in dict.fromkeys(missing): print("MISSING:", item)
    print("Input check:", "FAILED" if missing else "PASSED")
    return not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check inputs without running analysis; combine with --plot-only.")
    parser.add_argument("--plot-only", action="store_true", help="Replay main figure code from saved results; no fitting, inference or network requests.")
    parser.add_argument("--stage", action="append", choices=ALL_STAGES, help="Run selected stages in the supplied order; repeat this flag for multiple stages.")
    args = parser.parse_args()
    stages = args.stage or (PLOT_STAGES if args.plot_only else DEFAULT_STAGES)
    if args.plot_only and any(s not in [*PLOT_STAGES, "response"] for s in stages):
        parser.error("--plot-only supports overview, loo, core, response, lrko, ccc and go")
    if "response" in stages and not args.plot_only:
        parser.error("response is a saved-result plotting stage; add --plot-only, or use --stage core to recompute DEG/GSEA")
    if not check(stages, args.plot_only): return 2
    if args.check: return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = {"started":stamp, "python":sys.executable, "mode":"plot-only" if args.plot_only else "full", "requested_stages":stages, "completed_stages":[], "inputs":[]}
    manifest = OUTPUT_DIR / "executed" / ("run_" + stamp + ".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    plan = [[stage] for stage in stages]
    if not args.stage and not args.plot_only:
        plan = [["overview"], ["loo"], ["core", "enrichment", "lrko", "ccc"], ["go"]]
    try:
        for group in plan:
            if len(group) > 1:
                print("Running", ", ".join(group), "with shared model/state", flush=True)
                execute(combined_analysis_cells(), "perturbation_analysis")
                from paper_panels import publish_response_panels
                publish_response_panels(OUTPUT_DIR)
                run["completed_stages"].extend(group)
                continue
            stage = group[0]
            print("Running", stage, flush=True)
            if args.plot_only or stage == "go": run["inputs"] += copy_inputs(plot_inputs(stage))
            elif stage == "loo": run["inputs"] += copy_inputs(["celcomen_summary.csv"])
            if stage == "overview":
                from paper_panels import spatial_overview
                spatial_overview(PROCESSED_ROOT / "adata_all.h5ad", OUTPUT_DIR)
            elif stage == "go": run_go()
            else: execute(plot_cells(stage) if args.plot_only else full_cells(stage), stage + ("_plot" if args.plot_only else "_analysis"))
            if stage in {"core", "response"}:
                from paper_panels import publish_response_panels
                publish_response_panels(OUTPUT_DIR)
            if stage == "preprocess":
                for source in PROCESSED_ROOT.rglob("*"):
                    relative = source.relative_to(PROCESSED_ROOT)
                    if source.is_file() and not any(part.startswith("Leaveoneout_") for part in relative.parts) and source.suffix.lower() in {".csv", ".png", ".pdf", ".json"}:
                        target = OUTPUT_DIR / "preprocess" / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            run["completed_stages"].append(stage)
    except Exception as error:
        run["error"] = repr(error)
        raise
    finally:
        manifest.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
