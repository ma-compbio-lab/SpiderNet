"""Execute the preserved AgingBrain notebooks in the invoking Python environment."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from workflow_paths import HERE, DATA_DIR, RESULTS_ROOT, OUTPUT, RUNS, saved

TRAIN = "AgingBrain_training_analysis.ipynb"
TRANSFER = "AgingBrain_transfer_Sagittal_Hippocampus_analysis.ipynb"
PREPROCESS = "spidernet_dataloading_MIdimselection_AgingBrain.ipynb"
PRETRANSFER = "spidernet_dataloading_MIdimselection_AgingBrain_Sagittal_Hippocampus.ipynb"
CCC = "Aging_Tcell_sender_CCC_method_SMD_comparison"
DEFAULT_STAGES = ["coronal", "transfer", "go", "r-plots"]
STAGES = ["preprocess", "preprocess-transfer", "train", "mi-summary", "spatial", *DEFAULT_STAGES, "loso"]
CELL_TYPES = ["Astrocyte", "B cell", "Endothelial", "Ependymal", "Macrophage", "Microglia",
              "Neuroblast", "Neuron-Excitatory", "Neuron-Inhibitory", "Neuron-MSN", "Neutrophil", "NSC",
              "Oligodendrocyte", "OPC", "Pericyte", "T cell", "VLMC", "VSMC"]
BUNDLE = ["adata_all.h5ad", "adata_list.pkl", "SpiderNet_data_pyg_list.pkl", "LR_list.pkl",
          "LR_list_all.pkl", "LR_list_cellchatdb.pkl", "LR_meta_cellchatdb.pkl", "batch_cell_unique.pkl",
          "batch_cell.pkl", "genenames.pkl", "genenames_train.pkl", "cellclass_unique.pkl"]
R_REPORTS = ["MI_circleplot_visualization.Rmd", "MIOI_visualization.Rmd", "MI_GOenrichment_analysis.Rmd"]


def notebook(filename):
    import nbformat
    return nbformat.read(HERE / filename, as_version=4)


def source(filename, tag):
    matches = [c.source for c in notebook(filename).cells if tag in c.metadata.get("tags", [])]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cell tagged {tag} in {filename}")
    return matches[0]


def cells(filename, numbers, prefix):
    return [source(filename, f"{prefix}-{i}") for i in numbers]


def definitions(text, names=None):
    """Reuse exact function text without executing its surrounding analysis."""
    return "\n\n".join(ast.get_source_segment(text, n) for n in ast.parse(text).body
                        if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
                        and (names is None or not isinstance(n, ast.FunctionDef) or n.name in names))


def basic_setup():
    return '''from pathlib import Path
import os, pickle, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display
from workflow_paths import DATA_DIR, RESULTS_ROOT, OUTPUT, RUNS, input_path, output_path, saved
plt.rcdefaults()
# Existing exports use the white axes style; make notebook state explicit.
sns.set_style("white")
DATA_ROOT = DATA_DIR
OUTPUT_ROOT = RESULTS_ROOT
PROCESSED_DATA_DIR = RESULTS_ROOT / "ProcessedData"
run_dirs = {"run_dir": RUNS["coronal"], "model_dir": RUNS["coronal"] / "Model"}
DIM_ENVIR = 30
CELL_TYPE_COL = "celltype"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "arial"
'''


def age_plot_style():
    # Preserve the state established before the original age/regression/group plots.
    return source(TRAIN, "train-56").split("NUM_CV_SPLITS =", 1)[0] + '\nplt.rcParams["font.family"] = "arial"\n'


def coronal_setup(with_model=False):
    result = cells(TRAIN, [2, 4, 6, 12, 16], "train")
    result += ['''results = {
    "factor_envir_list": pd.read_pickle(input_path(run_dirs["run_dir"] / "Factor_envir_list.pkl")),
    "factor_envir": np.load(input_path(run_dirs["run_dir"] / "Factor_envir_use.npy")),
}
''']
    if with_model:
        result += ['''from SpiderNet.api import build_model
processed_train = processed
train_run_dirs = run_dirs
MODEL_CHECKPOINT = None
HIDDEN_CHANNELS = 256
import re
''']
        result += cells(TRANSFER, [11, 12], "transfer")
    return result


def analysis_cells(stage):
    if stage in ("preprocess", "preprocess-transfer"):
        filename = PREPROCESS if stage == "preprocess" else PRETRANSFER
        return [c.source for c in notebook(filename).cells if c.cell_type == "code"]
    if stage == "train":
        return cells(TRAIN, [2, 4, 6, 7, 9, 10, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28], "train")
    if stage == "transfer":
        return [c.source for c in notebook(TRANSFER).cells if c.cell_type == "code"]
    if stage == "go":
        return [basic_setup(), '''from types import SimpleNamespace
genes = pd.read_csv(saved("coronal", "loading_receiver_use.csv"), index_col=0, nrows=0).columns
processed = SimpleNamespace(adata_list=[SimpleNamespace(var_names=genes)])
MI_OI = "MI29"
'''] + cells(TRAIN, [132, 142, 143, 144, 145, 147, 148, 150, 151], "train")
    if stage == "mi-summary":
        return [basic_setup()] + coronal_setup() + cells(TRAIN, [41, 42, 43, 48, 70], "train") + [age_plot_style()] + cells(TRAIN, [75, 93, 96, 97, 98, 99, 100], "train")
    if stage == "spatial":
        return [basic_setup()] + coronal_setup() + [age_plot_style()] + cells(TRAIN, [89, 90, 91], "train")
    if stage == "coronal":
        # Model outputs/checkpoint are upstream inputs. This stage never retrains SpiderNet.
        tags = [int(t.split("-")[1]) for c in notebook(TRAIN).cells if c.cell_type == "code"
                for t in c.metadata.get("tags", []) if t.startswith("train-") and 32 <= int(t.split("-")[1]) <= 140]
        # LOSO is separate from the manuscript's 10-fold benchmark. Its full code remains available.
        tags = [i for i in tags if not 62 <= i <= 68]
        return [basic_setup()] + coronal_setup(with_model=True) + cells(TRAIN, tags, "train")
    if stage == "loso":
        return coronal_setup() + cells(TRAIN, [48, 49, 50, 51, 53, 54], "train") + [
            source(TRAIN, "train-56").split("pearson_corr_dict = {}", 1)[0],
        ] + cells(TRAIN, [62, 63], "train") + [
            definitions(source(TRAIN, "train-60"), ["safe_div"]),
            # Reuse the same method palette as the original comparison cell.
            'method_colors = ' + ast.unparse(next(n.value for n in ast.parse(source(TRAIN, "train-60")).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "method_colors" for t in n.targets))),
        ] + cells(TRAIN, [64, 67, 68], "train")
    raise ValueError(stage)


def plot_cells(stage):
    if stage == "spatial":
        # Spatial maps use saved MI activities and coordinates; no inference or fitting.
        return analysis_cells(stage)
    result = [basic_setup()]
    if stage == "coronal":
        result += [age_plot_style()]
        result += ['pearson_corr_dict_df = pd.read_csv(saved("coronal", "Celltype_age_prediction_pearsonr_comparison_wide.csv"), index_col=0)']
        result += cells(TRAIN, [60], "train")
        # Metric rows are already computed, but ranking, layout and export remain the original code.
        bubble = source(TRAIN, "train-37")
        result += [basic_setup(), bubble.split('factor_path =', 1)[0], '''full_metric_df = pd.read_csv(saved("coronal", "MI_sender_receiver_pair_full_metrics_maxnorm_minEdges30.csv"))
mi_names = [f"MI-{i+1}" for i in range(DIM_ENVIR)]
''', 'if RANK_BY == "maxnorm":' + bubble.split('if RANK_BY == "maxnorm":', 1)[1]]
        result += cells(TRAIN, [39], "train")
        prediction_files = [f"AgePred_Hist2D_SpiderNet_{ct}_sqrtDensity_regLine_noColorbar_predictions.csv" for ct in CELL_TYPES]
        if all(saved("coronal", name).is_file() for name in prediction_files):
            result += [source(TRAIN, "train-56").split("pearson_corr_dict = {}", 1)[0]]
            for ct, filename in zip(CELL_TYPES, prediction_files):
                result += [f'''prediction = pd.read_csv(saved("coronal", {filename!r}))
y = prediction["TrueAge"].to_numpy()
y_pred_all = prediction["PredictedAge"].to_numpy()
pearson_corr, _ = compute_metrics(y, y_pred_all)
plot_hist2d_true_vs_pred(y, y_pred_all, {ct!r}, "SpiderNet", pearson_corr,
    str(RUNS["coronal"] / {filename.replace('_predictions.csv', '')!r}), bins=HIST2D_BINS,
    cmin=HIST2D_CMIN, use_sqrt_density=USE_SQRT_DENSITY, reg_line_ls=REG_LINE_LS,
    reg_line_lw=REG_LINE_LW, reg_line_color=REG_LINE_COLOR)
''']
        else:
            result += ['print("Age-prediction assembly reuses the 18 original PDF/PNG panels; cell-level 10-fold predictions were not previously saved.")']
        result += cells(TRAIN, [58], "train")
        result += cells(TRAIN, [103, 105], "train")
        result += [f'''from matplotlib import rcParams
selected_summary_df = pd.read_csv(saved("coronal", "{CCC}/AgingBrain_Tcell_sender_any_receiver_CCC_selected_dimension_SMD_summary.csv"))
method_order_use = [m for m in METHODS_TO_RUN if m in selected_summary_df["Method"].unique()]
'''] + cells(TRAIN, [112], "train")
        for tag, name, key, value, variable in [
            (125, "NeighborAgeChange_plotdata.csv", "Permuted_CellType", "Mean_Age_Change", "diff_mean_all_dict"),
            (126, "NeighborMIOIChange_plotdata.csv", "Permuted_CellType", "Mean_MIOI_Change", "MI_OI_change_mean_all_dict"),
            (140, "NeighborAgeChange_plotdata_LRknockout.csv", "Permuted_Type", "Mean_Age_Change", "diff_mean_all_dict"),
        ]:
            result += [f'''_plot_input = pd.read_csv(saved("coronal", {name!r}))
{variable} = {{k: g[{value!r}].to_numpy() for k, g in _plot_input.groupby({key!r}, sort=False)}}
'''] + cells(TRAIN, [tag], "train")
        # New plot tables are exported by mi-summary/full analysis. Old exports are retained explicitly otherwise.
        if saved("coronal", "MI_mean_by_age_maxnorm.csv").is_file():
            text = source(TRAIN, "train-41")
            plot = text[text.index("    ## ==========================="):]
            import textwrap
            result += [basic_setup(), '''from matplotlib.colors import LinearSegmentedColormap
MI_mean_pd_maxnorm = pd.read_csv(saved("coronal", "MI_mean_by_age_maxnorm.csv"), index_col=0)
age_values = MI_mean_pd_maxnorm.index.to_numpy(dtype=float)
type_show = "SpiderNet"
''', textwrap.dedent(plot)] + cells(TRAIN, [42, 43], "train")
        if saved("coronal", "AgePrediction_RegressionCoefficients.csv").is_file():
            result += [age_plot_style(), '''coef_df = pd.read_csv(saved("coronal", "AgePrediction_RegressionCoefficients.csv"), index_col=0)
pval_df = pd.read_csv(saved("coronal", "AgePrediction_RegressionPvalues.csv"), index_col=0)
'''] + cells(TRAIN, [75], "train")
        if saved("coronal", "Tcell_MI29_aging_plot_data.csv").is_file():
            result += ['mean_receiver_MI29_fromTcell_Aging_all = pd.read_csv(saved("coronal", "Tcell_MI29_aging_plot_data.csv"))']
            result += cells(TRAIN, [99, 100], "train")
        return result
    if stage == "transfer":
        # The original import cell also initializes the global plotting theme.
        style = source(TRANSFER, "transfer-4")
        result += [style[style.index('plt.rcParams["pdf.fonttype"]'):]]
        result += cells(TRANSFER, [2, 14], "transfer")
        for label, dataset, ymin, ymax in [("sagittal", "Sagittal", 0.4, 1.4), ("hippocampus", "Hippocampus", 0.35, 2)]:
            # The combined table is the original plot input. Per-method CSVs can be from later runs.
            result += [f'''ratio_df_combined = pd.read_csv(saved({label!r}, "{dataset}_Old_vs_Young_ratio_combined.csv"))
plot_ratio_comparison(ratio_df_combined, {dataset!r}, RUNS[{label!r}], y_min={ymin}, y_max={ymax})
''']
        sagittal = source(TRANSFER, "transfer-29")
        result += [sagittal.split('if "sagittal_results" not in globals():', 1)[0], '''sagittal_outdir = RUNS["sagittal"]
sagittal_tcell_mi29_aging_df = pd.read_csv(saved("sagittal", "Sagittal_Tcell_MI29_sum_group_Aging_module_score_table.csv"))
''', 'sagittal_tcell_aging_stats = _sagittal_plot_grouped_aging_score(' + sagittal.split('sagittal_tcell_aging_stats = _sagittal_plot_grouped_aging_score(', 1)[1]]
        return result
    if stage in ["loso", "mi-summary"]:
        if stage == "mi-summary":
            result += ['''from matplotlib.colors import LinearSegmentedColormap
MI_mean_pd_maxnorm = pd.read_csv(saved("coronal", "MI_mean_by_age_maxnorm.csv"), index_col=0)
age_values = MI_mean_pd_maxnorm.index.to_numpy(dtype=float)
type_show = "SpiderNet"
coef_df = pd.read_csv(saved("coronal", "AgePrediction_RegressionCoefficients.csv"), index_col=0)
pval_df = pd.read_csv(saved("coronal", "AgePrediction_RegressionPvalues.csv"), index_col=0)
mean_receiver_MI29_fromTcell_Aging_all = pd.read_csv(saved("coronal", "Tcell_MI29_aging_plot_data.csv"))
''']
            import textwrap
            text = source(TRAIN, "train-41")
            result += [textwrap.dedent(text[text.index("    ## ==========================="):])]
            return result + cells(TRAIN, [42, 43], "train") + [age_plot_style()] + cells(TRAIN, [75, 99, 100], "train")
        result += [source(TRAIN, "train-56").split("pearson_corr_dict = {}", 1)[0],
                   definitions(source(TRAIN, "train-60"), ["safe_div"]),
                   'method_colors = ' + ast.unparse(next(n.value for n in ast.parse(source(TRAIN, "train-60")).body
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "method_colors" for t in n.targets))),
                   '''loso_outdir = RUNS["coronal"] / "AgePrediction_LOSO"
pearson_corr_loso_df = pd.read_csv(input_path(loso_outdir / "Celltype_age_prediction_LOSO_pearsonr.csv"), index_col=0)
'''] + cells(TRAIN, [64, 67, 68], "train")
        return result
    raise ValueError(f"--plot-only does not apply to {stage}")


def plot_inputs(stage):
    if stage == "coronal":
        files = ["Celltype_age_prediction_pearsonr_comparison_wide.csv",
                 "MI_sender_receiver_pair_full_metrics_maxnorm_minEdges30.csv",
                 f"{CCC}/AgingBrain_Tcell_sender_any_receiver_CCC_selected_dimension_SMD_summary.csv",
                 "NeighborAgeChange_plotdata.csv", "NeighborMIOIChange_plotdata.csv", "NeighborAgeChange_plotdata_LRknockout.csv"]
        predictions = [f"AgePred_Hist2D_SpiderNet_{ct}_sqrtDensity_regLine_noColorbar_predictions.csv"
                       for ct in CELL_TYPES]
        files += predictions if all(saved("coronal", f).is_file() for f in predictions) else [
            f"AgePred_Hist2D_SpiderNet_{ct}_sqrtDensity_regLine_noColorbar{ext}"
            for ct in CELL_TYPES for ext in [".pdf", ".png"]]
        return [("coronal", f) for f in files] + plot_inputs("mi-summary")
    if stage == "transfer":
        return [(label, f"{ds}_Old_vs_Young_ratio_combined.csv")
                for label, ds in [("sagittal", "Sagittal"), ("hippocampus", "Hippocampus")]] + [
            ("sagittal", "Sagittal_Tcell_MI29_sum_group_Aging_module_score_table.csv")]
    if stage == "r-plots":
        return [("coronal", f) for f in ["cell_meta_all.csv", "MI_edge_index_all_choose.csv",
            "MI-29_all_links.csv", "MI-29_Tcell_links.csv", "GO_enrichment_toptarget_MI29.csv", "GO_enrichment_topregulator_MI29.csv"]]
    if stage == "loso":
        return [("coronal", "AgePrediction_LOSO/" + f) for f in ["Celltype_age_prediction_LOSO_pearsonr.csv", "Celltype_age_prediction_LOSO_cell_predictions.csv"]]
    if stage == "mi-summary":
        return [("coronal", f) for f in ["MI_mean_by_age_maxnorm.csv", "LR_loading_pathway.csv", "AgePrediction_RegressionCoefficients.csv", "AgePrediction_RegressionPvalues.csv", "Tcell_MI29_aging_plot_data.csv"]]
    return []


def rscript():
    configured = os.environ.get("RSCRIPT")
    found = configured or shutil.which("Rscript")
    if found:
        return found
    candidates = sorted(Path("C:/Program Files/R").glob("*/bin/Rscript.exe"))
    return str(candidates[-1]) if candidates else None


def check(stages, plot_only):
    missing = []
    checked = 0
    packages = {"nbformat", "nbclient", "ipykernel", "numpy", "pandas", "matplotlib", "seaborn", "scipy", "sklearn", "pypdf"}
    if not plot_only or "spatial" in stages:
        packages |= {"SpiderNet", "torch", "torch_scatter", "scanpy", "statsmodels", "openpyxl"}
    if "go" in stages:
        packages.add("gseapy")
    for package in sorted(packages):
        if importlib.util.find_spec(package) is None:
            missing.append("Python package: " + package)
    def need(path):
        nonlocal checked
        checked += 1
        if not Path(path).is_file():
            missing.append(str(path))
    for stage in stages:
        if (plot_only and stage != "spatial") or stage == "r-plots":
            for label, name in plot_inputs(stage):
                need(saved(label, name))
        elif stage in ["preprocess", "preprocess-transfer", "train", "coronal", "mi-summary", "spatial", "loso", "transfer"]:
            labels = ["coronal", "sagittal", "hippocampus"] if stage == "transfer" else ["coronal"]
            for label in labels:
                bundle = RESULTS_ROOT / ("ProcessedData" if label == "coronal" else f"AgingBrain_{label.capitalize()}/ProcessedData")
                if stage not in ["preprocess", "preprocess-transfer"]:
                    # Transfer bundles need only their preserved bundle contract.
                    files = BUNDLE if label == "coronal" else ["adata_all.h5ad", "adata_list.pkl", "SpiderNet_data_pyg_list.pkl", "LR_list.pkl", "LR_list_cellchatdb.pkl", "LR_meta_cellchatdb.pkl", "genenames_train.pkl", "batch_cell.pkl", "batch_cell_unique.pkl"]
                    for f in files:
                        need(bundle / f)
            if stage in ["coronal", "mi-summary", "spatial", "loso"]:
                for f in BUNDLE:
                    need(RUNS["coronal"] / f)
                for f in ["Factor_envir_list.pkl", "Factor_envir_use.npy", "loading_LR_use.npy", "loading_receiver_use.npy", "loading_sender_use.npy"]:
                    need(saved("coronal", f))
            if stage in ["coronal", "transfer"] and not list((RUNS["coronal"] / "Model").glob("model_epoch*.pth")):
                missing.append(str(RUNS["coronal"] / "Model/model_epoch*.pth"))
            if stage in ["coronal", "mi-summary", "transfer"]:
                need(DATA_DIR / "Supp_table/2023-12-22736D-TableS1_MERFISHGenePanel.xlsx")
            if stage in ["coronal", "loso", "transfer"]:
                for directory, pattern in [("COMMOT", "*.h5ad"), ("Banksy", "*.mtx"), ("Banksy", "*_genes.csv"), ("Banksy", "*_barcodes.csv")]:
                    if not list((RESULTS_ROOT / directory).glob(pattern)):
                        missing.append(str(RESULTS_ROOT / directory / pattern))
                if stage != "transfer":
                    need(RESULTS_ROOT / "ScCChain/h5ad_files.txt")
                import pickle
                age_file = RESULTS_ROOT / "ProcessedData/batch_cell_unique.pkl"
                if age_file.is_file():
                    with age_file.open("rb") as stream:
                        ages = pickle.load(stream)
                    for age in ages:
                        need(RESULTS_ROOT / "COMMOT" / f"aging_coronal_age{age}_commot.h5ad")
                        for suffix in [".mtx", "_genes.csv", "_barcodes.csv"]:
                            need(RESULTS_ROOT / "Banksy" / f"aging_coronal_age{age}_Banksy_cellidentity_lambda0.2{suffix}")
                listing = RESULTS_ROOT / "ScCChain/h5ad_files.txt"
                if stage != "transfer" and listing.is_file():
                    for line in listing.read_text(encoding="utf-8-sig").splitlines():
                        candidate = Path(line.strip())
                        if not candidate.is_file():
                            candidate = RESULTS_ROOT / "ScCChain" / line.strip()
                        need(candidate)
                        score = RESULTS_ROOT / "ScCChain" / f"{candidate.stem}_ScCChain_edge_program_scores.csv"
                        need(score if score.is_file() else candidate.parent / score.name)
                if stage == "transfer":
                    for label, prefix in [("Sagittal", "aging_saggital_age"), ("Hippocampus", "")]:
                        base = RESULTS_ROOT / f"AgingBrain_{label}"
                        listing = base / "ProcessedData/batch_cell_unique.pkl"
                        if listing.is_file():
                            with listing.open("rb") as stream:
                                samples = pickle.load(stream)
                            for sample in samples:
                                for suffix in [".mtx", "_genes.csv", "_barcodes.csv"]:
                                    need(base / "Banksy" / f"{prefix}{sample}_Banksy_cellidentity_lambda0.2{suffix}")
                                if label == "Hippocampus":
                                    need(base / "COMMOT" / f"{sample}_commot.h5ad")
            if stage in ["preprocess", "train"] and not list((DATA_DIR / "adata").glob("*.h5ad")):
                missing.append(str(DATA_DIR / "adata/*.h5ad"))
            if stage == "preprocess-transfer":
                need(RESULTS_ROOT / "ProcessedData/genenames_train.pkl")
                for label in ["Sagittal", "Hippocampus"]:
                    raw = DATA_DIR / f"AgingBrain_{label}" / "adata"
                    if not list(raw.glob("*.h5ad")):
                        missing.append(str(raw / "*.h5ad"))
        elif stage == "go":
            for f in ["loading_receiver_use.csv", "loading_receiver_use.npy", "loading_sender_use.npy"]:
                need(saved("coronal", f))
    if "r-plots" in stages:
        exe = rscript()
        if not exe:
            missing.append("Rscript (set RSCRIPT)")
        else:
            probe = subprocess.run([exe, "--vanilla", "-e", 'p<-c("ggplot2","dplyr","stringr","circlize","scales","knitr","tidyr","readr"); m<-p[!sapply(p,requireNamespace,quietly=TRUE)]; if(length(m)){cat(paste(m,collapse=", "));quit(status=1)}'], capture_output=True, text=True)
            if probe.returncode:
                missing.append("R packages: " + probe.stdout.strip())
    for name in [TRAIN, TRANSFER, PREPROCESS, PRETRANSFER]:
        content = json.loads((HERE / name).read_text(encoding="utf-8"))
        for i, cell in enumerate(content["cells"]):
            if cell["cell_type"] == "code":
                try:
                    ast.parse("".join(cell["source"]), filename=f"{name}:cell{i}")
                except SyntaxError as exc:
                    missing.append(str(exc))
    print(f"Checked {checked} file inputs and notebook syntax; Python: {sys.executable}")
    for item in missing:
        print("MISSING:", item)
    return missing


def execute(stage, sources, plot_only):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    class CurrentEnvironmentKernel(KernelManager):
        def format_kernel_cmd(self, extra_arguments=None):
            return [sys.executable, "-m", "ipykernel_launcher", "-f", self.connection_file] + (extra_arguments or [])
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources])
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Current Python", "language": "python"}
    destination = OUTPUT / "executed" / f"{stage}_{'plot' if plot_only else 'analysis'}_{datetime.now():%Y%m%d_%H%M%S_%f}.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", kernel_manager_class=CurrentEnvironmentKernel,
                            resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, destination)
        print("Executed copy:", destination, flush=True)


def copy_plot_inputs(stage):
    records = []
    for label, name in plot_inputs(stage):
        src = saved(label, name)
        dest = OUTPUT / label / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        records.append({"source": str(src), "output": str(dest), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    return records


def run_r():
    exe = rscript()
    env = os.environ.copy()
    env["SPIDERNET_AGINGBRAIN_RESULT_DIR"] = str(OUTPUT / "coronal")
    for name in R_REPORTS:
        text = (HERE / name).read_text(encoding="utf-8")
        chunks = re.findall(r"```\{r[^\n]*\}\n(.*?)```", text, re.S)
        script = OUTPUT / "executed" / f"{Path(name).stem}_{datetime.now():%Y%m%d_%H%M%S}.R"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("\n\n".join(chunks), encoding="utf-8")
        log = script.with_suffix(".log")
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run([exe, "--vanilla", str(script)], cwd=OUTPUT / "executed", env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("R output:", log, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only input/environment check; combine with --plot-only")
    parser.add_argument("--plot-only", action="store_true", help="Use saved numerical results; no fitting or inference")
    parser.add_argument("--stage", choices=STAGES, action="append", help="Repeat to select ordered stages")
    args = parser.parse_args()
    stages = args.stage or (["coronal", "spatial", "transfer", "r-plots"] if args.plot_only else DEFAULT_STAGES)
    if args.plot_only and any(s not in ["coronal", "mi-summary", "spatial", "transfer", "r-plots", "loso"] for s in stages):
        parser.error("plot-only supports coronal, mi-summary, spatial, transfer, r-plots and loso")
    if check(stages, args.plot_only):
        return 1
    if args.check:
        print("Input check passed. Full analysis also validates shapes, feature names and sample alignment at runtime.")
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    report = {"python": sys.executable, "started_at": started.isoformat(),
              "mode": "plot-only" if args.plot_only else "analysis", "stages": [], "inputs": [],
              "upstream": {"data": str(DATA_DIR), "results": str(RESULTS_ROOT)},
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                  for name in [TRAIN, TRANSFER, PREPROCESS, PRETRANSFER, *R_REPORTS,
                               "run_benchmarks.py", "workflow_paths.py"]}}
    for stage in stages:
        print("Starting", stage, flush=True)
        if args.plot_only or stage == "r-plots":
            report["inputs"] += copy_plot_inputs(stage)
        if stage == "r-plots":
            run_r()
        else:
            execute(stage, plot_cells(stage) if args.plot_only else analysis_cells(stage), args.plot_only)
        report["stages"].append(stage)
        (OUTPUT / "last_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (OUTPUT / "executed" / f"run_{started:%Y%m%d_%H%M%S_%f}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    print("Outputs:", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
