# PerturbFISH

## Release data

Restore the `plot-perturbfish` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Analyze T-cell responses to melanoma perturbation, MI-17 molecular programs, cell communication and ligand-receptor perturbations.

## Usage

Run from this directory in `SpiderNet_env`:

```powershell
python run_benchmarks.py --check
python run_benchmarks.py --check --plot-only
python run_benchmarks.py --plot-only
python run_benchmarks.py
```

The default workflow runs `overview`, `loo`, `core`, `enrichment`, `lrko`, `ccc`, and `go`. `paper_panels.py` renders the overview; the leave-one-out and spatial-perturbation notebooks provide the main analyses; `go_enrichment.py` and `PerturbFISH_GO_enrichment.Rmd` provide GO analysis/plots. Repeat `--stage` to select execution order. `preprocess`, `train`, and `train-loo` are explicit optional stages. `--plot-only --stage overview --stage go --stage response` renders selected panels; `response` performs neither model inference nor GSEA permutations.

Full runs use upstream data/checkpoints/baselines and recalculate downstream analyses, including online GO requests. Core and LR-KO share the model; NMF/COMMOT reuse existing files. GO uses epoch19999 loading, reusing existing MI activity during full execution; standalone enrichment infers once. Plot-only uses local tables or copies upstream tables without inference, network requests or permutations. Core spatial plots still read processed graphs/coordinates; overview reads coordinates/labels from `adata_all.h5ad`. The full S14 DEG/GSEA chain belongs to `core`; diagnostics without saved null vectors require full analysis.

## Inputs and outputs

Roots are found through an ancestor containing `Data/` and `Results/` (locally `D:/SpiderNet`). Overrides:

| Variable | Input or output |
|---|---|
| `SPIDERNET_WORKSPACE_ROOT` | Workspace |
| `SPIDERNET_PERTURBFISH_DATA_ROOT` | `Data/PerturbFISH`, including `adata/*.h5ad` |
| `SPIDERNET_PERTURBFISH_OUTPUT_ROOT` | Upstream `Results/PerturbFISH` |
| `SPIDERNET_PERTURBFISH_PROCESSED_ROOT` | ProcessedData and 11 `Leaveoneout_<gene>` bundles |
| `SPIDERNET_PERTURBFISH_RESULTS_DIR` | V1/dim23, epoch19999, loadings, linear joblib and Celcomen CSV |
| `SPIDERNET_PERTURBFISH_SCCCHAIN_ROOT` | `Version_V1/dim_envir_23/Baseline_CCC_MI17_GO_SMD_melanoma_to_Tcell/ScCChain` |
| `SPIDERNET_PERTURBFISH_PLOT_DIR` / `SPIDERNET_RSCRIPT` | Local output / Rscript executable |

Figures and summary CSVs are under `output/`, with manuscript panels under `output/paper_figures/`. S14 panel files are byte copies of the corresponding PDF/PNG. GO inputs, selected genes and checkpoint/input hashes use `output/GO_MI17_inputs/`; mismatches require enrichment again. Execution copies and provenance use `output/executed/`. Explicit training writes `output/Model` or `output/Leaveoneout_*`; selecting those models requires setting RESULTS_DIR to that output. Preprocessing bundles remain upstream, with plots under `output/preprocess`.

Python needs SpiderNet, PyTorch/PyG/torch-scatter, scanpy/anndata, gseapy, scipy, statsmodels, scikit-learn, pandas, matplotlib, seaborn, nbclient and ipykernel. GO plots need R packages ggplot2, dplyr, stringr, scales and knitr. No other Tutorial directory is required. ScCChain runner, Celcomen fitting and leave-one-out split generation are not supplied; provide their upstream results. COMMOT is optional.

Multilabel overview cells retain all targets as color sectors; gray includes unassigned labels. GO bands indicate manuscript groups, not statistical magnitude. Model provenance and remaining scientific limitations are documented in validation_report.md (`output/validation_report.md`, supplied in `plot-perturbfish`).
