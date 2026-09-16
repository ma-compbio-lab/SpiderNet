# AgingBrain

## Release data

Restore the `plot-agingbrain` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Analyze MI-29 age associations, age prediction, transfer across brain regions, and fixed-model perturbations in ageing mouse brain.

## Usage

Run from this directory in the existing SpiderNet Python environment; notebooks use the calling interpreter. Rscript is also required (`RSCRIPT` can select it).

```powershell
python run_benchmarks.py --check
python run_benchmarks.py --plot-only
python run_benchmarks.py
```

The default order is `coronal`, `transfer`, `go`, `r-plots`, starting from processed data, model outputs and checkpoints without retraining SpiderNet. The coronal and transfer notebooks contain the analyses; the three Rmd files render circle, spatial and GO plots. Optional `preprocess`, `preprocess-transfer`, `train`, and `loso` stages must be selected explicitly. Training uses V1/30 MI/50,000 epochs; LOSO is separate from the manuscript's 10-fold analysis.

Repeat `--stage` to specify execution order. `--stage mi-summary` recomputes age summaries, regression coefficients and age scores; `--stage spatial --plot-only` redraws spatial maps. `--check --plot-only` checks plotting inputs. Continue GO/R plotting with `--stage go --stage r-plots`.

Coronal NMF-LR is refitted each run and reused within that run only when graph hashes and parameters match. Transfer loads existing caches or fits missing ones and selects the largest V1 checkpoint epoch. Plot-only uses saved results without fitting/inference; S20 is assembled from 18 saved panels, or redrawn when all cell-prediction tables are available. Spatial exports retain 1500 dpi and can take about ten minutes for ten PDFs. GO retries HTTP 429 at most three times with 30/60-second delays.

## Inputs and outputs

Default inputs are `D:/SpiderNet/Data/AgingBrain` and `D:/SpiderNet/Results/AgingBrain`: three ProcessedData bundles; coronal V1/dim30 factors and checkpoints; Banksy for each region; coronal/hippocampal COMMOT; coronal ScCChain; and gene panel Table S1. ScCChain `h5ad_files.txt` must identify the actual files. LR resources come from the installed package.

Configure roots through `SPIDERNET_WORKSPACE_ROOT`, `SPIDERNET_AGINGBRAIN_DATA_ROOT`, `SPIDERNET_AGINGBRAIN_RESULTS_ROOT`, and `SPIDERNET_AGINGBRAIN_OUTPUT`. Parameters remain in the notebooks. No other Tutorial directory is required.

Figures and tables are under `output/coronal/`, `output/sagittal/`, and `output/hippocampus/`; execution copies and logs use `output/executed/`, with `output/last_run.json`. Existing local results take precedence over the specified upstream Results; cache validity is not inferred automatically. Large preprocessing/training products remain upstream.

Python dependencies include SpiderNet, torch/torch_scatter, scanpy, numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, openpyxl, gseapy, pypdf, nbformat, nbclient and ipykernel. R uses ggplot2, dplyr, tidyr, readr, stringr, circlize, scales and knitr.

Fig. 5f/g comparisons use unpaired two-sided Wilcoxon rank-sum (Mann-Whitney U); tables are `output/coronal/Neighbor*_ranksum_tests.csv`. Schematics and ROI composition are manual. GO connection bands indicate manuscript groups, not statistical magnitude. See validation and limitations (`output/validation_report.md`, supplied in `plot-agingbrain`).
