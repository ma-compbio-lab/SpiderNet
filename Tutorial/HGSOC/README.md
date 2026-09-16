# HGSOC

## Release data

Restore the `plot-hgsoc` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Analyze malignant subtypes, CAF signaling, molecular concordance and MI cascades using HGSOC CosMx data and trained SpiderNet results.

## Usage

Run from this directory in the existing Python 3.11 environment with SpiderNet, PyTorch/PyG, Scanpy, GSEApy and nbclient:

```powershell
python run_benchmarks.py --check
python run_benchmarks.py --check --plot-only
python run_benchmarks.py --plot-only
python run_benchmarks.py
```

The default order is `model-summary`, `malignant`, `baseline`, `cascade`, `ccc`, `perturbation`. These use the training, subtype, baseline and cascade notebooks and existing trained results. Clustering, online KEGG enrichment, cascade permutations and perturbations can be expensive. NMF caching and checkpoint selection follow the notebook settings.

Plot-only uses saved scores, statistics and embeddings without fitting, rescoring or permutations. Missing KEGG annotation, sender-type aggregation or cascade heatmap matrices are reported, and existing figures are copied; spatial cascade figures are also copied. Full analyses save the small heatmap matrices. Copying and redrawing are recorded separately.

Select stages with repeated `--stage` flags, in execution order. Optional `preprocess` and `train` write upstream bundles/checkpoints. `--stage r-plots --plot-only` uses `MI_CAF_functionalstate_visualization_V2_SMD_annotation.Rmd` and requires R, ggplot2, dplyr, tibble, knitr, rmarkdown and Pandoc.

## Inputs and outputs

Default inputs are `D:/SpiderNet/Data/HGSOC/`, `D:/SpiderNet/Results/HGSOC/ProcessedData/`, and the run selected by `Results/HGSOC/run_dirs.json`: matching PyG/AnnData, factors/loadings, model configuration/checkpoint, CancerSEA, TIL and four-pathway KEGG references. Baselines require Banksy; CCC comparisons also require COMMOT/scCChain/Spacia. The four-pathway KEGG reference is required for scoring and is not replaced with new gene sets.

Paths are defined in `workflow_paths.py`. Overrides are `SPIDERNET_ROOT`, `HGSOC_DATA_ROOT`, `HGSOC_RESULTS_ROOT`, `HGSOC_PROCESSED_DATA`, `HGSOC_RUN_DIR`, and `HGSOC_OUTPUT`; explicit environment variables take precedence over `HGSOC_modeltraining_setup.json`. No other Tutorial directory is required.

Figures and CSVs use `output/V1/SpiderNet_Result_dim15/`; R outputs use `output/r-plots/`. Executed notebooks, HTML and run/copy records use `output/executed/`. Source notebooks are not overwritten.

The workflow supplies components of Fig. 3b-f and S6-S10, plus the five-method CCC comparison. Fig. 3a chord diagrams and final multipanel composition are not implemented here; this is not complete manuscript-figure reconstruction.
