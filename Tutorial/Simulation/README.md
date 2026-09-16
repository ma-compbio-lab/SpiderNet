# Simulation

## Release data

Restore the `plot-simulation` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Evaluate recovery of directional MI edges and molecular programs in simulated spatial transcriptomics, producing analysis figures and statistics associated with Fig. 2a-d, S3 and S4.

## Usage

Run from this directory in the existing SpiderNet environment with numpy, pandas, scipy, matplotlib, scikit-learn, scanpy, torch, nbformat, nbclient and ipykernel. Full spatial comparisons also need COMMOT and Julia with ScCChain/CSV/DataFrames.

```powershell
python run_benchmarks.py --check
python run_benchmarks.py --plot-only
python run_benchmarks.py
```

The default order is `summary`, `sample`, `insitu`: aggregate six settings by 30 repeats, calculate both loading-rank scopes from saved loadings/activity, plot a representative sample, then recompute its NMF-LR/COMMOT/ScCChain comparisons. The 180 SpiderNet models and full-grid comparator results are upstream inputs. Default sample selection is `--setting 0 --experiment 18` (Dropout1/Experiment_18); summary still uses all 180 repeats.

Plot-only reads saved summaries, rank tables/definitions/checksums, and spatial-score caches verified against input/source hashes. It neither fits nor merges upstream tables and fails without valid caches. First calculate rank tables with `--stage summary`. `--check --plot-only` checks plotting inputs. Notebooks use the calling interpreter and save executed copies under `output/executed/`.

Optional `--stage train` and `--stage generate` use `SpiderNet_Simulation.py` and `SimulationData_generation.py`. Without sample arguments they process all 180 samples and write upstream Results/Data; generation overwrites selected data. The generator uses `hash()` seeds, which are not guaranteed to match across processes.

## Inputs and outputs

Inputs are the five raw CSVs per sample in `D:/SpiderNet/Data/Simulation/<setting>/Experiment_<number>/`, plus `adata_simulation.h5ad` for spatial comparisons and `cell_neigh_metaIprop.csv` for training. `D:/SpiderNet/Results/Simulation/` supplies SpiderNet/COMMOT/NMF_LR metrics and ScCChain_analysis/Spacia_analysis `*_metrics.csv`. Rank summaries need gene/edge metadata, `ProcessedData/LR_list.pkl`, `loading_LR_use.npy`, `loading_sender_use.csv`, `loading_receiver_use.csv`, `Factor_envir_use.npy`, and `EdgeProgramScores.csv` for all samples. The representative sample also uses loading arrays/CSVs and processed model inputs.

Override roots with `--data-root`, `--result-root`, `--spacia-root`, `--output-root`, or `SIMULATION_DATA_ROOT`, `SIMULATION_RESULT_ROOT`, `SIMULATION_SPACIA_ROOT`. No other Tutorial directory is required; the Julia runner is local.

Outputs are under `output/`: `Merged_Benchmark/` for combined CSVs and rank definitions; `Merged_Benchmark/Summary_Plots/` for figures/statistics; `SpiderNet/<setting>/Experiment_<number>/SpiderNet_Result_Mode_cell_class/` for sample results; `Simulation_Data/` for diagnostics; and `ScCChain_analysis/` for Julia intermediates. Two diagnostics requiring unavailable latent expression are copied with provenance.

Sender/receiver ranks compare 10 versus 10 features within 20 genes and 10 versus 70 within all 80; LR uses 10 versus 10 in both. Ranks increase with loading and use average ties, column normalization and one-to-one activity-Spearman matching. Ratios are averaged across two MIs, then tested against 1 across 30 repeats using two-sided t tests. `_all80` identifies the second scope. `LoadingRankRatio_boxplot_by_setting_loading.pdf` is the separate share/top-1 measure, not the mean-rank ratio. Spatial comparisons retain `LOAD_COMPATIBLE_SPACIA=False`. See the 20-feature definition (`output/validation/rank_ratio_addition.md`, supplied in `plot-simulation`) and 80-gene definition (`output/validation/rank_ratio_all80_addition.md`, supplied in `plot-simulation`) for limitations.
