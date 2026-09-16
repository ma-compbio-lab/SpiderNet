# SpiderNet component ablation

## Release data

Restore the `plot-ablation-study` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Compare the full model and three component ablations for regulatory coupling and held-out perturbation response prediction (Supplementary Fig. S30).

## Usage

Run from this directory in the existing SpiderNet environment (locally `E:\ANACONDA\envs\SpiderNet_env`):

```powershell
python run_benchmarks.py --check
python run_benchmarks.py --plot-only
python run_benchmarks.py
```

The default order is `hgsoc`, `aging`, then `perturbfish`; select one with `--stage`. `--check --plot-only` checks saved plotting inputs. Checks do not load large pickles or establish numerical reproduction.

`regulatory_coupling.py` loads processed AnnData/PyG and full-model factors, loads or trains the three ablations, and computes coupling and summaries. `perturbation_prediction.py` calculates Spearman correlations for 11 perturbations from existing observed/predicted LFC. PerturbFISH training and counterfactual LFC generation are upstream. Plot-only uses local tables without fitting or recalculating correlations. Full computation selects CUDA/CPU automatically; floating-point aggregation is not guaranteed to be bitwise identical.

## Inputs and outputs

`ablation_settings.py` defines paths and parameters. The workspace is found through an ancestor `Results/` directory; override with `SPIDERNET_WORKSPACE_DIR`. `SPIDERNET_PROJECT_DIR` selects the package directory containing `SpiderNet/model.py`; `SPIDERNET_MODEL_PY` supplies a fallback model path.

Required inputs include processed `{adata_list,SpiderNet_data_pyg_list,LR_list}.pkl` (optional `adata_all.h5ad`, otherwise concatenation), full-model factors and `targets_by_LR.json` from HGSOC V1/dim15 and AgingBrain V1/dim30, and `Data/Database/OmnipathR/interactions_regulatory_{human,mouse}.csv`. HGSOC prefers valid `run_dirs.json`. Ablation models use `Component_ablation/*_reinit/` and initialization pickles. PerturbFISH requires both `Cached_Tcell_LFC_*seed10042/` CSV sets and `Correlation_Matrix_SpiderNet_Component_Ablations.csv` under V1/dim23.

Outputs are PNG/PDF/CSV under `output/HGSOC/Benchmark1_curated_gene_coupling/`, `output/AgingMousebrain/Benchmark1_curated_gene_coupling/`, and `output/PerturbFISH_component_ablation/`.

Plotting needs NumPy, pandas, SciPy and Matplotlib. Full coupling also needs Scanpy, scikit-learn, PyTorch/PyG, torch-scatter and SpiderNet. No other Tutorial directory is required.

## Limitations

Missing or incompatible ablations train with `RETRAIN_ABLATIONS=False`, 2000 warmup/20000 epochs and seed 123; the full model is not retrained. Coupling uses neighboring-edge means although manuscript A.3 describes sums. PerturbFISH plots use the canonical matrix and save recomputed LFC correlations and differences separately. The no-intrinsic initialization uses `MiniBatchNMF(n_init=1)`, incompatible with the local sklearn version; valid cached results remain usable. Cache validation does not check `source_notebook`; check the initialization environment before running without caches.
