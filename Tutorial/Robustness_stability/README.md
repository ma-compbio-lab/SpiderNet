# MI robustness and stability

## Release data

Restore the `plot-robustness-stability` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Reproduce SpiderNet MI dimensionality, neighborhood, label-error, random-seed and slice-subsampling analyses (Supplementary Figs. S31–S32).

Use the existing `SpiderNet_env` environment (Python 3.11; NumPy, pandas, SciPy, Matplotlib/seaborn, PyTorch/PyG, AnnData and the local SpiderNet package). No new dependencies are needed.

```powershell
conda activate SpiderNet_env
cd D:\SpiderNet\SpiderNet_proj\SpiderNet_Project\Tutorial\Robustness_stability
python run_benchmarks.py --check
python run_benchmarks.py --plot-only
python run_benchmarks.py
```

If conda activation is unavailable, replace `python` with `& 'E:\ANACONDA\envs\SpiderNet_env\python.exe'` in PowerShell.

`--check` checks imports and required upstream files; `--check --plot-only` checks saved plotting inputs. It does not load large graphs or certify every cached fit.
`--plot-only` reads local `output/tables/`, rebuilds the original plots and rank-sum table, and never fits models or loads processed graphs.
The default run performs all analyses from upstream processed data/reference outputs, reuses compatible results, and trains missing fits for **20,000 epochs**. It can take hours if fits are absent.
`--cached` retains the old notebook's no-training analysis mode; missing results cause a nonzero exit.

| Order | Analysis entry in `robustness_analysis.py` / `--stage` | Paper panels |
|---|---|---|
| 1 | `prepare_lr_panel` / `lr` | S31a: four LR component heatmaps |
| 2 | `dimension_sweep` / `dimension` | S31b: HGSOC M=12/15/18; PerturbFISH M=20/23/26 |
| 3 | `k_sweep` / `neighborhood` | S31c: HGSOC K=5/8/10 |
| 4 | `label_experiment` / `labels` | S32a: 10%/20%/30% label errors |
| 5 | `stability_experiment` / `seeds` | S32b: ten full-data seeds |
| 6 | `stability_experiment` / `subsampling` | S32c: ten 34-of-48-slice fits |

`robustness_plots.py` renders all panels. For a subset: `python run_benchmarks.py --stage labels seeds --cached`. Combined figures require all constituent stages in the same run; unavailable combined panels are reported as pending.

Outputs: `output/figures/` contains `Figure_S31`, `Figure_S31bc`, `Figure_S32` and individual panels, each as PNG/PDF; `output/tables/` contains correlations, alignments, repeat means, label/slice audits and LR components. Names are unchanged. Provenance and validation are in `output/manifests/` and `output/validation/`. No notebook execution is required.

Paths default to workspace `D:/SpiderNet`; override with `--workspace`, `--results-root`, `--package-root`, `--work-dir` or `--output-dir`.

Required upstream inputs under `Results/`:

- `HGSOC/ProcessedData` and `PerturbFISH/ProcessedData`: graph pickle/PT, genes, retained LR pairs and slice IDs; LR analysis also uses processed AnnData and bundle metadata.
- `AgingBrain/ProcessedData`, `Pancancer/ProcessedData_entire`: processed AnnData, LR list, bundle metadata, saved LR selection (or slice IDs for recomputation).
- References: `HGSOC/V1/SpiderNet_Result_dim15`, `PerturbFISH/V1/SpiderNet_Result_dim23`: factors and LR/sender/receiver loadings; these V1 references are fixed.
- Reusable fits: `HGSOC/MI_stability/HGSOC_MI_stability_dim15_V1`; fitting/scratch caches remain in `MI_robustness_S31_S32/{runs,cache}`. Fingerprints, seeds, checkpoint selection and cache rules are unchanged. Partial training restarts unless the final checkpoint exists.

The shared package defaults to `SpiderNet_proj/SpiderNet_Project/SpiderNet`. No other Tutorial directory is imported or executed. Large upstream data/models are not copied. The four small LR matrices and final CSV/PNG/PDF files are shareable; regenerable common-edge NPZ audits are ignored by Git.

The detailed scientific-discrepancy and validation report is not included in this folder. Source: `SpiderNet (39).pdf`, PDF pp. 21–22, 43, 77–78.
