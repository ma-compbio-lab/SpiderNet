# Real-tissue benchmarks

## Release data

Restore the `plot-coupling-benchmark` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Regulatory coupling, directionality, cell similarity and spatial specificity in
HGSOC CosMx and ageing mouse brain MERFISH data.

## Run

Run from this folder in the original SpiderNet Python environment
(on this workstation: `E:/ANACONDA/envs/SpiderNet_env/python.exe`).

```powershell
python run_benchmarks.py --check       # Check required input paths
python run_benchmarks.py --plot-only   # Redraw figures from saved results
python run_benchmarks.py               # Recalculate analyses and draw figures
```

**No manual notebook execution is needed.** `--plot-only` skips model fitting
and inference. The full run uses existing SpiderNet/comparator results and
rebuilds NMF-LR once per study; add `--reuse-nmflr` to reuse matching cached factors.

Run one stage with `--stage coupling`, `--stage similarity` or `--stage spatial`.
For example: `python run_benchmarks.py --stage spatial --plot-only`.

## Workflow and paper figures

The runner executes these three stages in order. Figure numbering follows
`SpiderNet (39).pdf`; final manuscript panel assembly is separate.

| Stage | Source | Paper figures |
| --- | --- | --- |
| 1. Curated regulatory coupling, edge reversal and reciprocal-edge correlations | `CCC_Coupling_benchmark_with_directionality.ipynb` | Fig. 2e, S5, S6 |
| 2. Within-cell-type communication/expression similarity | `ccc_cell_similarity.py` | Fig. S7 |
| 3. True-neighbor versus distant-pseudo-edge MI activity | `Spatial specificity.ipynb` | Fig. S8 |

Keep all seven code files together. The runner and three analysis entries above
use `benchmark_config.py` for paths, `benchmark_common.py` for coupling helpers,
and `ccc_directionality_two_studies.py` for combined directionality displays.

## Outputs

| Location relative to this folder | Contents |
| --- | --- |
| `output/CCC_Coupling_benchmark_with_directionality/` | Coupling and reciprocal-edge figures: PDF + PNG |
| `output/cell_similarity/` | Combined similarity figure: PDF + PNG; per-study and combined CSVs |
| `output/executed/` | Spatial-specificity PDF and combined CSVs; executed notebook copies |

Original notebooks are not overwritten. Repeated runs update same-named outputs.
Per-study coupling and spatial intermediate tables remain under `Results/`.

## Required inputs

Use the original processed data, SpiderNet factors/checkpoints, COMMOT/scCChain
results, HGSOC Spacia results, and curated target/regulator annotations. These
upstream inputs are not generated or downloaded by this folder. Plot-only mode
requires their existing per-study result tables.

Default roots are `D:/SpiderNet/Data` and `D:/SpiderNet/Results`, resolved through
`benchmark_config.py`. Set `SPIDERNET_DATA_ROOT` and `SPIDERNET_RESULTS_ROOT` for
another installation; update paths inside `run_dirs.json` and scCChain manifests
when relocating data. Keep the original software versions and analysis settings.
