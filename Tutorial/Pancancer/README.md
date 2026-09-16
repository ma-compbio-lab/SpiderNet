# Pancancer

## Release data

Restore the `plot-pancancer` profile and activate this checkout's paths before running the commands below. See the [data guide](../../docs/DATA.md) and [reproduction instructions](../../docs/REPRODUCIBILITY.md). The default author-machine paths documented below are overridden by the supplied activation script or `scripts/run_study.py`.

Spatial meta-interactions, fibroblast-to-tumor signaling, MI cascades, and tumor-only bulk projection for the SpiderNet pan-cancer study.

Run from this folder in the **SpiderNet analysis Python environment**. Rscript is resolved from `--rscript`/`RSCRIPT`, PATH, `R_HOME`, or a single standard Windows R installation; multiple installations require an explicit choice. Fresh CCC baselines additionally require COMMOT and Julia with ScCChain. `--check` lists missing inputs/packages without fitting or downloading.

```bash
python run_benchmarks.py --check
python run_benchmarks.py --check --plot-only
python run_benchmarks.py --prepare-plot-inputs
python run_benchmarks.py --plot-only
python run_benchmarks.py
```

| Order / stage | Analysis entry | Paper panels |
|---|---|---|
| Optional `preprocess` | `spidernet_dataloading_MIdimselection_Pancancer.ipynb` | Pancancer part of S31a |
| Optional `train` | `Pancancer_modeltraining.ipynb` | Upstream 11-MI model |
| 1 `spatial` | `Pancancer_analysis_V2.ipynb` + `ScCChain_Pancancer_runner.jl` | 6a-d, S24, S25a, S26a |
| 2 `cascade` | `Pancancer_MIcascade_analysis_V3.ipynb` | 6f, S28, S29 |
| 3 `lrko` | `Pancancer_MI4_Fibroblast_to_Tumor_LRKO.ipynb` | S25b |
| 4 `projection` | `TCGA_MI_decomposition_V2_updated_tumor_only.R` | S26b; inputs for 6e/S27 |
| 5 `survival` | `Survival_analysis_KM_tertile_groups_smaller_censor_tumor_only.R` | 6e, S27a-b |
| 6 `icb` | `ICB_analysis_V2.R` | S27c |

The default full command runs stages 1-6 from existing upstream data/model results. It executes complete notebooks and R analyses with their existing cache policies; it does not retrain SpiderNet. To rerun an individual stage:

```bash
python run_benchmarks.py --stage spatial
python run_benchmarks.py --stage projection --r-stage pseudobulk
python run_benchmarks.py --stage survival --mi MI4
python run_benchmarks.py --stage survival --plot-only --mi MI4
```

`--mi MI4` is explicit because the preserved KM default is MI2/MI3/MI9. `projection --r-stage tcga` runs TCGA only; `survival --plot-only --r-stage nested-lrt` redraws the adjusted scatter. Optional upstream stages are run separately; `train` retains its original preprocessing default.

Final figures and summary CSVs are under **`output/{spatial,cascade,lrko,projection,survival,icb}/`**. Executed notebook copies are timestamped under `output/executed/`; source notebooks are never overwritten. Large model/graph intermediates remain upstream; newly written notebook figures/tables are collected with their original relative filenames. `output/run_status.json` records failures and incomplete plotting.

Plot-only redraws supported figures from saved numeric tables using the same plotting code, without model fitting. Before the first replay of legacy results, `--prepare-plot-inputs` exports missing spatial/cascade/KM drawing caches into the selected Results directory. It reads graph metadata without loading expression/LR tensors, reuses saved cascade normalization/permutations, and verifies the small original Cox refits against all 99 saved HR/P pairs. It does not download TCGA or rerun MI inference/projection. Preparation requires matching legacy results; it also accepts `--stage spatial`, `cascade`, or `survival` and `--check`.

Keeping `output/` does not create missing caches. Missing panels yield a PARTIAL/incomplete status and nonzero exit code after available plots are saved. Repeated warnings are summarized in `run_status.json`; non-interactive display warnings do not prevent export. R plotting reads complete local output caches first, then matching upstream caches (including the original nested-LRT table); `--cache-dir` explicitly selects another source. Drawing all 160 high-resolution spatial/cascade samples can take substantial time. See [REVIEW.md](REVIEW.md) for validation and scientific differences.

Required upstream inputs:

- SpiderNet package and its CellChat/scSeqComm databases; annotated `.h5ad` files with `celltype_final`, `subslice_id`, and `obsm['spatial']` for preprocessing.
- `Data/Pancancer` (including CancerSEA/EcoTyper references); `Results/Pancancer/ProcessedData_entire`; matching 11-MI checkpoint, factors, loadings, and analysis caches under `Results/Pancancer/V1/SpiderNet_Result_dim11`.
- TCGA GDC data/clinical records and Hugo/Gide/Jung ICB expression/response data for the R stages. Their package requirements are checked by each R entry.

`SPIDERNET_ROOT` locates shared `Data/` and `Results/` (auto-detected from ancestors, with the original `D:/SpiderNet` fallback). Override individual roots with `--data-dir`, `--processed-dir`, `--results-dir`, `--output-dir`; R additionally accepts `SPIDERNET_TCGA_GDC`, `SPIDERNET_ICB_DATA`, and `SPIDERNET_CANCERSEA`. No other Tutorial analysis folder is required.

S23 cell-type annotation UMAP/marker validation, raw annotation/sub-slice preparation, and final illustration/ROI composition are not implemented here. Existing scientific differences from the PDF are preserved and listed in [REVIEW.md](REVIEW.md).

