# SpiderNet-Interactive

Restore the `interactive` data profile before opening the packaged datasets; see the [data guide](../../docs/DATA.md). Application source and static assets are included in Git, while prepared data and cached results are downloaded separately.

A local Flask application for exploring trained SpiderNet spatial-omics runs. Its four modules provide MI visualization and enrichment, cell-subtype discovery, MI cascade analysis, and in silico spatial perturbation.

## Usage

Use the environment that trained the runs, with SpiderNet and PyTorch/PyG installed as described in the [package README](../README.md#installation). From this directory:

```bash
python -m pip install -r ../requirements-UI.txt
python -m pip check
python app.py
```

Open http://localhost:8000. `config.py` defines `PORT` and `SEARCH_ROOT`. Alternatively, `bash run.sh` starts the app and attempts to activate the `spidernet` conda environment. GO/KEGG displays require `gseapy`.

## Datasets

The app discovers `<Name>_UI/` folders under `../Interactivetool/SpiderNet-interactive_V2/` at startup. Restart after adding datasets. Each dataset needs:

```text
<Name>_UI/
  <Name>_modeltraining_setup.json
  config.json
  run_dirs.json
  ProcessedData/
  <VERSION>/SpiderNet_Result_dim*/
    Factor_envir_list.pkl
    loading_LR_use.npy
    loading_sender_use.npy
    loading_receiver_use.npy
    Model/
      SpiderNet_model_config.json
      model_epoch*.pth
```

The setup file specifies MI dimension, version and species; `config.json` identifies cell-type, sample-ID and spatial-coordinate fields. ProcessedData contains the matching AnnData/PyG bundles. `core/datasets.py` resolves dataset paths and `core/loaders.py` loads the bundles.

## Analysis and caching

- M1: spatial MI activity, LR/sender/receiver loadings, pathways and cell-type-pair enrichment.
- M2: MI-guided PCA, kNN, Louvain and UMAP, followed by DEGs and GO/KEGG.
- M3: permutation-tested MI cascades, cell-type triplets, spatial displays and DEG/GO analysis.
- M4: gene knockdown or cell-type replacement with the trained model, baseline comparisons and GO/KEGG.

Modules are under `modules/`; templates and browser assets are under `templates/` and `static/`. Dataset bundles are reused in memory. M2/M3 cache results by parameters. M4 also caches model state and per-slice baseline predictions at `<run_dir>/UI_Exports/module4_spatial_perturbation_web/baseline_predictions.npz`; its first request can be slow because it runs inference across slices.

M2 GO/KEGG requires internet access to Enrichr (`https://maayanlab.cloud`). Only the selected cluster and direction are queried, using one gene-list submission for both libraries. Successful cached enrichment and DEG results are reused. After a network error or HTTP 429 rate limit, wait a few minutes and click **Compute DEG + GO/KEGG** again. Failed enrichment entries from older caches are retried without rebuilding the expression cache or rerunning Wilcoxon tests. Errors are displayed as messages, not enrichment terms; the response records the library used for new results.

M4 GO/KEGG also requires Enrichr internet access. The DE table shows genes passing the absolute log2 fold-change and adjusted-P thresholds in both directions; enrichment additionally applies **Enrichment direction** and **Enrichment scope**. At least 3 selected genes are required. The summary reports eligible up/down counts, and the panels distinguish insufficient genes, empty responses and request failures. Changing these filters reuses the current DE results. After a network error, wait and click **Retry enrichment** to retry failed libraries without rerunning perturbation; successful library responses are cached. This run cache lasts until the application restarts.

Theme changes restyle the existing plots in the browser and preserve selections and zoom; they do not rerun clustering, permutation tests or perturbation. Cascade's Top N selector changes only the number of displayed cell-type triples. Invalid numeric inputs and overlapping replacement/replaced cell types are reported before submission.

Cascade GO uses the original `GO_Biological_Process_2021` library and configured species. Request failures are shown explicitly; clicking **Compute DEG + GO** again retries failed GO requests using the cached DEG results.

If no datasets appear, check the search root, `_UI` suffix and required files. If the port is occupied, select another `PORT` in `config.py`. The interface supports light/dark themes and reduced motion.

Part of SpiderNet, distributed under the MIT License.
