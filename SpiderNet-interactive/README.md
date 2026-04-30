# SpiderNet-Interactive

A local web app for interactively exploring SpiderNet spatial omics analyses. Four interconnected modules let you visualize learned meta-interactions, discover cell subtypes, trace communication cascades, and run *in silico* perturbations against the trained model — all without leaving the browser.

## Quick start

```bash
cd SpiderNet/SpiderNet-interactive
bash run.sh
```

Then open **http://localhost:8000**.

## Modules

| | | |
|---|---|---|
| **M1** | Basic analysis | *In situ* MI rendering, top LR / sender / receiver loadings per MI, pathway and cell-type-pair enrichment. |
| **M2** | Subtype discovery | MI-guided clustering of one cell type — PCA → kNN → Louvain → UMAP — with DEGs and GO/KEGG per subcluster. |
| **M3** | MI cascade | Permutation-tested MI×MI cascade pairs, cell-type triplet stems, *in situ* cascade rendering, and DEG/GO at each cascade position. |
| **M4** | Spatial perturbation | *In silico* gene knockdown or cell-type replacement against the trained SpiderNet model, paired DEG vs. baseline, GO/KEGG enrichment. |

Each module is a Flask blueprint with its own routes, service layer (compute + caching), template, and JS controller. Heavy results (model state, baseline predictions, run outputs, enrichment tables) are cached in-process and on disk so re-renders with new thresholds don't re-run the analysis.

## Interface

The UI is editorial-scientific in tone — quiet, typographic, designed to disappear so the data can speak.

- **Type pairing**: Source Serif 4 (display, italic optical-size variants for emphasis), Geist (UI), Geist Mono (code and identifiers). All from Google Fonts.
- **Color**: a single deep-cobalt accent paired with a warm terracotta "ember" used sparingly for headlines, the active-nav indicator, and module-card numerals.
- **Themes**: full **light and dark mode** with an OKLCH token system. Toggle in the top-right; choice is persisted to `localStorage`. The page auto-follows OS preference if you haven't toggled. All server-rendered Plotly figures (M1–M4) re-fetch with the new theme on toggle, so plots stay readable in both modes.
- **Reduced motion**: respected — entrance animations and the body grain are dampened when `prefers-reduced-motion` is set.

## Requirements

- Python 3.11+
- Flask 3.0+
- The same Python environment that trained your SpiderNet runs (PyTorch, PyTorch Geometric, scanpy, plotly, scipy). M4 additionally needs `gseapy` for GO/KEGG enrichment — without it the bubble plots show "no terms" but the rest of M4 still works.

```bash
conda activate spidernet
pip install -r requirements.txt
```

## Datasets

The app discovers datasets under `../Interactivetool/SpiderNet-interactive_V2/`, looking for directories named `<Name>_UI/`. Each must contain:

```
<Name>_UI/
├── <Name>_modeltraining_setup.json   # DIM_ENVIR, VERSION, SPECIES
├── config.json                        # CELL_TYPE_COL, SAMPLE_ID_COL, SPATIAL_KEY
├── run_dirs.json                      # original run paths (only the schema is used)
├── ProcessedData/                     # adata_list.pkl, SpiderNet_data_pyg_list.pkl, ...
└── <VERSION>/
    └── SpiderNet_Result_dim*/
        ├── Factor_envir_list.pkl
        ├── loading_LR_use.npy
        ├── loading_sender_use.npy
        ├── loading_receiver_use.npy
        └── Model/
            ├── SpiderNet_model_config.json
            └── model_epoch*.pth
```

Restart the app after adding a new dataset — discovery runs once at startup. Each dataset shows up as a row on the home page with chips linking into M1–M4.

## Project layout

```
SpiderNet-interactive/
├── app.py                          # Flask app factory; registers the four blueprints
├── config.py                       # SEARCH_ROOT, PORT, APP_TITLE, DEBUG
├── run.sh                          # Start script
├── requirements.txt
├── core/
│   ├── datasets.py                 # *_UI discovery and path resolution
│   └── loaders.py                  # Process-wide cache for adata / pyg / factor lists
├── modules/
│   ├── module1_basic/              # routes + service + template per module
│   ├── module2_subtype/
│   ├── module3_cascade/
│   └── module4_perturb/
├── templates/                      # base, index, team, paper, 404, module_stub
└── static/
    ├── css/style.css               # OKLCH tokens + light/dark themes + bold layer
    └── js/                         # main.js + m1_basic.js / m2 / m3 / m4
```

## Caching

- **Datasets** are loaded into memory once at first request via `core/loaders.get_core_bundle`. Subsequent requests reuse the bundle.
- **M2 / M3** cache analysis results keyed by parameter hash; threshold sliders re-render from cache without re-running.
- **M4** additionally caches the trained-model state and the per-slice baseline predictions on disk (`<run_dir>/UI_Exports/module4_spatial_perturbation_web/baseline_predictions.npz`). First page load runs a forward pass over every slice — slow once, fast forever.

## Customizing

| Want to | Edit |
|---|---|
| Change port | `config.py` → `PORT` |
| Change dataset search root | `config.py` → `SEARCH_ROOT` |
| Tweak colors / spacing / type | `static/css/style.css` (`:root` token block + `[data-theme="dark"]`) |
| Module-specific compute | `modules/<moduleN>/service.py` |
| Module-specific UI | `modules/<moduleN>/templates/` + `static/js/m<N>_*.js` |

## Troubleshooting

**No datasets on the home page.** Check that your dataset folder is under `../Interactivetool/SpiderNet-interactive_V2/`, ends in `_UI`, and contains `*_modeltraining_setup.json`, `config.json`, `ProcessedData/`, and a `<VERSION>/SpiderNet_Result_dim*/` directory.

**M4 first request is slow.** Expected — it loads the trained model and runs baseline inference for every slice. Subsequent requests use the on-disk baseline cache and are fast.

**Plots look unstyled.** Hard-refresh to bust the CSS cache (Cmd-Shift-R / Ctrl-F5). The theme system is OKLCH-based and needs a modern browser (Chrome 119+, Safari 16.4+, Firefox 113+).

**Port already in use.** Change `PORT` in `config.py` or kill the process bound to 8000.

## About

Part of the SpiderNet framework — an interpretable deep-learning approach for learning cell–cell meta-interactions from spatial transcriptomics. The interactive tool lets researchers efficiently explore learned communication patterns across tissue contexts.

- **Repository**: https://github.com/ma-compbio-lab/SpiderNet
- **License**: MIT
