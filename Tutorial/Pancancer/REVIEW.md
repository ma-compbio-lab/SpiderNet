# Pancancer workflow validation and limitations

## 1. Scope

The implementation has known differences from the manuscript. The reference is `SpiderNet (39).pdf` (79 pages; SHA-256 `d5524a316db8e22a44114b4fc2f68d92996fca3e2674f69222657334851e3836`), including the pan-cancer text, methods A.7/A.8, main figure 6, supplementary figures S23-S29, and the Pancancer portion of S31a.

## 2. Analysis entries

The nine analysis entries are five notebooks (spatial analysis, cascade, LR perturbation, preprocessing/dimension selection, and training), three R scripts (projection, survival, ICB), and the Julia scCChain companion. `run_benchmarks.py` provides orchestration. Three Python plotting helpers reuse tagged notebook source; `plot_saved_R.R` shares exact plotting blocks with the R analyses.

## 3. Input and dependency chain

Annotated `.h5ad` data -> optional preprocessing/dimension selection -> optional 11-MI model training -> spatial inference and summaries. Cascade and LR perturbation consume matching spatial factors, graphs, expression and gene/LR annotations. Projection consumes the exported loadings and pseudo-bulk summaries plus TCGA expression. Survival consumes projected TCGA abundance and clinical records. ICB uses the same loadings with separate Hugo/Gide/Jung expression/response inputs.

Shared SpiderNet software, databases, `Data/`, `Results/`, and checkpoints remain upstream. No code is imported from another Tutorial analysis folder; the only local companions are the Julia script, plotting helpers, and notebooks. The preprocessing hook still requires existing `celltype_final` and `subslice_id`; raw annotation and construction of the sub-slices are not supplied here.

Full runs preserve the original checkpoint selection, normalization, statistical settings and cache policies. Training retains preprocessing enabled by default. Full TCGA/survival execution can query/download GDC as before. Saved-result R plotting first uses complete canonical local output caches, then matching canonical upstream caches. `--cache-dir` is an explicit override; it must identify the intended scientific variant. No inferred fallback to an older MI, clinical model or decomposition variant is made.

## 4. Paper coverage

Cell numbers below are one-based physical positions including Markdown, not execution counts. The figure map below records the implementation and its limits.

| Panel | Implementation and limits |
|---|---|
| 6a | Spatial notebook cell-count summaries; body/organ illustration and final composition are external. |
| 6b | Spatial cells 70, 74, 82: cell-pair recurrence, LR pathway heatmap, CancerSEA split triangles. |
| 6c / S24c | Spatial MI4 settings cell 121, plotting cell 123; representative ROIs/connectors are not assembled automatically. Cell 125 supplies an additional MI2 preview. |
| 6d | Spatial cells 93-113 and the local Julia companion implement the CCC comparison. |
| 6e / S27a | Projection -> survival; pass `--mi MI4` to select the manuscript KM panels. |
| 6f | Cascade cells 27-28 and 30-36: abundance and upstream-gated feature comparisons; topology illustration/composition are external. |
| S23a-h | Annotation UMAP/marker validation workflow not found in the original or current folder. Existing annotation is an input, not a replacement for these panels. |
| S24a-b | Spatial cells 118 and 115: median directed-pair activity and fibroblast-to-tumor edge distribution. |
| S25a-b | Spatial cells 90-92 for receiver programs; LRKO notebook cells 2-11 for perturbation, plots and tests. |
| S26a-b | Spatial cell 46 LR-proxy validation; projection script's integrated MI4 pseudo-bulk recovery. |
| S27b-c | Integrated nested-LRT survival analysis; independent ICB script. |
| S28a-c | Cascade enrichment/recurrence cells 11-16 and triplet composition cells 23-25. |
| S29 | Cascade cell 27 spatial relay maps; final ROI composition remains external. |
| S31a | Optional preprocessing notebook calls `SpiderNet.MI_dimension_selection`; only the Pancancer part belongs here. |

## 5. Execution and outputs

Use the commands in README.md. The default full command runs the six downstream stages, in order, using the current Python interpreter for notebooks and the selected Rscript for R analyses. Notebook execution creates a timestamped copy under `output/executed/`, including diagnostic output if a cell fails. The source notebook is not overwritten. Full runs skip dependent stages after a selected upstream stage fails; independent stages and saved-result plotting checks continue. The runner returns nonzero if any stage is incomplete.

Final figures/summary tables go below `output/`. Notebook stages retain their original scientific Results/cache layout; the runner collects newly written figures and bounded summary tables, preserving relative filenames. CSV/TSV intermediates larger than 64 MiB remain upstream and are recorded rather than duplicated by the collector. This size rule does not subsample or alter any table. Windows long filenames are supported. Local ignore rules explicitly include final PDF/PNG/SVG/CSV/TSV outputs while excluding bulky caches and executed copies.

Plot-only runs do not refit a model or repeat permutations/perturbations. They reuse original drawing code and saved tables. Spatial/cascade drawing caches added by full execution retain the exact drawing inputs. A missing cache is reported as incomplete; copying a historical image is never labeled regeneration. Some additional exploratory single-unit/GO grids remain full-analysis outputs rather than compact-cache plotting targets.

## 6. Validation performed

- Python cells/scripts and R sources passed the recorded syntax checks. Preprocessing, training, cascade, LRKO and R scientific functions were compared against reference implementations.
- Spatial cached rendering produced nine PDF/PNG pairs and 21 CSVs. Corresponding original-source execution produced byte-identical nine PNGs and 21 CSVs; five derived tables also matched existing Results numerically.
- Spatial drawing-cache regressions covered three MI4 cases and eight MI2 cancer previews: original full drawing, modified full drawing, and cache replay yielded byte-identical PNGs. MI2 ranking/selection tables were identical. These are controlled small inputs, not a full atlas redraw.
- Cascade cached rendering produced 41 figure files. Nineteen PNGs were pixel-identical to execution using untouched notebook plotting source in the same environment. Ten regenerated tables matched the established caches within `rtol=1e-12`.
- LRKO used all 8,601,570 saved rows, reading only the four plotted columns. The full panel was pixel-identical to the original plotting cell in the same environment. Three statistical summaries were copied unchanged; tests/inference were not repeated.
- The full 160-sample pseudo-bulk joint projection ran with the current and reference implementations: all 18 CSVs were identical, including estimates, offsets, weights and recovery statistics. Recovery PDF rasterization was pixel-identical.
- Cached TCGA summaries, four ICB heatmaps and the nested-LRT scatter were rendered. Available source tables and rank/order outputs matched existing results. The nested-LRT test explicitly selected the legacy Results root with `--cache-dir`; it did not assume the new canonical cache already existed.
- The unified runner's R plotting path and current-interpreter notebook execution were exercised, including a failing notebook that saves its diagnostic copy without changing the source. Read-only checks identify missing inputs/packages; a targeted KM fixture verifies that MI2-only caches do not pass an MI4 check. Spatial cached rendering also succeeded with access to every other Tutorial folder blocked. Julia syntax was parsed without loading or running the model.

Historical PNGs can differ because of the plotting environment. Same-environment comparisons distinguish rendering effects from implementation differences. Numerical and image-validation evidence is in `output/validation/`.

## 7. Validation limits and present missing inputs

No full atlas model inference, training, CCC baseline recomputation, cascade permutation run, LR perturbation run, or full TCGA download/projection was repeated. Large graph/expression inputs are tens of GB. Full ICB refitting was not repeated. Julia model execution was not tested. Cache preparation ran the small MI-only Cox models from the saved clinical/projection tables and verified all 99 adjusted HR/P pairs against the existing results.

With `survminer` installed, cache preparation reconstructed 99 exact KM inputs under Results. The nested-LRT plot can read its table at the Results root. `--prepare-plot-inputs` reconstructs drawing caches directly from saved analyses: all 160 graph/annotation cell orders, coordinates and labels are checked; stored cascade normalization and adjacency are checked against current factors/graphs; all 160 observed/potential cascade counts and 385 overlapping selected-triplet proportions match existing tables. It retains the original edge selection, display thresholds, sampling seeds, ordering and plotting functions. Graph tensor loading is deferred to avoid reading unused expression/LR arrays into RAM. Fresh COMMOT computation still needs the missing `commot` Python package; existing validated baseline caches can be reused under the original policy.

All six plot-only stages completed with exit code 0, exporting 160 MI4 spatial maps and 8 MI2 representative maps in PNG/PDF, 320 cascade spatial PNGs, and 36 KM PDFs (the original MI2/MI3/MI9 selection plus an explicit MI4 run). All 488 spatial/cascade PNGs passed file-integrity checks. Existing statistical figures were also regenerated. Evidence and exact validation limits are recorded in `output/validation/missing_plot_inputs_recovery.json`; this is saved-analysis reproduction, not a new full scientific analysis.

## 8. Scientific discrepancies preserved

| Topic | Preserved code behavior versus manuscript description |
|---|---|
| LR pathway heatmap | Mean normalized loading in the package helper; manuscript says mean loading rank. |
| CCC receiver activity | Incoming sum; manuscript describes max. Baseline feature selection uses mean rank across panels, rather than strongest mean coupling. |
| LR perturbation selection | Top five raw LR loadings; manuscript describes normalized loading >0.5. The low-loading random pool and gene-overlap exclusion remain unchanged. |
| KM selection | Default MI2/MI3/MI9; manuscript panels use MI4. Explicit `--mi MI4` changes only the selected KM exports. |
| Clinical adjustment | Age/gender/tumor grade rather than the described age/sex/stage. Nested LRT retains additional EMT and purity adjustment. |
| Cascade enrichment | Ratio threshold 1.2 rather than manuscript 1.5; detection, display and upstream grouping retain their separate thresholds. |
| Cohort offsets | Free gene/LR offsets retain the conditional translation ambiguity in absolute cohort abundance; no new anchoring or regularization was added. |
| Pseudo-bulk recovery scope | Current S26b output selects MI4 (160 entries; adjacent P = 0.103 and 0.150). The manuscript drawing matches the historical boxplot pooling all 11 MIs (1,760 entries; P = 4.89e-13 and 4.35e-05), despite its MI4 caption. Both selections were checked with historical and current matrices; see `output/validation/S26b_scope_diagnosis/REPORT.md`. The mismatch remains unresolved. |
| Representative MI2 preview | Ranks sub-slices by total eligible edge intensity, not intensity per unit area. It is additional to the MI4 manuscript maps. |

The three-panel cell-pair summary has a title overlap.

## 9. Reproduction status

The folder provides full analysis entries and a tested saved-result plotting route, with explicit missing-cache and dependency failures. It is not claimed to reproduce every manuscript panel from raw data: annotation validation, raw sub-slice preparation, final illustration/ROI assembly, unexecuted large-data stages and the preserved scientific differences above remain visible limitations.
