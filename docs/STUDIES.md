# Study guide

The entry point for a reader reproducing saved figures is `Tutorial/<study>/run_benchmarks.py`. Each study uses the notebooks and helpers in its own folder. `Tutorial/` is the single tutorial directory for this release; the data profiles below match these workflows.

| Study | Scientific purpose | Data profile | Workflow documentation |
|---|---|---|---|
| AgingBrain | Age-associated cell interactions, spatial MI patterns, and transfer to sagittal/hippocampal samples | `plot-agingbrain` | [README](../Tutorial/AgingBrain/README.md) |
| HGSOC | Malignant subtypes, CAF signaling, functional states, CCC comparisons, and MI cascades | `plot-hgsoc` | [README](../Tutorial/HGSOC/README.md) |
| PerturbFISH | Perturbation response, held-out predictions, MI-linked programs, and ligand-receptor perturbation comparisons | `plot-perturbfish` | [README](../Tutorial/PerturbFISH/README.md) |
| Pancancer | Cross-cancer spatial interactions, cascades, LR perturbations, bulk projection, survival, and ICB associations | `plot-pancancer` | [README](../Tutorial/Pancancer/README.md) |
| Simulation | Controlled synthetic-data benchmarks and a representative spatial-interaction example | `plot-simulation` | [README](../Tutorial/Simulation/README.md) |
| Coupling_benchmark | Directionality, cell-similarity concordance, and spatial specificity across tissues | `plot-coupling-benchmark` | [README](../Tutorial/Coupling_benchmark/README.md) |
| Ablation_study | Component contributions to coupling and perturbation prediction | `plot-ablation-study` | [README](../Tutorial/Ablation_study/README.md) |
| Robustness_stability | MI selection/alignment, annotation errors, resampling, and stability analyses | `plot-robustness-stability` | [README](../Tutorial/Robustness_stability/README.md) |

## How the pieces fit together

Prepared spatial data and a graph feed SpiderNet training and MI inference. Saved factors/loadings feed the study-specific statistics. Saved statistics, embeddings, drawing inputs, and selected existing figures feed `--plot-only`. The interactive application consumes prepared datasets and saved model outputs through its own export caches.

The release preserves these stages. `--plot-only` is the appropriate first step for matching saved figure inputs; it is not a request to train a fresh model. Individual READMEs document expensive steps, optional comparators, copied versus redrawn figures, and manuscript panels not implemented by that workflow.

## Other directories

- `SpiderNet/SpiderNet/`: public Python modules; `api.py` exposes model construction, training, inference, normalization and export functions.
- `SpiderNet/SpiderNet/resources/`: small ligand-receptor resources included in the installed package.
- `SpiderNet/SpiderNet-interactive/`: web application; restore the `interactive` profile before exploring the packaged datasets.
- `benchmarks/training_speed/`: optional implementation and benchmark code. Its archived runs and initialization caches use the `training-benchmark-results` bundle.
- Restored `Tutorial/<study>/output/`: exact saved products, numerical inputs and figures for that workflow.
- Restored `.spidernet/workspace/`: explicitly packaged upstream plot inputs. Environment helpers point to these paths rather than an author's machine.
- Restored `.spidernet/provenance/`: original execution records retained for reference. Old machine-specific execution identity files are kept here instead of activating them in a new output folder.
