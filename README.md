[SpiderNet_figure_1_v6.pdf](https://github.com/user-attachments/files/27071739/SpiderNet_figure_1_v6.pdf)# SpiderNet

[Uploading SpiderNet_figure_1_v6.pdf…]()

SpiderNet is an interpretable deep learning framework for learning directional cell-cell meta-interactions (MIs) from spatial omics data.

SpiderNet takes gene expression profiles and a spatial cell-cell graph as input. For each directed neighboring cell pair, the encoder maps sender and receiver expression into a non-negative vector of MI strengths. The decoder then uses these MIs to reconstruct ligand-receptor co-expression on cell-cell pairs and gene expression in cells. The non-negative model components make the learned MIs interpretable through intrinsic, ligand-receptor, sender-regulator, and receiver-target gene loadings.

The inferred MIs provide a compact and interpretable representation of multicellular communication programs, enabling downstream analyses such as MI-associated pathway inference, cell-type-pair enrichment, cell subtype discovery, MI cascade detection, in silico spatial perturbation, and phenotype prediction.

---

## Installation

SpiderNet requires Python 3.11. We recommend creating a clean conda environment.

```bash
conda create -n SpiderNet_env python=3.11.8
conda activate SpiderNet_env
```

### 1. Install PyTorch and PyTorch Geometric dependencies

PyTorch and PyTorch Geometric should be installed before installing SpiderNet, because their installation depends on the operating system, CUDA version, and PyTorch version.

For CUDA 11.7, install PyTorch and PyG-related packages with:

```bash
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

If you use a different CUDA version or a CPU-only environment, please install the corresponding PyTorch and PyG wheels.

### 2. Install SpiderNet

Clone the repository and install SpiderNet from the project root:

```bash
git clone https://github.com/ma-compbio-lab/SpiderNet.git
cd SpiderNet
pip install .
```

For development, use editable installation:

```bash
pip install -e .
```

Editable installation is recommended when modifying the source code, because changes in the `SpiderNet/` package are reflected without reinstalling.

### 3. Optional: install tutorial dependencies

To run tutorial notebooks, install additional lightweight notebook dependencies:

```bash
pip install -r requirements-tutorial.txt
```

Alternatively, install the tutorial optional dependency group defined in `pyproject.toml`:

```bash
pip install -e ".[tutorial]"
```

Some simulation benchmark tutorials require additional external-method dependencies. Install them only when needed:

```bash
pip install -r requirements-benchmark.txt
```

The repository also keeps `requirements-full-freeze.txt` as a complete snapshot of the development environment. This file is mainly for reproducibility and is not recommended for routine installation.

Some R/R Markdown tutorials require additional R packages. See `Tutorial/R_requirements.md` for installation instructions.

If you are using Jupyter, register the conda environment as a notebook kernel:

```bash
python -m ipykernel install --user --name SpiderNet_env --display-name "Python (SpiderNet_env)"
```

---

## Verify installation

After installation, test whether SpiderNet can be imported:

```bash
python -c "import SpiderNet; print(SpiderNet.__file__)"
```

Test whether PyTorch and PyG are available:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_geometric, torch_scatter, torch_sparse; print('PyG OK')"
```

Test whether the packaged ligand-receptor resources are available:

```bash
python -c "from SpiderNet.utils import get_default_cellchat_db; print(get_default_cellchat_db('human'))"
```

---

## Basic usage

A typical SpiderNet workflow includes:

1. Preparing spatial omics data and constructing a spatial cell-cell graph.
2. Loading processed data into SpiderNet-compatible objects.
3. Training the SpiderNet model.
4. Inferring MI strengths for directed neighboring cell pairs.
5. Running downstream analyses and visualizations.

Example:

```python
from pathlib import Path

from SpiderNet.config import TrainingConfig
from SpiderNet.io import load_processed_data
from SpiderNet.api import (
    build_model,
    run_training,
    infer_meta_interactions,
    normalize_outputs,
    export_results,
)

# Define input and output paths.
processed_data_dir = Path("path/to/ProcessedData")
output_dir = Path("path/to/SpiderNet_Result_dim15")
model_dir = output_dir / "Model"

# Load processed data.
processed = load_processed_data(processed_data_dir)

# Configure training.
training_config = TrainingConfig(
    dim_envir=15,
    max_epoch=50000,
    n_jobs=5,
    optimizer="adam",
)

# Build and train model.
model = build_model(processed, training_config)
trained_model = run_training(
    model=model,
    processed=processed,
    train_cfg=training_config,
    model_dir=model_dir,
)

# Infer, normalize, and export MI results.
outputs = infer_meta_interactions(trained_model, processed)
outputs = normalize_outputs(outputs)

export_results(
    results=outputs,
    processed=processed,
    precessed_data_dir=processed_data_dir,
    output_dir=output_dir,
)
```

Please see the tutorial notebooks for dataset-specific examples.

---

## Tutorial notes

The tutorial notebooks demonstrate SpiderNet analyses across multiple datasets, including HGSOC, AgingBrain, PerturbFISH, Pancancer, Simulation, and CCC coupling benchmark examples.

Some tutorial notebooks currently use local example paths such as:

```text
D:/SpiderNet/Data/...
D:/SpiderNet/Results/...
```

Before running these tutorials, please update the data and output paths to match your local directory structure. Some tutorials also require prepared spatial omics datasets, processed SpiderNet input bundles, trained model outputs, or external benchmark results. Please check the corresponding tutorial folder for required input files and expected directory structure.

The core SpiderNet package can be installed independently of these tutorial datasets. Dataset-specific notebooks may require additional optional dependencies beyond the core package.

---

## Main modules

SpiderNet contains the following core modules:

- `SpiderNet.config`: configuration dataclasses for paths, preprocessing, and model training.
- `SpiderNet.io`: loading and saving processed SpiderNet data objects.
- `SpiderNet.api`: high-level model training, inference, normalization, and export functions.
- `SpiderNet.model`: core SpiderNet model architecture.
- `SpiderNet.analysis`: downstream MI enrichment, cascade, perturbation, and phenotype-related analyses.
- `SpiderNet.visualization`: visualization utilities for MI results.
- `SpiderNet.MI_dimension_selection`: utilities for selecting the number of MI dimensions.
- `SpiderNet.utils`: helper functions and default ligand-receptor database paths.
- `SpiderNet.dataloading_unified`: unified data loading and preprocessing utilities.

---

## Package resources

SpiderNet includes default ligand-receptor resources under:

```text
SpiderNet/resources/
```

These include CellChatDB- and scSeqComm-derived ligand-receptor pairs for human and mouse datasets.

The default resource paths can be accessed with:

```python
from SpiderNet.utils import get_default_cellchat_db, get_default_scseqcomm_db

cellchat_human = get_default_cellchat_db("human")
scseqcomm_human = get_default_scseqcomm_db("human")
```

---

## Notes for users

The tutorial notebooks may require additional dependencies beyond the core SpiderNet package. Lightweight notebook dependencies are provided in `requirements-tutorial.txt`. Heavier benchmark or external-method dependencies are provided in `requirements-benchmark.txt` and should be installed only when needed.

R/R Markdown tutorials require separate R packages; see `Tutorial/R_requirements.md`.

The PyTorch / PyG installation commands above are examples for CUDA 11.7. Please adapt them to your own CUDA and PyTorch environment.

---

## Citation

If you use SpiderNet in your research, please cite the corresponding SpiderNet manuscript.

---

## License

This project is distributed under the MIT License.
