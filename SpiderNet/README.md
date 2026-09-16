# SpiderNet

SpiderNet is an interpretable deep learning framework for directional cell-cell meta-interactions (MIs) in spatial omics data. It provides preprocessing, model training, MI inference, downstream analysis, and visualization.

## Contents

- `SpiderNet/`: Python package and ligand-receptor resources.
- [../Tutorial/](../Tutorial/): the repository's eight study workflows, notebooks and R analyses.
- [SpiderNet-interactive/](SpiderNet-interactive/README.md): local web application for trained runs.
- `requirements*.txt`: core, tutorial, benchmark, UI, and recorded environment dependencies. `requirements-full-freeze.txt` is a reproducibility snapshot.

## Installation

The package requires Python 3.11. Install PyTorch and PyTorch Geometric for the selected CPU/CUDA environment before installing SpiderNet. The recorded CUDA 11.7 setup is:

```bash
conda create -n SpiderNet_env python=3.11.8
conda activate SpiderNet_env
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install -e .
```

Run installation commands from this directory. Optional dependencies are listed in `requirements-tutorial.txt`, `requirements-benchmark.txt`, and `requirements-UI.txt`; R packages are listed in [../Tutorial/R_requirements.md](../Tutorial/R_requirements.md).

## Usage

The workflow is preprocessing, training, MI inference, and downstream analysis. `SpiderNet.api` provides `build_model`, `run_training`, `infer_meta_interactions`, `normalize_outputs`, and `export_results`. Tutorials require matching prepared data and checkpoints and contain local input/output paths.

Verify the package location with `python -c "import SpiderNet; print(SpiderNet.__file__)"`. For the web application, install `requirements-UI.txt`, then run `python app.py` from `SpiderNet-interactive/`.

Distributed under the [MIT License](LICENSE). Please cite the corresponding SpiderNet manuscript when using this work.
