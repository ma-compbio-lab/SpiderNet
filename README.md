# SpiderNet
---

# Overview

<img width="1222" height="713" alt="SpiderNet_V2" src="https://github.com/user-attachments/assets/bf13203c-be25-4554-b98f-ec885f47db1f" />



**SpiderNet** is an interpretable deep learning framework for learning cell-cell meta-interactions (MIs) from spatial omics data. Specifically, SpiderNet inputs a spatial cell--cell graph and gene expression. The encoder maps sender/receiver expression to MI strengths. The decoder aggregates MIs to reconstruct ligand--receptor co-expression and gene expression. Nonnegative loadings make intrinsic, regulator, and target components interpretable. The resulting MIs and loadings enable diverse downstream applications including MI-associated pathway inference, cell subtype identification, in silico perturbation, MI cascade detection, and phenotype prediction.


## Installation

You can install **SpiderNet** directly from PyPI:
```bash
pip install SpiderNet
```

### Software dependencies
| Package      | Tested with          |
|--------------|----------------------|
| Python       | 3.11.7               |
| torch        | 2.0.1 (w/ cuda 11.7) |
| torchvision  | 0.15.2 (w/ cuda 11.7)|
| torchaudio   | 2.0.2 (w/ cuda 11.7) |
| torch-scatter   | 2.1.2                 |
| torch-geometric | 2.6.1                 |
| numpy           | 1.26.4                |
| scipy           | 1.15.3                |
| pandas          | 2.2.3                 |
| scikit-learn    | 1.5.2                 |
| scikit-misc     | 0.5.1                 |
| matplotlib      | 3.10.3                |
| seaborn         | 0.13.2                |
| scanpy          | 1.11.1                |
| squidpy         | 1.6.5                 |
| igraph          | 0.11.8                |
| louvain         | 0.8.2                 |
| gseapy          | 1.1.10                |

## Tutorials
A few examples in Jupyter notebook are included in the examples folder:
1. Application to overian cancer data
   - [Part0: Model training](https://github.com/junjie-sml/SpiderNet/blob/main/Examples/HGSC_Part0_modeltraining.ipynb)
   - [Part1: Basic analysis of inferred meta-interactions](https://github.com/junjie-sml/SpiderNet/blob/main/Examples/HGSC_Part1_basicanalysis.ipynb)
   - [Part2: Meta-interaction-guided malignant subtype identification](https://github.com/junjie-sml/SpiderNet/blob/main/Examples/HGSC_Part2_Malignantsubtype_analysis_P1.ipynb)
   - [Part3: In-silico perturbation analysis](https://github.com/junjie-sml/SpiderNet/blob/main/Examples/HGSC_Part3_insilicospatialperturbation.ipynb)
   - [Part3: MI cascade detection](https://github.com/junjie-sml/SpiderNet/blob/main/Examples/HGSC_Part4_cascadedetection.ipynb)
2. Application to Perturb-FISH human melanoma data
   - [Part0: Model training](https://github.com/junjie-sml/SpiderNet/blob/main/Examples/PerturbFISH_Part0_modeltraining.ipynb)
   - [Part1: In-silico perturbation analysis](https://github.com/junjie-sml/SpiderNet/blob/main/Examples/PerturbFISH_Part1_insilicospatialperturbation.ipynb)

Data used in these examples are available in [Google drive](https://drive.google.com/drive/folders/15tC6j2cQdNUNZHs1Zjmw-XnDx0Y7PapE?usp=sharing). Trained models are also uploaded.

## SpiderNet-Interactive: Web-Based Results Viewer

SpiderNet includes **SpiderNet-Interactive**, a local web-based visualization tool for exploring analysis outputs. The viewer provides an intuitive interface to navigate through plots from Basic Analysis, Subtype Analysis, and MI Cascade Analysis.

### Quick Start
```bash
cd SpiderNet/SpiderNet-interactive
bash run.sh
```

Then open your browser to **http://localhost:8000**

### Features
- **Multi-Dataset Support**: Automatically discovers all SpiderNet result directories
- **Three Analysis Modules**:
  - **Basic Analysis**: MI correlation, LR loading pathway enrichment, cell-type pair associations
  - **Subtype Analysis**: MI-guided clustering, functional states, clinical metadata integration
  - **MI Cascade Analysis**: Multi-hop communication, spatial visualization, gene program analysis
- **Advanced Search & Filtering**: Real-time search, category filtering, and sorting
- **Interactive Exploration**: Click-to-enlarge modals, keyboard shortcuts, quick actions
- **Local Server**: Runs on your machine, no internet required, accessible via browser
- **Cross-Platform**: Works on macOS, Linux, and Windows

### Results Location
Place your SpiderNet analysis results (directories matching `SpiderNet_Result_dim*`) in either:
- `./Interactivetool/`
- `./SpiderNet/Results/`

The viewer will automatically discover and categorize all PNG plots from your analysis notebooks.

For more details, see [SpiderNet-interactive/README.md](SpiderNet-interactive/README.md).
