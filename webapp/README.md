# SpiderNet-Interactive

A local web-based visualization tool for exploring SpiderNet spatial omics analysis results.

## Quick Start

```bash
cd /Users/wenduoc/SpiderNet/webapp
bash run.sh
```

Then open your browser to: **http://localhost:8000**

## Features

- **Multi-Dataset Support**: Automatically discovers all SpiderNet result directories
- **Three Analysis Modules**:
  - Basic Analysis (MI correlation, LR loading pathway enrichment, cell-type pair associations)
  - Subtype Analysis (MI-guided clustering, functional states, clinical metadata integration)
  - MI Cascade Analysis (multi-hop communication, spatial visualization, gene program analysis)
- **Advanced Search & Filtering**: Real-time search, category filtering, and sorting
- **Interactive Exploration**: Click-to-enlarge modals, keyboard shortcuts, quick actions
- **Local Server**: Runs on your machine, no internet required, accessible via browser
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Requirements

- Python 3.11+
- Flask 3.0.0
- Pillow 10.1.0

## Installation

The dependencies should already be in your `spidernet` conda environment. If not:

```bash
conda activate spidernet
pip install -r requirements.txt
```

## Usage

### Start the server
```bash
bash run.sh
```

### Stop the server
Press `Ctrl+C` in the terminal

## Structure

```
webapp/
├── app.py              # Flask application
├── config.py           # Configuration
├── utils.py            # Plot discovery logic
├── requirements.txt    # Python dependencies
├── run.sh             # Start script
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── basic_analysis.html
│   ├── subtype_analysis.html
│   └── cascade_analysis.html
└── static/             # CSS and JavaScript
    ├── css/style.css
    └── js/main.js
```

## Adding New Datasets

1. Place your SpiderNet results in either:
   - `./Interactivetool/`
   - `./SpiderNet/Results/`
2. Ensure the results are in a folder named `SpiderNet_Result_dim*`
3. Restart the application
4. Your new dataset will appear on the home page

## Troubleshooting

**No datasets found:**
- Check that results are in `SpiderNet_Result_dim*` folders
- Verify search paths in `config.py`

**Images not loading:**
- Check file permissions
- Ensure PNG files exist in results directory

**Port already in use:**
- Change `PORT` in `config.py`
- Or stop other applications using port 8000

## Development

To modify the application:

1. **Change port**: Edit `PORT` in `config.py`
2. **Add new categories**: Edit `categorize_plots()` in `utils.py`
3. **Customize styling**: Edit `static/css/style.css`
4. **Add features**: Edit `static/js/main.js`

## About

SpiderNet-Interactive is part of the SpiderNet framework, an interpretable deep learning approach for learning cell-cell meta-interactions from spatial transcriptomics data. This visualization tool enables researchers to efficiently explore and interpret learned communication patterns across tissue contexts.

**Repository**: https://github.com/ma-compbio-lab/SpiderNet
**Documentation**: See main SpiderNet README for analysis pipeline details
**License**: MIT License
