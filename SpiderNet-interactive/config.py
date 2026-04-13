"""
Configuration settings for SpiderNet Results Viewer
"""
from pathlib import Path

# Root directory for SpiderNet project
SPIDERNET_ROOT = Path("/Users/wenduoc/SpiderNet")

# Directories to search for results
SEARCH_DIRS = [
    SPIDERNET_ROOT / "Interactivetool",
    SPIDERNET_ROOT / "SpiderNet" / "Results",
]

# Pattern to identify SpiderNet result directories
RESULT_DIR_PATTERN = "SpiderNet_Result_dim*"

# Supported image formats
IMAGE_FORMATS = ['.png', '.pdf']

# Default port
PORT = 8000

# Application title
APP_TITLE = "SpiderNet: Interpretable modeling of intercellular meta-interactions"

# Debug mode
DEBUG = True
