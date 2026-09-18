"""
Configuration settings for SpiderNet-Interactive.

The app discovers datasets in the V2 *_UI layout under SEARCH_ROOT.
"""
from pathlib import Path

# Project root (parent of SpiderNet-interactive/)
SPIDERNET_ROOT = Path(__file__).parent.parent.resolve()

# Where to look for <Name>_UI/ dataset directories
SEARCH_ROOT = SPIDERNET_ROOT / "Interactivetool" / "SpiderNet-interactive_V2"

# Default port
PORT = 8000

# Application title
APP_TITLE = "SpiderNet: Interpretable modeling of intercellular meta-interactions"

# Normal analysis must not be interrupted by the development auto-reloader.
# Set True explicitly only for development.
DEBUG = False
