#!/bin/bash
# SpiderNet-Interactive - Run Script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set working directory to script location
cd "$SCRIPT_DIR"

# Activate conda environment (if spidernet environment exists)
if command -v conda &> /dev/null; then
    # Try to activate spidernet environment
    eval "$(conda shell.bash hook)"
    conda activate spidernet 2>/dev/null || echo "Note: conda environment 'spidernet' not found, using current environment"
fi

# Display startup message
echo "🧬 Starting SpiderNet-Interactive..."
echo "📂 Working directory: $(pwd)"
echo "🌐 Server will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Flask app with unbuffered stdout so progress logs from the modules
# (e.g. "[m2] Building embedding store...") appear in real time instead of
# sitting in Python's default block buffer until the process exits.
PYTHONUNBUFFERED=1 python -u app.py
