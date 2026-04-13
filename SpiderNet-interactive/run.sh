#!/bin/bash
# SpiderNet-Interactive - Run Script

# Set working directory
cd /Users/wenduoc/SpiderNet/SpiderNet-interactive

# Activate conda environment
source ~/miniconda3/bin/activate spidernet

# Display startup message
echo "🧬 Starting SpiderNet-Interactive..."
echo "📂 Working directory: $(pwd)"
echo "🌐 Server will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Flask app
python app.py
