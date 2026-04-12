#!/bin/bash
# SpiderNet Results Viewer - Run Script

# Set working directory
cd /Users/wenduoc/SpiderNet/webapp

# Activate conda environment
source ~/miniconda3/bin/activate spidernet

# Display startup message
echo "🧬 Starting SpiderNet Results Viewer..."
echo "📂 Working directory: $(pwd)"
echo "🌐 Server will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Flask app
python app.py
