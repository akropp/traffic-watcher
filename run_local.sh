#!/bin/bash

# Script to run both the traffic watcher and web interface locally

set -e  # Exit on error

echo "Starting Traffic Watcher Web Interface..."
echo "=========================================="
echo ""

# Check if conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "traffic-watcher" ]]; then
    echo "Error: Please activate the conda environment first:"
    echo "  conda activate traffic-watcher"
    echo ""
    echo "If you haven't created it yet, run:"
    echo "  conda create -n traffic-watcher python=3.11 -y"
    echo "  conda activate traffic-watcher"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "Starting web server on http://localhost:5050"
echo "Press Ctrl+C to stop both services"
echo ""

# Initialize database
python database.py

# Start web server in background
python web_app.py &
WEB_PID=$!

# Start main tracker (will run in foreground)
python main.py

# Cleanup: kill web server when main.py exits
kill $WEB_PID 2>/dev/null
