#!/bin/bash

# Script to set up a conda environment for traffic-watcher

echo "Creating conda environment: traffic-watcher"
echo "==========================================="

# Create conda environment with Python 3.11
conda create -n traffic-watcher python=3.11 -y

echo ""
echo "Activating environment..."
conda activate traffic-watcher

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "==========================================="
echo "Setup complete!"
echo ""
echo "To use this environment:"
echo "  conda activate traffic-watcher"
echo ""
echo "To run the application:"
echo "  python main.py              # Start tracker"
echo "  python web_app.py           # Start web interface"
echo "  ./run_local.sh              # Run both together"
echo ""
