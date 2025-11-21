#!/bin/bash

# Script to run both the traffic watcher and web interface locally

echo "Starting Traffic Watcher Web Interface..."
echo "=========================================="
echo ""
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
