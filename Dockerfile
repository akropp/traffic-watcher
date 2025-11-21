# Use Debian base with Python 3.11 for system OpenCV compatibility
FROM debian:bookworm-slim

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for OpenCV and RTSP/H.264 support
# python3-opencv: System OpenCV with GStreamer support (pip version doesn't have it)
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ffmpeg \
    libavcodec-extra \
    libavformat-dev \
    libswscale-dev \
    libva2 \
    libva-drm2 \
    i965-va-driver \
    intel-media-va-driver \
    vainfo \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-vaapi \
    gstreamer1.0-libav \
    gstreamer1.0-va \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# CRITICAL ORDER: Install torch first (no opencv dep), then manual deps, then ultralytics with --no-deps
# This prevents opencv-python from being installed, allowing system opencv with GStreamer to be used
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir --no-deps ultralytics

# Copy application files
COPY main.py .
COPY database.py .
COPY web_app.py .
COPY static/ ./static/

# Pre-download YOLO model (this might auto-install opencv-python, we'll remove it after)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true && \
    echo "OpenCV location:" && python3 -c "import cv2; print(cv2.__file__)" && \
    echo "GStreamer support:" && python3 -c "import cv2; print('YES' if 'gstreamer' in cv2.getBuildInformation().lower() else 'NO')"

# Create directories for outputs
RUN mkdir -p /app/logs /app/snapshots

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HEADLESS=true

# Run in headless mode by default
CMD ["python3", "-u", "main.py"]
