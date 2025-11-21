# Use Python slim image for smaller footprint
FROM python:3.11-slim

# Install system dependencies for OpenCV and RTSP/H.264 support
RUN apt-get update && apt-get install -y \
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
    gstreamer1.0-vaapi \
    gstreamer1.0-libav \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Use CPU-only version of PyTorch for smaller size and better compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY database.py .
COPY web_app.py .
COPY static/ ./static/

# Pre-download YOLO model to avoid runtime download
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create directories for outputs
RUN mkdir -p /app/logs /app/snapshots

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HEADLESS=true

# Run in headless mode by default
CMD ["python", "-u", "main.py"]
