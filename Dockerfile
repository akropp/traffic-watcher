# Use Debian base with Python 3.11 for system OpenCV compatibility
FROM debian:bookworm-slim

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies and system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libva2 \
    libva-drm2 \
    libva-dev \
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
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-gi \
    gir1.2-gst-rtsp-server-1.0 \
    gir1.2-gstreamer-1.0 \
    && rm -rf /var/lib/apt/lists/*

# Install numpy for OpenCV build
RUN python3 -m pip install --no-cache-dir --break-system-packages "numpy<2.0"

# Build OpenCV 4.10 from source with GStreamer support
# System package 4.6.0 has poor GStreamer appsink integration
RUN cd /tmp && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip && \
    unzip opencv.zip && \
    cd opencv-4.10.0 && \
    mkdir build && cd build && \
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_GSTREAMER=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_V4L=ON \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.11 \
        -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.11/dist-packages \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D WITH_CUDA=OFF \
        -D WITH_GTK=OFF \
        -D WITH_QT=OFF \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/opencv*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# CRITICAL ORDER: Install torch first (no opencv dep), then manual deps, then ultralytics with --no-deps
# This prevents opencv-python from being installed, allowing system opencv with GStreamer to be used
# --break-system-packages is safe in Docker containers (isolated environment)
RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip && \
    python3 -m pip install --no-cache-dir --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    python3 -m pip install --no-cache-dir --break-system-packages --no-deps ultralytics

# Copy application files
COPY main.py .
COPY database.py .
COPY web_app.py .
COPY static/ ./static/

# Pre-download YOLO model (this might auto-install opencv-python, we'll remove it after)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true && \
    echo "=== OpenCV Configuration ===" && \
    python3 -c "import cv2; print(f'Version: {cv2.__version__}'); print(f'Location: {cv2.__file__}')" && \
    echo "GStreamer:" && python3 -c "import cv2; print('YES' if 'gstreamer' in cv2.getBuildInformation().lower() else 'NO')" && \
    echo "==========================="

# Create directories for outputs
RUN mkdir -p /app/logs /app/snapshots

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HEADLESS=true

# Run in headless mode by default
CMD ["python3", "-u", "main.py"]
