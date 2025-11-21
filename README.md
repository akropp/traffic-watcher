# RTSP Car Counter and Speed Estimator

This project consumes a video feed (RTSP or file), detects vehicles using YOLOv8, counts them, and estimates their speed based on a defined zone.

**Features:**
- Detects and classifies vehicles (Car, Truck, Motorcycle, Bus)
- Estimates speed using dual-line timing method
- Bi-directional traffic support (northbound/southbound)
- Headless mode for server deployment
- Docker support for easy deployment on low-power devices

## Quick Start with Docker (Recommended)

**1. Build and run with Docker Compose:**

```bash
docker-compose up -d
```

**2. View logs:**

```bash
docker-compose logs -f
```

**3. Stop the container:**

```bash
docker-compose down
```

The logs and snapshots will be saved to `./logs` and `./snapshots` on your host machine.

## Manual Setup (Without Docker)

**1. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the script:**

```bash
# GUI mode (default when not in Docker)
HEADLESS=false python main.py

# Headless mode (no window)
HEADLESS=true python main.py
```

## Configuration

Open `main.py` and edit the configuration section at the top:

* `VIDEO_SOURCE`: Your RTSP URL (e.g., `rtsp://user:pass@ip:port/path`) or path to a video file.
* `DISTANCE_METERS`: The real-world distance between the two measurement lines. You need to measure this on the ground.
* `LINE_LEFT_X_RATIO` and `LINE_RIGHT_X_RATIO`: The horizontal positions of the left and right vertical lines (0.0 to 1.0, relative to image width).
* `ROI_TOP_LEFT_X`, `ROI_TOP_LEFT_Y`, `ROI_BOTTOM_RIGHT_X`, `ROI_BOTTOM_RIGHT_Y`: Coordinates for the Region of Interest (0.0 to 1.0) where detection happens.
* `MAX_CROSSING_DURATION`: Maximum time (seconds) allowed between line crossings to prevent false positives.
* `HEADLESS`: Set via environment variable - `true` for no GUI (Docker default), `false` for GUI window.

## How it Works

1. **Detection**: Uses YOLOv8 (medium model by default) to detect cars, trucks, buses, and motorcycles.
2. **Tracking**: Uses the built-in tracker in Ultralytics to track objects across frames.
3. **Speed Estimation**:
    * When a vehicle crosses the first line, a timer starts.
    * When it crosses the second line, the timer stops.
    * Speed is calculated as `Distance / Time`.
    * Vehicles can temporarily leave the ROI and still complete measurements (with 10-second timeout).
4. **Counting**: Vehicles are counted when they successfully cross the second line after crossing the first.
5. **Vehicle Classification**: Displays the type of vehicle (Car, Truck, Motorcycle, Bus) in logs, snapshots, and on-screen.

## Docker Deployment on Low-Power Devices

The Docker image uses CPU-only PyTorch and is optimized for devices like Raspberry Pi, NUC, or other low-power servers:

**Resource Limits (configurable in `docker-compose.yml`):**
- CPU: 1-2 cores
- Memory: 1-2 GB

**For very low-power devices:**

Edit `main.py` and change the model to a smaller one:
```python
def __init__(self, video_source, model_path='yolov8n.pt'):  # nano model (smallest)
```

Or use the small model:
```python
def __init__(self, video_source, model_path='yolov8s.pt'):  # small model
```

## Tips for Accuracy

* **Camera Angle**: The camera should ideally be high up and looking down at an angle. The "lines" approach assumes the road is somewhat straight in the vertical axis of the frame.
* **Calibration**: The `DISTANCE_METERS` is crucial. If you can't measure it physically, you can estimate it using standard road markings (e.g., dashes are usually a standard length).
* **Performance**: 
  - For slow performance, use a smaller model (`yolov8n.pt` or `yolov8s.pt`)
  - Reduce the video resolution/FPS at the RTSP source
  - Adjust the ROI to be smaller (less area to process)
  - In Docker, adjust CPU/memory limits in `docker-compose.yml`

## Output

- **Console/Logs**: Real-time detection events with vehicle type, ID, speed, and direction
- **Snapshots**: Saved to `./snapshots/` directory with vehicle info overlaid
- **Log File**: `./logs/car_log.txt` contains all detected vehicle events with timestamps
