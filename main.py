import cv2
import time
import os
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import signal
import sys
from datetime import datetime
import database
import threading
import queue

# Try to import GStreamer capture (only available in Docker)
try:
    from gstreamer_capture import GStreamerCapture
    GSTREAMER_AVAILABLE = True
except ImportError:
    GSTREAMER_AVAILABLE = False
    print("GStreamer Python bindings not available - using OpenCV backend only")

# --- CONFIGURATION ---
# RTSP Stream URL (Replace with your camera's URL)
# Example: "rtsp://username:password@ip_address:port/stream"
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'rtsp://frigate.thekropp.com:8554/driveway')

# Real world distance between the two lines (in meters)
DISTANCE_METERS = 10.0

# Maximum time allowed between line crossings (seconds) - prevents false positives from stale timers
MAX_CROSSING_DURATION = 10.0

# Headless mode (no GUI window) - useful for Docker/server deployment
HEADLESS = os.getenv('HEADLESS', 'true').lower() == 'true'

# Skip YOLO inference (for testing decode-only CPU usage)
SKIP_INFERENCE = os.getenv('SKIP_INFERENCE', 'false').lower() == 'true'

# Motion detection settings
MOTION_DETECTION = os.getenv('MOTION_DETECTION', 'true').lower() == 'true'
MOTION_THRESHOLD = int(os.getenv('MOTION_THRESHOLD', '500'))  # Pixels changed threshold
MOTION_MIN_AREA = int(os.getenv('MOTION_MIN_AREA', '100'))    # Minimum contour area

# Video downsampling scale (0.5 = half resolution, 1.0 = original)
# Lower values = faster processing but less accurate detection
# Recommended: 0.5 for CPU, 1.0 for GPU
VIDEO_SCALE = float(os.getenv('VIDEO_SCALE', '1.0'))

# Lines positions (0-1 relative to frame height/width)
# Vertical lines for left-to-right / right-to-left traffic
LINE_LEFT_X_RATIO = 0.64
LINE_RIGHT_X_RATIO = 0.77

# Region of Interest (ROI) for detection (0-1 relative to frame dimensions)
# Top-Left (x, y) and Bottom-Right (x, y)
ROI_TOP_LEFT_X = 0.52
ROI_TOP_LEFT_Y = 0.01
ROI_BOTTOM_RIGHT_X = 0.86
ROI_BOTTOM_RIGHT_Y = 0.26 # Top half of the frame

# Vehicle classes to detect (COCO dataset class IDs)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Class ID to name mapping
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ---------------------

class CarCounter:
    def __init__(self, video_source, model_path='yolov8s.pt', headless=HEADLESS):
        self.video_source = video_source
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.first_seen = {}       # track_id -> (x, time)
        self.last_seen = {}        # track_id -> (x, time)
        self.vehicle_speeds = {}   # track_id -> speed
        self.vehicle_types = {}    # track_id -> vehicle type
        self.ids_in_roi = set()    # track IDs currently in ROI
        self.recent_exits = {}     # track_id -> (x, time, vehicle_type) - for merging rapid re-entries
        self.id_mapping = {}       # new_id -> original_id - for tracking ID reassignments
        self.counted_ids = set()   # track IDs that have already been counted (prevents duplicate counts)
        self.car_count = 0
        self.headless = headless
        self.running = True
        
        # Thresholds for merging rapid re-entries
        self.reentry_time_threshold = 1.5  # seconds
        self.reentry_distance_threshold = 400  # pixels
        
        # Create output directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("snapshots", exist_ok=True)
        self.log_file = "logs/car_log.txt"
        
        # Initialize database
        database.init_db()
        
        # Open video
        print(f"Opening video source: {self.video_source[:80]}...")
        
        # Use native GStreamer if pipeline is detected AND available (Docker only)
        if GSTREAMER_AVAILABLE and ('rtspsrc' in self.video_source or '!' in self.video_source):
            print("Detected GStreamer pipeline, using native GStreamer bindings")
            print(f"Full pipeline: {self.video_source}")
            
            try:
                self.cap = GStreamerCapture(self.video_source)
            except Exception as e:
                print(f"Failed to create GStreamer pipeline: {e}")
                print("Falling back to OpenCV with plain RTSP URL")
                # Extract RTSP URL from pipeline if possible
                if 'location=' in self.video_source:
                    rtsp_url = self.video_source.split('location=')[1].split(' ')[0]
                    self.cap = cv2.VideoCapture(rtsp_url)
                else:
                    return
        else:
            # OpenCV backend (local development or plain RTSP URL)
            if 'rtspsrc' in self.video_source or '!' in self.video_source:
                print("GStreamer not available, extracting RTSP URL from pipeline")
                # Extract RTSP URL from pipeline
                if 'location=' in self.video_source:
                    rtsp_url = self.video_source.split('location=')[1].split(' ')[0]
                    print(f"Using OpenCV with: {rtsp_url}")
                    self.cap = cv2.VideoCapture(rtsp_url)
                else:
                    print("Could not extract RTSP URL from pipeline")
                    return
            else:
                print("Using OpenCV FFmpeg backend")
                self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            self.cap.release()
            return

        # Set buffer size to 1 to reduce latency and avoid H.264 decoding errors on old frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Check backend being used
        try:
            backend = self.cap.getBackendName()
            print(f"Video backend: {backend}")
        except:
            print("Video backend: Unknown (could not query)")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Apply video scaling to dimensions
        if VIDEO_SCALE != 1.0:
            self.width = int(self.width * VIDEO_SCALE)
            self.height = int(self.height * VIDEO_SCALE)
        
        # Calculate line positions in pixels
        self.line_left_x = int(self.width * LINE_LEFT_X_RATIO)
        self.line_right_x = int(self.width * LINE_RIGHT_X_RATIO)
        
        # Calculate pixels per meter for speed calculation
        self.pixels_between_lines = self.line_right_x - self.line_left_x
        self.pixels_per_meter = self.pixels_between_lines / DISTANCE_METERS
        
        # Calculate ROI in pixels
        self.roi_x1 = int(self.width * ROI_TOP_LEFT_X)
        self.roi_y1 = int(self.height * ROI_TOP_LEFT_Y)
        self.roi_x2 = int(self.width * ROI_BOTTOM_RIGHT_X)
        self.roi_y2 = int(self.height * ROI_BOTTOM_RIGHT_Y)
        
        print(f"Video Source: {self.width}x{self.height} @ {self.fps} FPS")
        if GSTREAMER_AVAILABLE and hasattr(self.cap, 'pipeline'):
            print(f"  (Note: GStreamer reports stream metadata FPS, actual delivery may vary)")
        print(f"  Using frame interval: {1.0/self.fps:.4f}s ({self.fps:.2f} FPS) for speed calculations")
        print(f"Speed Measurement Lines at X={self.line_left_x} and X={self.line_right_x}")
        print(f"ROI: ({self.roi_x1}, {self.roi_y1}) to ({self.roi_x2}, {self.roi_y2})")
        print(f"Mode: {'Headless' if self.headless else 'GUI'}")
        print(f"Video Scale: {VIDEO_SCALE}")
        print(f"Motion Detection: {'Enabled' if MOTION_DETECTION else 'Disabled'}")
        
        # Motion detection state
        self.prev_gray = None
        self.last_motion_time = time.time()  # Time-based instead of frame-based
        self.motion_detected_count = 0
        self.inference_skipped_count = 0
        
        # Frame queue for async reading (prevents frame drops during inference)
        self.frame_queue = queue.Queue(maxsize=10)  # Buffer up to 10 frames
        self.frame_reader_thread = None
        self.reader_running = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self.running = False
        self.reader_running = False
    
    def _frame_reader_loop(self):
        """Background thread that continuously reads frames to prevent drops"""
        print("[Frame Reader] Thread started")
        while self.reader_running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            success, frame = self.cap.read()
            if success and frame is not None:
                # Downsample frame if needed
                if VIDEO_SCALE != 1.0:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                
                # Try to add to queue (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Queue full, drop oldest frame and try again
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                    except (queue.Empty, queue.Full):
                        pass
            else:
                time.sleep(0.01)  # Brief pause on read failure
        
        print("[Frame Reader] Thread stopped")
    
    def detect_motion(self, frame):
        """Detect motion in ROI using frame differencing"""
        if not MOTION_DETECTION:
            return True  # Always run inference if motion detection disabled
        
        # Always run inference if we're actively tracking vehicles
        if len(self.ids_in_roi) > 0:
            self.last_motion_time = time.time()  # Reset timer while tracking
            return True
        
        # Extract ROI
        roi = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # First frame initialization
        if self.prev_gray is None:
            self.prev_gray = gray
            return True
        
        # Compute absolute difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Count changed pixels
        changed_pixels = cv2.countNonZero(thresh)
        
        # Update previous frame
        self.prev_gray = gray
        
        # Motion detected if enough pixels changed
        if changed_pixels > MOTION_THRESHOLD:
            self.last_motion_time = time.time()
            return True
        
        # Continue running inference for 10 seconds after motion stops
        # Time-based grace period works regardless of FPS variations
        time_since_motion = time.time() - self.last_motion_time
        if time_since_motion < 10.0:  # 10 second grace period
            return True
        
        return False

    def find_matching_recent_exit(self, center_x, vehicle_type, current_time):
        """Check if this entry matches a recent exit (likely same vehicle with new ID)"""
        for exit_id, (exit_x, exit_time, exit_type) in list(self.recent_exits.items()):
            # Check if exit was recent and vehicle type matches
            time_diff = current_time - exit_time
            distance_diff = abs(center_x - exit_x)
            
            if (time_diff <= self.reentry_time_threshold and 
                distance_diff <= self.reentry_distance_threshold and
                vehicle_type == exit_type):
                # Found a match - remove from recent exits and return the original ID
                del self.recent_exits[exit_id]
                return exit_id
        
        return None

    def get_original_id(self, track_id):
        """Get the original track ID if this ID was remapped"""
        return self.id_mapping.get(track_id, track_id)

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        print(full_msg)
        
        # Write vehicle detection events to the log file
        if "SAW" in message and any(v in message for v in ["CAR", "TRUCK", "MOTORCYCLE", "BUS"]):
            with open(self.log_file, "a") as f:
                f.write(full_msg + "\n")

    def save_snapshot(self, frame, track_id, speed_mph, direction, duration, distance):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        vehicle_type = self.vehicle_types.get(track_id, "Vehicle").lower()
        filename = f"snapshots/{vehicle_type}_{track_id}_{timestamp}.jpg"
        
        # Draw info on frame copy
        snapshot = frame.copy()
        cv2.putText(snapshot, f"ID: {track_id}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(snapshot, f"Type: {self.vehicle_types.get(track_id, 'Vehicle')}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(snapshot, f"Speed: {speed_mph:.1f} MPH", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(snapshot, f"Count: {self.car_count}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imwrite(filename, snapshot)
        print(f"[Snapshot] Saved to {filename}")
        
        # Save to database
        try:
            db_timestamp = datetime.now().isoformat()
            database.add_observation(
                observation_number=self.car_count,
                vehicle_type=self.vehicle_types.get(track_id, 'Vehicle'),
                timestamp=db_timestamp,
                direction=direction,
                duration=duration,
                distance=distance,
                speed=speed_mph,
                image_filename=filename
            )
        except Exception as e:
            print(f"[Database] Error saving observation: {e}")
        
        return filename

    def reconnect_stream(self):
        """Reconnect to the video stream"""
        print("Attempting to reconnect to stream...")
        self.cap.release()
        time.sleep(2)
        
        # Use same backend as initial connection
        if GSTREAMER_AVAILABLE and ('rtspsrc' in self.video_source or '!' in self.video_source):
            try:
                self.cap = GStreamerCapture(self.video_source)
            except Exception as e:
                print(f"Failed to recreate GStreamer pipeline: {e}")
                return False
        else:
            # OpenCV backend
            if 'rtspsrc' in self.video_source or '!' in self.video_source:
                # Extract RTSP URL from pipeline
                if 'location=' in self.video_source:
                    rtsp_url = self.video_source.split('location=')[1].split(' ')[0]
                    self.cap = cv2.VideoCapture(rtsp_url)
                else:
                    return False
            else:
                self.cap = cv2.VideoCapture(self.video_source)
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("Successfully reconnected to stream")
            return True
        print("Failed to reconnect to stream")
        return False

    def process_video(self):
        retry_count = 0
        reconnect_attempts = 0
        frame_count = 0
        
        # Use frame-based timestamps instead of wall clock for accurate speed calculations
        # This accounts for frame drops and processing delays
        frame_timestamp = 0.0  # seconds from start
        frame_interval = 1.0 / self.fps if self.fps > 0 else 0.167  # fallback to ~6fps
        
        # Start background frame reader thread to prevent drops during inference
        self.reader_running = True
        self.frame_reader_thread = threading.Thread(target=self._frame_reader_loop, daemon=True)
        self.frame_reader_thread.start()
        print("[Main] Frame reader thread started, reading from queue...")
        
        while self.running:
            if not self.cap.isOpened():
                if not self.reconnect_stream():
                    reconnect_attempts += 1
                    if reconnect_attempts > 3:
                        print("Error: Failed to reconnect after 3 attempts. Exiting.")
                        break
                    time.sleep(5)
                    continue
                reconnect_attempts = 0
            
            # Read frame from queue (already downsampled by reader thread)
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                retry_count += 1
                if retry_count > 10:
                    print("Warning: No frames received for 10 seconds, stream may be disconnected")
                    retry_count = 0
                continue
            
            # Reset retry count on success
            retry_count = 0
            
            # Increment frame timestamp (accounts for actual frame delivery, not processing time)
            frame_timestamp += frame_interval

            # Crop frame to ROI for faster processing
            roi_frame = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            if roi_frame.size == 0:
                continue

            # Skip inference if SKIP_INFERENCE mode is enabled (for testing)
            if SKIP_INFERENCE:
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Decode-only mode: Processed {frame_count} frames (no inference)")
                continue

            # Check for motion before running expensive YOLO inference
            has_motion = self.detect_motion(frame)
            frame_count += 1
            
            if not has_motion:
                # No motion detected - skip inference
                self.inference_skipped_count += 1
                if frame_count % 100 == 0:
                    skip_percent = (self.inference_skipped_count / frame_count) * 100
                    queue_size = self.frame_queue.qsize()
                    print(f"Processed {frame_count} frames (skipped {skip_percent:.1f}% due to no motion) [queue: {queue_size}/10]")
            
            else:
                # Motion detected - run inference
                self.motion_detected_count += 1

                # Run YOLOv8 tracking on the CROP
                # We don't need imgsz=1280 anymore because the crop is small and focused
                results = self.model.track(roi_frame, persist=True, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
                
                current_roi_ids = set()
                
                if results[0].boxes.id is not None:
                    boxes_xywh = results[0].boxes.xywh.cpu()
                    boxes_xyxy = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes = results[0].boxes.cls.int().cpu().tolist()
                    
                    for xywh, xyxy, track_id, class_id in zip(boxes_xywh, boxes_xyxy, track_ids, classes):
                        # Adjust coordinates from ROI-relative to Frame-relative
                        x_roi, y_roi, w, h = xywh
                        x1_roi, y1_roi, x2_roi, y2_roi = xyxy
                        
                        x = x_roi + self.roi_x1
                        y = y_roi + self.roi_y1
                        
                        x1 = x1_roi + self.roi_x1
                        y1 = y1_roi + self.roi_y1
                        x2 = x2_roi + self.roi_x1
                        y2 = y2_roi + self.roi_y1
                        
                        center_y = float(y)
                        center_x = float(x)
                        
                        # Check if center point is inside ROI (It should be, since we cropped, but keep safety)
                        if not (self.roi_x1 <= center_x <= self.roi_x2 and self.roi_y1 <= center_y <= self.roi_y2):
                            continue
                        
                        current_roi_ids.add(track_id)
                        
                        # Store vehicle type
                        vehicle_type = CLASS_NAMES.get(class_id, "Vehicle")
                        self.vehicle_types[track_id] = vehicle_type
                        
                        # Track first and last seen positions (using frame timestamps, not wall clock)
                        
                        # Log entry and record first position
                        if track_id not in self.ids_in_roi:
                            # Check if this is likely a re-entry of a recently exited vehicle (ID reassignment)
                            original_id = self.find_matching_recent_exit(center_x, vehicle_type, frame_timestamp)
                            
                            if original_id is not None:
                                # This is likely the same vehicle with a new ID
                                self.id_mapping[track_id] = original_id
                                # Copy tracking data from original ID
                                if original_id in self.first_seen:
                                    self.first_seen[track_id] = self.first_seen[original_id]
                                else:
                                    # Original didn't have first_seen (shouldn't happen now, but safety)
                                    self.first_seen[track_id] = (center_x, frame_timestamp)
                                if original_id in self.vehicle_speeds:
                                    self.vehicle_speeds[track_id] = self.vehicle_speeds[original_id]
                                self.log_message(f"{vehicle_type} {track_id} re-entered (merged with ID {original_id}) at X={center_x:.1f}")
                            else:
                                # New vehicle entry
                                self.first_seen[track_id] = (center_x, frame_timestamp)
                                self.log_message(f"{vehicle_type} {track_id} entered ROI at X={center_x:.1f}")
                        
                        # Always update last seen position
                        self.last_seen[track_id] = (center_x, frame_timestamp)
                        
                        # Store track history for visualization
                        track = self.track_history[track_id]
                        track.append((center_x, center_y))
                        if len(track) > 30:  # retain 30 frames
                            track.pop(0)

                        # Draw bounding box and ID
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        
                        # Draw label with type and speed if available
                        vehicle_type = self.vehicle_types.get(track_id, "Vehicle")
                        label = f"{vehicle_type} {track_id}"
                        if track_id in self.vehicle_speeds:
                            label += f" {self.vehicle_speeds[track_id]:.1f} MPH"
                        
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Check exits and calculate speed
                for track_id in self.ids_in_roi:
                    if track_id not in current_roi_ids:
                        vehicle_type = self.vehicle_types.get(track_id, "Vehicle")
                        
                        # Calculate speed if we have first and last positions
                        if track_id in self.first_seen and track_id in self.last_seen:
                            first_x, first_time = self.first_seen[track_id]
                            last_x, last_time = self.last_seen[track_id]
                            
                            duration = last_time - first_time
                            distance_pixels = abs(last_x - first_x)
                            distance_meters = distance_pixels / self.pixels_per_meter
                            
                            # Ignore tracking flicker - if vehicle hasn't moved >50 pixels (~2m) from entry in <5s, it's not really gone
                            # This prevents spam from vehicles stopping/slow-moving with unstable tracking
                            # BUT: if duration is exactly 0, the vehicle was only seen for 1 frame - let it exit cleanly
                            if distance_pixels < 50 and 0.1 < duration < 5.0:
                                # Tracking flicker - log once per vehicle ID to help debug
                                if track_id not in getattr(self, '_flicker_logged', set()):
                                    if not hasattr(self, '_flicker_logged'):
                                        self._flicker_logged = set()
                                    self._flicker_logged.add(track_id)
                                    self.log_message(f"{vehicle_type} {track_id} appears stationary (moved {distance_meters:.1f}m in {duration:.1f}s), continuing to track")
                                continue
                            
                            # Only calculate if vehicle moved reasonable distance OR time
                            # Ultra-low thresholds to catch brief detections (0.05s, 0.2m = ~8 inches)
                            if duration > 0.05 and distance_meters > 0.2:
                                speed_mps = distance_meters / duration
                                speed_mph = speed_mps * 2.23694
                                
                                # Determine direction
                                if last_x > first_x:
                                    direction = "northbound"
                                else:
                                    direction = "southbound"
                                
                                self.vehicle_speeds[track_id] = speed_mph
                                
                                # Get the original ID (in case this is a re-entry with new ID)
                                original_id = self.id_mapping.get(track_id, track_id)
                                
                                # Only count if this vehicle hasn't been counted yet
                                if original_id not in self.counted_ids:
                                    self.car_count += 1
                                    self.counted_ids.add(original_id)
                                    self.log_message(f"SAW {vehicle_type.upper()}: id={track_id}, speed={speed_mph:.1f} MPH, dir={direction}, distance={distance_meters:.1f}m, time={duration:.1f}s, total count={self.car_count}")
                                    self.save_snapshot(frame, track_id, speed_mph, direction, duration, distance_meters)
                                else:
                                    self.log_message(f"DUPLICATE AVOIDED: {vehicle_type} {track_id} (original_id={original_id}) already counted, speed={speed_mph:.1f} MPH")
                            else:
                                # Insufficient data - might be a tracking glitch or edge detection
                                if track_id in self.last_seen:
                                    exit_x, exit_time = self.last_seen[track_id]
                                    self.recent_exits[track_id] = (exit_x, exit_time, vehicle_type)
                                # Log with more detail for debugging
                                if duration == 0.0:
                                    self.log_message(f"{vehicle_type} {track_id} exited ROI (single-frame detection at X={first_x:.1f})")
                                else:
                                    self.log_message(f"{vehicle_type} {track_id} exited ROI (insufficient data: {distance_meters:.1f}m in {duration:.1f}s)")
                        else:
                            # No tracking data - add to recent exits
                            if track_id in self.last_seen:
                                exit_x, exit_time = self.last_seen[track_id]
                                self.recent_exits[track_id] = (exit_x, exit_time, vehicle_type)
                            self.log_message(f"{vehicle_type} {track_id} exited ROI")
                        
                        # DON'T cleanup tracking data yet - keep it for re-entry merging
                        # It will be cleaned up with recent_exits after the reentry window expires
                
                self.ids_in_roi = current_roi_ids
                
                # Clean up old recent exits (older than threshold, using frame timestamps)
                expired_exits = [
                    exit_id for exit_id, (_, exit_time, _) in self.recent_exits.items()
                    if frame_timestamp - exit_time > self.reentry_time_threshold
                ]
                for exit_id in expired_exits:
                    del self.recent_exits[exit_id]
                    # Also clean up tracking data for this expired exit
                    self.first_seen.pop(exit_id, None)
                    self.last_seen.pop(exit_id, None)
                    self.vehicle_speeds.pop(exit_id, None)

            # Visualize ROI
            cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 0, 255), 1)
            
            # Visualize lines (Vertical)
            cv2.line(frame, (self.line_left_x, self.roi_y1), (self.line_left_x, self.roi_y2), (0, 255, 255), 2)
            cv2.line(frame, (self.line_right_x, self.roi_y1), (self.line_right_x, self.roi_y2), (0, 255, 0), 2)

            # Draw Total Count
            cv2.putText(frame, f"Count: {self.car_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the frame (only in GUI mode)
            if not self.headless:
                cv2.imshow("Car Counter", frame)
                # Break on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # In headless mode, just keep processing
                if frame_count % 100 == 0:
                    queue_size = self.frame_queue.qsize()
                    if MOTION_DETECTION:
                        skip_percent = (self.inference_skipped_count / frame_count) * 100
                        print(f"Processed {frame_count} frames, detected {self.car_count} vehicles (skipped {skip_percent:.1f}% due to no motion) [queue: {queue_size}/10]")
                    else:
                        print(f"Processed {frame_count} frames, detected {self.car_count} vehicles [queue: {queue_size}/10]")
        
        # Stop frame reader thread
        self.reader_running = False
        if self.frame_reader_thread and self.frame_reader_thread.is_alive():
            print("[Main] Waiting for frame reader thread to stop...")
            self.frame_reader_thread.join(timeout=2.0)
        
        self.cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        print(f"\nFinal count: {self.car_count} vehicles detected")

if __name__ == "__main__":
    # Check for GPU
    import torch
    if torch.cuda.is_available():
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        print("Using MPS (Mac Metal)")
    else:
        print("Using CPU")

    counter = CarCounter(VIDEO_SOURCE)
    counter.process_video()
