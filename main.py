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
DISTANCE_METERS = 13.0

# Maximum time allowed between line crossings (seconds) - prevents false positives from stale timers
MAX_CROSSING_DURATION = 10.0

# Headless mode (no GUI window) - useful for Docker/server deployment
HEADLESS = os.getenv('HEADLESS', 'true').lower() == 'true'

# Skip YOLO inference (for testing decode-only CPU usage)
SKIP_INFERENCE = os.getenv('SKIP_INFERENCE', 'false').lower() == 'true'

# Motion detection settings
MOTION_DETECTION = os.getenv('MOTION_DETECTION', 'true').lower() == 'true'
MOTION_THRESHOLD = int(os.getenv('MOTION_THRESHOLD', '2000'))  # Pixels changed threshold
MOTION_MIN_AREA = int(os.getenv('MOTION_MIN_AREA', '500'))    # Minimum contour area

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
ROI_TOP_LEFT_Y = 0.0
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
        self.snapshot_frames = {}  # track_id -> frame (captured at midpoint for best view)
        self.ids_in_roi = set()    # track IDs currently in ROI
        self.pending_exits = {}    # track_id -> (exit_time, vehicle_type) - vehicles that left ROI, waiting to confirm and count
        self.id_mapping = {}       # new_id -> original_id - for tracking ID reassignments
        self.counted_ids = set()   # track IDs that have already been counted (prevents duplicate counts)
        self.exit_confirmation_time = 0.5  # seconds to wait before finalizing count (allow tracking glitches to resolve)
        self.car_count = 0
        self.headless = headless
        self.running = True
        
        # Frame timestamp tracking
        self.last_frame_capture_time = None  # Capture timestamp of last processed frame
        
        # Motion detection state
        self.last_motion_bbox = None  # Last detected motion bounding box
        self.motion_bbox_reuse_count = 0  # How many frames we've reused the bbox
        
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
        
        # Thresholds for merging rapid re-entries (scaled with VIDEO_SCALE)
        self.reentry_time_threshold = 2.0  # seconds - increased to catch more re-entries
        self.reentry_distance_threshold = int(800 * VIDEO_SCALE)  # pixels, scaled - increased for moving vehicles
        self.stationary_distance_threshold = int(50 * VIDEO_SCALE)  # pixels, scaled
        
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
        self.last_motion_time = 0.0  # Frame timestamp of last motion
        self.motion_detected_count = 0
        self.inference_skipped_count = 0
        
        # Frame queue for async reading (prevents frame drops during inference)
        self.frame_queue = queue.Queue(maxsize=40)  # Buffer up to 20 frames (4 seconds)
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
                # Capture wall clock timestamp when frame is read
                capture_time = time.time()
                # Downsample frame if needed
                if VIDEO_SCALE != 1.0:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                
                # Try to add frame and its capture timestamp to queue (non-blocking)
                try:
                    self.frame_queue.put((frame, capture_time), block=False)
                except queue.Full:
                    # Queue full, drop oldest frame and try again
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame, capture_time), block=False)
                    except (queue.Empty, queue.Full):
                        pass
            else:
                time.sleep(0.01)  # Brief pause on read failure
        
        print("[Frame Reader] Thread stopped")
    
    def detect_motion(self, frame, frame_timestamp):
        """Detect motion in ROI using frame differencing
        Returns: (has_motion, motion_debug_info)
        """
        if not MOTION_DETECTION:
            return True, None  # Always run inference if motion detection disabled
        
        # Always run inference if we're actively tracking vehicles
        if len(self.ids_in_roi) > 0:
            self.last_motion_time = frame_timestamp  # Reset timer while tracking
            return True, {'reason': 'tracking', 'changed_pixels': 0, 'largest_contour': 0}
        
        # Extract ROI
        roi = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # First frame initialization
        if self.prev_gray is None:
            self.prev_gray = gray
            return True, {'reason': 'init', 'changed_pixels': 0, 'largest_contour': 0}
        
        # Compute absolute difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours of motion regions
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count total changed pixels and check for large contours
        changed_pixels = cv2.countNonZero(thresh)
        largest_contour_area = max([cv2.contourArea(c) for c in contours], default=0)
        has_large_contour = largest_contour_area > MOTION_MIN_AREA
        
        # Calculate bounding box around all motion (for targeted inference)
        motion_bbox = None
        if has_large_contour and len(contours) > 0:
            # Get bounding boxes of all significant contours
            significant_contours = [c for c in contours if cv2.contourArea(c) > MOTION_MIN_AREA]
            if significant_contours:
                # Find overall bounding box encompassing all motion
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = 0, 0
                for contour in significant_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
                
                # Add generous padding for YOLO context (150 pixels on each side)
                # YOLO needs to see the whole vehicle, not just the motion pixels
                x_padding = 150
                y_padding = 100
                roi_width = self.roi_x2 - self.roi_x1
                roi_height = self.roi_y2 - self.roi_y1
                x_min = max(0, x_min - x_padding)
                y_min = max(0, y_min - y_padding)
                x_max = min(roi_width + x_padding, x_max + x_padding)
                y_max = min(roi_height + y_padding, y_max + y_padding)
                
                # Ensure minimum size for YOLO to work properly (at least 400x400)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                min_size = 400
                if bbox_width < min_size:
                    expand = (min_size - bbox_width) / 2
                    x_min = max(0, x_min - expand)
                    x_max = min(roi_width + x_padding, x_max + expand)
                if bbox_height < min_size:
                    expand = (min_size - bbox_height) / 2
                    y_min = max(0, y_min - expand)
                    y_max = min(roi_height + y_padding, y_max + expand)
                
                motion_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                # Fresh detection - reset reuse count and store bbox
                self.last_motion_bbox = motion_bbox
                self.motion_bbox_reuse_count = 0
        else:
            # No contours detected - reuse last bbox for 1 frame if available
            if self.last_motion_bbox is not None and self.motion_bbox_reuse_count < 1:
                motion_bbox = self.last_motion_bbox
                self.motion_bbox_reuse_count += 1
            else:
                # Either no previous bbox or already reused it, clear everything
                self.last_motion_bbox = None
                self.motion_bbox_reuse_count = 0
        
        # Store debug info
        debug_info = {
            'changed_pixels': changed_pixels,
            'largest_contour': largest_contour_area,
            'thresh_mask': thresh,
            'contours': contours,
            'has_large_contour': has_large_contour,
            'motion_bbox': motion_bbox,
            'bbox_reused': self.motion_bbox_reuse_count > 0  # Flag if we're reusing last frame's bbox
        }
        
        # Update previous frame
        self.prev_gray = gray
        
        # Motion detected if enough pixels changed OR we have a solid motion region
        if changed_pixels > MOTION_THRESHOLD or has_large_contour:
            self.last_motion_time = frame_timestamp
            debug_info['reason'] = 'motion'
            return True, debug_info
        
        # Continue running inference for 5 seconds after motion stops
        # Frame-based grace period ensures consistent behavior regardless of processing speed
        time_since_motion = frame_timestamp - self.last_motion_time
        if time_since_motion < 5.0:  # 5 second grace period
            debug_info['reason'] = f'grace ({5.0 - time_since_motion:.1f}s left)'
            return True, debug_info
        
        debug_info['reason'] = 'no_motion'
        return False, debug_info

    def find_matching_pending_exit(self, center_x, vehicle_type, current_time):
        """Check if this entry matches a pending exit (vehicle re-entering, or tracking glitch resolved)"""
        best_match = None
        best_distance = float('inf')
        
        for exit_id, (exit_time, exit_type) in list(self.pending_exits.items()):
            # Check if exit was recent and vehicle type matches
            time_diff = current_time - exit_time
            
            if vehicle_type != exit_type:
                continue
                
            if time_diff > self.reentry_time_threshold:
                continue
            
            # Use last known position for distance check
            if exit_id in self.last_seen:
                exit_x, _ = self.last_seen[exit_id]
                distance_diff = abs(center_x - exit_x)
                
                # If moving in same direction (vehicle progressing through ROI), be more lenient
                # Northbound = X increasing, Southbound = X decreasing
                is_progressing = (center_x > exit_x) if exit_x < self.line_left_x or center_x > self.line_right_x else (center_x < exit_x)
                
                # Use larger threshold if vehicle is progressing in expected direction
                threshold = self.reentry_distance_threshold * 1.5 if is_progressing else self.reentry_distance_threshold
                
                if distance_diff <= threshold:
                    # Track best match (closest vehicle)
                    if distance_diff < best_distance:
                        best_distance = distance_diff
                        best_match = exit_id
        
        # Return and remove the best match
        if best_match is not None:
            del self.pending_exits[best_match]
            return best_match
        
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
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        timestamp_display = time.strftime("%Y-%m-%d %H:%M:%S")
        vehicle_type = self.vehicle_types.get(track_id, "Vehicle").lower()
        
        # Save original full frame (as-is, no overlay)
        filename_full = f"snapshots/{vehicle_type}_{track_id}_{timestamp_str}_full.jpg"
        cv2.imwrite(filename_full, frame)
        
        # Create ROI-only snapshot with text overlay
        filename_roi = f"snapshots/{vehicle_type}_{track_id}_{timestamp_str}.jpg"
        roi_snapshot = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2].copy()
        
        # Add text overlay to ROI snapshot in corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color_white = (255, 255, 255)
        color_black = (0, 0, 0)
        
        # Text with black outline for visibility
        y_offset = 30
        
        # Format speed/direction line
        if speed_mph > 0 and direction != "unknown":
            speed_text = f"Speed: {speed_mph:.1f} MPH {direction}"
        else:
            speed_text = "Speed: No data"
        
        texts = [
            f"{timestamp_display}",
            f"ID: {track_id}",
            f"Type: {self.vehicle_types.get(track_id, 'Vehicle')}",
            speed_text,
            f"Count: {self.car_count}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = y_offset + (i * 35)
            # Black outline
            cv2.putText(roi_snapshot, text, (12, y_pos), font, font_scale, color_black, thickness + 1)
            # White text
            cv2.putText(roi_snapshot, text, (10, y_pos - 2), font, font_scale, color_white, thickness)
        
        cv2.imwrite(filename_roi, roi_snapshot)
        print(f"[Snapshot] Saved full: {filename_full}")
        print(f"[Snapshot] Saved ROI: {filename_roi}")
        
        # Save to database (use ROI filename for website display)
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
                image_filename=filename_roi
            )
        except Exception as e:
            print(f"[Database] Error saving observation: {e}")
        
        return filename_roi

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
        motion_debug = None  # For motion detection visualization
        
        # Use frame-based timestamps instead of wall clock for accurate speed calculations
        # This accounts for frame drops and processing delays
        frame_timestamp = 0.0  # seconds from start
        frame_interval = 1.0 / self.fps if self.fps > 0 else 0.167  # initial estimate, will be updated
        interval_reported = False  # Track if we've logged the interval choice
        
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
            
            # Read frame and capture timestamp from queue
            try:
                frame, capture_time = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                retry_count += 1
                if retry_count > 10:
                    print("Warning: No frames received for 10 seconds, stream may be disconnected")
                    retry_count = 0
                continue
            
            # Reset retry count on success
            retry_count = 0
            
            # Calculate actual interval from capture timestamps
            if self.last_frame_capture_time is not None:
                frame_interval = capture_time - self.last_frame_capture_time
                # Log FPS once for diagnostics
                if not interval_reported:
                    interval_reported = True
                    actual_fps = 1.0 / frame_interval if frame_interval > 0 else 0
                    print(f"[Frame Timing] Actual frame capture interval: {frame_interval:.4f}s ({actual_fps:.2f} FPS)")
                    print(f"[Frame Timing] Reported stream FPS: {self.fps:.2f} FPS")
                    print(f"[Frame Timing] Using ACTUAL capture timestamps for accurate timing")
            else:
                # First frame - use reported FPS as initial estimate
                frame_interval = 1.0 / self.fps if self.fps > 0 else 0.167
            
            self.last_frame_capture_time = capture_time
            
            # Increment frame timestamp by actual interval between captures
            frame_timestamp += frame_interval

            # Crop frame to ROI for faster processing
            roi_frame = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            if roi_frame.size == 0:
                continue

            # Skip inference if SKIP_INFERENCE mode is enabled (for testing)
            if SKIP_INFERENCE:
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Decode-only mode: Processed {frame_count} frames (no inference) {1.0 / frame_interval:.2f} FPS")
                continue

            # Check for motion before running expensive YOLO inference
            has_motion, motion_debug = self.detect_motion(frame, frame_timestamp)
            frame_count += 1
            
            if not has_motion:
                # No motion detected - skip inference
                self.inference_skipped_count += 1
            
            else:
                # Motion detected - run inference
                self.motion_detected_count += 1

                # Use motion bounding box only when NOT actively tracking
                # When tracking, use full ROI to maintain consistent context for YOLO tracker
                inference_frame = roi_frame
                motion_offset_x = 0
                motion_offset_y = 0
                
                if len(self.ids_in_roi) == 0 and motion_debug and motion_debug.get('motion_bbox'):
                    # No active tracking - use motion bbox for faster inference
                    mx1, my1, mx2, my2 = motion_debug['motion_bbox']
                    inference_frame = roi_frame[my1:my2, mx1:mx2]
                    motion_offset_x = mx1
                    motion_offset_y = my1
                    print(f"Inference frame size: {inference_frame.shape} (motion crop), queue size: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")
                else:
                    # Active tracking - use full ROI for stable tracking
                    if len(self.ids_in_roi) > 0:
                        print(f"Inference frame size: {inference_frame.shape} (full ROI, tracking {len(self.ids_in_roi)} vehicles), queue size: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")
                    else:
                        print(f"Inference frame size: {inference_frame.shape} (full ROI), queue size: {self.frame_queue.qsize()}/{self.frame_queue.maxsize}")

                # Run YOLOv8 tracking on the cropped area
                results = self.model.track(inference_frame, persist=True, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)
                
                current_roi_ids = set()
                
                if results[0].boxes.id is not None:
                    boxes_xywh = results[0].boxes.xywh.cpu()
                    boxes_xyxy = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes = results[0].boxes.cls.int().cpu().tolist()
                    
                    for xywh, xyxy, track_id, class_id in zip(boxes_xywh, boxes_xyxy, track_ids, classes):
                        # Adjust coordinates from inference-crop to ROI to Frame
                        x_crop, y_crop, w, h = xywh
                        x1_crop, y1_crop, x2_crop, y2_crop = xyxy
                        
                        # First adjust from crop to ROI space
                        x_roi = x_crop + motion_offset_x
                        y_roi = y_crop + motion_offset_y
                        x1_roi = x1_crop + motion_offset_x
                        y1_roi = y1_crop + motion_offset_y
                        x2_roi = x2_crop + motion_offset_x
                        y2_roi = y2_crop + motion_offset_y
                        
                        # Then adjust from ROI to frame space
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
                        
                        # Store vehicle type
                        vehicle_type = CLASS_NAMES.get(class_id, "Vehicle")
                        self.vehicle_types[track_id] = vehicle_type
                        
                        # Skip tracking if this ID was already counted (prevents re-counting same vehicle)
                        if track_id in self.counted_ids:
                            # This ID was already counted - likely the same car re-entering or lingering
                            # Don't track it again to avoid duplicate counts
                            if track_id not in self.ids_in_roi:
                                self.log_message(f"{vehicle_type} {track_id} re-entered ROI but was already counted, ignoring")
                            continue
                        
                        current_roi_ids.add(track_id)
                        
                        # Track first and last seen positions (using frame timestamps, not wall clock)
                        
                        # Log entry and record first position
                        if track_id not in self.ids_in_roi:
                            
                            # Check if this is a re-entry of a pending exit (tracking glitch or ID reassignment)
                            original_id = self.find_matching_pending_exit(center_x, vehicle_type, frame_timestamp)
                            
                            if original_id is not None:
                                # This is the same vehicle - tracking glitch resolved or ID reassignment
                                self.id_mapping[track_id] = original_id
                                # Copy tracking data from original ID
                                if original_id in self.first_seen:
                                    self.first_seen[track_id] = self.first_seen[original_id]
                                else:
                                    # Original didn't have first_seen (shouldn't happen now, but safety)
                                    self.first_seen[track_id] = (center_x, frame_timestamp)
                                if original_id in self.vehicle_speeds:
                                    self.vehicle_speeds[track_id] = self.vehicle_speeds[original_id]
                                if original_id in self.snapshot_frames:
                                    self.snapshot_frames[track_id] = self.snapshot_frames[original_id]
                                # Calculate distance for logging
                                if original_id in self.last_seen:
                                    exit_x, _ = self.last_seen[original_id]
                                    distance = abs(center_x - exit_x)
                                    self.log_message(f"{vehicle_type} {track_id} resumed tracking (merged with ID {original_id}, moved {distance:.0f}px) at X={center_x:.1f}")
                                else:
                                    self.log_message(f"{vehicle_type} {track_id} resumed tracking (merged with ID {original_id}) at X={center_x:.1f}")
                            else:
                                # New vehicle entry
                                self.first_seen[track_id] = (center_x, frame_timestamp)
                                self.log_message(f"{vehicle_type} {track_id} entered ROI at X={center_x:.1f}")
                                # Debug: show pending exits to help diagnose merge failures
                                if len(self.pending_exits) > 0:
                                    pending_info = []
                                    for pid, (ptime, ptype) in self.pending_exits.items():
                                        if pid in self.last_seen:
                                            px, _ = self.last_seen[pid]
                                            dist = abs(center_x - px)
                                            time_diff = frame_timestamp - ptime
                                            pending_info.append(f"ID {pid} ({ptype}) at X={px:.0f}, dist={dist:.0f}px, time={time_diff:.1f}s")
                                    if pending_info:
                                        self.log_message(f"  Pending exits: {'; '.join(pending_info)}")
                        
                        # Always update last seen position
                        self.last_seen[track_id] = (center_x, frame_timestamp)
                        
                        # Capture snapshot frame when vehicle is well into ROI for best image
                        # Capture at midpoint between entry and current position, or on first few frames
                        if track_id in self.first_seen and track_id not in self.snapshot_frames:
                            # Capture on first detection
                            self.snapshot_frames[track_id] = frame.copy()
                        elif track_id in self.first_seen:
                            # Update snapshot if vehicle is near middle of travel distance
                            first_x, _ = self.first_seen[track_id]
                            distance_traveled = abs(center_x - first_x)
                            roi_width = self.roi_x2 - self.roi_x1
                            # Update snapshot when vehicle is roughly in middle third of ROI
                            if distance_traveled > roi_width * 0.2 and distance_traveled < roi_width * 0.6:
                                self.snapshot_frames[track_id] = frame.copy()
                        
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
                
                # Check for vehicles that left ROI - add to pending exits
                for track_id in self.ids_in_roi:
                    if track_id not in current_roi_ids:
                        vehicle_type = self.vehicle_types.get(track_id, "Vehicle")
                        
                        # Vehicle left ROI - add to pending exits for confirmation
                        # Don't count yet, wait to see if it re-enters (tracking glitch) or truly exits
                        self.pending_exits[track_id] = (frame_timestamp, vehicle_type)
                        
                        # Log exit for debugging
                        if track_id in self.last_seen:
                            last_x, _ = self.last_seen[track_id]
                            self.log_message(f"{vehicle_type} {track_id} left ROI at X={last_x:.1f} (pending confirmation)")
                        else:
                            self.log_message(f"{vehicle_type} {track_id} left ROI (pending confirmation)")
                
                self.ids_in_roi = current_roi_ids
                
                # Process pending exits - finalize count for vehicles confirmed gone
                confirmed_exits = []
                for track_id, (exit_time, vehicle_type) in list(self.pending_exits.items()):
                    time_since_exit = frame_timestamp - exit_time
                    
                    # Vehicle has been gone long enough - finalize the count
                    if time_since_exit >= self.exit_confirmation_time:
                        confirmed_exits.append(track_id)
                        
                        # Calculate final speed if we have tracking data
                        if track_id in self.first_seen and track_id in self.last_seen:
                            first_x, first_time = self.first_seen[track_id]
                            last_x, last_time = self.last_seen[track_id]
                            
                            duration = last_time - first_time
                            distance_pixels = abs(last_x - first_x)
                            distance_meters = distance_pixels / self.pixels_per_meter
                            
                            # Check for stationary/tracking glitch
                            if distance_pixels < self.stationary_distance_threshold and 0.1 < duration < 5.0:
                                self.log_message(f"{vehicle_type} {track_id} appears stationary (moved {distance_meters:.1f}m in {duration:.1f}s), not counting")
                                continue
                            
                            # Only count if vehicle moved reasonable distance
                            if duration > 0.05 and distance_meters > 0.2:
                                speed_mps = distance_meters / duration
                                speed_mph = speed_mps * 2.23694
                                
                                # Determine direction
                                direction = "northbound" if last_x > first_x else "southbound"
                                
                                self.vehicle_speeds[track_id] = speed_mph
                                
                                # Get original ID and check if already counted
                                original_id = self.id_mapping.get(track_id, track_id)
                                
                                if original_id not in self.counted_ids:
                                    self.car_count += 1
                                    self.counted_ids.add(original_id)
                                    self.log_message(f"SAW {vehicle_type.upper()}: id={track_id}, speed={speed_mph:.1f} MPH, dir={direction}, distance={distance_meters:.1f}m, time={duration:.1f}s, total count={self.car_count}")
                                    # Use stored snapshot frame if available, otherwise use current frame
                                    snapshot_frame = self.snapshot_frames.get(track_id, frame)
                                    self.save_snapshot(snapshot_frame, track_id, speed_mph, direction, duration, distance_meters)
                                else:
                                    self.log_message(f"DUPLICATE AVOIDED: {vehicle_type} {track_id} (original_id={original_id}) already counted, speed={speed_mph:.1f} MPH")
                            else:
                                # Insufficient movement data - but still count as detected
                                original_id = self.id_mapping.get(track_id, track_id)
                                
                                if original_id not in self.counted_ids:
                                    self.car_count += 1
                                    self.counted_ids.add(original_id)
                                    
                                    if duration == 0.0:
                                        self.log_message(f"SAW {vehicle_type.upper()}: id={track_id} (single-frame detection, no speed data), total count={self.car_count}")
                                    else:
                                        self.log_message(f"SAW {vehicle_type.upper()}: id={track_id} (insufficient data: {distance_meters:.1f}m in {duration:.1f}s, no speed), total count={self.car_count}")
                                    
                                    # Save snapshot without speed/direction info
                                    snapshot_frame = self.snapshot_frames.get(track_id, frame)
                                    self.save_snapshot(snapshot_frame, track_id, 0.0, "unknown", duration, distance_meters)
                                else:
                                    self.log_message(f"DUPLICATE AVOIDED: {vehicle_type} {track_id} (original_id={original_id}) already counted (insufficient data)")
                        else:
                            # No tracking data - but still count as detected
                            original_id = self.id_mapping.get(track_id, track_id)
                            
                            if original_id not in self.counted_ids:
                                self.car_count += 1
                                self.counted_ids.add(original_id)
                                self.log_message(f"SAW {vehicle_type.upper()}: id={track_id} (no tracking data), total count={self.car_count}")
                                
                                # Try to save snapshot if we have one
                                snapshot_frame = self.snapshot_frames.get(track_id, frame)
                                self.save_snapshot(snapshot_frame, track_id, 0.0, "unknown", 0.0, 0.0)
                            else:
                                self.log_message(f"DUPLICATE AVOIDED: {vehicle_type} {track_id} (original_id={original_id}) already counted (no tracking data)")
                
                # Clean up confirmed exits
                for track_id in confirmed_exits:
                    self.pending_exits.pop(track_id, None)
                    self.first_seen.pop(track_id, None)
                    self.last_seen.pop(track_id, None)
                    self.vehicle_speeds.pop(track_id, None)
                    self.vehicle_types.pop(track_id, None)
                    self.snapshot_frames.pop(track_id, None)
                    # Clean up counted_ids to allow ID reuse
                    original_id = self.id_mapping.get(track_id, track_id)
                    self.counted_ids.discard(original_id)
                    self.id_mapping.pop(track_id, None)

            # Visualize ROI
            cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 0, 255), 1)
            
            # Visualize lines (Vertical)
            cv2.line(frame, (self.line_left_x, self.roi_y1), (self.line_left_x, self.roi_y2), (0, 255, 255), 2)
            cv2.line(frame, (self.line_right_x, self.roi_y1), (self.line_right_x, self.roi_y2), (0, 255, 0), 2)

            # Draw Total Count
            cv2.putText(frame, f"Count: {self.car_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw motion detection debug info (only in GUI mode)
            if not self.headless and MOTION_DETECTION and motion_debug:
                y_offset = 80
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Motion status with colored background
                if motion_debug.get('reason') == 'motion':
                    status_color = (0, 255, 0)  # Green for active motion
                    status_text = "MOTION DETECTED"
                elif motion_debug.get('reason') == 'no_motion':
                    status_color = (0, 0, 255)  # Red for no motion
                    status_text = "NO MOTION"
                else:
                    status_color = (0, 255, 255)  # Yellow for other states
                    status_text = motion_debug.get('reason', '').upper()
                
                cv2.putText(frame, status_text, (20, y_offset), font, 0.7, status_color, 2)
                
                # Motion metrics
                changed_px = motion_debug.get('changed_pixels', 0)
                largest_contour = motion_debug.get('largest_contour', 0)
                
                cv2.putText(frame, f"Changed Pixels: {changed_px} / {MOTION_THRESHOLD}", 
                           (20, y_offset + 35), font, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Largest Contour: {int(largest_contour)} / {MOTION_MIN_AREA}", 
                           (20, y_offset + 70), font, 0.6, (0, 0, 255), 2)
                if changed_px > 10:
                    print(f"Changed Pixels: {changed_px} / {MOTION_THRESHOLD}")
                if largest_contour > 10:
                    print(f"Largest Contour: {int(largest_contour)} / {MOTION_MIN_AREA}")
                
                # Show inference area reduction
                if motion_debug.get('motion_bbox'):
                    mx1, my1, mx2, my2 = motion_debug['motion_bbox']
                    motion_area = (mx2 - mx1) * (my2 - my1)
                    roi_area = (self.roi_x2 - self.roi_x1) * (self.roi_y2 - self.roi_y1)
                    reduction_pct = 100.0 * (1 - motion_area / roi_area)
                    bbox_w = mx2 - mx1
                    bbox_h = my2 - my1
                    cv2.putText(frame, f"Inference: {bbox_w}x{bbox_h} ({reduction_pct:.0f}% smaller)", 
                               (20, y_offset + 105), font, 0.6, (255, 255, 0), 2)
                    print(f"Inference Area: {bbox_w}x{bbox_h} ({reduction_pct:.0f}% smaller)")
                
                # Draw motion contours on ROI
                if 'contours' in motion_debug and len(motion_debug['contours']) > 0:
                    for contour in motion_debug['contours']:
                        # Adjust contour coordinates from ROI to full frame
                        contour_adjusted = contour + [self.roi_x1, self.roi_y1]
                        area = cv2.contourArea(contour)
                        # Color code: green if above threshold, red if below
                        color = (0, 255, 0) if area > MOTION_MIN_AREA else (0, 0, 255)
                        cv2.drawContours(frame, [contour_adjusted], -1, color, 2)
                
                # Draw motion bounding box (inference area)
                if 'motion_bbox' in motion_debug and motion_debug['motion_bbox']:
                    mx1, my1, mx2, my2 = motion_debug['motion_bbox']
                    # Adjust from ROI space to frame space
                    bbox_x1 = mx1 + self.roi_x1
                    bbox_y1 = my1 + self.roi_y1
                    bbox_x2 = mx2 + self.roi_x1
                    bbox_y2 = my2 + self.roi_y1
                    
                    # Color and label based on usage
                    is_reused = motion_debug.get('bbox_reused', False)
                    is_tracking = len(self.ids_in_roi) > 0
                    
                    if is_tracking:
                        # Gray - motion bbox exists but not used (tracking active)
                        bbox_color = (128, 128, 128)
                        label = "Motion detected (using full ROI for tracking)"
                    elif is_reused:
                        # Orange - reused from previous frame
                        bbox_color = (0, 165, 255)
                        label = "Inference Area (reused)"
                    else:
                        # Cyan - fresh detection, actively used
                        bbox_color = (255, 255, 0)
                        label = "Inference Area"
                    
                    cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 3)
                    cv2.putText(frame, label, (bbox_x1 + 5, bbox_y1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
                
                # Optionally show threshold mask as overlay in corner
                if 'thresh_mask' in motion_debug:
                    thresh_resized = cv2.resize(motion_debug['thresh_mask'], 
                                               (int((self.roi_x2 - self.roi_x1) * 0.3), 
                                                int((self.roi_y2 - self.roi_y1) * 0.3)))
                    thresh_colored = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
                    # Place in top-right corner
                    h, w = thresh_resized.shape
                    frame[10:10+h, frame.shape[1]-w-10:frame.shape[1]-10] = thresh_colored

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
                        print(f"Processed {frame_count} frames, detected {self.car_count} vehicles (skipped {skip_percent:.1f}% due to no motion) [queue: {queue_size}/{self.frame_queue.maxsize}] {1.0 / frame_interval:.2f} FPS")
                    else:
                        print(f"Processed {frame_count} frames, detected {self.car_count} vehicles [queue: {queue_size}/{self.frame_queue.maxsize}] {1.0 / frame_interval:.2f} FPS")
        
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
