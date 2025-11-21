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
ROI_TOP_LEFT_X = 0.5
ROI_TOP_LEFT_Y = 0.0
ROI_BOTTOM_RIGHT_X = 0.86
ROI_BOTTOM_RIGHT_Y = 0.27 # Top half of the frame

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
        self.car_count = 0
        self.headless = headless
        self.running = True
        
        # Thresholds for merging rapid re-entries
        self.reentry_time_threshold = 1.0  # seconds
        self.reentry_distance_threshold = 200  # pixels
        
        # Create output directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("snapshots", exist_ok=True)
        self.log_file = "logs/car_log.txt"
        
        # Initialize database
        database.init_db()
        
        # Open video
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return

        # Set buffer size to 1 to reduce latency and avoid H.264 decoding errors on old frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        print(f"Speed Measurement Lines at X={self.line_left_x} and X={self.line_right_x}")
        print(f"ROI: ({self.roi_x1}, {self.roi_y1}) to ({self.roi_x2}, {self.roi_y2})")
        print(f"Mode: {'Headless' if self.headless else 'GUI'}")
        print(f"Video Scale: {VIDEO_SCALE}")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self.running = False

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
            
            success, frame = self.cap.read()
            
            if not success or frame is None:
                retry_count += 1
                print(f"Warning: Failed to read frame (Attempt {retry_count}/10)")
                if retry_count > 10:
                    print("Stream appears disconnected. Will attempt reconnection...")
                    retry_count = 0
                    self.cap.release()
                    time.sleep(1)
                    continue
                time.sleep(0.5)
                continue
            
            # Reset retry count on success
            retry_count = 0
            
            # Downsample frame if VIDEO_SCALE < 1.0
            if VIDEO_SCALE != 1.0:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            # Crop frame to ROI for faster processing
            roi_frame = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            if roi_frame.size == 0:
                continue

            # Run YOLOv8 tracking on the CROP
            # We don't need imgsz=1280 anymore because the crop is small and focused
            results = self.model.track(roi_frame, persist=True, classes=VEHICLE_CLASSES, verbose=False, conf=0.25)

            # Visualize ROI
            cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 0, 255), 1)
            
            # Visualize lines (Vertical)
            cv2.line(frame, (self.line_left_x, self.roi_y1), (self.line_left_x, self.roi_y2), (0, 255, 255), 2)
            cv2.line(frame, (self.line_right_x, self.roi_y1), (self.line_right_x, self.roi_y2), (0, 255, 0), 2)
            
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
                    
                    # Track first and last seen positions
                    current_time = time.time()
                    
                    # Log entry and record first position
                    if track_id not in self.ids_in_roi:
                        # Check if this is likely a re-entry of a recently exited vehicle (ID reassignment)
                        original_id = self.find_matching_recent_exit(center_x, vehicle_type, current_time)
                        
                        if original_id is not None:
                            # This is likely the same vehicle with a new ID
                            self.id_mapping[track_id] = original_id
                            # Copy tracking data from original ID
                            if original_id in self.first_seen:
                                self.first_seen[track_id] = self.first_seen[original_id]
                            if original_id in self.vehicle_speeds:
                                self.vehicle_speeds[track_id] = self.vehicle_speeds[original_id]
                            self.log_message(f"{vehicle_type} {track_id} re-entered (merged with ID {original_id}) at X={center_x:.1f}")
                        else:
                            # New vehicle entry
                            self.first_seen[track_id] = (center_x, current_time)
                            self.log_message(f"{vehicle_type} {track_id} entered ROI at X={center_x:.1f}")
                    
                    # Always update last seen position
                    self.last_seen[track_id] = (center_x, current_time)
                    
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
                        
                        # Only calculate if vehicle moved reasonable distance and time
                        if duration > 0.5 and distance_meters > 1.0:
                            speed_mps = distance_meters / duration
                            speed_mph = speed_mps * 2.23694
                            
                            # Determine direction
                            if last_x > first_x:
                                direction = "northbound"
                            else:
                                direction = "southbound"
                            
                            self.vehicle_speeds[track_id] = speed_mph
                            self.car_count += 1
                            self.log_message(f"SAW {vehicle_type.upper()}: id={track_id}, speed={speed_mph:.1f} MPH, dir={direction}, distance={distance_meters:.1f}m, time={duration:.1f}s, total count={self.car_count}")
                            self.save_snapshot(frame, track_id, speed_mph, direction, duration, distance_meters)
                        else:
                            # Insufficient data - might be a tracking glitch, add to recent exits
                            if track_id in self.last_seen:
                                exit_x, exit_time = self.last_seen[track_id]
                                self.recent_exits[track_id] = (exit_x, exit_time, vehicle_type)
                            self.log_message(f"{vehicle_type} {track_id} exited ROI (insufficient data: {distance_meters:.1f}m in {duration:.1f}s)")
                    else:
                        # No tracking data - add to recent exits
                        if track_id in self.last_seen:
                            exit_x, exit_time = self.last_seen[track_id]
                            self.recent_exits[track_id] = (exit_x, exit_time, vehicle_type)
                        self.log_message(f"{vehicle_type} {track_id} exited ROI")
                    
                    # Cleanup tracking data
                    self.first_seen.pop(track_id, None)
                    self.last_seen.pop(track_id, None)
            
            self.ids_in_roi = current_roi_ids
            
            # Clean up old recent exits (older than threshold)
            current_cleanup_time = time.time()
            expired_exits = [
                exit_id for exit_id, (_, exit_time, _) in self.recent_exits.items()
                if current_cleanup_time - exit_time > self.reentry_time_threshold
            ]
            for exit_id in expired_exits:
                del self.recent_exits[exit_id]

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
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames, detected {self.car_count} vehicles")
        
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
