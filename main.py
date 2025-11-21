import cv2
import time
import os
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import signal
import sys

# --- CONFIGURATION ---
# RTSP Stream URL (Replace with your camera's URL)
# Example: "rtsp://username:password@ip_address:port/stream"
VIDEO_SOURCE = "rtsp://frigate.thekropp.com:8554/driveway" # Public test stream or local file path

# Real world distance between the two lines (in meters)
DISTANCE_METERS = 10.0

# Maximum time allowed between line crossings (seconds) - prevents false positives from stale timers
MAX_CROSSING_DURATION = 10.0

# Headless mode (no GUI window) - useful for Docker/server deployment
HEADLESS = os.getenv('HEADLESS', 'true').lower() == 'true'

# Lines positions (0-1 relative to frame height/width)
# Vertical lines for left-to-right / right-to-left traffic
LINE_LEFT_X_RATIO = 0.63
LINE_RIGHT_X_RATIO = 0.76

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
    def __init__(self, video_source, model_path='yolov8n.pt', headless=HEADLESS):
        self.video_source = video_source
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.left_to_right_start_times = {} # track_id -> start_time
        self.right_to_left_start_times = {} # track_id -> start_time
        self.vehicle_speeds = {}   # track_id -> speed
        self.vehicle_types = {}    # track_id -> vehicle type
        self.ids_in_roi = set()    # track IDs currently in ROI
        self.car_count = 0
        self.headless = headless
        self.running = True
        
        # Create output directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("snapshots", exist_ok=True)
        self.log_file = "logs/car_log.txt"
        
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
        
        # Calculate line positions in pixels
        self.line_left_x = int(self.width * LINE_LEFT_X_RATIO)
        self.line_right_x = int(self.width * LINE_RIGHT_X_RATIO)
        
        # Calculate ROI in pixels
        self.roi_x1 = int(self.width * ROI_TOP_LEFT_X)
        self.roi_y1 = int(self.height * ROI_TOP_LEFT_Y)
        self.roi_x2 = int(self.width * ROI_BOTTOM_RIGHT_X)
        self.roi_y2 = int(self.height * ROI_BOTTOM_RIGHT_Y)
        
        print(f"Video Source: {self.width}x{self.height} @ {self.fps} FPS")
        print(f"Speed Measurement Lines at X={self.line_left_x} and X={self.line_right_x}")
        print(f"ROI: ({self.roi_x1}, {self.roi_y1}) to ({self.roi_x2}, {self.roi_y2})")
        print(f"Mode: {'Headless' if self.headless else 'GUI'}")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self.running = False

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        print(full_msg)
        
        # Write vehicle detection events to the log file
        if "SAW" in message and any(v in message for v in ["CAR", "TRUCK", "MOTORCYCLE", "BUS"]):
            with open(self.log_file, "a") as f:
                f.write(full_msg + "\n")

    def save_snapshot(self, frame, track_id, speed_mph):
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

    def process_video(self):
        retry_count = 0
        frame_count = 0
        while self.cap.isOpened() and self.running:
            success, frame = self.cap.read()
            
            if not success or frame is None:
                retry_count += 1
                print(f"Warning: Failed to read frame (Attempt {retry_count}/5)")
                if retry_count > 5:
                    print("Error: Video stream ended or failed repeatedly.")
                    break
                time.sleep(0.5)
                continue
            
            # Reset retry count on success
            retry_count = 0

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
                    
                    # Log entry
                    if track_id not in self.ids_in_roi:
                        self.log_message(f"{vehicle_type} {track_id} entered ROI at X={center_x:.1f}")
                    
                    # Store track history
                    track = self.track_history[track_id]
                    track.append((center_x, center_y))
                    if len(track) > 30:  # retain 30 frames
                        track.pop(0)

                    # Speed Estimation Logic (X-axis movement)
                    # This continues even if vehicle temporarily leaves ROI
                    if len(track) > 1:
                        prev_x = track[-2][0]
                        curr_x = track[-1][0]
                        
                        # Debug position (commented out to reduce noise)
                        # self.log_message(f"ID {track_id} X: {curr_x:.1f} (L:{self.line_left_x} R:{self.line_right_x})")

                        # --- Moving Left to Right (Increasing X) ---
                        # Crosses Left Line (Start Timer)
                        if prev_x < self.line_left_x and curr_x >= self.line_left_x:
                            self.left_to_right_start_times[track_id] = time.time()
                            self.log_message(f"[Debug] Car {track_id} started L->R timer")
                        
                        # Crosses Right Line (End Timer)
                        if prev_x < self.line_right_x and curr_x >= self.line_right_x:
                            if track_id in self.left_to_right_start_times:
                                start_time = self.left_to_right_start_times[track_id]
                                end_time = time.time()
                                duration = end_time - start_time
                                
                                # Check duration is reasonable (not too fast or too slow)
                                if 0 < duration <= MAX_CROSSING_DURATION:
                                    speed_mps = DISTANCE_METERS / duration
                                    speed_mph = speed_mps * 2.23694
                                    
                                    self.vehicle_speeds[track_id] = speed_mph
                                    self.car_count += 1
                                    vehicle_type = self.vehicle_types.get(track_id, "Vehicle")
                                    self.log_message(f"SAW {vehicle_type.upper()}: id={track_id}, speed={speed_mph:.1f} MPH, dir=northbound, total count={self.car_count}")
                                    self.save_snapshot(frame, track_id, speed_mph)
                                elif duration > MAX_CROSSING_DURATION:
                                    self.log_message(f"[Debug] Car {track_id} L->R timer expired ({duration:.1f}s)")
                                del self.left_to_right_start_times[track_id]

                        # --- Moving Right to Left (Decreasing X) ---
                        # Crosses Right Line (Start Timer)
                        if prev_x > self.line_right_x and curr_x <= self.line_right_x:
                            self.right_to_left_start_times[track_id] = time.time()
                            self.log_message(f"[Debug] Car {track_id} started R->L timer")
                        
                        # Crosses Left Line (End Timer)
                        if prev_x > self.line_left_x and curr_x <= self.line_left_x:
                            if track_id in self.right_to_left_start_times:
                                start_time = self.right_to_left_start_times[track_id]
                                end_time = time.time()
                                duration = end_time - start_time
                                
                                # Check duration is reasonable (not too fast or too slow)
                                if 0 < duration <= MAX_CROSSING_DURATION:
                                    speed_mps = DISTANCE_METERS / duration
                                    speed_mph = speed_mps * 2.23694
                                    
                                    self.vehicle_speeds[track_id] = speed_mph
                                    self.car_count += 1
                                    vehicle_type = self.vehicle_types.get(track_id, "Vehicle")
                                    self.log_message(f"SAW {vehicle_type.upper()}: id={track_id}, speed={speed_mph:.1f} MPH, dir=southbound, total count={self.car_count}")
                                    self.save_snapshot(frame, track_id, speed_mph)
                                elif duration > MAX_CROSSING_DURATION:
                                    self.log_message(f"[Debug] Car {track_id} R->L timer expired ({duration:.1f}s)")
                                del self.right_to_left_start_times[track_id]

                    # Draw bounding box and ID
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    # Draw label with type and speed if available
                    vehicle_type = self.vehicle_types.get(track_id, "Vehicle")
                    label = f"{vehicle_type} {track_id}"
                    if track_id in self.vehicle_speeds:
                        label += f" {self.vehicle_speeds[track_id]:.1f} MPH"
                    
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Check exits and cleanup stale timers
            for track_id in self.ids_in_roi:
                if track_id not in current_roi_ids:
                    vehicle_type = self.vehicle_types.get(track_id, "Vehicle")
                    self.log_message(f"{vehicle_type} {track_id} exited ROI")
            self.ids_in_roi = current_roi_ids
            
            # Clean up expired timers (vehicles that started but never finished)
            current_time = time.time()
            expired_l2r = [tid for tid, start_time in self.left_to_right_start_times.items() 
                          if current_time - start_time > MAX_CROSSING_DURATION]
            for tid in expired_l2r:
                self.log_message(f"[Debug] Cleaning expired L->R timer for vehicle {tid}")
                del self.left_to_right_start_times[tid]
            
            expired_r2l = [tid for tid, start_time in self.right_to_left_start_times.items() 
                          if current_time - start_time > MAX_CROSSING_DURATION]
            for tid in expired_r2l:
                self.log_message(f"[Debug] Cleaning expired R->L timer for vehicle {tid}")
                del self.right_to_left_start_times[tid]

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
