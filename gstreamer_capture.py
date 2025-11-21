"""
GStreamer-based video capture that supports hardware acceleration.
Bypasses OpenCV's limited GStreamer backend to enable VA-API hardware decode.
"""
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import threading
import queue
import time

Gst.init(None)

class GStreamerCapture:
    """Video capture using GStreamer with hardware decode support"""
    
    def __init__(self, pipeline_str):
        self.pipeline_str = pipeline_str
        self.pipeline = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.width = 0
        self.height = 0
        self.fps = 0.0
        
        # Create pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get appsink element
        self.appsink = self.pipeline.get_by_name('appsink0')
        if not self.appsink:
            # Pipeline might auto-name it differently, find it
            for element in self.pipeline.iterate_elements():
                if element.get_factory().get_name() == 'appsink':
                    self.appsink = element
                    break
        
        if not self.appsink:
            raise RuntimeError("No appsink element found in pipeline")
        
        # Configure appsink
        self.appsink.set_property('emit-signals', True)
        self.appsink.set_property('drop', True)
        self.appsink.set_property('max-buffers', 2)
        self.appsink.connect('new-sample', self._on_new_sample)
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")
        
        # Wait for pipeline to negotiate caps (get video properties)
        # RTSP streams can take several seconds to connect
        print("[GStreamer] Waiting for stream to connect...")
        max_wait = 10  # seconds
        for i in range(max_wait * 2):  # Check every 0.5 seconds
            time.sleep(0.5)
            self._get_video_properties()
            if self.width > 0 and self.height > 0:
                break
        
        if self.width == 0 or self.height == 0:
            raise RuntimeError("Failed to get video properties from pipeline after 10 seconds")
        
        self.running = True
        print(f"[GStreamer] Pipeline started: {self.width}x{self.height} @ {self.fps} FPS")
    
    def _get_video_properties(self):
        """Extract video properties from pipeline caps"""
        pad = self.appsink.get_static_pad('sink')
        if pad:
            caps = pad.get_current_caps()
            if caps and caps.get_size() > 0:
                structure = caps.get_structure(0)
                
                # Get width and height
                success, self.width = structure.get_int('width')
                success, self.height = structure.get_int('height')
                
                # Get framerate (it's a fraction: numerator/denominator)
                success, fps_num, fps_denom = structure.get_fraction('framerate')
                if success:
                    self.fps = float(fps_num) / float(fps_denom)
    
    def _on_new_sample(self, appsink):
        """Callback when new frame is available"""
        sample = appsink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                try:
                    # Get actual dimensions from this sample's caps (might differ during startup)
                    structure = caps.get_structure(0)
                    success, width = structure.get_int('width')
                    success, height = structure.get_int('height')
                    
                    # Convert to numpy array (BGR format: height x width x 3)
                    frame_data = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    )
                    # Make a copy before unmapping
                    frame = frame_data.copy()
                    
                    # Add to queue (non-blocking, drop if full)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Drop frame if queue is full
                finally:
                    buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def read(self):
        """Read a frame (OpenCV-compatible interface)"""
        if not self.running:
            return False, None
        
        try:
            # Wait up to 5 seconds for a frame
            frame = self.frame_queue.get(timeout=5.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def isOpened(self):
        """Check if capture is open"""
        return self.running and self.pipeline is not None
    
    def get(self, prop_id):
        """Get video property (OpenCV-compatible interface)"""
        import cv2
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0
    
    def set(self, prop_id, value):
        """Set video property (not implemented for GStreamer)"""
        return False
    
    def release(self):
        """Release capture and stop pipeline"""
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
    
    def getBackendName(self):
        """Return backend name for compatibility"""
        return "GSTREAMER_NATIVE"
