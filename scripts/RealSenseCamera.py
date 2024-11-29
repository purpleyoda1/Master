import pyrealsense2 as rs
import numpy as np
import cv2
from config import Config
import utilities
from typing import List
from enum import Enum, auto
import os
import datetime
import time
import matplotlib.pyplot as plt

class StreamView(Enum):
    """Enum for controlling what to include when displaying stream"""
    COLOR = auto()
    DEPTH = auto()
    DEPTH_COLORMAP = auto()
    COLOR_OVERLAY = auto()
    DEPTH_OVERLAY = auto()

class RealSenseCamera:
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = None
        self.depth_intrinsic = None
        self.model = None
        self.is_running = False
        self.save_dir = "saved_frames"
        os.makedirs(self.save_dir, exist_ok=True)

    def _enable_stream(self, rs_config: rs.config) -> None:
        # Enable depth stream
        rs_config.enable_stream(
            rs.stream.depth,
            self.config.depth_stream_width,
            self.config.depth_stream_height,
            rs.format.z16,
            self.config.depth_stream_fps
        )
        # Enable depth stream
        rs_config.enable_stream(
            rs.stream.color,
            self.config.color_stream_width,
            self.config.color_stream_height,
            rs.format.bgr8,
            self.config.color_stream_fps
        )

    def _initialize_YOLO_model(self) -> None:
        """Initialize YOLO model within camera"""
        from YOLO import YoloModel
        self.model = YoloModel(self.config.model_path, self.config.confidence_threshold)
        self.model.load_model()
        print("[Camera] YOLO model initialized")

    def initialize(self) -> None:
        """Initialize camera with settings from config"""
        rs_config = rs.config()
        self._enable_stream(rs_config)
        
        # Start stream
        self.profile = self.pipeline.start(rs_config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        print(f"[Camera] Depth scale: {self.depth_scale} meters/unit")
        print(f"[Camera] Depth intrinsics: {self.depth_intrinsics}")

        self._initialize_YOLO_model()

        self.is_running = True

    def get_frames(self, pad: bool= False):
        """Get depth and color frames from camera"""
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        #Align and turn into numpy arrays
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if pad: #TODO: Can this be removed?
            depth_image, _, _ = utilities.pad_to_square(depth_image)
            color_image, _, _ = utilities.pad_to_square(color_image)

        return depth_image, color_image
    
    def apply_YOLO(self):
        """Apply YOLO model to frames and draw bounding boxes"""
         # Get frames
        depth_image, color_image = self.get_frames(pad= False)
        if depth_image is None or color_image is None:
            print("no images found")

        # Get prediciton results
        depth_3channel = np.stack((depth_image,)*3, axis=-1)
        results = self.model.predict(depth_3channel)

        # Process depth image for display
        depth_image_float = depth_image.astype(float)
        # Normalize between 0 and 1
        depth_mask = depth_image_float > 0
        if depth_mask.any():
            depth_min = depth_image_float[depth_mask].min()
            depth_max = depth_image_float[depth_mask].max()
            depth_image_float[depth_mask] = (depth_image_float[depth_mask] - depth_min) / (depth_max - depth_min)

        cmap = plt.get_cmap('viridis')
        depth_colored = cmap(depth_image_float)
        depth_colormap = (depth_colored[:, :, :3] * 255).astype(np.uint8)

        # Draw bounding boxes
        depth_overlay, color_overlay = self.model.draw_detections(depth_colormap.copy(), color_image.copy(), results, self.depth_scale)

        return {
            StreamView.COLOR: color_image,
            StreamView.DEPTH: depth_image,
            StreamView.DEPTH_COLORMAP: depth_colormap,
            StreamView.COLOR_OVERLAY: color_overlay,
            StreamView.DEPTH_OVERLAY: depth_overlay
        }
    
    def prepare_display(self, processed_frames: dict, views: List[StreamView], scale_factor: float= 1.5):
        """Stack requested images nicely"""
        if not views:
            raise ValueError("[Camera] Must request atleast one view")

        # Stack images horizontally
        combined_image = np.hstack([processed_frames[view] for view in views])

        # Resize
        width = int(combined_image.shape[1] * scale_factor)
        height = int(combined_image.shape[0] * scale_factor)

        return cv2.resize(combined_image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def save_frame(self, processed_frames, views: List[StreamView]):
        """ """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for view in views:
                frame = processed_frames.get(view)
                if frame is not None:
                    filename = f"{view.name.lower()}_{timestamp}.png"
                    filepath = os.path.join(self.save_dir, filename)

                    cv2.imwrite(filepath, frame)
                    print(f"[Camera] Saved {view.name} to {filepath}")
                else:
                    print(f"[Camera] Frame not available")

        except Exception as e:
            print(f"[Camera] Error saving frame: {e}")
    
    def run(self, views: List[StreamView]= None, window_name: str= "Realsense camera", scale_factor: float= 1.5):
        """Run camera in a loop"""

        # Set default output
        if views is None:
            views = [StreamView.COLOR, StreamView.DEPTH_COLORMAP]

        # Cooldown for save
        last_save = 0
        save_cooldown = 0.5

        try:
            while self.is_running:
                try:
                    processed_frames = self.apply_YOLO()
                    display_image = self.prepare_display(processed_frames, views, scale_factor)
                    cv2.imshow(window_name, display_image)

                    key = cv2.waitKey(1) & 0xFF
                    current_time = time.time()
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        if (current_time - last_save) > save_cooldown:
                            self.save_frame(processed_frames, views)
                            last_save = current_time
                        
                except RuntimeError as e:
                    print(f"Frame processing error: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("[Camera] Interrupted by user")
            
        finally:
            self.stop()

    
    def stop(self):
        self.is_running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("[Camera] Stopped Realsense pipeline and closed all windows")
