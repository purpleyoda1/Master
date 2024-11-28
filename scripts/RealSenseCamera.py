import pyrealsense2 as rs
import numpy as np
import cv2
from config import Config
import utilities

class RealSenseCamera:
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = None
        self.depth_intrinsic = None
    
    def initialize(self):
        rs_config = rs.config()
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
        # Start stream
        self.profile = self.pipeline.start(rs_config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        print(f"[Realsense] Depth scale: {self.depth_scale} meters/unit")
        print(f"[Realsense] Depth intrinsics: {self.depth_intrinsics}")

    def get_frames(self, pad= False):
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        #Align and turn into numpy arrays
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if pad:
            depth_image, _, _ = utilities.pad_to_square(depth_image)
            color_image, _, _ = utilities.pad_to_square(color_image)
        
        #depth_image = depth_image.astype(np.uint16)
        depth_image = np.stack((depth_image,)*3, axis=-1)

        return depth_image, color_image
    
    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
