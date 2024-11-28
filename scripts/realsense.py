import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO



###################################################################
###############            MISC                ####################
###################################################################

def check_depth_sensor():
    # Configure realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    
    # Start stream
    profile = pipeline.start(config)

    # Get depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()


###################################################################
###############         SAVE IMAGES            ####################
###################################################################

def capture_and_save_depth_map(output_path):
    # Configure realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    
    # Start stream
    pipeline.start(config)

    try:
        # Skip first 30 frames to stabilize
        for i in range(30):
            pipeline.wait_for_frames()
        
        # Get depth frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise Exception("Could not get depth frame")
        
        # Convert to numpy array and save
        depth_map = np.asanyarray(depth_frame.get_data())
        cv2.imwrite(output_path, depth_map)
        print(f"Depth image saved to {output_path}")

    finally:
        pipeline.stop()

def capture_and_save_raw_depth_map(output_path):
    # Configure realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    
    # Start stream
    profile = pipeline.start(config)
    
    try:
        # Skip first 30 frames to stabilize
        for i in range(30):
            pipeline.wait_for_frames()
        
        # Get depth frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise Exception("Could not get depth frame")
        
        # Convert to numpy array and save
        depth_map = np.asanyarray(depth_frame.get_data())

        cv2.imwrite(output_path, depth_map)
        print(f"Depth map saved to {output_path}")

    finally:
        pipeline.stop()

def capture_and_save_normalized_depth_map(output_path, min_depth=0.1, max_depth=3.0):
    """
    Captures a single depth frame from the RealSense D435i and saves it as a grayscale PNG image.

    Parameters:
    - output_path (str): Path where the depth image will be saved.
    - min_depth (float): Minimum depth (in meters) to display.
    - max_depth (float): Maximum depth (in meters) to display.
    """
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    
    # Start streaming
    profile = pipeline.start(config)

    # Retrieve the depth scale (meters per unit)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[Realsense] Depth scale: {depth_scale} meters/unit")
    
    try:
        # Allow the camera to warm up and stabilize
        for i in range(30):
            frames = pipeline.wait_for_frames()
        print("[Realsense] Warming up complete.")

        # Capture a single frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise Exception("Could not get depth frame")
        
        # Convert depth frame to numpy array
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_meters = depth_raw * depth_scale  # Convert to meters
        print(f"[Depth] Raw depth data type: {depth_raw.dtype}, shape: {depth_raw.shape}")
        print(f"[Depth] Depth in meters: min={depth_meters.min()}, max={depth_meters.max()}")

        # Clip depth values to the specified range
        depth_clipped = np.clip(depth_meters, min_depth, max_depth)
        print(f"[Depth] Clipped depth: min={depth_clipped.min()}, max={depth_clipped.max()}")

        # Normalize the clipped depth to 0-255
        # Mapping: min_depth -> 0, max_depth -> 255
        depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        depth_normalized = np.clip(depth_normalized, 0, 255)  # Ensure values are within [0, 255]
        print(f"[Depth] Normalized depth: min={depth_normalized.min()}, max={depth_normalized.max()}")

        # Save the depth image
        cv2.imwrite(output_path, depth_normalized)
        print(f"[Info] Depth image saved to {output_path}")

    except Exception as e:
        print(f"[Error] {e}")

    finally:
        # Stop the pipeline
        pipeline.stop()
        print("[Realsense] Pipeline stopped.")


###################################################################
###############         RUN INFERENCE          ####################
###################################################################

def display_inference(model_path, confident_threshold= 0.5):
    """
    
    """
    # Initialize Realsense
    pipeline = rs.pipeline()
    config = rs.config()

    # Set pipeline configurations
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.z16, 30)

    # Start stream
    profile = pipeline.start(config)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} meters/unit")

    # Align RGB and depth stream
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Load YOLO model
    try:
        model = YOLO(model_path)
        model.conf = confident_threshold
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        pipeline.stop()
        return
    
    try:
        while True:
            # Wait for aligned frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get images and turn them into numpy arrays
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame)
            color_image = np.asanyarray(color_frame)

            # Make copies for overlaying bboxes
            depth_copy = depth_image.copy()
            color_copy = color_image.copy()

            # Run inference on depth input
    except:
        pass





def main():
    filename = 'realsense_report.png'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    path = os.path.join(script_dir, 'Images', filename)
    capture_and_save_raw_depth_map(path)

    #check_depth_sensor()
    pass


if __name__=="__main__":
    main()