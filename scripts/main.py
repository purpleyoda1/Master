import cv2
import numpy as np
from config import Config
from RealSenseCamera import RealSenseCamera
from YOLO import YoloModel
import utilities


def main():
    # Load config
    config = Config()

    # Initialize camera
    realsense = RealSenseCamera(config)
    realsense.initialize()

    # Initialize YOLO model
    model = YoloModel(config.model_path, config.confidence_threshold)
    model.load_model()

    try: 
        while True: 
            # Get frames
            depth_image, color_image = realsense.get_frames(pad= False)
            if depth_image is None or color_image is None:
                print("no images found")
                continue

            # Apply model
            results = None
            results = model.predict(depth_image)

            # Process depth image for display
            depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

            # Draw bounding boxes
            depth_overlay, color_overlay = model.draw_detections(depth_colormap, color_image, results, realsense.depth_scale)

            # Stack and display images
            combined_image = np.hstack((depth_image_8bit*2, color_overlay))

            # Resize the combined image
            scale_factor = 1.5 
            width = int(combined_image.shape[1] * scale_factor)
            height = int(combined_image.shape[0] * scale_factor)
            resized_image = cv2.resize(combined_image, (width, height), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("YOLOv8 on Realsense D435i", resized_image)

            # Set up exit strategy
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")

    finally:
        # Cleanup
        realsense.stop()
        print("[Main] Stopped RealSense pipeline and closed all windows.")


if __name__ == "__main__":
    main()