import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
from ultralytics.engine.results import Results

class YoloModel:
    def __init__(self, model_path: str, confidence_threshold= 0.8):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
    
    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            print(f"[YOLO] Model loaded succesfully")
        except Exception as e:
            print(f"[YOLO] Error loading model: {e}")
            raise e
        
    def predict(self, image: np.ndarray):
        results = self.model.predict(source= image, verbose= False, conf = self.confidence_threshold)
        return results
    
    def draw_detections(self,
                        depth_image: np.ndarray, 
                        color_image: np.ndarray,
                        results: List[Results],
                        depth_scale: float
                        ):
        """
        
        """
        # Create copies of images for overlaying
        depth_copy = depth_image.copy()
        color_copy = color_image.copy()

        # Iterate through the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                label = result.names[int(cls)]

                # Set colors
                bbox_color = (0, 255, 0)
                text_color = (0, 0, 0)

                # Draw on overlays
                self._draw_box_and_label(depth_copy, x1, y1, x2, y2, label, conf, bbox_color, text_color)
                self._draw_box_and_label(color_copy, x1, y1, x2, y2, label, conf, bbox_color, text_color)

        return depth_copy, color_copy



                

    def _draw_box_and_label(self, image, x1: int, y1: int, x2: int, y2: int, label: str, conf: float, bbox_color: tuple, text_color: tuple):
        """
        
        """
        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

        # Drwa label and confidence
        label_text = f"{label} {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1-baseline-text_height), (x1+text_height, y1), bbox_color, cv2.FILLED)
        cv2.putText(image, label_text, (x1, y1-baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def _get_center(self, x1: int, y1: int, x2: int, y2: int):
        """
        
        """
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y
