import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
from ultralytics.engine.results import Results
import os
import time

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

    def evaluate(self, yaml_path, test_path, conf_thres=0.5, iou_thresh=0.5):
        """
        
        """
        start_time = time.perf_counter()
        predictions = self.model.predict(source=test_path, conf=conf_thres, iou=iou_thresh)
        total_time = time.perf_counter() - start_time
        avg_inference_time = (total_time / 284) * 1000

        metrics = self.model.val(data=yaml_path, conf=conf_thres, iou=iou_thresh, verbose=True)
        results = {
            'Precision': metrics.results_dict['metrics/precision(B)'],
            'Recall': metrics.results_dict['metrics/recall(B)'],
            'mAP50': metrics.results_dict['metrics/mAP50(B)'],
            'mAP50-95': metrics.results_dict['metrics/mAP50-95(B)'],
            'F1 Score': 2 * (metrics.results_dict['metrics/precision(B)'] * metrics.results_dict['metrics/recall(B)']) / (metrics.results_dict['metrics/precision(B)'] + metrics.results_dict['metrics/recall(B)']),
            'Inference time': avg_inference_time
        }

        print("\n" + "="*50)
        print(f"Evaluation report on test data at {test_path}")
        print("="*50)

        print(f"\nDetection Metrics:")
        print(f"{'Metric':<20} {'Value':<10}")
        print("-"*30)
        
        metrics_to_print = ['Precision', 'Recall', 'F1 Score', 'mAP50', 'mAP50-95', 'Inference time']
        for metric in metrics_to_print:
            print(f"{metric:<20} {results[metric]:.4f}")

        print("\n" + "="*50)

        return results
    

if __name__ == "__main__":
    base_path = os.path.dirname( os.path.abspath(__file__))
    yaml_path = os.path.join(base_path, 'data.yaml')
    model_path = os.path.join(base_path, 'runs/detect/train8/weights/best.pt')
    test_path = os.path.join(os.path.dirname(base_path), 'synthetic_data/training_424x240/images/test')

    model = YoloModel(model_path)
    model.load_model()   
    model.evaluate(yaml_path, test_path)
