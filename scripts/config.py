from dataclasses import dataclass
import os

@dataclass
class Config:
    # Realsense camera
    depth_stream_width: int = 424
    depth_stream_height: int = 240
    depth_stream_fps: int = 30
    color_stream_width: int = 424
    color_stream_height: int = 240
    color_stream_fps: int = 30

    # YOLO model
    confidence_threshold: float = 0.8
    @property
    def model_path(self) -> str:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = "scripts/runs/detect/train8/weights/best.pt"
        return os.path.join(base_path, relative_path)


