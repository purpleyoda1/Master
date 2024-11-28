from dataclasses import dataclass

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
    model_path: str = 'C:\\Users\\sondr\\Documents\\NTNU\\9_semester\\prosjekt\\scripts\\runs\\detect\\train4\\weights\\best.pt'
    confidence_threshold: float = 0.8

