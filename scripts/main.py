import cv2
import numpy as np
from config import Config
from RealSenseCamera import RealSenseCamera, StreamView
from YOLO import YoloModel
import utilities


def main():
    config = Config()
    camera = RealSenseCamera(config)
    camera.initialize()

    camera.run(views=[
        StreamView.COLOR_OVERLAY,
        StreamView.DEPTH_OVERLAY
    ])


if __name__ == "__main__":
    main()