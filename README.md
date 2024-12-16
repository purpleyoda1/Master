# Synthetic depth data for object detection in robotic assembly
This repository contains code for my semesterproject and later on master thesis on using synthetic depth data for training object detectors for use in roboitc assembly

## Repository structure
- 'Blender/': Contains the blender project files used for generating synthetic depth maps
- 'Realsense/': Outpout of testing camera functionalities
- 'STL_CAD/': CAD files used in the project
- 'saved_frames/': Saved frames while running the camera with object detector applied and overlayed
- 'scripts/': Main code implementation files

## Scripts directory
- 'RealSenseCamera.py': Custom class for running RealSense camera, with YOLO functionality
- 'YOLO.py': Ultralytics YOLO class with some added functionality
- 'YOLO_trainer.ipynb': Notebook for training YOLO model
- 'config.py': Used to configure datastream and model path in a seperate and modifiable file
- 'data.yaml': Defines data location and other parameters for model training
- 'main.py': Runs camera with detector overlaid
- 'realsense.py': Testing and experiment with RealSense SDK, not used in main pipeline
- 'utilities.py': Various functions used for testing and exploring
- 'visualize_and_compare.py': Used to analyze depth maps, images, and more

## Main dependencies
- Python 3.8
- opencv 4.10.0
- pyrealsense2 2.55.1.6486
- matplotlib 3.7.2
- ultralytics 8.3.15
- torchvision 0.18.1

NOTE: This project requires Python 3.8 due to compatibility constraints with PyRealSense2.
