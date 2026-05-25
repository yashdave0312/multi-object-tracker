# Multi Object Tracker

Real-time multi-object tracker built with YOLOv8, DeepSORT, 
Kalman Filter and OpenCV.

## Features
- YOLOv8 object detection
- Kalman filter for position prediction
- Hungarian algorithm for detection assignment
- Re-ID CNN for occlusion handling
- Threaded video capture pipeline

## Installation
pip install -r requirements.txt

## Usage
python main.py

## Tech Stack
- Python 3.13
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- SciPy