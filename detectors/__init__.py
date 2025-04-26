"""
Object Detection Modules

This package contains implementations of object detection algorithms.
Currently supports YOLOv8 for real-time object detection.
"""

from .yolov8_detector import YOLOv8Detector

__all__ = ['YOLOv8Detector']
