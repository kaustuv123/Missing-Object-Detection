"""
Video Analytics Pipeline

A real-time video analytics system for detecting missing and new objects in video streams.
"""

from .detectors.yolov8_detector import YOLOv8Detector
from .trackers.deep_sort_tracker import DeepSORTTracker
from .engine.scene_monitor import SceneMonitor
from .engine.inference import InferenceEngine
from .utils.visualizer import Visualizer
from .alerts.notify import AlertNotifier

__all__ = [
    'YOLOv8Detector',
    'DeepSORTTracker',
    'SceneMonitor',
    'InferenceEngine',
    'Visualizer',
    'AlertNotifier'
]
