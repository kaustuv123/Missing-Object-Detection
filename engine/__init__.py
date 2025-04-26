"""
Core Engine Modules

This package contains the main processing engine components:
- Scene monitoring for change detection
- Inference engine for coordinating the pipeline
"""

from .scene_monitor import SceneMonitor
from .inference import InferenceEngine

__all__ = ['SceneMonitor', 'InferenceEngine']
