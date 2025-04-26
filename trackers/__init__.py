"""
Object Tracking Modules

This package contains implementations of object tracking algorithms.
Currently supports DeepSORT for multi-object tracking.
"""

from .deep_sort_tracker import DeepSORTTracker

__all__ = ['DeepSORTTracker']
