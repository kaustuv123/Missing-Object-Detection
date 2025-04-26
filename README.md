# Real-Time Video Analytics Pipeline

This project implements a real-time video analytics pipeline that can detect:

1. **Missing Object Detection** - When an object that was previously in the frame is no longer visible
2. **New Object Placement Detection** - When a new object appears in the scene

## Features

- Real-time object detection using YOLOv8 nano model
- Multi-object tracking with DeepSORT algorithm
- Scene memory to track objects across frames and detect changes
- Visualization with bounding boxes, labels, and status information
- On-screen notifications for missing and new objects
- Optimized for CPU performance (though CUDA is used if available)

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 nano model (if it doesn't download automatically):
   ```
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

## Usage

Run the pipeline with default settings (webcam input):

```
python main.py
```

Run with a video file:

```
python main.py --video path/to/video.mp4
```

Run with a custom configuration:

```
python main.py --config path/to/config.yaml
```

### Controls

- Press 'q' to quit the application

## Configuration

The pipeline can be configured through the `config.yaml` file. Key parameters include:

- Video source and resolution
- Detection confidence threshold
- Tracking parameters
- Scene monitoring thresholds
- Visualization options

## Project Structure

```
video_analytics_pipeline/
├── main.py               # Main entry point
├── config.yaml           # Configuration settings
├── requirements.txt      # Dependencies
├── detectors/            # Object detection modules
│   └── yolov8_detector.py
├── trackers/             # Object tracking modules
│   └── deep_sort_tracker.py
├── engine/               # Core processing modules
│   ├── baseline_memory.py
│   ├── scene_monitor.py
│   └── inference.py
├── utils/                # Utility functions
│   ├── visualizer.py
│   └── timer.py
└── alerts/               # Notification system
    └── notify.py
```

## How It Works

1. The system establishes a baseline of objects in the scene
2. As new frames arrive, objects are detected and tracked
3. The scene monitor compares current objects with the baseline
4. When an object is missing for more than N frames, it's flagged as missing
5. When a new object appears and remains stable for N frames, it's flagged as new
6. Notifications appear on screen for missing and new objects

## Performance

- Optimized for real-time processing on CPU
- Target performance: >5 FPS at 640x480 resolution
- Performance will vary based on hardware capabilities
