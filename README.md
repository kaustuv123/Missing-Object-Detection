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

### Running the Pipeline

1. **Using Webcam (Default)**

   ```
   python main.py
   ```

   This will start the pipeline using your default webcam (usually device index 0).

2. **Using a Video File**

   ```
   python main.py --video path/to/your/video.mp4
   ```

   Replace `path/to/your/video.mp4` with the actual path to your video file.

3. **Using Custom Configuration**
   ```
   python main.py --config path/to/config.yaml
   ```
   This allows you to use custom settings defined in your configuration file.

### Controls

- Press 'q' to quit the application
- Press 's' to save the current frame
- Press 'p' to pause/resume the video

## Pipeline Architecture

The video analytics pipeline consists of several interconnected components:

1. **Input Layer**

   - Handles video input from webcam or video file
   - Performs initial frame preprocessing

2. **Detection Layer**

   - Uses YOLOv8 for real-time object detection
   - Identifies objects and their bounding boxes
   - Assigns confidence scores to detections

3. **Tracking Layer**

   - Implements DeepSORT algorithm for object tracking
   - Maintains object IDs across frames
   - Handles object re-identification

4. **Scene Memory**

   - Maintains a baseline of objects in the scene
   - Tracks object presence/absence over time
   - Implements persistence thresholds

5. **Change Detection**

   - Monitors for missing objects
   - Detects newly placed objects
   - Implements temporal smoothing

6. **Visualization Layer**
   - Draws bounding boxes and labels
   - Displays status information
   - Shows real-time notifications

## Screenshots

The `Screenshots` folder contains example outputs of the pipeline in action:

### Object Detection and Tracking

![Object Detection](Screenshots/Screenshot%202025-04-27%20104059.png)

### Missing Object Detection

![Missing Object](Screenshots/Screenshot%202025-04-27%20104111.png)

### New Object Placement

![New Object](Screenshots/Screenshot%202025-04-27%20104118.png)

### Tracking Visualization

![Tracking](Screenshots/Screenshot%202025-04-27%20104148.png)

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
├── alerts/               # Notification system
│   └── notify.py
└── Screenshots/          # Example outputs
```

## Performance

- Optimized for real-time processing on CPU
- Target performance: >5 FPS at 640x480 resolution
- Performance will vary based on hardware capabilities
- CUDA acceleration available if supported hardware is present
