import os
import argparse
import yaml
import cv2

from detectors.yolov8_detector import YOLOv8Detector
from trackers.deep_sort_tracker import DeepSORTTracker
from engine.scene_monitor import SceneMonitor
from engine.inference import InferenceEngine
from utils.visualizer import Visualizer
from alerts.notify import AlertNotifier


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration settings.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main entry point for the video analytics pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Video Analytics Pipeline for Object Change Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--video', type=str, default=None,
                        help='Video source (0 for webcam, or path to video file)')
    args = parser.parse_args()
    
    # Set up config path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_dir, args.config)
    
    # Load configuration
    config = load_config(config_path)
    
    # Override video source if provided
    if args.video:
        # Convert to integer if it's a webcam index
        if args.video.isdigit():
            config['video']['source'] = int(args.video)
        else:
            config['video']['source'] = args.video
    
    # Initialize components
    # 1. Detector
    print("Initializing YOLOv8 Detector...")
    detector = YOLOv8Detector(config['detector'])
    
    # 2. Tracker
    print("Initializing Deep SORT Tracker...")
    tracker = DeepSORTTracker(config['tracker'])
    
    # 3. Scene Monitor
    print("Initializing Scene Monitor...")
    scene_monitor = SceneMonitor(config['scene_monitor'])
    
    # 4. Visualizer
    print("Initializing Visualizer...")
    visualizer = Visualizer(config['visualization'])
    
    # 5. AlertNotifier
    print("Initializing Alert Notifier...")
    alert_notifier = AlertNotifier()
    
    # 6. Inference Engine
    print("Initializing Inference Engine...")
    engine = InferenceEngine(
        config_path, 
        detector, 
        tracker, 
        scene_monitor, 
        visualizer,
        alert_notifier
    )
    
    # Run the pipeline
    print(f"Starting video analytics pipeline on source: {config['video']['source']}")
    print("Press 'q' to quit.")
    try:
        engine.run()
    except KeyboardInterrupt:
        print("Pipeline stopped by user.")
    except Exception as e:
        print(f"Error running pipeline: {e}")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        print("Pipeline shutdown complete.")


if __name__ == "__main__":
    main() 