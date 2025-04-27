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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Video Analytics Pipeline for Object Change Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--video', type=str, default=None,
                        help='Video source (0 for webcam, or path to video file)')
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_dir, args.config)
    
    config = load_config(config_path)
    
    if args.video:
        if args.video.isdigit():
            config['video']['source'] = int(args.video)
        else:
            config['video']['source'] = args.video
    
    print("Initializing YOLOv8 Detector...")
    detector = YOLOv8Detector(config['detector'])
    
    print("Initializing Deep SORT Tracker...")
    tracker = DeepSORTTracker(config['tracker'])
    
    print("Initializing Scene Monitor...")
    scene_monitor = SceneMonitor(config['scene_monitor'])
    
    print("Initializing Visualizer...")
    visualizer = Visualizer(config['visualization'])
    
    print("Initializing Alert Notifier...")
    alert_notifier = AlertNotifier()
    
    print("Initializing Inference Engine...")
    engine = InferenceEngine(
        config_path, 
        detector, 
        tracker, 
        scene_monitor, 
        visualizer,
        alert_notifier
    )
    
    print(f"Starting video analytics pipeline on source: {config['video']['source']}")
    print("Press 'q' to quit.")
    try:
        engine.run()
    except KeyboardInterrupt:
        print("Pipeline stopped by user.")
    except Exception as e:
        print(f"Error running pipeline: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Pipeline shutdown complete.")


if __name__ == "__main__":
    main() 