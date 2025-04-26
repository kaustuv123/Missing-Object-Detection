import cv2
import yaml
import time
import numpy as np
from ..utils.timer import FPSTimer, PerformanceTimer
from ..alerts.notify import AlertNotifier


class InferenceEngine:
    """
    Main inference engine that coordinates detection, tracking, and scene monitoring.
    Manages the video capture, processing pipeline, and visualization.
    """
    def __init__(self, config_path, detector, tracker, scene_monitor, visualizer, alert_notifier=None):
        """
        Initialize the inference engine with components.
        
        Args:
            config_path (str): Path to the configuration YAML file.
            detector: Object detector instance.
            tracker: Object tracker instance.
            scene_monitor: Scene monitor for change detection.
            visualizer: Visualization utility for rendering results.
            alert_notifier (AlertNotifier, optional): Alert notification system.
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Get video configuration
        video_config = self.config.get('video', {})
        self.source = video_config.get('source', 0)
        self.width = video_config.get('width', 640)
        self.height = video_config.get('height', 480)
        
        # Initialize components
        self.detector = detector
        self.tracker = tracker
        self.scene_monitor = scene_monitor
        self.visualizer = visualizer
        self.alert_notifier = alert_notifier if alert_notifier else AlertNotifier()
        
        # Create FPS timer
        self.fps_timer = FPSTimer(buffer_size=30)
        
        # Performance timers
        self.detection_timer = PerformanceTimer('Detection')
        self.tracking_timer = PerformanceTimer('Tracking')
        self.monitoring_timer = PerformanceTimer('Monitoring')
        self.visualization_timer = PerformanceTimer('Visualization')
        
        # State
        self.is_running = False
        self.frame_count = 0
        self.last_fps = 0
        
    def _setup_video_capture(self):
        """
        Set up the video capture source.
        
        Returns:
            cv2.VideoCapture: Configured video capture object.
        """
        # Create video capture object
        if isinstance(self.source, str) and (self.source.startswith('rtsp://') or 
                                            self.source.startswith('http://') or
                                            self.source.endswith('.mp4') or
                                            self.source.endswith('.avi')):
            # Video file or stream
            cap = cv2.VideoCapture(self.source)
        else:
            # Webcam
            cap = cv2.VideoCapture(int(self.source) if str(self.source).isdigit() else 0)
            
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Check if opened successfully
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")
            
        return cap
        
    def run(self):
        """
        Run the inference pipeline on the video source.
        """
        # Set up video capture
        cap = self._setup_video_capture()
        
        # Mark as running
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to grab frame, stream may have ended.")
                    break
                    
                # Increment counter
                self.frame_count += 1
                
                # Resize frame if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                    
                # Process frame
                result_frame, status_panel = self._process_frame(frame)
                
                # Update FPS
                self.last_fps = self.fps_timer.update()
                
                # Show the processed frame
                cv2.imshow('Video Analytics Pipeline', result_frame)
                if status_panel is not None:
                    cv2.imshow('Status Panel', status_panel)
                    
                # Check for exit key (q)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # Calculate final stats
            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print(f"Pipeline finished processing {self.frame_count} frames in {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False
            
    def _process_frame(self, frame):
        """
        Process a single frame through the pipeline.
        
        Args:
            frame (numpy.ndarray): Input video frame.
            
        Returns:
            tuple: (processed_frame, status_panel)
        """
        # 1. Object Detection
        with self.detection_timer:
            detections = self.detector.detect(frame)
            
        # 2. Object Tracking
        with self.tracking_timer:
            tracked_objects = self.tracker.update(detections)
            
        # 3. Scene Monitoring
        with self.monitoring_timer:
            tracked_objects, missing_ids, new_ids, missing_objects, new_objects = self.scene_monitor.process_frame(tracked_objects)
            
            # Update the alert notifier with scene changes
            self.alert_notifier.update_with_scene_changes(missing_objects, new_objects)
            
        # 4. Visualization
        with self.visualization_timer:
            # Draw detections and tracking on frame
            result_frame = self.visualizer.draw_results(
                frame, 
                tracked_objects, 
                missing_ids, 
                new_ids, 
                fps=self.last_fps
            )
            
            # Add alerts to the frame
            result_frame = self.alert_notifier.draw_alerts(result_frame)
            
            # Create status panel
            status_panel = self.visualizer.create_status_panel(
                missing_objects,
                new_objects,
                frame.shape[1]
            )
            
        return result_frame, status_panel
        
    def get_performance_metrics(self):
        """
        Get performance metrics from all components.
        
        Returns:
            dict: Dictionary of performance metrics.
        """
        metrics = {
            'fps': self.last_fps,
            'frames_processed': self.frame_count,
            'detection_time_ms': self.detection_timer.get_elapsed_ms(),
            'tracking_time_ms': self.tracking_timer.get_elapsed_ms(),
            'monitoring_time_ms': self.monitoring_timer.get_elapsed_ms(),
            'visualization_time_ms': self.visualization_timer.get_elapsed_ms()
        }
        
        # Add scene monitor metrics
        monitor_metrics = self.scene_monitor.get_metrics()
        metrics.update(monitor_metrics)
        
        return metrics 