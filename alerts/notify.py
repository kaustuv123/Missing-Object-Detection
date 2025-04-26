import cv2
import time
import numpy as np


class AlertNotifier:
    """
    Handles on-screen notifications and alerts for object changes.
    """
    def __init__(self):
        """
        Initialize the alert notifier.
        """
        self.alerts = []  # List of active alerts
        self.alert_duration = 3.0  # Duration to show each alert in seconds
        
    def add_missing_object_alert(self, obj_data):
        """
        Add an alert for a missing object.
        
        Args:
            obj_data (dict): Data about the missing object.
        """
        class_id = obj_data.get('class_id', 0)
        obj_id = obj_data.get('id', -1)
        
        # Create alert
        alert = {
            'type': 'missing',
            'message': f"MISSING: Object {obj_id} has disappeared",
            'time': time.time(),
            'color': (0, 0, 255),  # BGR (red)
            'obj_id': obj_id,
            'class_id': class_id
        }
        
        # Add to alerts list
        self.alerts.append(alert)
        
    def add_new_object_alert(self, obj_data):
        """
        Add an alert for a new object.
        
        Args:
            obj_data (dict): Data about the new object.
        """
        class_id = obj_data.get('class_id', 0)
        obj_id = obj_data.get('id', -1)
        
        # Create alert
        alert = {
            'type': 'new',
            'message': f"NEW: Object {obj_id} has appeared",
            'time': time.time(),
            'color': (0, 255, 0),  # BGR (green)
            'obj_id': obj_id,
            'class_id': class_id
        }
        
        # Add to alerts list
        self.alerts.append(alert)
        
    def draw_alerts(self, frame):
        """
        Draw active alerts on the frame.
        
        Args:
            frame (numpy.ndarray): The input frame to draw alerts on.
            
        Returns:
            numpy.ndarray: The frame with alerts drawn.
        """
        # Make a copy of the frame
        result_frame = frame.copy()
        
        # Current time
        current_time = time.time()
        
        # Filter out expired alerts
        self.alerts = [alert for alert in self.alerts if current_time - alert['time'] < self.alert_duration]
        
        # Draw remaining alerts at the top of the frame
        y_pos = 70  # Start position for alerts
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i, alert in enumerate(self.alerts):
            # Create background rectangle for better visibility
            text_size = cv2.getTextSize(
                alert['message'], 
                font, 
                1.0, 
                2
            )[0]
            
            # Draw semi-transparent background for the alert
            overlay = result_frame.copy()
            cv2.rectangle(
                overlay, 
                (10, y_pos - 30), 
                (10 + text_size[0] + 20, y_pos + 10), 
                alert['color'], 
                -1
            )
            
            # Apply transparency
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
            
            # Draw alert text
            cv2.putText(
                result_frame, 
                alert['message'], 
                (20, y_pos), 
                font, 
                1.0, 
                (255, 255, 255),  # White text
                2
            )
            
            y_pos += 50  # Space between alerts
            
        return result_frame
        
    def update_with_scene_changes(self, missing_objects, new_objects):
        """
        Update alerts based on scene changes.
        
        Args:
            missing_objects (list): List of missing object data.
            new_objects (list): List of new object data.
        """
        # Check if we need to add alerts for missing objects
        for obj in missing_objects:
            # Only add if we don't already have an alert for this object
            if not any(alert['obj_id'] == obj['id'] and alert['type'] == 'missing' for alert in self.alerts):
                if obj.get('missing_frames', 0) == obj.get('missing_frames_threshold', 15):
                    self.add_missing_object_alert(obj)
                    
        # Check if we need to add alerts for new objects
        for obj in new_objects:
            # Only add if we don't already have an alert for this object
            if not any(alert['obj_id'] == obj['id'] and alert['type'] == 'new' for alert in self.alerts):
                if obj.get('new_frames', 0) == obj.get('stability_frames', 5):
                    self.add_new_object_alert(obj) 