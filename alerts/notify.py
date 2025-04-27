import cv2
import time
import numpy as np


class AlertNotifier:
    def __init__(self):
        self.alerts = []
        self.alert_duration = 3.0
        
    def add_missing_object_alert(self, obj_data):
        class_id = obj_data.get('class_id', 0)
        obj_id = obj_data.get('id', -1)
        
        alert = {
            'type': 'missing',
            'message': f"MISSING: Object {obj_id} has disappeared",
            'time': time.time(),
            'color': (0, 0, 255),
            'obj_id': obj_id,
            'class_id': class_id
        }
        
        self.alerts.append(alert)
        
    def add_new_object_alert(self, obj_data):
        class_id = obj_data.get('class_id', 0)
        obj_id = obj_data.get('id', -1)
        
        alert = {
            'type': 'new',
            'message': f"NEW: Object {obj_id} has appeared",
            'time': time.time(),
            'color': (0, 255, 0),
            'obj_id': obj_id,
            'class_id': class_id
        }
        
        self.alerts.append(alert)
        
    def draw_alerts(self, frame):
        result_frame = frame.copy()
        
        current_time = time.time()
        
        self.alerts = [alert for alert in self.alerts if current_time - alert['time'] < self.alert_duration]
        
        y_pos = 70
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i, alert in enumerate(self.alerts):
            text_size = cv2.getTextSize(
                alert['message'], 
                font, 
                1.0, 
                2
            )[0]
            
            overlay = result_frame.copy()
            cv2.rectangle(
                overlay, 
                (10, y_pos - 30), 
                (10 + text_size[0] + 20, y_pos + 10), 
                alert['color'], 
                -1
            )
            
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
            
            cv2.putText(
                result_frame, 
                alert['message'], 
                (20, y_pos), 
                font, 
                1.0, 
                (255, 255, 255),
                2
            )
            
            y_pos += 50
            
        return result_frame
        
    def update_with_scene_changes(self, missing_objects, new_objects):
        for obj in missing_objects:
            if not any(alert['obj_id'] == obj['id'] and alert['type'] == 'missing' for alert in self.alerts):
                if obj.get('missing_frames', 0) == obj.get('missing_frames_threshold', 15):
                    self.add_missing_object_alert(obj)
                    
        for obj in new_objects:
            if not any(alert['obj_id'] == obj['id'] and alert['type'] == 'new' for alert in self.alerts):
                if obj.get('new_frames', 0) == obj.get('stability_frames', 5):
                    self.add_new_object_alert(obj) 