import cv2
import numpy as np


class Visualizer:
    """
    Utility class for visualizing detection and tracking results on frames.
    Handles bounding boxes, labels, and performance metrics like FPS.
    """
    def __init__(self, config):
        """
        Initialize the visualizer with configuration settings.
        
        Args:
            config (dict): Configuration settings for visualization.
        """
        self.config = config
        self.box_thickness = config.get('box_thickness', 2)
        self.text_size = config.get('text_size', 0.5)
        self.text_thickness = config.get('text_thickness', 2)
        self.show_fps = config.get('show_fps', True)
        self.show_boxes = config.get('show_boxes', True)
        self.show_labels = config.get('show_labels', True)
        
        # BGR colors
        self.missing_color = tuple(config.get('missing_color', [0, 0, 255]))  # Red for missing
        self.new_color = tuple(config.get('new_color', [0, 255, 0]))  # Green for new
        self.normal_color = tuple(config.get('normal_color', [255, 0, 0]))  # Blue for normal
        
        # Class names for COCO dataset (default for YOLOv8)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    def draw_results(self, frame, tracked_objects, missing_ids, new_ids, fps=None):
        """
        Draw detection and tracking results on the frame.
        
        Args:
            frame (numpy.ndarray): The input frame to draw on.
            tracked_objects (list): List of tracked objects with bounding boxes and IDs.
            missing_ids (list): List of IDs of objects considered missing.
            new_ids (list): List of IDs of newly detected objects.
            fps (float, optional): FPS value to display.
            
        Returns:
            numpy.ndarray: The annotated frame.
        """
        output_frame = frame.copy()
        
        # Draw tracked objects
        if self.show_boxes and tracked_objects:
            for obj in tracked_objects:
                # Extract object properties
                track_id = obj.get('id', -1)
                box = obj.get('bbox', [0, 0, 0, 0])  # [x1, y1, x2, y2]
                class_id = obj.get('class_id', 0)
                confidence = obj.get('confidence', 0)
                
                # Determine color based on object status
                if track_id in missing_ids:
                    color = self.missing_color
                    status = "MISSING"
                elif track_id in new_ids:
                    color = self.new_color
                    status = "NEW"
                else:
                    color = self.normal_color
                    status = ""
                
                # Draw bounding box
                cv2.rectangle(
                    output_frame, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    color, 
                    self.box_thickness
                )
                
                # Draw label if enabled
                if self.show_labels:
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    label = f"{class_name} {int(confidence * 100)}% ID:{track_id}"
                    if status:
                        label = f"{status}: {label}"
                        
                    # Text background
                    text_size = cv2.getTextSize(
                        label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        self.text_size, 
                        self.text_thickness
                    )[0]
                    cv2.rectangle(
                        output_frame, 
                        (int(box[0]), int(box[1] - 20)), 
                        (int(box[0] + text_size[0]), int(box[1])), 
                        color, 
                        -1
                    )
                    
                    # Text
                    cv2.putText(
                        output_frame, 
                        label, 
                        (int(box[0]), int(box[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        self.text_size, 
                        (255, 255, 255), 
                        self.text_thickness
                    )
        
        # Draw FPS if enabled
        if self.show_fps and fps is not None:
            cv2.putText(
                output_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255), 
                2
            )
            
        return output_frame
        
    def create_status_panel(self, missing_objects, new_objects, frame_width):
        """
        Create a status panel showing details of missing and new objects.
        
        Args:
            missing_objects (list): List of missing object details.
            new_objects (list): List of new object details.
            frame_width (int): Width of the frame for panel sizing.
            
        Returns:
            numpy.ndarray: Status panel image.
        """
        # Create blank panel with fixed height and frame width
        panel_height = max(len(missing_objects), len(new_objects)) * 30 + 40
        panel_height = max(panel_height, 400)  # Minimum height
        panel = np.zeros((panel_height, frame_width, 3), dtype=np.uint8)
        
        # Draw headers
        cv2.putText(
            panel, 
            "Status Monitor", 
            (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Draw missing objects
        y_pos = 60
        cv2.putText(
            panel, 
            f"Missing Objects ({len(missing_objects)}):", 
            (10, y_pos), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            self.missing_color, 
            2
        )
        
        y_pos += 25
        for i, obj in enumerate(missing_objects[:10]):  # Limit to 10 items
            class_name = self.class_names[obj.get('class_id', 0)] if obj.get('class_id', 0) < len(self.class_names) else "object"
            track_id = obj.get('id', -1)
            cv2.putText(
                panel, 
                f"- ID {track_id}: {class_name}", 
                (30, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                self.missing_color, 
                1
            )
            y_pos += 20
            
        # Draw new objects
        y_pos = 60
        half_width = frame_width // 2
        cv2.putText(
            panel, 
            f"New Objects ({len(new_objects)}):", 
            (half_width, y_pos), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            self.new_color, 
            2
        )
        
        y_pos += 25
        for i, obj in enumerate(new_objects[:10]):  # Limit to 10 items
            class_name = self.class_names[obj.get('class_id', 0)] if obj.get('class_id', 0) < len(self.class_names) else "object"
            track_id = obj.get('id', -1)
            cv2.putText(
                panel, 
                f"- ID {track_id}: {class_name}", 
                (half_width + 20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                self.new_color, 
                1
            )
            y_pos += 20
            
        return panel 