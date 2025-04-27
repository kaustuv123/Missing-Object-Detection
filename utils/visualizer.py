import cv2
import numpy as np


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.box_thickness = config.get('box_thickness', 2)
        self.text_size = config.get('text_size', 0.5)
        self.text_thickness = config.get('text_thickness', 2)
        self.show_fps = config.get('show_fps', True)
        self.show_boxes = config.get('show_boxes', True)
        self.show_labels = config.get('show_labels', True)
        
        self.missing_color = tuple(config.get('missing_color', [0, 0, 255]))
        self.new_color = tuple(config.get('new_color', [0, 255, 0]))
        self.normal_color = tuple(config.get('normal_color', [255, 0, 0]))
        
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
        output_frame = frame.copy()
        
        if self.show_boxes and tracked_objects:
            for obj in tracked_objects:
                track_id = obj.get('id', -1)
                box = obj.get('bbox', [0, 0, 0, 0])
                class_id = obj.get('class_id', 0)
                confidence = obj.get('confidence', 0)
                
                if track_id in missing_ids:
                    color = self.missing_color
                    status = "MISSING"
                elif track_id in new_ids:
                    color = self.new_color
                    status = "NEW"
                else:
                    color = self.normal_color
                    status = ""
                
                cv2.rectangle(
                    output_frame, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    color, 
                    self.box_thickness
                )
                
                if self.show_labels:
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    label = f"{class_name} {int(confidence * 100)}% ID:{track_id}"
                    if status:
                        label = f"{status}: {label}"
                        
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
                    
                    cv2.putText(
                        output_frame, 
                        label, 
                        (int(box[0]), int(box[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        self.text_size, 
                        (255, 255, 255), 
                        self.text_thickness
                    )
        
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
        panel_height = max(len(missing_objects), len(new_objects)) * 30 + 40
        panel_height = max(panel_height, 400)
        panel = np.zeros((panel_height, frame_width, 3), dtype=np.uint8)
        
        cv2.putText(
            panel, 
            "Status Monitor", 
            (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
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
        for i, obj in enumerate(missing_objects[:10]):
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
        for i, obj in enumerate(new_objects[:10]):
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