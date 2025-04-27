import torch
import numpy as np
from ultralytics import YOLO


class YOLOv8Detector:
    def __init__(self, config):
        self.model_path = config.get('model_path', 'yolov8n.pt')
        self.conf_threshold = config.get('confidence_threshold', 0.7)
        self.classes = config.get('classes', None)
        
        self.model = self._load_model()
        
    def _load_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLOv8 model on {device}...")
        
        model = YOLO(self.model_path)
        model.to(device)
        
        return model
    
    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, classes=self.classes, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                class_id = int(box.cls.cpu().numpy()[0])
                
                confidence = float(box.conf.cpu().numpy()[0])
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id
                }
                
                detections.append(detection)
        
        return detections 