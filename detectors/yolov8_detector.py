import torch
import numpy as np
from ultralytics import YOLO


class YOLOv8Detector:
    """
    Object detector using Ultralytics YOLOv8 model optimized for CPU or GPU.
    """
    def __init__(self, config):
        """
        Initialize the YOLOv8 detector.
        
        Args:
            config (dict): Configuration settings for the detector.
        """
        self.model_path = config.get('model_path', 'yolov8n.pt')
        self.conf_threshold = config.get('confidence_threshold', 0.7)
        self.classes = config.get('classes', None)  # None means all classes
        
        # Load the model
        self.model = self._load_model()
        
    def _load_model(self):
        """
        Load the YOLOv8 model with appropriate settings.
        
        Returns:
            YOLO: The loaded YOLOv8 model.
        """
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLOv8 model on {device}...")
        
        # Load the model
        model = YOLO(self.model_path)
        model.to(device)
        
        return model
    
    def detect(self, frame):
        """
        Perform object detection on a single frame.
        
        Args:
            frame (numpy.ndarray): The input frame for detection.
            
        Returns:
            list: List of detection results, each containing:
                 - bbox: [x1, y1, x2, y2] coordinates
                 - confidence: detection confidence
                 - class_id: detected class ID
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, classes=self.classes, verbose=False)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            
            # Extract bounding boxes, classes and confidence scores
            for i, box in enumerate(boxes):
                # Extract box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Extract class ID
                class_id = int(box.cls.cpu().numpy()[0])
                
                # Extract confidence
                confidence = float(box.conf.cpu().numpy()[0])
                
                # Create detection object
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id
                }
                
                detections.append(detection)
        
        return detections 