import numpy as np
import cv2


class KalmanTracker:
    """
    Simple Kalman filter-based tracker for object tracking when Deep SORT is not available.
    """
    def __init__(self, bbox, track_id, class_id, confidence, max_age=30):
        """
        Initialize a tracker for a single object.
        
        Args:
            bbox (list): Initial bounding box [x1, y1, x2, y2].
            track_id (int): Unique tracking ID.
            class_id (int): Class ID of the object.
            confidence (float): Detection confidence.
            max_age (int): Maximum number of frames to keep the track alive without detections.
        """
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = confidence
        self.max_age = max_age
        self.age = 0
        self.time_since_update = 0
        
        # Initialize Kalman filter
        self.kf = cv2.KalmanFilter(8, 4)
        
        # State transition matrix (8x8)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 1, 0],  # w
            [0, 0, 0, 1, 0, 0, 0, 1],  # h
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw
            [0, 0, 0, 0, 0, 0, 0, 1],  # dh
        ], dtype=np.float32)
        
        # Measurement matrix (4x8)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance matrix (8x8)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise covariance matrix (4x4)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Error covariance matrix (8x8)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        
        # Initial state
        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        
        # Track history
        self.history = [bbox]
        
    def _bbox_to_xywh(self, bbox):
        """
        Convert [x1, y1, x2, y2] bounding box to [x, y, w, h] format.
        
        Args:
            bbox (list): Bounding box in [x1, y1, x2, y2] format.
            
        Returns:
            tuple: (x, y, w, h) where (x, y) is the center.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2
        y = y1 + h/2
        return x, y, w, h
        
    def _xywh_to_bbox(self, x, y, w, h):
        """
        Convert [x, y, w, h] format to [x1, y1, x2, y2] bounding box.
        
        Args:
            x (float): Center x coordinate.
            y (float): Center y coordinate.
            w (float): Width.
            h (float): Height.
            
        Returns:
            list: Bounding box in [x1, y1, x2, y2] format.
        """
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return [x1, y1, x2, y2]
        
    def predict(self):
        """
        Predict the new state using the Kalman filter.
        
        Returns:
            list: Predicted bounding box in [x1, y1, x2, y2] format.
        """
        if self.time_since_update > 0:
            self.age += 1
            self.time_since_update += 1
            
        # Predict
        self.kf.predict()
        
        # Get state
        x, y, w, h = self.kf.statePost[0:4, 0]
        
        # Convert to bbox
        bbox = self._xywh_to_bbox(x, y, w, h)
        
        return bbox
        
    def update(self, bbox, confidence=None, class_id=None):
        """
        Update the tracker with a new detection.
        
        Args:
            bbox (list): New bounding box [x1, y1, x2, y2].
            confidence (float, optional): New detection confidence.
            class_id (int, optional): New class ID.
            
        Returns:
            list: Updated bounding box in [x1, y1, x2, y2] format.
        """
        # Reset counters
        self.time_since_update = 0
        self.age += 1
        
        # Update confidence and class if provided
        if confidence is not None:
            self.confidence = confidence
        if class_id is not None:
            self.class_id = class_id
            
        # Convert to measurement format
        x, y, w, h = self._bbox_to_xywh(bbox)
        measurement = np.array([x, y, w, h], dtype=np.float32).reshape(4, 1)
        
        # Update Kalman filter
        self.kf.correct(measurement)
        
        # Add to history
        self.history.append(bbox)
        if len(self.history) > 5:  # Keep last 5 positions
            self.history.pop(0)
            
        # Get current state
        x, y, w, h = self.kf.statePost[0:4, 0]
        
        # Convert to bbox
        bbox = self._xywh_to_bbox(x, y, w, h)
        
        return bbox
        
    def get_state(self):
        """
        Get the current state of the tracker.
        
        Returns:
            dict: Current state including ID, bbox, confidence, etc.
        """
        # Get current state
        x, y, w, h = self.kf.statePost[0:4, 0]
        
        # Convert to bbox
        bbox = self._xywh_to_bbox(x, y, w, h)
        
        return {
            'id': self.track_id,
            'bbox': bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'age': self.age,
            'time_since_update': self.time_since_update
        }


class DeepSORTTracker:
    """
    Implementation of a tracker using Kalman filter similar to Deep SORT principles.
    """
    def __init__(self, config):
        """
        Initialize the tracker with configuration settings.
        
        Args:
            config (dict): Configuration settings for the tracker.
        """
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
        self.trackers = []
        self.next_id = 1
        self.frame_count = 0
        
    def _calculate_iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            boxA (list): First bounding box [x1, y1, x2, y2].
            boxB (list): Second bounding box [x1, y1, x2, y2].
            
        Returns:
            float: IoU value.
        """
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute the IoU by taking the intersection area and dividing it by the sum of both box areas
        # minus the intersection area (to avoid double counting)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        return iou
    
    def _assign_detections_to_trackers(self, detections, trackers):
        """
        Assign detections to existing trackers using IoU.
        
        Args:
            detections (list): List of detection dictionaries.
            trackers (list): List of tracker objects.
            
        Returns:
            tuple: (matches, unmatched_detections, unmatched_trackers)
        """
        if not trackers or not detections:
            return [], list(range(len(detections))), list(range(len(trackers)))
            
        # Build cost matrix based on IoU
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                trk_state = trk.get_state()
                iou_matrix[d, t] = self._calculate_iou(det['bbox'], trk_state['bbox'])
                
        # Use Hungarian algorithm for assignment (simplified greedy match here)
        matches = []
        unmatched_detections = []
        unmatched_trackers = list(range(len(trackers)))
        
        # For each detection, find the tracker with the highest IoU
        for d, det in enumerate(detections):
            best_iou = self.iou_threshold
            best_t = -1
            
            for t in unmatched_trackers:
                if iou_matrix[d, t] > best_iou:
                    best_iou = iou_matrix[d, t]
                    best_t = t
                    
            if best_t >= 0:
                matches.append((d, best_t))
                unmatched_trackers.remove(best_t)
            else:
                unmatched_detections.append(d)
                
        return matches, unmatched_detections, unmatched_trackers
        
    def update(self, detections):
        """
        Update trackers with new detections.
        
        Args:
            detections (list): List of detection dictionaries from the detector.
            
        Returns:
            list: List of tracked objects with IDs and states.
        """
        self.frame_count += 1
        
        # Predict new locations for existing trackers
        for trk in self.trackers:
            trk.predict()
            
        # Get trackers that are active (not lost for too long)
        active_trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]
        
        # Match detections to trackers
        matched_idx, unmatched_detections, unmatched_trackers = self._assign_detections_to_trackers(
            detections, active_trackers
        )
        
        # Update matched trackers with assigned detections
        for d_idx, t_idx in matched_idx:
            active_trackers[t_idx].update(
                detections[d_idx]['bbox'],
                detections[d_idx]['confidence'],
                detections[d_idx]['class_id']
            )
            
        # Increment age and time_since_update for unmatched trackers
        for t_idx in unmatched_trackers:
            active_trackers[t_idx].age += 1
            active_trackers[t_idx].time_since_update += 1
            
        # Create new trackers for unmatched detections
        for d_idx in unmatched_detections:
            new_tracker = KalmanTracker(
                detections[d_idx]['bbox'],
                self.next_id,
                detections[d_idx]['class_id'],
                detections[d_idx]['confidence'],
                self.max_age
            )
            self.next_id += 1
            active_trackers.append(new_tracker)
            
        # Filter out dead trackers
        self.trackers = [trk for trk in active_trackers if trk.time_since_update <= self.max_age]
        
        # Prepare result
        tracked_objects = []
        for trk in self.trackers:
            trk_state = trk.get_state()
            
            # Only return objects that have been tracked for min_hits frames
            if trk.age >= self.min_hits and trk.time_since_update <= 1:
                tracked_objects.append(trk_state)
                
        return tracked_objects 