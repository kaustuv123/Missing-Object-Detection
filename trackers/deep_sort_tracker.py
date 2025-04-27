import numpy as np
import cv2


class KalmanTracker:
    def __init__(self, bbox, track_id, class_id, confidence, max_age=30):
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = confidence
        self.max_age = max_age
        self.age = 0
        self.time_since_update = 0
        
        self.kf = cv2.KalmanFilter(8, 4)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        
        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        
        self.history = [bbox]
        
    def _bbox_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2
        y = y1 + h/2
        return x, y, w, h
        
    def _xywh_to_bbox(self, x, y, w, h):
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return [x1, y1, x2, y2]
        
    def predict(self):
        if self.time_since_update > 0:
            self.age += 1
            self.time_since_update += 1
            
        self.kf.predict()
        
        x, y, w, h = self.kf.statePost[0:4, 0]
        
        bbox = self._xywh_to_bbox(x, y, w, h)
        
        return bbox
        
    def update(self, bbox, confidence=None, class_id=None):
        self.time_since_update = 0
        self.age += 1
        
        if confidence is not None:
            self.confidence = confidence
        if class_id is not None:
            self.class_id = class_id
            
        x, y, w, h = self._bbox_to_xywh(bbox)
        measurement = np.array([x, y, w, h], dtype=np.float32).reshape(4, 1)
        
        self.kf.correct(measurement)
        
        self.history.append(bbox)
        if len(self.history) > 5:
            self.history.pop(0)
            
        x, y, w, h = self.kf.statePost[0:4, 0]
        
        bbox = self._xywh_to_bbox(x, y, w, h)
        
        return bbox
        
    def get_state(self):
        x, y, w, h = self.kf.statePost[0:4, 0]
        
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
    def __init__(self, config):
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
        self.trackers = []
        self.next_id = 1
        self.frame_count = 0
        
    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        return iou
    
    def _assign_detections_to_trackers(self, detections, trackers):
        if not trackers or not detections:
            return [], list(range(len(detections))), list(range(len(trackers)))
            
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                trk_state = trk.get_state()
                iou_matrix[d, t] = self._calculate_iou(det['bbox'], trk_state['bbox'])
                
        matches = []
        unmatched_detections = []
        unmatched_trackers = list(range(len(trackers)))
        
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
        self.frame_count += 1
        
        for trk in self.trackers:
            trk.predict()
            
        active_trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]
        
        matched_idx, unmatched_detections, unmatched_trackers = self._assign_detections_to_trackers(
            detections, active_trackers
        )
        
        for d_idx, t_idx in matched_idx:
            active_trackers[t_idx].update(
                detections[d_idx]['bbox'],
                detections[d_idx]['confidence'],
                detections[d_idx]['class_id']
            )
            
        for t_idx in unmatched_trackers:
            active_trackers[t_idx].age += 1
            active_trackers[t_idx].time_since_update += 1
            
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
            
        self.trackers = [trk for trk in active_trackers if trk.time_since_update <= self.max_age]
        
        tracked_objects = []
        for trk in self.trackers:
            trk_state = trk.get_state()
            
            if trk.age >= self.min_hits and trk.time_since_update <= 1:
                tracked_objects.append(trk_state)
                
        return tracked_objects 