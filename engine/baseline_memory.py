import numpy as np
import time
from collections import defaultdict


class BaselineMemory:
    def __init__(self, config):
        self.missing_frames_threshold = config.get('missing_object_frames', 15)
        self.stability_frames = config.get('stability_frames', 5)
        
        self.tracked_objects = {}
        self.object_history = defaultdict(list)
        self.missing_objects = {}
        self.new_objects = {}
        
        self.baseline_time = time.time()
        self.is_baseline_established = False
        self.frame_count = 0
        
    def reset(self):
        self.tracked_objects = {}
        self.object_history = defaultdict(list)
        self.missing_objects = {}
        self.new_objects = {}
        self.baseline_time = time.time()
        self.is_baseline_established = False
        self.frame_count = 0
        
        print("Baseline memory has been reset.")
        
    def update(self, tracked_objects):
        self.frame_count += 1
        
        current_objects = {obj['id']: obj for obj in tracked_objects}
        current_ids = set(current_objects.keys())
        
        if not self.is_baseline_established and self.frame_count >= self.stability_frames:
            self.tracked_objects = current_objects.copy()
            self.is_baseline_established = True
            print(f"Baseline established with {len(self.tracked_objects)} objects")
            
        if self.is_baseline_established:
            previous_ids = set(self.tracked_objects.keys())
            
            potentially_missing = previous_ids - current_ids
            
            self._update_missing_objects(potentially_missing, current_objects)
            
            potentially_new = current_ids - previous_ids
            
            self._update_new_objects(potentially_new, current_objects)
            
            for obj_id, obj_data in current_objects.items():
                self.tracked_objects[obj_id] = obj_data
                
                self.object_history[obj_id].append({
                    'frame': self.frame_count,
                    'bbox': obj_data['bbox'],
                    'confidence': obj_data['confidence'],
                    'timestamp': time.time()
                })
                
                if len(self.object_history[obj_id]) > 30:
                    self.object_history[obj_id].pop(0)
            
        missing_ids = list(self.missing_objects.keys())
        new_ids = list(self.new_objects.keys())
        
        missing_objects_data = [obj for _, obj in self.missing_objects.items()]
        new_objects_data = [obj for _, obj in self.new_objects.items()]
        
        return missing_ids, new_ids, missing_objects_data, new_objects_data
        
    def _update_missing_objects(self, potentially_missing, current_objects):
        for obj_id in potentially_missing:
            if obj_id in self.missing_objects:
                self.missing_objects[obj_id]['missing_frames'] += 1
                
                if self.missing_objects[obj_id]['missing_frames'] >= self.missing_frames_threshold:
                    if self.missing_objects[obj_id]['missing_frames'] == self.missing_frames_threshold:
                        print(f"Object ID {obj_id} is now considered MISSING")
            else:
                if obj_id in self.tracked_objects:
                    obj_data = self.tracked_objects[obj_id].copy()
                    obj_data['missing_frames'] = 1
                    self.missing_objects[obj_id] = obj_data
                    
        reappeared = []
        for obj_id in list(self.missing_objects.keys()):
            if obj_id in current_objects:
                print(f"Object ID {obj_id} has reappeared")
                reappeared.append(obj_id)
                
        for obj_id in reappeared:
            del self.missing_objects[obj_id]
                
    def _update_new_objects(self, potentially_new, current_objects):
        for obj_id in potentially_new:
            if obj_id in self.new_objects:
                self.new_objects[obj_id]['new_frames'] += 1
                
                if self.new_objects[obj_id]['new_frames'] >= self.stability_frames:
                    if self.new_objects[obj_id]['new_frames'] == self.stability_frames:
                        print(f"NEW object detected with ID {obj_id}")
            else:
                obj_data = current_objects[obj_id].copy()
                obj_data['new_frames'] = 1
                self.new_objects[obj_id] = obj_data
                
        expired_new = []
        for obj_id, obj_data in self.new_objects.items():
            if obj_data['new_frames'] > self.stability_frames * 3:
                expired_new.append(obj_id)
                
        for obj_id in expired_new:
            del self.new_objects[obj_id]
            
    def get_object_history(self, obj_id):
        return self.object_history.get(obj_id, []) 