import time
from .baseline_memory import BaselineMemory


class SceneMonitor:
    def __init__(self, config):
        self.config = config
        self.baseline_memory = BaselineMemory(config)
        self.last_update_time = time.time()
        
        self.total_missing_objects = 0
        self.total_new_objects = 0
        self.frame_count = 0
        
    def reset(self):
        self.baseline_memory.reset()
        self.last_update_time = time.time()
        self.total_missing_objects = 0
        self.total_new_objects = 0
        self.frame_count = 0
        
    def process_frame(self, tracked_objects):
        self.frame_count += 1
        
        missing_ids, new_ids, missing_objects, new_objects = self.baseline_memory.update(tracked_objects)
        
        if len(missing_objects) > 0:
            self.total_missing_objects = max(self.total_missing_objects, len(missing_objects))
            
        if len(new_objects) > 0:
            self.total_new_objects = max(self.total_new_objects, len(new_objects))
            
        return tracked_objects, missing_ids, new_ids, missing_objects, new_objects
        
    def get_metrics(self):
        return {
            'total_missing_objects': self.total_missing_objects,
            'total_new_objects': self.total_new_objects,
            'frames_processed': self.frame_count,
            'uptime': time.time() - self.last_update_time
        }
        
    def get_object_history(self, obj_id):
        return self.baseline_memory.get_object_history(obj_id) 