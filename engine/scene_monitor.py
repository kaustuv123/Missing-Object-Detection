import time
from .baseline_memory import BaselineMemory


class SceneMonitor:
    """
    Monitors the scene for changes, such as missing objects and new objects.
    Uses the baseline memory to track objects over time.
    """
    def __init__(self, config):
        """
        Initialize the scene monitor with configuration settings.
        
        Args:
            config (dict): Configuration settings for scene monitoring.
        """
        self.config = config
        self.baseline_memory = BaselineMemory(config)
        self.last_update_time = time.time()
        
        # Metrics for monitoring
        self.total_missing_objects = 0
        self.total_new_objects = 0
        self.frame_count = 0
        
    def reset(self):
        """
        Reset the scene monitor to start fresh.
        """
        self.baseline_memory.reset()
        self.last_update_time = time.time()
        self.total_missing_objects = 0
        self.total_new_objects = 0
        self.frame_count = 0
        
    def process_frame(self, tracked_objects):
        """
        Process a frame of tracked objects to detect scene changes.
        
        Args:
            tracked_objects (list): List of tracked objects from the tracker.
            
        Returns:
            tuple: (tracked_objects, missing_ids, new_ids, missing_objects, new_objects)
                - tracked_objects: The list of currently tracked objects
                - missing_ids: IDs of objects considered missing
                - new_ids: IDs of newly detected objects
                - missing_objects: Full data of missing objects
                - new_objects: Full data of new objects
        """
        self.frame_count += 1
        
        # Update the baseline memory with current objects
        missing_ids, new_ids, missing_objects, new_objects = self.baseline_memory.update(tracked_objects)
        
        # Update metrics
        if len(missing_objects) > 0:
            self.total_missing_objects = max(self.total_missing_objects, len(missing_objects))
            
        if len(new_objects) > 0:
            self.total_new_objects = max(self.total_new_objects, len(new_objects))
            
        return tracked_objects, missing_ids, new_ids, missing_objects, new_objects
        
    def get_metrics(self):
        """
        Get monitoring metrics.
        
        Returns:
            dict: Dictionary of monitoring metrics.
        """
        return {
            'total_missing_objects': self.total_missing_objects,
            'total_new_objects': self.total_new_objects,
            'frames_processed': self.frame_count,
            'uptime': time.time() - self.last_update_time
        }
        
    def get_object_history(self, obj_id):
        """
        Get the history of a specific object.
        
        Args:
            obj_id (int): ID of the object to get history for.
            
        Returns:
            list: History of the object's tracking data.
        """
        return self.baseline_memory.get_object_history(obj_id) 