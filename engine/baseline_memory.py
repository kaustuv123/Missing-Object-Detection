import numpy as np
import time
from collections import defaultdict


class BaselineMemory:
    """
    Maintains a memory of tracked objects across frames to detect missing 
    and newly placed objects in the scene.
    """
    def __init__(self, config):
        """
        Initialize the baseline memory with configuration settings.
        
        Args:
            config (dict): Configuration settings for scene monitoring.
        """
        self.missing_frames_threshold = config.get('missing_object_frames', 15)
        self.stability_frames = config.get('stability_frames', 5)
        
        # Object tracking data
        self.tracked_objects = {}  # {object_id: object_data}
        self.object_history = defaultdict(list)  # {object_id: [frame_appearances]}
        self.missing_objects = {}  # {object_id: object_data with missing frame count}
        self.new_objects = {}  # {object_id: object_data with frame count}
        
        # Timestamp when memory was last reset
        self.baseline_time = time.time()
        self.is_baseline_established = False
        self.frame_count = 0
        
    def reset(self):
        """
        Reset the memory to start fresh.
        """
        self.tracked_objects = {}
        self.object_history = defaultdict(list)
        self.missing_objects = {}
        self.new_objects = {}
        self.baseline_time = time.time()
        self.is_baseline_established = False
        self.frame_count = 0
        
        print("Baseline memory has been reset.")
        
    def update(self, tracked_objects):
        """
        Update the memory with new tracked objects from the current frame.
        
        Args:
            tracked_objects (list): List of tracked objects from the tracker.
            
        Returns:
            tuple: (missing_ids, new_ids, missing_objects_data, new_objects_data)
                - missing_ids: IDs of objects considered missing
                - new_ids: IDs of newly detected objects
                - missing_objects_data: Full data of missing objects
                - new_objects_data: Full data of new objects
        """
        self.frame_count += 1
        
        # Convert tracked objects list to a dictionary for easier lookup
        current_objects = {obj['id']: obj for obj in tracked_objects}
        current_ids = set(current_objects.keys())
        
        # If baseline isn't established yet, establish it after a few frames
        if not self.is_baseline_established and self.frame_count >= self.stability_frames:
            self.tracked_objects = current_objects.copy()
            self.is_baseline_established = True
            print(f"Baseline established with {len(self.tracked_objects)} objects")
            
        # After baseline is established, track objects
        if self.is_baseline_established:
            # Get previously tracked object IDs
            previous_ids = set(self.tracked_objects.keys())
            
            # Find missing objects
            potentially_missing = previous_ids - current_ids
            
            # Update missing objects
            self._update_missing_objects(potentially_missing, current_objects)
            
            # Find new objects
            potentially_new = current_ids - previous_ids
            
            # Update new objects
            self._update_new_objects(potentially_new, current_objects)
            
            # Update tracked objects with current frame data
            for obj_id, obj_data in current_objects.items():
                self.tracked_objects[obj_id] = obj_data
                
                # Update object history
                self.object_history[obj_id].append({
                    'frame': self.frame_count,
                    'bbox': obj_data['bbox'],
                    'confidence': obj_data['confidence'],
                    'timestamp': time.time()
                })
                
                # Keep history bounded
                if len(self.object_history[obj_id]) > 30:  # Keep last 30 frames
                    self.object_history[obj_id].pop(0)
            
        # Prepare result
        missing_ids = list(self.missing_objects.keys())
        new_ids = list(self.new_objects.keys())
        
        # Get full data for missing and new objects
        missing_objects_data = [obj for _, obj in self.missing_objects.items()]
        new_objects_data = [obj for _, obj in self.new_objects.items()]
        
        return missing_ids, new_ids, missing_objects_data, new_objects_data
        
    def _update_missing_objects(self, potentially_missing, current_objects):
        """
        Update the missing objects based on the current frame.
        
        Args:
            potentially_missing (set): Set of IDs of potentially missing objects.
            current_objects (dict): Dictionary of currently tracked objects.
        """
        # Check objects that are potentially missing
        for obj_id in potentially_missing:
            if obj_id in self.missing_objects:
                # Object was already marked as potentially missing, increment count
                self.missing_objects[obj_id]['missing_frames'] += 1
                
                # If object has been missing for too long, confirm it as missing
                if self.missing_objects[obj_id]['missing_frames'] >= self.missing_frames_threshold:
                    # Object is officially missing, keep it in the missing_objects dict
                    if self.missing_objects[obj_id]['missing_frames'] == self.missing_frames_threshold:
                        print(f"Object ID {obj_id} is now considered MISSING")
            else:
                # Object is potentially missing for the first time
                if obj_id in self.tracked_objects:
                    obj_data = self.tracked_objects[obj_id].copy()
                    obj_data['missing_frames'] = 1
                    self.missing_objects[obj_id] = obj_data
                    
        # Check if any previously missing objects have reappeared
        reappeared = []
        for obj_id in list(self.missing_objects.keys()):
            if obj_id in current_objects:
                print(f"Object ID {obj_id} has reappeared")
                reappeared.append(obj_id)
                
        # Remove reappeared objects from missing list
        for obj_id in reappeared:
            del self.missing_objects[obj_id]
                
    def _update_new_objects(self, potentially_new, current_objects):
        """
        Update the newly detected objects based on the current frame.
        
        Args:
            potentially_new (set): Set of IDs of potentially new objects.
            current_objects (dict): Dictionary of currently tracked objects.
        """
        # Check objects that are potentially new
        for obj_id in potentially_new:
            if obj_id in self.new_objects:
                # Object was already marked as potentially new, increment count
                self.new_objects[obj_id]['new_frames'] += 1
                
                # If object has been stable for enough frames, confirm it as new
                if self.new_objects[obj_id]['new_frames'] >= self.stability_frames:
                    # Object is officially new, print once when it first becomes stable
                    if self.new_objects[obj_id]['new_frames'] == self.stability_frames:
                        print(f"NEW object detected with ID {obj_id}")
            else:
                # Object is potentially new for the first time
                obj_data = current_objects[obj_id].copy()
                obj_data['new_frames'] = 1
                self.new_objects[obj_id] = obj_data
                
        # Remove objects from new list if they've been around for a while
        expired_new = []
        for obj_id, obj_data in self.new_objects.items():
            if obj_data['new_frames'] > self.stability_frames * 3:  # After 3x stability frames
                expired_new.append(obj_id)
                
        # Remove expired new objects
        for obj_id in expired_new:
            del self.new_objects[obj_id]
            
    def get_object_history(self, obj_id):
        """
        Get the history of a specific object.
        
        Args:
            obj_id (int): ID of the object to get history for.
            
        Returns:
            list: History of the object's tracking data.
        """
        return self.object_history.get(obj_id, []) 