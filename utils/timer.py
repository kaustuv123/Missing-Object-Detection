import time
from collections import deque


class FPSTimer:
    def __init__(self, buffer_size=30):
        self.prev_time = time.time()
        self.frame_times = deque(maxlen=buffer_size)
        
    def update(self):
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time
        self.frame_times.append(delta)
        
        return self.get_fps()
    
    def get_fps(self):
        if not self.frame_times:
            return 0.0
            
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


class PerformanceTimer:
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.stop()
        return self.elapsed
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        if self.start_time is None:
            return 0
            
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
        
    def get_elapsed_ms(self):
        return self.elapsed * 1000 