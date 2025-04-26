import time
from collections import deque


class FPSTimer:
    """
    Utility class for calculating and tracking frames per second.
    """
    def __init__(self, buffer_size=30):
        """
        Initialize the FPS timer.
        
        Args:
            buffer_size (int): Number of frames to use for FPS calculation.
        """
        self.prev_time = time.time()
        self.frame_times = deque(maxlen=buffer_size)
        
    def update(self):
        """
        Update the timer with a new frame.
        
        Returns:
            float: Current FPS value.
        """
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time
        self.frame_times.append(delta)
        
        return self.get_fps()
    
    def get_fps(self):
        """
        Calculate the current FPS based on the time buffer.
        
        Returns:
            float: Current FPS value.
        """
        if not self.frame_times:
            return 0.0
            
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


class PerformanceTimer:
    """
    Utility class for measuring execution time of code blocks.
    Can be used as a context manager or as start/stop methods.
    """
    def __init__(self, name=""):
        """
        Initialize the performance timer.
        
        Args:
            name (str): Optional name for the timer.
        """
        self.name = name
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        """
        Start timing when entering a context block.
        
        Returns:
            PerformanceTimer: Self reference for use in context manager.
        """
        self.start()
        return self
        
    def __exit__(self, *args):
        """
        Stop timing when exiting a context block.
        
        Returns:
            float: The elapsed time in seconds.
        """
        self.stop()
        return self.elapsed
        
    def start(self):
        """
        Start the timer.
        """
        self.start_time = time.time()
        
    def stop(self):
        """
        Stop the timer and calculate elapsed time.
        
        Returns:
            float: The elapsed time in seconds.
        """
        if self.start_time is None:
            return 0
            
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
        
    def get_elapsed_ms(self):
        """
        Get the elapsed time in milliseconds.
        
        Returns:
            float: The elapsed time in milliseconds.
        """
        return self.elapsed * 1000 