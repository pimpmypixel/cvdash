import time
import threading
from queue import Queue
import config.config as c

class GraphData:
    def __init__(self, color_queue: Queue):
        self.history = []
        self.status = {'fps': 0, 'mode': 'RUN'}
        self.lock = threading.Lock()
        self.color_queue = color_queue
        self.last_queue_size = 0
        self.max_history_size = c.window_width // 2  # Set max_history_size based on window width

    def update_loop(self):
        while True:
            with self.lock:
                try:
                    # Get the latest color from the queue
                    if not self.color_queue.empty():
                        item = self.color_queue.get()
                        # Add to history
                        self.history.append(item)
                        # Keep history size limited
                        if len(self.history) > self.max_history_size:
                            self.history = self.history[-self.max_history_size:]
                        # Put the item back in the queue
                        # self.color_queue.put(item) # REMOVED: Item should be consumed, not put back
                except Exception as e:
                    print(f"Error in update_loop: {e}")
                
                self.status['fps'] = 30  # Placeholder
            time.sleep(0.1)

    def get_history(self):
        with self.lock:
            return list(self.history)

    def get_status(self):
        with self.lock:
            return dict(self.status)
