import time
import threading
from queue import Queue

class GraphData:
    def __init__(self, color_queue: Queue):
        self.history = []
        self.status = {'fps': 0, 'mode': 'RUN'}
        self.lock = threading.Lock()
        self.color_queue = color_queue
        self.last_queue_size = 0

    def update_loop(self):
        while True:
            with self.lock:
                current_size = self.color_queue.qsize()
                
                # Only process new items
                if current_size > self.last_queue_size:
                    # Get only the new items
                    for _ in range(current_size - self.last_queue_size):
                        if not self.color_queue.empty():
                            item = self.color_queue.get()
                            self.history.append(item)
                            # Put the item back
                            self.color_queue.put(item)
                    
                    # Keep only the last 100 colors
                    if len(self.history) > 100:
                        self.history = self.history[-100:]
                    
                    self.last_queue_size = current_size
                
                self.status['fps'] = 30  # Placeholder
            time.sleep(0.1)

    def get_history(self):
        with self.lock:
            return list(self.history)

    def get_status(self):
        with self.lock:
            return dict(self.status)
