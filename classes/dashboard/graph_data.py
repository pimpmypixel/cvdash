import time
import threading
from queue import Queue

class GraphData:
    def __init__(self, color_queue: Queue):
        self.history = []
        self.status = {'fps': 0, 'mode': 'RUN'}
        self.lock = threading.Lock()
        self.color_queue = color_queue

    def update_loop(self):
        while True:
            with self.lock:
                if not self.color_queue.empty():
                    color = self.color_queue.get()
                    self.history.append(color)
                    if len(self.history) > 100:  # Keep last 100 colors
                        self.history.pop(0)
                self.status['fps'] = 30  # Placeholder
            time.sleep(0.1)

    def get_history(self):
        with self.lock:
            return list(self.history)

    def get_status(self):
        with self.lock:
            return dict(self.status)
