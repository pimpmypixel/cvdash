import cv2
import time
from queue import Queue
import config.config as c
from classes.utils.utils import add_log

def capture_webcam(queue: Queue):
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            queue.put(frame)
        time.sleep(0.1)
    cap.release()

def is_webcam_accessible():
    cap = cv2.VideoCapture(c.webcam_index)
    if not cap.isOpened():
        add_log(f"Error: Could not access webcam")
        return False
    
    # Read a test frame to get dimensions
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        cap.release()
        return height, width
    
    cap.release()
    return False