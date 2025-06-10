import cv2
import time
from queue import Queue

def capture_webcam(queue: Queue):
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            queue.put(frame)
        time.sleep(0.1)
    cap.release()
