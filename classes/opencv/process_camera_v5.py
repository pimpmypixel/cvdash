import cv2
import numpy as np
from ultralytics import YOLO
from classes.utils.logger import add_log
import time
import os
import json
import queue

# Constants
ROI_MAX_AGE = 86400  # 24 hours
ROI_CHECK_INTERVAL = 0  # Disable automatic ROI checks since webcam is fixed
CENTER_BIAS_THRESHOLD = 0.3  # TV should be within 30% of center distance from image center
CENTER_STABILITY_THRESHOLD = 200  # Much more lenient since webcam is fixed
LOCK_CONFIDENCE_THRESHOLD = 1.0
CONF_INCREASE_STABLE = 0.3
CONF_DECREASE_UNSTABLE = 0.1
CONF_DECREASE_MISSED = 0.2
MIN_TV_AREA = 2000  # To avoid very small false positives
ROI_SAVE_PATH = 'storage/tv_roi.json'
YOLO_CONFIDENCE = 0.5

# State
tv_box = None
detection_confidence = 0.0
locked_roi = None
last_roi_check_time = 0
model = YOLO("config/yolov10n.pt")


def save_roi(roi):
    if roi is None or len(roi) != 4:
        return
    try:
        os.makedirs('storage', exist_ok=True)
        with open(ROI_SAVE_PATH, 'w') as f:
            json.dump({'roi': list(roi), 'timestamp': time.time()}, f)
        add_log("TV ROI saved")
    except Exception as e:
        add_log(f"Error saving ROI: {e}")


def load_saved_roi():
    global locked_roi, detection_confidence
    try:
        if not os.path.exists(ROI_SAVE_PATH):
            return False
        with open(ROI_SAVE_PATH, 'r') as f:
            data = json.load(f)
        if time.time() - data.get('timestamp', 0) < ROI_MAX_AGE:
            locked_roi = tuple(data['roi'])
            detection_confidence = LOCK_CONFIDENCE_THRESHOLD
            add_log("Loaded saved ROI")
            return True
        else:
            os.remove(ROI_SAVE_PATH)
            add_log("Saved ROI expired")
    except Exception as e:
        add_log(f"Error loading ROI: {e}")
    return False


def is_centered(box, frame_shape):
    """Check if box is sufficiently centered"""
    frame_center = np.array([frame_shape[1] / 2, frame_shape[0] / 2])
    box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    norm_dist = np.linalg.norm(box_center - frame_center) / max(frame_shape[:2])
    return norm_dist < CENTER_BIAS_THRESHOLD


def confirm_roi(frame):
    """Since webcam is fixed, we only invalidate ROI on manual reset"""
    return True  # Always return True unless manually reset


def process_webcam_frame(frame, avg_color_queue):
    global tv_box, detection_confidence, locked_roi

    if locked_roi is None:
        load_saved_roi()

    if locked_roi:  # Removed confirm_roi check since webcam is fixed
        x1, y1, x2, y2 = locked_roi
        roi = frame[y1:y2, x1:x2]
        avg_bgr, avg_rgb = calculate_average_color(roi)
        if avg_rgb is not None:
            if avg_color_queue.full():
                try:
                    avg_color_queue.get_nowait()
                    add_log("Removed old item from webcam color queue.")
                except queue.Empty:
                    add_log("Attempted to remove from empty webcam color queue (shouldn't happen if full()).")
                    pass
            avg_color_queue.put(avg_rgb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return frame

    try:
        results = model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)
        best_tv = None
        best_score = -1

        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = result.names[int(cls)].lower()
                area = (x2 - x1) * (y2 - y1)
                if label == "tv" and area > MIN_TV_AREA and is_centered((x1, y1, x2, y2), frame.shape):
                    center_score = 1.0 - (np.linalg.norm(
                        np.array([(x1 + x2)/2, (y1 + y2)/2]) - np.array([frame.shape[1]/2, frame.shape[0]/2])
                    ) / max(frame.shape[:2]))
                    if center_score > best_score:
                        best_score = center_score
                        best_tv = (x1, y1, x2, y2)

        if best_tv:
            tv_box = best_tv
            update_detection_confidence(tv_box)
            x1, y1, x2, y2 = tv_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, "TV", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            roi = frame[y1:y2, x1:x2]
            avg_bgr, avg_rgb = calculate_average_color(roi)
            if avg_rgb is not None:
                if avg_color_queue.full():
                    try:
                        avg_color_queue.get_nowait()
                        add_log("Removed old item from webcam color queue.")
                    except queue.Empty:
                        add_log("Attempted to remove from empty webcam color queue (shouldn't happen if full()).")
                        pass
                avg_color_queue.put(avg_rgb)
    except Exception as e:
        add_log(f"YOLO inference error: {e}")

    return frame


def update_detection_confidence(current_box):
    global detection_confidence, locked_roi, tv_box

    if current_box is None:
        detection_confidence = max(0, detection_confidence - CONF_DECREASE_MISSED)
        return

    if tv_box is None:
        detection_confidence = 0.6
        return

    current_center = np.array([(current_box[0] + current_box[2]) / 2, (current_box[1] + current_box[3]) / 2])
    last_center = np.array([(tv_box[0] + tv_box[2]) / 2, (tv_box[1] + tv_box[3]) / 2])
    dist = np.linalg.norm(current_center - last_center)

    if dist < 10:
        detection_confidence = min(1.0, detection_confidence + CONF_INCREASE_STABLE)
    else:
        detection_confidence = max(0, detection_confidence - CONF_DECREASE_UNSTABLE)

    if detection_confidence >= LOCK_CONFIDENCE_THRESHOLD:
        locked_roi = current_box
        save_roi(current_box)
        add_log("TV ROI locked")


def calculate_average_color(roi):
    if roi is None or roi.size == 0:
        return None, None
    small_roi = roi[::max(roi.shape[0] // 100, 1), ::max(roi.shape[1] // 100, 1)]
    avg_bgr = np.mean(small_roi, axis=(0, 1))
    avg_rgb = avg_bgr[::-1]
    return avg_bgr, avg_rgb


def reset_detection():
    global tv_box, detection_confidence, locked_roi, last_roi_check_time
    tv_box = None
    detection_confidence = 0.0
    locked_roi = None
    last_roi_check_time = 0
    try:
        if os.path.exists(ROI_SAVE_PATH):
            os.remove(ROI_SAVE_PATH)
    except Exception as e:
        add_log(f"Error removing ROI file: {e}")
    add_log("TV detection reset")
