import cv2
import numpy as np
from ultralytics import YOLO
from classes.utils.logger import add_log
import time
import os
import json

# Global variables for TV detection
tv_box = None
detection_confidence = 0.0
locked_roi = None
last_processed_roi_frame = None
last_roi_check_time = 0
ROI_CHECK_INTERVAL = 60  # Check ROI every 10 seconds
ROI_SAVE_PATH = 'storage/tv_roi.json'

model = YOLO("config/yolov8n.pt")  # or path to a lighter custom-trained model

def save_roi(roi):
    """Save ROI coordinates to JSON file"""
    if roi is not None and len(roi) == 4:
        try:
            os.makedirs('storage', exist_ok=True)
            roi_list = roi.tolist() if isinstance(roi, np.ndarray) else list(roi)
            with open(ROI_SAVE_PATH, 'w') as f:
                json.dump({
                    'roi': roi_list,
                    'timestamp': time.time()
                }, f)
            add_log("TV ROI saved")
        except Exception as e:
            add_log(f"Error saving ROI: {e}")

def load_saved_roi():
    """Load saved ROI from JSON file with validation"""
    global locked_roi, detection_confidence
    try:
        if os.path.exists(ROI_SAVE_PATH):
            with open(ROI_SAVE_PATH, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'roi' in data:
                    roi_list = data['roi']
                    # Validate age of saved ROI (24 hours max)
                    if time.time() - data.get('timestamp', 0) < 86400:
                        locked_roi = tuple(roi_list)
                        detection_confidence = 1.0
                        add_log("Loaded valid saved TV ROI")
                        return True
                    else:
                        add_log("Saved ROI expired, will re-detect")
                        os.remove(ROI_SAVE_PATH)
    except Exception as e:
        add_log(f"Error loading ROI: {e}")
    return False

def confirm_roi(frame):
    """Periodically confirm the locked ROI is still valid"""
    global locked_roi, last_roi_check_time
    
    current_time = time.time()
    if current_time - last_roi_check_time < ROI_CHECK_INTERVAL:
        return True
    
    last_roi_check_time = current_time
    
    try:
        results = model.predict(frame, conf=0.3, verbose=False)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = result.names[int(cls)]
                if label.lower() == "tv":
                    # Check if detected TV is close to locked ROI
                    current_center = np.array([(x1 + x2)/2, (y1 + y2)/2])
                    locked_center = np.array([(locked_roi[0] + locked_roi[2])/2, (locked_roi[1] + locked_roi[3])/2])
                    center_distance = np.linalg.norm(current_center - locked_center)
                    
                    if center_distance < 50:  # Allow some movement
                        return True
    except Exception as e:
        add_log(f"ROI confirmation error: {e}")
    
    # If we get here, ROI is no longer valid
    add_log("ROI no longer valid, resetting detection")
    reset_detection()
    return False

def process_webcam_frame(frame, avg_color_queue):
    global tv_box, detection_confidence, locked_roi, last_processed_roi_frame
    
    # Try to load saved ROI first if not already loaded
    if locked_roi is None:
        load_saved_roi()
    
    # If ROI is locked, use it directly
    if locked_roi is not None:
        # Periodically confirm ROI is still valid
        if not confirm_roi(frame):
            return frame
            
        x1, y1, x2, y2 = locked_roi
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            avg_bgr, avg_rgb = calculate_average_color(frame, locked_roi)
            if avg_rgb is not None:
                avg_color_queue.put(avg_rgb)
            # Draw thin bounding box for locked ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return frame
    
    # Otherwise, try to detect TV
    try:
        results = model.predict(frame, conf=0.3, verbose=False)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = result.names[int(cls)]
                if label.lower() == "tv":
                    tv_box = (x1, y1, x2, y2)
                    # Update confidence
                    update_detection_confidence(tv_box)
                    # Draw thin bounding box for detected TV
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, "TV", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print("YOLO inference error:", e)

    if tv_box:
        x1, y1, x2, y2 = tv_box
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            avg_bgr, avg_rgb = calculate_average_color(frame, tv_box)
            if avg_rgb is not None:
                avg_color_queue.put(avg_rgb)

    return frame

def update_detection_confidence(current_box):
    """Update confidence in TV detection"""
    global detection_confidence, locked_roi, tv_box
    
    if current_box is None:
        detection_confidence = max(0, detection_confidence - 0.2)
        return
    
    # For first detection, set initial confidence
    if tv_box is None:
        detection_confidence = 0.6
        return
    
    # Calculate stability
    current_center = np.array([(current_box[0] + current_box[2])/2, (current_box[1] + current_box[3])/2])
    last_center = np.array([(tv_box[0] + tv_box[2])/2, (tv_box[1] + tv_box[3])/2])
    center_distance = np.linalg.norm(current_center - last_center)
    
    # Update confidence based on stability
    if center_distance < 10:  # Very stable
        confidence_increase = 0.3
        detection_confidence = min(1.0, detection_confidence + confidence_increase)
    else:
        detection_confidence = max(0, detection_confidence - 0.15)
    
    # Lock when confident
    if detection_confidence >= 1.0:
        locked_roi = current_box
        save_roi(current_box)
        add_log("TV ROI locked")

def calculate_average_color(frame, roi_box):
    """Calculate average color of the ROI"""
    if roi_box is None or len(roi_box) != 4:
        return None, None
    
    x1, y1, x2, y2 = roi_box
    roi = frame[y1:y2, x1:x2]
    
    if roi.size > 0:
        # Sample pixels for faster calculation
        if roi.size > 10000:
            roi = roi[::roi.shape[0]//100, ::roi.shape[1]//100]
        
        avg_bgr = np.mean(roi, axis=(0, 1))
        avg_rgb = avg_bgr[::-1]
        return avg_bgr, avg_rgb
    
    return None, None

def reset_detection():
    """Reset the TV detection system"""
    global tv_box, detection_confidence, locked_roi, last_processed_roi_frame, last_roi_check_time
    tv_box = None
    detection_confidence = 0.0
    locked_roi = None
    last_processed_roi_frame = None
    last_roi_check_time = 0
    
    try:
        if os.path.exists(ROI_SAVE_PATH):
            os.remove(ROI_SAVE_PATH)
    except Exception as e:
        add_log(f"Error removing ROI file: {e}")
    
    add_log("TV ROI detection reset")

def draw_info(frame, avg_bgr, avg_rgb, roi_contour=None):
    """Draw information overlay on the frame"""
    display_frame = frame.copy()
    
    # Scale down by 50% for display
    height, width = display_frame.shape[:2]
    display_frame = cv2.resize(display_frame, (width//2, height//2))
    
    if locked_roi:
        # Scale down the locked ROI coordinates
        x1, y1, x2, y2 = [coord//2 for coord in locked_roi]
        
        # Create dark mask for locked ROI
        mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        overlay = display_frame.copy()
        overlay[mask == 0] = overlay[mask == 0] * 0.3
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Draw locked ROI info
        cv2.putText(display_frame, "ROI LOCKED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif tv_box:
        # Scale down the TV box coordinates
        x1, y1, x2, y2 = [coord//2 for coord in tv_box]
        
        # Draw TV detection info
        cv2.putText(display_frame, f"DETECTING {detection_confidence:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Progress bar
        bar_width = 200
        bar_height = 15
        cv2.rectangle(display_frame, (10, 40), (10 + bar_width, 40 + bar_height), (50, 50, 50), -1)
        conf_width = int(bar_width * detection_confidence)
        cv2.rectangle(display_frame, (10, 40), (10 + conf_width, 40 + bar_height), (0, 255, 255), -1)
    
    # Draw average color info if available
    if avg_rgb is not None:
        color_text = f"RGB: ({int(avg_rgb[0])}, {int(avg_rgb[1])}, {int(avg_rgb[2])})"
        cv2.putText(display_frame, color_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw color sample
        color_sample = np.zeros((30, 100, 3), dtype=np.uint8)
        color_sample[:] = avg_bgr
        display_frame[80:110, 10:110] = color_sample
    
    return display_frame
