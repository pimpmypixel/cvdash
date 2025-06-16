import cv2
import numpy as np
import time
import os
from datetime import datetime
from classes.utils.logger import add_log
import config.config as c
from queue import Queue
import queue

roi_x=30
roi_y=10
roi_width=25
roi_height=25

render_frames_since_state_change = 0
last_detection_state = None
consecutive_detections = 0
consecutive_no_detections = 0
confidence = c.min_confidence  # Initialize with minimum confidence
frames_since_state_change = 0
continuous_update_interval = 10  # Publish continuous updates every N frames

def process_browser_frame(frame, stream_avg_colors_q: Queue):
    if len(frame.shape) == 3:
        r, g, b = average_color(frame)
        roi = make_roi(frame)
        logo_confidence, color_mask, colored_pixels = detect_red_logo(roi)
        logo_detected, state_changed, render_frames_since_state_change, should_publish = publish_change(logo_confidence, colored_pixels)
        
        q = [r,g,b,None,None]
        if state_changed and should_publish:
            status = "TV" if logo_detected else "ADS"
            add_log(f"{status} conf: {logo_confidence:.2f}")
            q[3] = status
            q[4] = time.time()
        
        if stream_avg_colors_q.full():
            try:
                stream_avg_colors_q.get_nowait()
                add_log("Removed old item from queue.")
            except queue.Empty:
                add_log("Attempted to remove from empty queue (shouldn't happen if full()).")
                pass
        stream_avg_colors_q.put(q)
        return render_image(frame, logo_detected)
    
    return np.zeros((c.window_height, c.window_width, 3), dtype=np.uint8)

def render_image(frame, logo_detected):
        # Create a copy of the full screenshot for visualization
        display_img = frame.copy()
        
        # Draw ROI rectangle on display image
        color = (0, 255, 0) if logo_detected else (0, 0, 255)
        cv2.rectangle(display_img, 
                    (roi_x, roi_y),
                    (roi_x + roi_width, roi_y + roi_height),
                    color, 1)
        return display_img

def average_color(img_np):
    avg_bgr = np.mean(img_np, axis=(0, 1))  # Average across both height and width
    avg_rgb = avg_bgr[::-1]  # Convert BGR to RGB
    return int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])

def make_roi(img_np):
    # Extract ROI directly from the full image
    return img_np[roi_y:roi_y+roi_height, 
                    roi_x:roi_x+roi_width]

def publish_change(logo_confidence, colored_pixels): 
    global last_detection_state
    global consecutive_detections
    global consecutive_no_detections
    global confidence
    global render_frames_since_state_change
    global frames_since_state_change
    global continuous_update_interval
    
    # Update logo detection state based on confidence threshold
    logo_detected = logo_confidence > c.detection_threshold
    
    # Calculate confidence adjustment based on detection quality
    confidence_factor = min(1.0, logo_confidence * 1.2)  # Slightly amplify high confidence detections
    
    # Update confidence with smoother transitions
    if logo_detected:
        # When logo is detected, increase confidence more if detection is strong
        confidence = min(c.max_confidence, 
                        confidence + (c.confidence_increment * confidence_factor))
    else:
        # When no logo is detected, decrease confidence more if no-logo confidence is high
        confidence = max(c.min_confidence, 
                        confidence - (c.confidence_decrement * confidence_factor))
    
    # Track state changes
    state_changed = False
    if last_detection_state is None:
        # First detection
        last_detection_state = logo_detected
        state_changed = True
        frames_since_state_change = 0
    elif last_detection_state != logo_detected:
        # State changed
        state_changed = True
        last_detection_state = logo_detected
        frames_since_state_change = 0
    else:
        frames_since_state_change += 1
        render_frames_since_state_change += 1
    
    # Update consecutive detection counters
    if logo_detected:
        consecutive_detections += 1
        consecutive_no_detections = 0
    else:
        consecutive_no_detections += 1
        consecutive_detections = 0
    
    # Determine if we should publish based on stability requirements
    should_publish = False
    if logo_detected and consecutive_detections >= 2:
        should_publish = True
    elif not logo_detected and consecutive_no_detections >= 2:
        should_publish = True
    
    # Return state information for rendering
    if should_publish and (state_changed or frames_since_state_change % continuous_update_interval == 0):
        return logo_detected, state_changed, render_frames_since_state_change, True
    
    # If we shouldn't publish yet, return current state with should_publish=False
    return logo_detected, state_changed, render_frames_since_state_change, False

def detect_red_logo(roi):
        """
        Detect red/blue circular logo using color-based detection
        Returns detection confidence (0.0 to 1.0)
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create masks for both blue and red color ranges
        blue_mask = cv2.inRange(hsv, c.lower_blue, c.upper_blue)
        red_mask1 = cv2.inRange(hsv, c.lower_red1, c.upper_red1)
        red_mask2 = cv2.inRange(hsv, c.lower_red2, c.upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine blue and red masks
        combined_mask = cv2.bitwise_or(blue_mask, red_mask)
        
        # Count colored pixels
        colored_pixel_count = cv2.countNonZero(combined_mask)
        
        # Calculate base confidence from pixel count
        max_possible_pixels = roi.shape[0] * roi.shape[1]
        pixel_confidence = min(1.0, colored_pixel_count / (c.min_red_pixels * 2))
        
        # Enhanced confidence calculation for better balance
        if colored_pixel_count < c.min_red_pixels:
            # Calculate no-logo confidence based on pixel count and distribution
            # Lower pixel count = higher confidence of no logo
            no_logo_confidence = 1.0 - min(1.0, colored_pixel_count / c.min_red_pixels)
            
            # Add shape analysis for no-logo case
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shape_confidence = 0.0
            
            if contours:
                # Check if any contours are too small or irregular to be the logo
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < c.min_contour_area:
                        shape_confidence += 0.2  # Small contours increase no-logo confidence
                    else:
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity < 0.5:  # Less circular = more likely not logo
                                shape_confidence += 0.3
            
            # Combine pixel and shape confidence for no-logo case
            final_confidence = (no_logo_confidence * 0.7) + (min(1.0, shape_confidence) * 0.3)
            return final_confidence, combined_mask, colored_pixel_count
        
        # Find contours in the mask to check for circular shapes
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return pixel_confidence * 0.5, combined_mask, colored_pixel_count
        
        # Check for circular contours and calculate shape confidence
        max_shape_confidence = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < c.min_contour_area:
                continue
            
            # Calculate circularity (4*pi*area/perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Convert circularity to confidence (0.5 -> 0.5, 1.0 -> 1.0)
            shape_confidence = min(1.0, max(0.0, (circularity - 0.3) / 0.7))
            max_shape_confidence = max(max_shape_confidence, shape_confidence)
        
        # Combine pixel and shape confidence
        final_confidence = (pixel_confidence * 0.4) + (max_shape_confidence * 0.6)
        return final_confidence, combined_mask, colored_pixel_count