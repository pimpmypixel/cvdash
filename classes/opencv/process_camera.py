import cv2
import numpy as np
import queue
import time
import os
import json
from classes.utils.utils import add_log
from queue import Queue
import config.config as c

# Global variables for TV detection
detection_confidence = 0.0
last_detected_roi = None
start_time = time.time()
locked_roi = None  # Store the locked ROI once confidence is 1.0
ROI_SAVE_PATH = 'storage/tv_roi.json'

# Global variables for tracking TV ROI and logo detection
tv_roi_locked = False
logo_detected = False

def save_roi(roi):
    """Save ROI coordinates to JSON file"""
    if roi is not None and len(roi) == 4:
        try:
            os.makedirs('storage', exist_ok=True)
            # Convert numpy array to list of lists
            roi_list = roi.tolist()
            with open(ROI_SAVE_PATH, 'w') as f:
                json.dump(roi_list, f)
            add_log("TV ROI saved")
        except Exception as e:
            add_log(f"Error saving ROI: {e}")

def load_saved_roi():
    """Load saved ROI from JSON file"""
    global locked_roi, detection_confidence
    try:
        if os.path.exists(ROI_SAVE_PATH):
            with open(ROI_SAVE_PATH, 'r') as f:
                roi_list = json.load(f)
                locked_roi = np.array(roi_list, dtype=np.int32)
                detection_confidence = 1.0
                add_log("Loaded saved TV ROI")
                return True
    except Exception as e:
        add_log(f"Error loading ROI: {e}")
    return False

def process_webcam_frame(frame, webcam_avg_colors_q: Queue):
    global detection_confidence, last_detected_roi, locked_roi, tv_roi_locked, logo_detected
    
    if frame is None:
        h = c.window_height
        w = c.window_width
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Try to load saved ROI if we don't have one
    if locked_roi is None:
        load_saved_roi()
    
    # Use locked ROI if available, otherwise detect
    roi_contour = locked_roi if locked_roi is not None else detect_tv_rectangle(frame)
    
    # Update detection confidence
    update_detection_confidence(roi_contour)
    
    # Calculate average color if we're recording
    avg_bgr, avg_rgb = None, None
    if roi_contour is not None:
        avg_bgr, avg_rgb = calculate_average_color(frame, roi_contour)
        
        # Store color data with timestamp
        if avg_rgb is not None:
            # Remove old item if queue is full
            if webcam_avg_colors_q.full():
                try:
                    webcam_avg_colors_q.get_nowait()
                except queue.Empty:
                    pass
            webcam_avg_colors_q.put(avg_rgb)
    
    # Draw information overlay and get display frame
    display_frame = draw_info(frame, avg_bgr, avg_rgb, roi_contour)
    
    # Update TV ROI detection status
    tv_roi_locked = True  # Set to True when ROI is detected
    
    # Update logo detection status
    logo_detected = True  # Set to True when logo is detected
    
    return display_frame

def detect_tv_rectangle(frame):
    """Detect TV/rectangle using edge detection, contour finding, and 16:9 aspect ratio validation"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection with multiple thresholds for better detection
    edges1 = cv2.Canny(blurred, 30, 100)
    edges2 = cv2.Canny(blurred, 50, 150)
    edges3 = cv2.Canny(blurred, 80, 200)
    edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    
    # Morphological operations to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the best TV-like rectangular contour
    best_score = 0
    best_contour = None
    best_corrected_corners = None
    
    frame_center = np.array([frame.shape[1]/2, frame.shape[0]/2])
    
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 15000:  # Minimum area threshold for TV
                # Calculate the score for this contour
                score, corrected_corners = evaluate_tv_candidate(approx, area, frame_center)
                if score > best_score:
                    best_score = score
                    best_contour = approx
                    best_corrected_corners = corrected_corners
    
    return best_corrected_corners if best_corrected_corners is not None else best_contour

def evaluate_tv_candidate(corners, area, frame_center):
    """Evaluate how well a 4-corner contour matches a 16:9 TV with perspective correction"""
    if len(corners) != 4:
        return 0, None
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners.reshape(4, 2))
    
    # Calculate center of the contour
    contour_center = np.mean(corners, axis=0)
    
    # Calculate distance from frame center (normalized by frame size)
    center_distance = np.linalg.norm(contour_center - frame_center) / np.linalg.norm(frame_center)
    center_score = max(0, 1 - center_distance)  # Higher score for more centered rectangles
    
    # Calculate current dimensions in image space
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    
    if avg_height == 0:
        return 0, None
    
    # Calculate aspect ratio
    aspect_ratio = avg_width / avg_height
    
    # Score based on how close to 16:9 (1.778) the aspect ratio is
    target_ratio = 16.0 / 9.0
    ratio_score = max(0, 1 - abs(aspect_ratio - target_ratio) / target_ratio)
    
    # Score based on area (larger is generally better for TV detection)
    area_score = min(1.0, area / 100000)  # Normalize to reasonable TV size
    
    # Score based on rectangle regularity (parallel sides, right angles)
    regularity_score = calculate_rectangle_regularity(corners)
    
    # Perspective correction score (less skew is better)
    perspective_score = calculate_perspective_score(corners)
    
    # Combined score with weights
    total_score = (ratio_score * 0.3 + 
                    area_score * 0.2 + 
                    regularity_score * 0.2 + 
                    perspective_score * 0.2 +
                    center_score * 0.1)  # Added center score
    
    # Apply perspective correction to get ideal 16:9 rectangle
    corrected_corners = correct_perspective_to_16_9(corners, avg_width, avg_height)
    
    return total_score, corrected_corners

def order_corners(pts):
    """Order corners as: top-left, top-right, bottom-right, bottom-left"""
    # Sort by y-coordinate
    sorted_pts = pts[np.argsort(pts[:, 1])]
    
    # Top two points
    top_pts = sorted_pts[:2]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
    
    # Bottom two points
    bottom_pts = sorted_pts[2:]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
    
    return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)

def calculate_rectangle_regularity(corners):
    """Calculate how regular/rectangular the shape is"""
    # Calculate side lengths
    sides = []
    for i in range(4):
        side_length = np.linalg.norm(corners[(i+1)%4] - corners[i])
        sides.append(side_length)
    
    # Check if opposite sides are similar (parallel sides should be equal)
    opposite_side_diff1 = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
    opposite_side_diff2 = abs(sides[1] - sides[3]) / max(sides[1], sides[3])
    
    # Calculate angles (should be close to 90 degrees)
    angles = []
    for i in range(4):
        v1 = corners[i] - corners[(i-1)%4]
        v2 = corners[(i+1)%4] - corners[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(abs(90 - angle))
    
    # Score based on how close sides and angles are to perfect rectangle
    side_regularity = 1 - (opposite_side_diff1 + opposite_side_diff2) / 2
    angle_regularity = 1 - np.mean(angles) / 90
    
    return (side_regularity + angle_regularity) / 2

def calculate_perspective_score(corners):
    """Calculate perspective distortion score (less distortion = higher score)"""
    # Calculate the degree of perspective distortion
    # In a non-distorted rectangle, parallel lines should remain parallel
    
    # Top and bottom edges
    top_edge = corners[1] - corners[0]
    bottom_edge = corners[2] - corners[3]
    
    # Left and right edges  
    left_edge = corners[3] - corners[0]
    right_edge = corners[2] - corners[1]
    
    # Calculate parallelism (dot product of normalized vectors)
    top_norm = top_edge / np.linalg.norm(top_edge)
    bottom_norm = bottom_edge / np.linalg.norm(bottom_edge)
    left_norm = left_edge / np.linalg.norm(left_edge)  
    right_norm = right_edge / np.linalg.norm(right_edge)
    
    horizontal_parallel = abs(np.dot(top_norm, bottom_norm))
    vertical_parallel = abs(np.dot(left_norm, right_norm))
    
    return (horizontal_parallel + vertical_parallel) / 2

def correct_perspective_to_16_9(corners, width, height):
    """Apply perspective correction to create ideal 16:9 rectangle"""
    # Calculate center point
    center_x = np.mean(corners[:, 0])
    center_y = np.mean(corners[:, 1])
    
    # Determine the corrected dimensions maintaining 16:9 ratio
    if width / height > 16.0 / 9.0:
        # Width is limiting factor
        corrected_width = width
        corrected_height = width * 9.0 / 16.0
    else:
        # Height is limiting factor
        corrected_height = height
        corrected_width = height * 16.0 / 9.0
    
    # Create ideal rectangle centered at the detected center
    half_width = corrected_width / 2
    half_height = corrected_height / 2
    
    ideal_corners = np.array([
        [center_x - half_width, center_y - half_height],  # top-left
        [center_x + half_width, center_y - half_height],  # top-right
        [center_x + half_width, center_y + half_height],  # bottom-right
        [center_x - half_width, center_y + half_height]   # bottom-left
    ], dtype=np.float32)
    
    # Apply weighted blend between detected corners and ideal rectangle
    # This preserves the general position while correcting aspect ratio
    blend_factor = 0.3  # How much to blend towards ideal rectangle
    corrected_corners = (1 - blend_factor) * corners + blend_factor * ideal_corners
    
    return corrected_corners.astype(np.int32)

def calculate_average_color(frame, roi_contour=None):
    """Calculate average color in ROI with perspective correction"""
    if roi_contour is not None:
        # Use detected TV rectangle with perspective correction
        if len(roi_contour) == 4:
            # Apply perspective transformation to get undistorted view
            corrected_frame = apply_perspective_correction(frame, roi_contour)
            if corrected_frame is not None:
                # Sample from the perspective-corrected frame
                roi_pixels = corrected_frame.reshape(-1, 3)
                # Filter out black pixels that might be from transformation padding
                roi_pixels = roi_pixels[np.sum(roi_pixels, axis=1) > 30]
            else:
                # Fallback to simple mask
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [roi_contour], 255)
                roi_pixels = frame[mask > 0]
        else:
            # Fallback for non-4-corner contours
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_contour], 255)
            roi_pixels = frame[mask > 0]
    else:
        return None, None
    
    if len(roi_pixels) > 0:
        avg_bgr = np.mean(roi_pixels, axis=0)
        avg_rgb = avg_bgr[::-1]  # Convert BGR to RGB
        return avg_bgr, avg_rgb
    
    return None, None

def apply_perspective_correction(frame, corners):
    """Apply perspective transformation to correct skewed TV view"""
    try:
        # Ensure corners are properly ordered
        if len(corners) == 4:
            corners = order_corners(corners.reshape(4, 2))
            
            # Calculate dimensions of the corrected rectangle
            top_width = np.linalg.norm(corners[1] - corners[0])
            bottom_width = np.linalg.norm(corners[2] - corners[3])  
            left_height = np.linalg.norm(corners[3] - corners[0])
            right_height = np.linalg.norm(corners[2] - corners[1])
            
            max_width = int(max(top_width, bottom_width))
            max_height = int(max(left_height, right_height))
            
            # Ensure 16:9 aspect ratio for output
            if max_width / max_height > 16.0 / 9.0:
                output_width = max_width
                output_height = int(max_width * 9.0 / 16.0)
            else:
                output_height = max_height
                output_width = int(max_height * 16.0 / 9.0)
            
            # Define destination points for perspective correction
            dst_points = np.array([
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1], 
                [0, output_height - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transformation matrix
            src_points = corners.astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transformation
            corrected = cv2.warpPerspective(frame, matrix, (output_width, output_height))
            
            return corrected
            
    except Exception as e:
        print(f"Perspective correction failed: {e}")
        
    return None

def update_detection_confidence(current_roi):
    """Update detection confidence based on stability of ROI detection"""
    global detection_confidence, last_detected_roi, locked_roi
    
    # If we have a locked ROI, don't update confidence
    if locked_roi is not None:
        return
        
    if current_roi is None:
        detection_confidence = max(0, detection_confidence - 0.1)
        return
    
    if last_detected_roi is None:
        last_detected_roi = current_roi
        detection_confidence = 0.3  # Start with higher initial confidence
        return
    
    # Calculate similarity between current and last ROI
    current_center = np.mean(current_roi, axis=0)
    last_center = np.mean(last_detected_roi, axis=0)
    
    # Calculate center distance (normalized)
    center_distance = np.linalg.norm(current_center - last_center) / np.linalg.norm(current_center)
    
    # Calculate area similarity
    current_area = cv2.contourArea(current_roi)
    last_area = cv2.contourArea(last_detected_roi)
    area_ratio = min(current_area, last_area) / max(current_area, last_area)
    
    # Update confidence based on stability with faster increase
    stability_score = (1 - center_distance) * area_ratio
    detection_confidence = min(1.0, detection_confidence + stability_score * 0.2)
    
    last_detected_roi = current_roi
    
    # Lock ROI when confidence reaches 1.0
    if detection_confidence >= 1.0:
        locked_roi = current_roi
        add_log("TV ROI lock")
        # Save ROI to JSON file
        save_roi(current_roi)

def reset_detection():
    """Reset the detection system"""
    global detection_confidence, last_detected_roi, locked_roi, start_time, tv_roi_locked, logo_detected
    detection_confidence = 0.0
    last_detected_roi = None
    locked_roi = None
    start_time = time.time()
    # Remove saved ROI file if it exists
    try:
        if os.path.exists(ROI_SAVE_PATH):
            os.remove(ROI_SAVE_PATH)
    except Exception as e:
        add_log(f"Error removing ROI file: {e}")
    add_log("Webcam TV ROI reset")
    tv_roi_locked = False
    logo_detected = False

def draw_info(frame, avg_bgr, avg_rgb, roi_contour=None):
    """Draw information overlay on frame"""
    # Create a copy of the frame for drawing
    display_frame = frame.copy()
    
    # Scale down the frame by 50%
    height, width = display_frame.shape[:2]
    display_frame = cv2.resize(display_frame, (width//2, height//2))
    
    # Scale down ROI coordinates if present
    if roi_contour is not None:
        roi_contour = roi_contour // 2
    
    # Draw detected TV rectangle and apply dark mask outside ROI
    if roi_contour is not None:
        # Only apply dark mask if we have full confidence
        if detection_confidence >= 1.0:
            # Create mask for the ROI
            mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_contour], 255)
            
            # Create dark overlay
            overlay = display_frame.copy()
            overlay[mask == 0] = overlay[mask == 0] * 0.3  # Darken non-ROI area
            
            # Blend the overlay with the original frame
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Draw ROI contour
        cv2.drawContours(display_frame, [roi_contour], -1, (0, 255, 255), 1)
    
    return display_frame

def is_tv_roi_locked():
    """Return whether the TV ROI is currently locked."""
    global tv_roi_locked
    return tv_roi_locked

def is_logo_detected():
    """Return whether a logo is currently detected."""
    global logo_detected
    return logo_detected