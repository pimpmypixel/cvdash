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
last_processed_roi_frame = None # New global variable to store the last ROI frame

# Additional globals for improved detection
frame_history = []  # Store recent frames for temporal consistency
detection_history = []  # Store recent detection results

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
    global detection_confidence, last_detected_roi, locked_roi, tv_roi_locked, logo_detected, last_processed_roi_frame
    
    if frame is None:
        return frame
        h = c.window_height
        w = c.window_width
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Try to load saved ROI if we don't have one
    if locked_roi is None:
        load_saved_roi()
    
    # Use locked ROI if available, otherwise detect
    roi_contour = locked_roi if locked_roi is not None else detect_tv_rectangle(frame)
    
    # Initialize content_change_score
    content_change_score = 0.0

    # Calculate average color and content change if we have an ROI
    avg_bgr, avg_rgb = None, None
    if roi_contour is not None:
        corrected_roi_frame = apply_perspective_correction(frame, roi_contour)
        if corrected_roi_frame is not None:
            # Calculate content change score if previous frame exists
            if last_processed_roi_frame is not None and corrected_roi_frame.shape == last_processed_roi_frame.shape:
                content_change_score = calculate_content_change(corrected_roi_frame, last_processed_roi_frame)
            
            # Update last processed ROI frame
            last_processed_roi_frame = corrected_roi_frame.copy()

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

    # Update detection confidence (pass content_change_score here)
    update_detection_confidence(roi_contour, content_change_score)
    
    # Draw information overlay and get display frame
    display_frame = draw_info(frame, avg_bgr, avg_rgb, roi_contour)
    
    # Update TV ROI detection status based on confidence
    tv_roi_locked = detection_confidence >= 1.0
    
    # Update logo detection status
    logo_detected = True  # Set to True when logo is detected
    
    return display_frame

def detect_tv_rectangle(frame):
    """Enhanced TV/rectangle detection with improved edge detection and temporal consistency"""
    global frame_history, detection_history
    
    # Store frame history for temporal analysis
    frame_history.append(frame.copy())
    if len(frame_history) > 5:  # Keep last 5 frames
        frame_history.pop(0)
    
    # Multi-scale and multi-method detection
    candidates = []
    
    # Method 1: Enhanced edge detection
    edge_candidates = detect_tv_by_edges(frame)
    candidates.extend(edge_candidates)
    
    # Method 2: Template-based detection for TV screens
    template_candidates = detect_tv_by_template(frame)
    candidates.extend(template_candidates)
    
    # Method 3: Adaptive brightness detection
    brightness_candidates = detect_tv_by_brightness(frame)
    candidates.extend(brightness_candidates)
    
    # Combine and evaluate all candidates
    best_candidate = evaluate_and_select_best_candidate(candidates, frame)
    
    # Apply temporal consistency check
    if len(detection_history) > 0:
        best_candidate = apply_temporal_consistency(best_candidate, detection_history)
    
    # Store detection result
    detection_history.append(best_candidate)
    if len(detection_history) > 10:  # Keep last 10 detections
        detection_history.pop(0)
    
    return best_candidate

def detect_tv_by_edges(frame):
    """Enhanced edge-based TV detection with multiple preprocessing methods"""
    candidates = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Standard Canny with morphology
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges1 = cv2.Canny(blurred, 30, 150)
    edges2 = cv2.Canny(blurred, 50, 200)
    edges3 = cv2.Canny(blurred, 80, 240)
    edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    
    # Enhanced morphological operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)
    
    candidates.extend(find_rectangular_contours(edges, frame, "edge_standard"))
    
    # Method 2: Adaptive threshold + edge detection
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges_adaptive = cv2.Canny(adaptive_thresh, 50, 150)
    edges_adaptive = cv2.morphologyEx(edges_adaptive, cv2.MORPH_CLOSE, kernel_close)
    
    candidates.extend(find_rectangular_contours(edges_adaptive, frame, "edge_adaptive"))
    
    # Method 3: Gradient-based edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude / magnitude.max() * 255)
    _, edges_gradient = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    edges_gradient = cv2.morphologyEx(edges_gradient, cv2.MORPH_CLOSE, kernel_close)
    
    candidates.extend(find_rectangular_contours(edges_gradient, frame, "edge_gradient"))
    
    return candidates

def detect_tv_by_template(frame):
    """Detect TV screens using brightness and contrast patterns typical of displays"""
    candidates = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Look for rectangular regions with high variance (typical of TV content)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Calculate local variance using different kernel sizes
    for kernel_size in [15, 25, 35]:
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean = cv2.filter2D(blurred.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((blurred.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # Normalize variance
        variance = np.uint8(variance / variance.max() * 255)
        
        # Threshold to find high-variance regions
        _, variance_thresh = cv2.threshold(variance, 100, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        variance_thresh = cv2.morphologyEx(variance_thresh, cv2.MORPH_CLOSE, kernel_morph)
        variance_thresh = cv2.morphologyEx(variance_thresh, cv2.MORPH_OPEN, kernel_morph)
        
        candidates.extend(find_rectangular_contours(variance_thresh, frame, f"template_{kernel_size}"))
    
    return candidates

def detect_tv_by_brightness(frame):
    """Detect TV screens by analyzing brightness patterns and content"""
    candidates = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Look for regions with moderate to high brightness (active screens)
    _, bright_thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    
    # Method 2: Look for regions with significant brightness variation
    blur_heavy = cv2.GaussianBlur(gray, (21, 21), 0)
    brightness_diff = cv2.absdiff(gray, blur_heavy)
    _, diff_thresh = cv2.threshold(brightness_diff, 20, 255, cv2.THRESH_BINARY)
    
    # Combine both methods
    combined = cv2.bitwise_or(bright_thresh, diff_thresh)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    candidates.extend(find_rectangular_contours(combined, frame, "brightness"))
    
    return candidates

def find_rectangular_contours(binary_image, frame, method_name):
    """Find rectangular contours in binary image and evaluate them"""
    candidates = []
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_center = np.array([frame.shape[1]/2, frame.shape[0]/2])
    
    for contour in contours:
        # Filter by area early
        area = cv2.contourArea(contour)
        if area < 10000:  # Minimum area for TV
            continue
            
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for 4-sided shapes or close to it
        if len(approx) >= 4:
            if len(approx) > 4:
                # If more than 4 points, try to find the best 4 corners
                approx = find_best_4_corners(approx)
            
            if len(approx) == 4:
                score, corrected_corners = evaluate_tv_candidate(approx, area, frame_center, frame)
                if score > 0.3:  # Minimum score threshold
                    candidates.append({
                        'corners': corrected_corners if corrected_corners is not None else approx,
                        'score': score,
                        'method': method_name,
                        'area': area
                    })
    
    return candidates

def find_best_4_corners(approx):
    """Extract the best 4 corners from a polygon with more than 4 points"""
    if len(approx) <= 4:
        return approx
    
    # Convert to simple array
    points = approx.reshape(-1, 2)
    
    # Find convex hull
    hull = cv2.convexHull(points)
    
    # If hull has 4 or fewer points, use it
    if len(hull) <= 4:
        return hull
    
    # Otherwise, find the 4 corners that form the largest quadrilateral
    max_area = 0
    best_quad = None
    
    # Try different combinations of 4 points
    from itertools import combinations
    for quad_points in combinations(hull.reshape(-1, 2), 4):
        quad = np.array(quad_points, dtype=np.int32).reshape(-1, 1, 2)
        area = cv2.contourArea(quad)
        if area > max_area:
            max_area = area
            best_quad = quad
    
    return best_quad if best_quad is not None else approx[:4]

def evaluate_and_select_best_candidate(candidates, frame):
    """Evaluate all candidates and select the best one"""
    if not candidates:
        return None
    
    # Score each candidate with additional TV-specific criteria
    for candidate in candidates:
        # Add content analysis score
        content_score = analyze_tv_content(frame, candidate['corners'])
        candidate['content_score'] = content_score
        
        # Add temporal consistency bonus if we have history
        if detection_history:
            consistency_score = calculate_consistency_score(candidate['corners'], detection_history)
            candidate['consistency_score'] = consistency_score
        else:
            candidate['consistency_score'] = 0.0
        
        # Calculate final weighted score
        candidate['final_score'] = (
            candidate['score'] * 0.4 +           # Base geometric score
            content_score * 0.3 +                # Content analysis
            candidate['consistency_score'] * 0.2 + # Temporal consistency
            min(1.0, candidate['area'] / 50000) * 0.1  # Size preference
        )
    
    # Select candidate with highest final score
    best_candidate = max(candidates, key=lambda x: x['final_score'])
    
    # Only return if score is above threshold
    if best_candidate['final_score'] > 0.4:
        return best_candidate['corners']
    
    return None

def analyze_tv_content(frame, corners):
    """Analyze the content inside the detected region to confirm it's a TV"""
    if corners is None or len(corners) != 4:
        return 0.0
    
    try:
        # Get the region inside the corners
        corrected_frame = apply_perspective_correction(frame, corners)
        if corrected_frame is None:
            return 0.0
        
        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
        
        # Check for content characteristics typical of TV screens
        
        # 1. Brightness distribution (TVs usually have varied brightness)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()
        
        # Calculate entropy (measure of brightness variation)
        entropy = -np.sum(hist_normalized[hist_normalized > 0] * np.log2(hist_normalized[hist_normalized > 0]))
        entropy_score = min(1.0, entropy / 6.0)  # Normalize entropy
        
        # 2. Edge density (TVs usually have varied content with edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        edge_score = min(1.0, edge_density * 20)  # Scale edge density
        
        # 3. Variance (active TV content has high variance)
        variance = np.var(gray)
        variance_score = min(1.0, variance / 2000)  # Normalize variance
        
        # 4. Mean brightness (TVs are usually moderately bright)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128  # Prefer moderate brightness
        
        # Combine scores
        content_score = (entropy_score * 0.3 + edge_score * 0.3 + 
                        variance_score * 0.2 + brightness_score * 0.2)
        
        return content_score
        
    except Exception as e:
        return 0.0

def calculate_consistency_score(current_corners, history):
    """Calculate how consistent current detection is with recent history"""
    if not history or current_corners is None:
        return 0.0
    
    consistency_scores = []
    
    # Check consistency with recent detections
    for past_detection in history[-5:]:  # Check last 5 detections
        if past_detection is None:
            continue
            
        # Calculate center similarity
        current_center = np.mean(current_corners.reshape(-1, 2), axis=0)
        past_center = np.mean(past_detection.reshape(-1, 2), axis=0)
        
        center_distance = np.linalg.norm(current_center - past_center)
        max_distance = np.linalg.norm([50, 50])  # Allow 50 pixel movement
        center_score = max(0, 1 - center_distance / max_distance)
        
        # Calculate area similarity
        current_area = cv2.contourArea(current_corners)
        past_area = cv2.contourArea(past_detection)
        area_ratio = min(current_area, past_area) / max(current_area, past_area)
        
        consistency_scores.append(center_score * 0.6 + area_ratio * 0.4)
    
    return np.mean(consistency_scores) if consistency_scores else 0.0

def apply_temporal_consistency(current_detection, history):
    """Apply temporal smoothing to reduce detection jitter"""
    if current_detection is None or not history:
        return current_detection
    
    # If we have a good detection history, smooth the corners
    recent_detections = [d for d in history[-3:] if d is not None]
    
    if len(recent_detections) >= 2:
        # Calculate average position of recent detections
        all_corners = np.array([d.reshape(-1, 2) for d in recent_detections])
        avg_corners = np.mean(all_corners, axis=0)
        
        # Blend current detection with average (smoothing)
        current_corners = current_detection.reshape(-1, 2)
        smoothed_corners = 0.7 * current_corners + 0.3 * avg_corners
        
        return smoothed_corners.reshape(current_detection.shape).astype(np.int32)
    
    return current_detection

def calculate_content_change(current_roi_frame, last_roi_frame):
    """Calculate the difference between current and last ROI frames"""
    # Convert to grayscale for difference calculation
    gray_current = cv2.cvtColor(current_roi_frame, cv2.COLOR_BGR2GRAY) if len(current_roi_frame.shape) == 3 else current_roi_frame
    gray_last = cv2.cvtColor(last_roi_frame, cv2.COLOR_BGR2GRAY) if len(last_roi_frame.shape) == 3 else last_roi_frame

    # Calculate absolute difference
    diff = cv2.absdiff(gray_current, gray_last)
    
    # Threshold the difference image to get areas of significant change
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Calculate the percentage of changed pixels
    change_percentage = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])
    
    # Normalize to a score between 0 and 1
    return min(1.0, change_percentage * 2) # Multiply to make smaller changes more impactful

def check_black_boundary(frame, corners):
    """Enhanced black boundary detection with better edge analysis"""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Create mask for the rectangle
    mask = np.zeros(gray.shape, dtype=np.uint8)
    corners_int = corners.astype(np.int32)
    corners_reshaped = corners_int.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [corners_reshaped], 255)
    
    # Analyze boundary at multiple widths
    boundary_scores = []
    
    for width in [3, 5, 7]:
        # Create boundary mask
        kernel = np.ones((width, width), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        boundary_mask = cv2.subtract(dilated_mask, mask)
        
        # Get boundary pixels
        boundary_pixels = gray[boundary_mask > 0]
        
        if len(boundary_pixels) > 0:
            # Multiple criteria for TV bezels
            dark_threshold = 60  # Adjusted for TV bezels
            dark_percentage = np.sum(boundary_pixels < dark_threshold) / len(boundary_pixels)
            
            # Check for uniformity (TV bezels are usually uniform)
            boundary_std = np.std(boundary_pixels)
            uniformity_score = max(0, 1 - boundary_std / 64)
            
            # Check for low variance (solid bezels)
            boundary_var = np.var(boundary_pixels)
            low_variance_score = max(0, 1 - boundary_var / 400)
            
            # Combine criteria
            score = (dark_percentage * 0.4 + uniformity_score * 0.3 + low_variance_score * 0.3)
            boundary_scores.append(score)
    
    return np.mean(boundary_scores) if boundary_scores else 0.0

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
    """Enhanced rectangle regularity calculation"""
    # Calculate side lengths
    sides = []
    for i in range(4):
        side_length = np.linalg.norm(corners[(i+1)%4] - corners[i])
        sides.append(side_length)
    
    # Check if opposite sides are similar (with tolerance for perspective)
    opposite_side_diff1 = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
    opposite_side_diff2 = abs(sides[1] - sides[3]) / max(sides[1], sides[3])
    
    # Calculate angles (should be close to 90 degrees, with perspective tolerance)
    angles = []
    for i in range(4):
        v1 = corners[i] - corners[(i-1)%4]
        v2 = corners[(i+1)%4] - corners[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(abs(90 - angle))
    
    # More tolerant scoring for perspective-skewed rectangles
    side_regularity = 1 - min(1.0, (opposite_side_diff1 + opposite_side_diff2) / 2)
    angle_regularity = 1 - min(1.0, np.mean(angles) / 45)  # More tolerant angle threshold
    
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

def evaluate_tv_candidate(corners, area, frame_center, frame):
    """Enhanced TV candidate evaluation with better scoring"""
    if len(corners) != 4:
        return 0, None
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners.reshape(4, 2))
    
    # Calculate center of the contour
    contour_center = np.mean(corners, axis=0)
    
    # Calculate distance from frame center (normalized by frame size)
    center_distance = np.linalg.norm(contour_center - frame_center) / np.linalg.norm(frame_center)
    center_score = max(0, 1 - center_distance * 0.8)  # Less penalty for off-center
    
    # Calculate current dimensions in image space
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    
    if avg_height == 0:
        return 0, None
    
    # Calculate aspect ratio with more tolerance for perspective
    aspect_ratio = avg_width / avg_height
    
    # Score based on how close to 16:9 (1.778) the aspect ratio is
    target_ratio = 16.0 / 9.0
    ratio_diff = abs(aspect_ratio - target_ratio) / target_ratio
    ratio_score = max(0, 1 - ratio_diff * 1.2)  # More tolerant ratio scoring
    
    # Score based on area (prefer larger TVs but not too large)
    optimal_area = frame.shape[0] * frame.shape[1] * 0.3  # 30% of frame
    area_ratio = min(area, optimal_area) / optimal_area
    area_score = area_ratio * (2 - area_ratio)  # Peak at optimal_area
    
    # Score based on rectangle regularity
    regularity_score = calculate_rectangle_regularity(corners)
    
    # Score based on black boundary detection (enhanced)
    boundary_score = check_black_boundary(frame, corners)
    
    # Perspective correction score (less skew is better)
    perspective_score = calculate_perspective_score(corners)
    
    # Combined score with adjusted weights for TV detection
    total_score = (ratio_score * 0.25 +        # Aspect ratio is important for TVs
                  area_score * 0.15 +          # Size preference
                  regularity_score * 0.15 +   # Rectangle shape
                  perspective_score * 0.1 +   # Perspective distortion
                  center_score * 0.1 +        # Position preference
                  boundary_score * 0.25)      # Black bezel detection is key for TVs
    
    # Apply perspective correction to get ideal 16:9 rectangle
    corrected_corners = correct_perspective_to_16_9(corners, avg_width, avg_height)
    
    return total_score, corrected_corners

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

def update_detection_confidence(current_roi, content_change_score):
    """Enhanced detection confidence update with content analysis"""
    global detection_confidence, last_detected_roi, locked_roi
    
    # If we have a locked ROI, don't update confidence
    if locked_roi is not None:
        return
        
    if current_roi is None:
        detection_confidence = max(0, detection_confidence - 0.15)  # Faster decrease
        return
    
    if last_detected_roi is None:
        last_detected_roi = current_roi
        detection_confidence = 0.4  # Higher initial confidence for good detections
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
    
    # Calculate shape similarity using corner positions
    try:
        current_corners = current_roi.reshape(-1, 2)
        last_corners = last_detected_roi.reshape(-1, 2)
        corner_distances = [np.linalg.norm(c1 - c2) for c1, c2 in zip(current_corners, last_corners)]
        avg_corner_distance = np.mean(corner_distances)
        shape_similarity = max(0, 1 - avg_corner_distance / 50)  # Normalize by 50 pixels
    except:
        shape_similarity = area_ratio
    
    # Update confidence based on multiple factors
    stability_score = (1 - center_distance) * area_ratio * shape_similarity
    
    # Boost confidence for active TV content (changing content indicates active screen)
    content_boost = 0
    if content_change_score > 0.03:  # Lower threshold for content change
        content_boost = min(0.2, content_change_score * 0.5)  # Significant boost for changing content
    
    # Apply confidence update with different rates for increase/decrease
    if stability_score > 0.7:  # Good stability
        confidence_increase = stability_score * 0.25 + content_boost
        detection_confidence = min(1.0, detection_confidence + confidence_increase)
    else:  # Poor stability
        detection_confidence = max(0, detection_confidence - 0.1)
    
    last_detected_roi = current_roi
    
    # Lock ROI when confidence reaches 1.0
    if detection_confidence >= 1.0:
        locked_roi = current_roi
        add_log("TV ROI locked with enhanced detection")
        # Save ROI to JSON file
        save_roi(current_roi)

def reset_detection():
    """Reset the detection system"""
    global detection_confidence, last_detected_roi, locked_roi, start_time, tv_roi_locked, logo_detected, last_processed_roi_frame, frame_history, detection_history
    detection_confidence = 0.0
    last_detected_roi = None
    locked_roi = None
    start_time = time.time()
    last_processed_roi_frame = None
    frame_history = []  # Reset frame history
    detection_history = []  # Reset detection history
    
    # Remove saved ROI file if it exists
    try:
        if os.path.exists(ROI_SAVE_PATH):
            os.remove(ROI_SAVE_PATH)
    except Exception as e:
        add_log(f"Error removing ROI file: {e}")
    add_log("Enhanced webcam TV ROI reset")
    tv_roi_locked = False
    logo_detected = False

def draw_info(frame, avg_bgr, avg_rgb, roi_contour=None):
    """Draw information overlay on frame with enhanced visualization"""
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
        
        # Draw ROI contour with color based on detection status
        if detection_confidence >= 1.0:
            contour_color = (0, 255, 0)  # Green for locked
            thickness = 1
        elif detection_confidence >= 0.7:
            contour_color = (0, 255, 255)  # Yellow for high confidence
            thickness = 3
        else:
            contour_color = (0, 165, 255)  # Orange for detecting
            thickness = 3
            
        cv2.drawContours(display_frame, [roi_contour], -1, contour_color, thickness)
        
        # Only show confidence bar and status text when not locked
        if detection_confidence < 1.0:
            # Add confidence bar
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, 60
            
            # Background bar
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Confidence bar
            conf_width = int(bar_width * detection_confidence)
            bar_color = (0, 255, 0) if detection_confidence >= 1.0 else (0, 255, 255)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), bar_color, -1)
            
            # Add status text
            status_text = f"DETECTING {detection_confidence:.2f}"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, contour_color, 2)
            
            # Add confidence percentage
            cv2.putText(display_frame, f"{detection_confidence*100:.0f}%", (bar_x + bar_width + 10, bar_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display_frame

def is_tv_roi_locked():
    """Return whether the TV ROI is currently locked."""
    global tv_roi_locked
    return tv_roi_locked

def is_logo_detected():
    """Return whether the logo is currently detected."""
    global logo_detected
    return logo_detected