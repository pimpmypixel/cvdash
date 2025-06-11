import numpy as np

ratio_16_9 = .5625
ratio_webcam = .75

webcam_index=1

window_height = 300
window_width = int(window_height / ratio_16_9)   # 16:9
window_width_webcam = int(window_height / ratio_webcam)   # 16:9


# Balanced confidence tracking
confidence = 0.5  # Start at neutral confidence
confidence_increment = 0.08  # Slightly reduced for smoother transitions
confidence_decrement = 0.08  # Made equal to increment for balance
max_confidence = 1.0
min_confidence = 0.0

# Detection thresholds
min_red_pixels = 15  # Minimum red pixels to consider logo present
min_contour_area = 20  # Minimum contour area for circular shape
detection_threshold = 0.4  # Confidence threshold for detection

# Color ranges
lower_blue = np.array([100, 150, 50])    # Blue range
upper_blue = np.array([130, 255, 255])

lower_red1 = np.array([0, 120, 70])      # Red range (backup)
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])    
upper_red2 = np.array([180, 255, 255])


# last_detection_state = None  # Track previous state (True/False/None)
# frames_since_state_change = 0
# continuous_update_interval = 10  # Publish continuous updates every N frames
# consecutive_detections = 0
# consecutive_no_detections = 0