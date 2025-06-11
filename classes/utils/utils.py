import cv2
import numpy as np
import time
import os
import config.config as c
from classes.utils.logger import add_log, draw_log_panel
from classes.opencv.process_camera_v4 import tv_box, detection_confidence, locked_roi

# Add any utility functions here that don't create circular dependencies
def format_timestamp(timestamp):
    """Format a timestamp into a readable string."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def validate_config(config_dict):
    """Validate configuration dictionary."""
    required_keys = ['window_width', 'window_height']
    return all(key in config_dict for key in required_keys)

def draw_status_overlay_column(frame, status):
    y0 = 20
    dy = 15
    for i, (key, value) in enumerate(status.items()):
        y = y0 + i * dy
        text = f"{key}: {value}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1, cv2.LINE_8)
    return frame

def draw_graph_column(stream_history, webcam_history=None, compare_colors=False, stream_enabled=False, webcam_enabled=False):
    graph = np.zeros((c.window_height, c.window_width, 3), dtype=np.uint8)
    if not stream_history and not webcam_history:
        return graph

    # Draw status table at the top
    y_start = 20
    dy = 15  # Reduced line height
    font_scale = 0.3  # Reduced font size by 25%
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get current time and NTP time
    current_time = time.strftime("%H:%M:%S")
    ntp_time = time.strftime("%H:%M:%S")  # TODO: Replace with actual NTP time
    
    # Get TV ROI status from v4
    if locked_roi is not None:
        tv_roi_status = "Locked"
    elif tv_box is not None:
        tv_roi_status = f"Detecting ({detection_confidence*100:.0f}%)"
    else:
        tv_roi_status = "Not Detected"
    
    # Get logo detection status from stream history
    logo_status = "TV"  # Default to TV
    if stream_history:
        # Get the most recent entry that has a status
        for entry in reversed(stream_history):
            if entry[3] is not None:  # Check if status exists
                logo_status = entry[3]
                break
    
    # Table rows
    rows = [
        ("Logo Detected", logo_status),
        ("Compare Colors [c]", "Enabled" if compare_colors else "Disabled"),
        ("Browser stream [b]", "Enabled" if stream_enabled else "Disabled"),
        ("Webcam [w]", "Enabled" if webcam_enabled else "Disabled"),
        ("NTP Clock", ntp_time),
        ("TV ROI [r]", tv_roi_status)
    ]
    
    # Draw table
    for i, (label, value) in enumerate(rows):
        y = y_start + i * dy
        # Draw label
        cv2.putText(graph, label, (10, y), font, font_scale, (200,200,200), font_thickness, cv2.LINE_8)
        # Draw value
        cv2.putText(graph, value, (c.window_width // 2 + 10, y), font, font_scale, (200,200,200), font_thickness, cv2.LINE_8)

    # Draw color bars from right to left
    bar_width = 1  # 1px width for each bar
    bar_height = 20  # 20px height for each bar
    x_start = c.window_width - 1
    
    # Draw webcam color history
    if webcam_history:
        y_start = c.window_height - bar_height - 70
        cv2.putText(graph, 'Webcam', (c.window_width - 40, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1, cv2.LINE_8)
        
        for color_data in reversed(webcam_history):  # Process colors from newest to oldest
            if x_start < 0:  # Stop if we've reached the left edge
                break
                
            # Draw the color bar
            cv2.rectangle(graph, 
                         (x_start, y_start),
                         (x_start + bar_width, y_start + bar_height),
                         color_data,  # Use the RGB color from history
                         -1)  # Fill the rectangle
            
            x_start -= bar_width  # Move left for next bar
    
    # Reset x_start for stream history
    x_start = c.window_width - 1
    
    # Draw stream color history
    if stream_history:
        y_start = c.window_height - bar_height - 10  # Position above webcam history
        cv2.putText(graph, 'Stream', (c.window_width - 40, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1, cv2.LINE_8)
        
        for color_data in reversed(stream_history):  # Process colors from newest to oldest
            if x_start < 0:  # Stop if we've reached the left edge
                break
                
            # Extract color and state data
            r, g, b, state, timestamp = color_data
            
            # Draw the color bar
            cv2.rectangle(graph, 
                         (x_start, y_start),
                         (x_start + bar_width, y_start + bar_height),
                         (r, g, b),  # Use the RGB color from history
                         -1)  # Fill the rectangle
            
            # If this is a state change point (state is not None), draw a white marker
            if state is not None:
                marker_y = y_start - 5  # Position marker above the color bar
                cv2.line(graph,
                        (x_start, marker_y),
                        (x_start, marker_y + 10),  # 10px tall marker
                        (255, 255, 255),  # White color
                        1)  # 1px width
            
            x_start -= bar_width  # Move left for next bar
    
    return graph

# Add other utility functions as needed