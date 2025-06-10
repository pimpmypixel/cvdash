import cv2
import numpy as np
import time
import os
import config.config as c

# Log buffer for the 4th panel
LOG_LINES = []
MAX_LOG_LINES = 20

def append_to_log_file(message):
    """Append a message to the storage/log.txt file. Create the file if it doesn't exist."""
    os.makedirs('storage', exist_ok=True)
    log_message = f"{message}\n"
    with open('storage/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_message)

def add_log(message):
    global LOG_LINES
    timestamp = time.strftime("%H:%M:%S")
    message = f"[{timestamp}] {message}"
    LOG_LINES.append(message)
    append_to_log_file(message)
    if len(LOG_LINES) > MAX_LOG_LINES:
        LOG_LINES.pop(0)

def draw_log_panel():
    panel = np.zeros((c.window_height, c.window_width // 2, 3), dtype=np.uint8) + 20
    y0 = 20
    dy = 10
    for i, line in enumerate(LOG_LINES):
        y = y0 + i * dy
        # cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1, cv2.LINE_8)
    return panel

def draw_status_overlay_column(frame, status):
    y0 = 20
    dy = 15
    for i, (key, value) in enumerate(status.items()):
        y = y0 + i * dy
        text = f"{key}: {value}"
        # cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1, cv2.LINE_8)
    return frame

def draw_graph_column(stream_history, webcam_history=None):
    graph = np.zeros((c.window_height, c.window_width, 3), dtype=np.uint8)
    if not stream_history and not webcam_history:
        return graph

    # Draw color bars from right to left
    bar_width = 1  # 1px width for each bar
    bar_height = 20  # 20px height for each bar
    x_start = c.window_width - 1  # Start from right edge
    
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