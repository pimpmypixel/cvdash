import logging
import os
import time
import numpy as np
import cv2
import config.config as c
from datetime import datetime

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Create a logger
logger = logging.getLogger('CVDash')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    os.path.join(log_dir, f'cvdash_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log buffer for the UI panel
LOG_LINES = []
MAX_LOG_LINES = 20

def add_log(message, level='info'):
    """Add a log message with the specified level and update UI panel."""
    global LOG_LINES
    
    # Add timestamp to message
    timestamp = time.strftime("%H:%M:%S")
    message = f"[{timestamp}] {message}"
    
    # Update UI log lines
    LOG_LINES.append(message)
    if len(LOG_LINES) > MAX_LOG_LINES:
        LOG_LINES.pop(0)
    
    # Log with appropriate level
    if level.lower() == 'debug':
        logger.debug(message)
    elif level.lower() == 'info':
        logger.info(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    elif level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'critical':
        logger.critical(message)
    else:
        logger.info(message)  # Default to info level

def draw_log_panel():
    """Draw the log panel for the UI."""
    panel = np.zeros((c.window_height, c.window_width // 2, 3), dtype=np.uint8) + 20
    y0 = 20
    dy = 10
    for i, line in enumerate(LOG_LINES):
        y = y0 + i * dy
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1, cv2.LINE_8)
    return panel 