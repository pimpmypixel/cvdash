import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
from classes.browser.browser_capture import capture_browser
from classes.camera.webcam_capture import capture_webcam, is_webcam_accessible
from classes.dashboard.graph_data import GraphData
from classes.utils.utils import draw_status_overlay_column, draw_graph_column
from classes.utils.logger import add_log
from classes.opencv.process_stream import process_browser_frame
from classes.opencv.process_camera_v5 import process_webcam_frame, reset_detection
from classes.opencv.process_color_history import compare_color_fluctuations
import config.config as c
import json
from datetime import datetime

SHOW_WINDOW = True
USE_WEBCAM = True
USE_BROWSER = True
HEADLESS_BROWSER = True
COMPARE_COLORS = False

# --- Display Frame Rate Control ---
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS  # Time in seconds for each frame

def main():
    global USE_WEBCAM, USE_BROWSER, COMPARE_COLORS, LOG_PANEL, HEADLESS_BROWSER, SHOW_WINDOW
    window_width_webcam = c.window_width_webcam

    webcam_dimensions = is_webcam_accessible()
    if USE_WEBCAM:
        if not webcam_dimensions:
            USE_WEBCAM = False
            add_log("Webcam disabled due to accessibility issues")
        else:
            window_width_webcam = int((c.window_height / webcam_dimensions[0]) * webcam_dimensions[1])

    max_history_size = c.window_width // 2  # Maximum number of color bars that can be displayed
    stream_avg_colors_q = Queue(maxsize=max_history_size)
    webcam_avg_colors_q = Queue(maxsize=max_history_size)
    browser_q = Queue(maxsize=5)
    webcam_q = Queue(maxsize=5)
    stats = GraphData(stream_avg_colors_q)
    webcam_stats = GraphData(webcam_avg_colors_q)

    if USE_BROWSER:
        Thread(target=capture_browser, args=(browser_q, HEADLESS_BROWSER), daemon=True).start()
    if USE_WEBCAM:
        Thread(target=capture_webcam, args=(webcam_q,), daemon=True).start()

    Thread(target=stats.update_loop, daemon=True).start()
    Thread(target=webcam_stats.update_loop, daemon=True).start()

    last_frame_time = time.time() # Initialize time tracking

    while True:
        current_time = time.time()

        browser_frame_display = np.zeros((c.window_height, c.window_width, 3), dtype=np.uint8)
        webcam_frame_display = np.zeros((c.window_height, window_width_webcam, 3), dtype=np.uint8)

        if USE_BROWSER and not browser_q.empty():
            raw_browser = browser_q.get()
            browser_frame = process_browser_frame(raw_browser, stream_avg_colors_q)
            browser_frame_display = cv2.resize(browser_frame, (c.window_width, c.window_height))

        if USE_WEBCAM and not webcam_q.empty():
            raw_webcam = webcam_q.get()
            webcam_frame = process_webcam_frame(raw_webcam, webcam_avg_colors_q)
            webcam_frame_display = cv2.resize(webcam_frame, (window_width_webcam, c.window_height))

        stream_size = stream_avg_colors_q.qsize()
        webcam_size = webcam_avg_colors_q.qsize()

        if COMPARE_COLORS and len(stats.get_history()) >= 50 and len(webcam_stats.get_history()) >= 50:
            add_log('Comparing colors...')
            # Create temporary queues with history data from GraphData
            temp_queue1 = Queue()
            temp_queue2 = Queue()
            for item in stats.get_history():
                temp_queue1.put(item)
            for item in webcam_stats.get_history():
                temp_queue2.put(item)
            
            color_comparison = compare_color_fluctuations(temp_queue1, temp_queue2, similarity_threshold=.1)
            for key, value in color_comparison.items():
                print(f"{key}: {value}")

        stats_column = draw_graph_column(
            stats.get_history(),
            webcam_stats.get_history(),
            compare_colors=COMPARE_COLORS,
            webcam_enabled=USE_WEBCAM,
            stream_enabled=USE_BROWSER,
        )
        
        # Assemble all frames for display, ensuring consistent dimensions
        display_frames = []
        if USE_BROWSER:
            display_frames.append(browser_frame_display)
        if USE_WEBCAM:
            display_frames.append(webcam_frame_display)
        display_frames.append(stats_column) # Stats column is always present

        # Calculate the total width for the combined frame
        total_width = sum(frame.shape[1] for frame in display_frames)

        try:
            combined = np.hstack(display_frames)
        except ValueError as e:
            add_log(f"Error combining frames: {e}")
            for i, frame in enumerate(display_frames):
                add_log(f"Frame {i} dimensions: {frame.shape}")
            # Fallback to a black frame if combination fails
            combined = np.zeros((c.window_height, total_width, 3), dtype=np.uint8)
        
        if SHOW_WINDOW:
            cv2.imshow("Dashboard", combined)

        key = cv2.waitKey(1) & 0xFF # Process key presses immediately

        # --- Frame Rate Control Logic ---
        time_elapsed = time.time() - current_time # Calculate time elapsed for the current frame
        time_to_sleep = FRAME_INTERVAL - time_elapsed
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        last_frame_time = time.time() # Update last frame time for next iteration

        if key == ord('q'):
            add_log("Quitting application.")
            add_log("-" * 50)
            break
        elif key == ord('w'):
            USE_WEBCAM = not USE_WEBCAM
            add_log(f"Webcam toggled {'ON' if USE_WEBCAM else 'OFF'}")
            if USE_WEBCAM:
                Thread(target=capture_webcam, args=(webcam_q,), daemon=True).start()
        elif key == ord('l'):
            LOG_PANEL = not LOG_PANEL
            add_log(f"Log panel toggled {'ON' if LOG_PANEL else 'OFF'}")
        elif key == ord('b'):
            USE_BROWSER = not USE_BROWSER
            add_log(f"Browser toggled {'ON' if USE_BROWSER else 'OFF'}")
            if USE_BROWSER:
                Thread(target=capture_browser, args=(browser_q, HEADLESS_BROWSER), daemon=True).start()
        elif key == ord('r'):
            reset_detection()
            while not webcam_avg_colors_q.empty():
                webcam_avg_colors_q.get()
            while not stream_avg_colors_q.empty():
                stream_avg_colors_q.get()
            add_log("TV ROI reset and color queues cleared")
        elif key == ord('c'):
            COMPARE_COLORS = not COMPARE_COLORS
            add_log(f"Color comparison {'ENABLED' if COMPARE_COLORS else 'DISABLED'}")

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
