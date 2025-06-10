import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
from classes.browser.browser_capture import capture_browser
from classes.camera.webcam_capture import capture_webcam
from classes.dashboard.graph_data import GraphData
from classes.utils.utils import draw_status_overlay_column, draw_graph_column
from classes.utils.utils import add_log, draw_log_panel
from classes.opencv.process_stream import process_browser_frame
from classes.opencv.process_camera import process_webcam_frame
from classes.opencv.process_color_history import compare_color_fluctuations
import config.config as c

SHOW_WINDOW = True
USE_WEBCAM = True
USE_BROWSER = True
HEADLESS_BROWSER = True

def main():
    global USE_WEBCAM, USE_BROWSER

# Color tracking
    stream_avg_colors_q = Queue(400)
    webcam_avg_colors_q = Queue(400)
    browser_q = Queue(maxsize=5)
    webcam_q = Queue(maxsize=5)
    stats = GraphData(stream_avg_colors_q)
    webcam_stats = GraphData(webcam_avg_colors_q)

    # Launch threads
    if USE_BROWSER:
        Thread(target=capture_browser, args=(browser_q, HEADLESS_BROWSER), daemon=True).start()
    if USE_WEBCAM:
        Thread(target=capture_webcam, args=(webcam_q,), daemon=True).start()

    Thread(target=stats.update_loop, daemon=True).start()
    Thread(target=webcam_stats.update_loop, daemon=True).start()

    while True:
        frames = []
        # Update frames if enabled and available
        if USE_BROWSER and not browser_q.empty():
            raw_browser = browser_q.get()
            browser_frame = process_browser_frame(raw_browser, stream_avg_colors_q)
            browser_resized = cv2.resize(browser_frame, (c.window_width, c.window_height))
            frames.append(browser_resized)

        if USE_WEBCAM and not webcam_q.empty():
            raw_webcam = webcam_q.get()
            webcam_frame = process_webcam_frame(raw_webcam, webcam_avg_colors_q)
            webcam_resized = cv2.resize(webcam_frame, (c.window_width_webcam, c.window_height))
            frames.append(webcam_resized)

        color_comparison = compare_color_fluctuations(stream_avg_colors_q, webcam_avg_colors_q)
        for key, value in color_comparison.items():
            add_log(f"{key}: {value}")
        

        # Status/graph column
        stats_column = draw_graph_column(stats.get_history(), webcam_stats.get_history())
        # stats_column = draw_status_overlay_column(stats_column, stats.get_status())
        frames.append(stats_column)

        # Log panel column
        log_panel = draw_log_panel()
        frames.append(log_panel)

        if SHOW_WINDOW and frames:
            combined = np.hstack(frames)
            cv2.imshow("Real-Time Dashboard", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                add_log("Quitting application.")
                add_log("-" * 50)
                break
            elif key == ord('w'):
                USE_WEBCAM = not USE_WEBCAM
                add_log(f"Webcam toggled {'ON' if USE_WEBCAM else 'OFF'}")
                # Start or stop webcam thread
                if USE_WEBCAM:
                    Thread(target=capture_webcam, args=(webcam_q,), daemon=True).start()
            elif key == ord('b'):
                USE_BROWSER = not USE_BROWSER
                add_log(f"Browser toggled {'ON' if USE_BROWSER else 'OFF'}")
                # Start or stop browser thread
                if USE_BROWSER:
                    Thread(target=capture_browser, args=(browser_q,), daemon=True).start()
            time.sleep(.5)
        else:
            time.sleep(0.1)

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()