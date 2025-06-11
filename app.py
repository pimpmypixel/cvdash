import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
from classes.browser.browser_capture import capture_browser
from classes.camera.webcam_capture import capture_webcam, is_webcam_accessible
from classes.dashboard.graph_data import GraphData
from classes.utils.utils import draw_status_overlay_column, draw_graph_column
from classes.utils.utils import add_log, draw_log_panel
from classes.opencv.process_stream import process_browser_frame
from classes.opencv.process_camera_v4 import process_webcam_frame, reset_detection
from classes.opencv.process_color_history import compare_color_fluctuations
import config.config as c

SHOW_WINDOW = True
LOG_PANEL = True
USE_WEBCAM = False
USE_BROWSER = True
HEADLESS_BROWSER = True
COMPARE_COLORS = False

def main():
    global USE_WEBCAM, USE_BROWSER, COMPARE_COLORS, LOG_PANEL, HEADLESS_BROWSER, SHOW_WINDOW
    window_width_webcam = c.window_width_webcam

    webcam_dimensions = is_webcam_accessible()
    if USE_WEBCAM and not webcam_dimensions:
        USE_WEBCAM = False
        add_log("Webcam disabled due to accessibility issues")
    else:
        window_width_webcam = int((c.window_height / webcam_dimensions[0]) * webcam_dimensions[1])

    stream_avg_colors_q = Queue(100)
    webcam_avg_colors_q = Queue(100)
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

    while True:
        frames = []

        if USE_BROWSER and not browser_q.empty():
            raw_browser = browser_q.get()
            browser_frame = process_browser_frame(raw_browser, stream_avg_colors_q)
            browser_resized = cv2.resize(browser_frame, (c.window_width, c.window_height))
            frames.append(browser_resized)

        if USE_WEBCAM and not webcam_q.empty():
            raw_webcam = webcam_q.get()
            webcam_frame = process_webcam_frame(raw_webcam, webcam_avg_colors_q)
            webcam_resized = cv2.resize(webcam_frame, (window_width_webcam, c.window_height))
            frames.append(webcam_resized)

        stream_size = stream_avg_colors_q.qsize()
        webcam_size = webcam_avg_colors_q.qsize()

        if COMPARE_COLORS and stream_size >= 50 and webcam_size >= 50:
            add_log('Comparing colors...')
            color_comparison = compare_color_fluctuations(stream_avg_colors_q, webcam_avg_colors_q, similarity_threshold=.1)
            for key, value in color_comparison.items():
                print(f"{key}: {value}")

        stats_column = draw_graph_column(
            stats.get_history(),
            webcam_stats.get_history(),
            compare_colors=COMPARE_COLORS,
            webcam_enabled=USE_WEBCAM,
            stream_enabled=USE_BROWSER,
        )
        frames.append(stats_column)

        if LOG_PANEL:
            log_panel = draw_log_panel()
            frames.append(log_panel)

        if SHOW_WINDOW and frames:
            target_height = c.window_height
            for i in range(len(frames)):
                if frames[i].shape[0] != target_height:
                    frames[i] = cv2.resize(frames[i], (frames[i].shape[1], target_height))
            try:
                combined = np.hstack(frames)
                cv2.imshow("Dashboard", combined)
                key = cv2.waitKey(1) & 0xFF

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
            except ValueError as e:
                add_log(f"Error combining frames: {e}")
                for i, frame in enumerate(frames):
                    add_log(f"Frame {i} dimensions: {frame.shape}")
            time.sleep(.5)
        else:
            time.sleep(0.1)

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
