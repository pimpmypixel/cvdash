import os
import pickle
import cv2
import numpy as np
import time
import pickle
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from classes.utils.utils import add_log

def capture_browser(queue, headless):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=headless)
        
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            viewport={'width': 600, 'height': 338},  # 16:9 aspect ratio for 600px width
            permissions=['notifications'],
            ignore_https_errors=True,
            java_script_enabled=True,
            has_touch=False,
            is_mobile=False,
            locale='da-DK',
            timezone_id='Europe/Copenhagen',
            geolocation={'latitude': 55.676098, 'longitude': 12.568337},
            color_scheme='dark',
            reduced_motion='no-preference',
            forced_colors='none',
            accept_downloads=False,
            extra_http_headers={
                'Accept-Language': 'da-DK,en;q=0.9',
            }
        )

        page = context.new_page()

        if os.path.exists('./storage/tv2play.pkl'):
            add_log("Found cookies!")
            with open('./storage/tv2play.pkl', 'rb') as f:
                cookies = pickle.load(f)
                context.add_cookies(cookies)
        else:
            add_log("No cookies!")
            page = handle_login(page)
        
        add_log("Navigating to stream URL")
        page.goto("https://play.tv2.dk/afspil/TV2NEWS")
        add_log("Waiting for stream to load...")
        time.sleep(2)
        
        # Mute
        try:
            add_log("Looking for mute button...")
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').wait_for(timeout=5000)
            add_log("Found mute button, clicking...")
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').click()
            add_log("Stream muted successfully")
        except Exception as e:
            add_log(f"Failed to find or click mute button: {e}")
            add_log('Waiting one minute before retrying...')
            time.sleep(60)
            raise

        if is_video_playing(page, "video"):
            add_log("Video is playing")
        else:
            add_log("Video is not playing")

        # Play
        try:
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/button[1]').wait_for(timeout=1000)
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/button[1]').click()
            add_log("Stream playing")
        except Exception as e:
            add_log(f"Failed to find or click play button: {e}")
            add_log('Waiting one minute before retrying...')
            time.sleep(60)
            raise

        while True:
            screenshot = page.screenshot()
            img_array = np.frombuffer(screenshot, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is not None:
                queue.put(frame)
            time.sleep(0.1) 

def handle_login(page):
    """Handle TV2-specific login process."""
    add_log("Starting login process...")
    page.goto('https://play.tv2.dk')
    add_log("Navigated to TV2 website")

    time.sleep(1)
    try:
        page.click("xpath=/html/body/div[1]/div/div[4]/div[1]/div/div[2]/button[1]")
        add_log("Closed cookies popup")
    except Exception as e:
        add_log(f"Could not close cookies popup: {e}")
    time.sleep(1)

    if page.get_by_role("link", name="Log ind").is_visible():
        add_log("Need to log in")
        page.get_by_role("link", name="Log ind").click()
        page.fill("xpath=/html/body/div/div/div/div[3]/div[1]/form/div/div[1]/div/div/label/input", os.getenv('TV2_USERNAME'))
        page.fill("xpath=/html/body/div/div/div/div[3]/div[1]/form/div/div[2]/div/label/input", os.getenv('TV2_PASSWORD'))
        time.sleep(1)
        add_log("Login credentials entered")
    return page

def start_stream(page):
        print("Navigating to stream URL")
        page.goto("https://play.tv2.dk/afspil/TV2NEWS")
        print("Waiting for stream to load...")
        time.sleep(2)
        
        # Mute
        try:
            print("Looking for mute button...")
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').wait_for(timeout=5000)
            print("Found mute button, clicking...")
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/div[1]/button[1]').click()
            print("Stream muted successfully")
        except Exception as e:
            print(f"Failed to find or click mute button: {e}")
            print('Waiting one minute before retrying...')
            time.sleep(60)
            raise

        if is_video_playing(page, "video"):
            print("Video is playing")
        else:
            print("Video is not playing")

        # Play
        try:
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/button[1]').wait_for(timeout=1000)
            page.locator('xpath=/html/body/div/div/div/div/div[3]/div[5]/button[1]').click()
            print("Stream playing")
        except Exception as e:
            print(f"Failed to find or click play button: {e}")
            print('Waiting one minute before retrying...')
            time.sleep(60)
            raise

        return page

def is_video_playing(page, video_selector):
    """Check if video is playing by examining the paused property"""
    is_paused = page.evaluate(f"""
        () => {{
            const video = document.querySelector('{video_selector}');
            return video ? video.paused : null;
        }}
    """)
    return is_paused is False  