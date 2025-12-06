# simple_record_headless.py
import time
import os
import cv2
from picamera2 import Picamera2

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DURATION = 5  # seconds

# Create media folder
video_folder = "media/videos"
os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, "simple_capture.mp4")

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config)
picam2.set_controls({"FrameRate": 30})
picam2.start()

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

start_time = time.time()
try:
    while time.time() - start_time < DURATION:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
finally:
    out.release()
    picam2.stop()
    print(f"Video saved as '{video_path}'")
