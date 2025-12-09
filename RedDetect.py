import cv2
import time
import os
from picamera2 import Picamera2

# ============================
# Camera settings
# ============================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 80   # Used for calculations only, *not* enforced

# ============================
# Video save location
# ============================
video_folder = "media/videos"
os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, "red_detect_capture.mp4")

# ============================
# Initialize camera
# ============================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)

# picam2.set_controls({"FrameRate": FPS})  # optional, try if stable
picam2.start()

# ============================
# Red color HSV ranges
# (red wraps around hue, so two ranges)
# ============================
lower_red1 = (0, 120, 70)
upper_red1 = (10, 255, 255)

lower_red2 = (170, 120, 70)
upper_red2 = (180, 255, 255)

# ============================
# Video writer (OpenCV requires a number)
# ============================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

# ============================
# Processing Loop
# ============================
frames_processed = 0
start_time = time.time()

try:
    while frames_processed < 100:  # ~100 frames
        frame = picam2.capture_array()

        # Convert RGB → BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Mask for red (two ranges)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            print("Contour area:", area)

            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 500:  # ignore tiny noise
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save frame
        out.write(frame_bgr)
        frames_processed += 1

finally:
    # ============================
    # FPS Measurement
    # ============================
    end_time = time.time()
    elapsed = end_time - start_time
    fps = frames_processed / elapsed

    print(f"\nProcessed {frames_processed} frames")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Measured FPS: {fps:.2f}")
    print(f"Saved video as {video_path}")

    out.release()
    picam2.stop()
