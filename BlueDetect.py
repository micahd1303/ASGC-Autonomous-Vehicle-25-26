import cv2
import time
from picamera2 import Picamera2

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 80

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)
#picam2.set_controls({"FrameRate": FPS})
picam2.start()

# BLUE color range (your comment said blue but your code used yellow)
lower_blue = (100, 150, 50)
upper_blue = (110, 255, 255)

frames_processed = 0
start_time = time.time()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('blue_detect_output.mp4', fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

try:
    while frames_processed < 100:  # ~10 seconds at 10 FPS
        frame = picam2.capture_array()

        # Convert RGB → BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert to HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Mask for BLUE
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            print("Contour area:", area)

            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 500:  # ignore noise
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        out.write(frame_bgr)
        frames_processed += 1

finally:
    end_time = time.time()
    elapsed = end_time - start_time
    fps = frames_processed / elapsed

    print(f"Processed {frames_processed} frames")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"FPS: {fps:.2f}")
    print("Saved video as blue_detect_output.mp4")

    out.release()
    picam2.stop()
