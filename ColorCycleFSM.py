import cv2
import time
import os
from picamera2 import Picamera2

# -----------------------------------
# TOGGLE FSM PARTS
# -----------------------------------
RUN_BALLS = False
RUN_BUCKETS = True

# -----------------------------------
# CAMERA SETTINGS
# -----------------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 80

# -----------------------------------
# VIDEO SAVE LOCATION
# -----------------------------------
video_folder = "media/videos"
os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, "fsm_detect_output.mp4")

picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(cfg)
picam2.start()

# -----------------------------------
# HSV COLOR RANGES (loosened for red & split green)
# -----------------------------------
COLOR_RANGES = {
    "RED": [
        ((0, 80, 50), (10, 255, 255)),
        ((165, 80, 50), (180, 255, 255)),
    ],
    "GREEN": [
        ((40, 30, 30), (90, 255, 120)),   # dark green (balls)
        ((40, 80, 120), (90, 255, 255)),  # bright green (buckets)
    ],
    "BLUE":  [((100, 150, 0), (140, 255, 255))],
    "YELLOW":[((20, 100, 100), (35, 255, 255))]
}

# -----------------------------------
# DETECTION PARAMETERS
# -----------------------------------
BALL_MIN_AREA = 100
BUCKET_MIN_AREA = 2000

BALL_AR_MIN, BALL_AR_MAX = 0.9, 1.1      # nearly square (ignored for now)
BUCKET_AR_MIN, BUCKET_AR_MAX = 1.3, 3.5  # rectangle-ish (ignored for now)

# -----------------------------------
# FSM STATE SEQUENCE
# -----------------------------------
FSM_SEQUENCE = []

if RUN_BALLS:
    FSM_SEQUENCE += [
        ("BALL", "RED"),
        ("BALL", "GREEN"),
        ("BALL", "BLUE"),
        ("BALL", "YELLOW"),
    ]

if RUN_BUCKETS:
    FSM_SEQUENCE += [
        ("BUCKET", "RED"),
        ("BUCKET", "GREEN"),
        ("BUCKET", "BLUE"),
        ("BUCKET", "YELLOW"),
    ]

FRAMES_PER_STATE = 75

# -----------------------------------
# VIDEO OUTPUT
# -----------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

print("Starting FSM object detection...\n")

# -----------------------------------
# FSM LOOP
# -----------------------------------
for obj_type, color_name in FSM_SEQUENCE:

    print(f"--- FSM STATE: Detecting {color_name} {obj_type} ---")
    frames = 0
    hsv_ranges = COLOR_RANGES[color_name]

    while frames < FRAMES_PER_STATE:

        # Capture & convert
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Build mask
        mask = None
        for low, high in hsv_ranges:
            piece = cv2.inRange(hsv, low, high)
            mask = piece if mask is None else (mask | piece)

        # Morphology to clean up small speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Extract contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # -----------------------------------
        # BALL LOGIC
        # -----------------------------------
        if obj_type == "BALL":
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < BALL_MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                ar = w / float(h)

                # Temporarily ignore aspect ratio
                # if not (BALL_AR_MIN <= ar <= BALL_AR_MAX):
                #     continue

                cv2.rectangle(bgr, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(bgr, f"{color_name} BALL", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # -----------------------------------
        # BUCKET LOGIC
        # -----------------------------------
        if obj_type == "BUCKET":
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < BUCKET_MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                ar = w / float(h)

                # Temporarily ignore aspect ratio
                # if not (BUCKET_AR_MIN <= ar <= BUCKET_AR_MAX):
                #     continue

                cv2.rectangle(bgr, (x, y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(bgr, f"{color_name} BUCKET", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Write frame
        out.write(bgr)
        frames += 1

# -----------------------------------
# Cleanup
# -----------------------------------
out.release()
picam2.stop()

print("\nFSM complete! Saved as:", video_path)
