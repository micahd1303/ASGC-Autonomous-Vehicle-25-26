import cv2
import time
import os
from picamera2 import Picamera2

# -----------------------------
# CAMERA SETUP
# -----------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 80

# ============================
# Video save location
# ============================
video_folder = "media/videos"
os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, "fsm_detect_output.mp4")

picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(cfg)
picam2.start()

# -----------------------------
# HSV RANGES
# -----------------------------

COLOR_RANGES = {
    "RED": [
        ((0,   120, 70), (10, 255, 255)),   # red range 1
        ((170, 120, 70), (180, 255, 255))   # red range 2
    ],
    "GREEN": [((40, 40, 40), (90, 255, 255))],
    "BLUE":  [((100, 150, 0), (140, 255, 255))],
    "YELLOW":[((20, 100, 100), (35, 255, 255))]
}

# -----------------------------
# DETECTION PARAMETERS
# -----------------------------
BALL_MIN_AREA = 100
BUCKET_MIN_AREA = 2000

# Aspect ratio requirements
BALL_AR_MIN, BALL_AR_MAX = 0.9, 1.1        # nearly square
BUCKET_AR_MIN, BUCKET_AR_MAX = 1.3, 3.5     # medium rectangle

# -----------------------------
# FSM ORDER
# -----------------------------
FSM_SEQUENCE = [
    ("BALL",  "RED"),
    ("BALL",  "GREEN"),
    ("BALL",  "BLUE"),
    ("BALL",  "YELLOW"),
    ("BUCKET","RED"),
    ("BUCKET","GREEN"),
    ("BUCKET","BLUE"),
    ("BUCKET","YELLOW"),
]

FRAMES_PER_STATE = 75



# -----------------------------
# VIDEO OUTPUT
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

print("Starting FSM object detection...")

# ---------------------------------
# MAIN FSM LOOP
# ---------------------------------
for obj_type, color_name in FSM_SEQUENCE:
    print(f"\n--- FSM STATE: Detecting {color_name} {obj_type} ---")
    frames = 0

    # HSV ranges for current color
    hsv_ranges = COLOR_RANGES[color_name]

    while frames < FRAMES_PER_STATE:

        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Build mask (some colors have 2 ranges)
        mask_total = None
        for low, high in hsv_ranges:
            mask_piece = cv2.inRange(hsv, low, high)
            mask_total = mask_piece if mask_total is None else (mask_total | mask_piece)

        # Find contours
        contours, _ = cv2.findContours(mask_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (obj_type == "BALL" and area < BALL_MIN_AREA) or \
               (obj_type == "BUCKET" and area < BUCKET_MIN_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)

            # BALL = nearly square
            if obj_type == "BALL":
                if not (BALL_AR_MIN <= aspect <= BALL_AR_MAX):
                    continue

            # BUCKET = medium rectangle
            if obj_type == "BUCKET":
                if not (BUCKET_AR_MIN <= aspect <= BUCKET_AR_MAX):
                    continue

            # Draw detection
            cv2.rectangle(bgr, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(bgr, f"{color_name} {obj_type}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Write frame to the combined video
        out.write(bgr)
        frames += 1

out.release()
picam2.stop()

print("\nFSM complete! Saved as fsm_detect_output.mp4")
