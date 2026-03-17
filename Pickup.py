import cv2
import numpy as np
import time
from picamera2 import Picamera2

# -------------------------------
# CAMERA CONFIG
# -------------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

picam2 = Picamera2()
cfg = picam2.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(cfg)
picam2.start()

# -------------------------------
# HSV RANGE FOR BLUE
# -------------------------------
BLUE_LOW = (100,150,0)
BLUE_HIGH = (140,255,255)

BALL_MIN_AREA = 300
AR_MIN = 0.7
AR_MAX = 4.5

# -------------------------------
# AREA TIERS
# -------------------------------
FAR_THRESH = 3000
MED_THRESH = 20000
CLOSE_THRESH = 40000

# Deadzone sizes (fraction of frame width)
DEADZONE_FAR = 0.10     # very sensitive
DEADZONE_MED = 0.20
DEADZONE_CLOSE = 0.35   # very stable

STEER_BAR_PIXELS = 250

# -------------------------------
# WINDOW
# -------------------------------
cv2.namedWindow("Steering Debug", cv2.WINDOW_NORMAL)

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:

    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # -------------------------------
    # MASK
    # -------------------------------
    mask = cv2.inRange(hsv, np.array(BLUE_LOW), np.array(BLUE_HIGH))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # -------------------------------
    # CONTOURS
    # -------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < BALL_MIN_AREA:
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        ar = w/float(h)

        if not (AR_MIN <= ar <= AR_MAX):
            continue

        if area > best_area:
            best_area = area
            best_contour = cnt

    # -------------------------------
    # STEERING DEBUG FRAME
    # -------------------------------
    debug = bgr.copy()

    center_x = FRAME_WIDTH // 2

    norm_error = 0.0
    command = "SEARCHING"
    tier = "NONE"
    deadzone_frac = DEADZONE_MED

    if best_contour is not None:

        x,y,w,h = cv2.boundingRect(best_contour)
        obj_cx = x + w//2

        # -------------------------------
        # DETERMINE TIER
        # -------------------------------
        if best_area < FAR_THRESH:
            tier = "FAR"
            deadzone_frac = DEADZONE_FAR

        elif best_area < MED_THRESH:
            tier = "MED"
            deadzone_frac = DEADZONE_MED

        elif best_area > CLOSE_THRESH:
            tier = "CLOSE"
            deadzone_frac = DEADZONE_CLOSE

        else:
            tier = "MED"
            deadzone_frac = DEADZONE_MED

        # -------------------------------
        # ERROR CALC
        # -------------------------------
        error = obj_cx - center_x
        norm_error = error / center_x
        norm_error = np.clip(norm_error, -1.0, 1.0)

        # Deadzone boundaries
        deadzone_half = int(FRAME_WIDTH * deadzone_frac / 2)
        left_bound = center_x - deadzone_half
        right_bound = center_x + deadzone_half

        # -------------------------------
        # COMMAND
        # -------------------------------
        if obj_cx < left_bound:
            command = "LEFT"
        elif obj_cx > right_bound:
            command = "RIGHT"
        else:
            command = "STRAIGHT"
            norm_error = 0.0

        # draw bounding box
        cv2.rectangle(debug,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(debug,(obj_cx,y+h//2),6,(0,255,255),-1)

    else:
        deadzone_half = int(FRAME_WIDTH * deadzone_frac / 2)
        left_bound = center_x - deadzone_half
        right_bound = center_x + deadzone_half

    # -------------------------------
    # DRAW ZONES
    # -------------------------------
    overlay = debug.copy()

    cv2.rectangle(overlay,(0,0),(left_bound,FRAME_HEIGHT),(255,0,0),-1)
    cv2.rectangle(overlay,(left_bound,0),(right_bound,FRAME_HEIGHT),(0,255,0),-1)
    cv2.rectangle(overlay,(right_bound,0),(FRAME_WIDTH,FRAME_HEIGHT),(0,0,255),-1)

    debug = cv2.addWeighted(overlay,0.15,debug,0.85,0)

    # boundary lines
    cv2.line(debug,(left_bound,0),(left_bound,FRAME_HEIGHT),(255,255,255),2)
    cv2.line(debug,(right_bound,0),(right_bound,FRAME_HEIGHT),(255,255,255),2)

    # -------------------------------
    # STEERING BAR
    # -------------------------------
    bar_len = int(norm_error * STEER_BAR_PIXELS)

    cv2.line(
        debug,
        (center_x,FRAME_HEIGHT-40),
        (center_x+bar_len,FRAME_HEIGHT-40),
        (0,255,255),
        6
    )

    # -------------------------------
    # TEXT
    # -------------------------------
    cv2.putText(debug,f"Cmd: {command}",(40,60),
                cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)

    cv2.putText(debug,f"Tier: {tier}",(40,110),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)

    cv2.putText(debug,f"Area: {int(best_area)}",(40,150),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)

    cv2.putText(debug,f"Err: {norm_error:+.2f}",(40,190),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)

    # -------------------------------
    # SHOW
    # -------------------------------
    cv2.imshow("Steering Debug", debug)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
