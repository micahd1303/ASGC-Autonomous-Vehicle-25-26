import cv2
import numpy as np
from picamera2 import Picamera2

# -----------------------------------
# CAMERA SETTINGS
# -----------------------------------
FRAME_WIDTH = 1080
FRAME_HEIGHT = 720

# -----------------------------------
# HSV RANGE (START WITH GREEN)
# -----------------------------------
HSV_RANGES = [ # THESE ARE ALL MONEY as of 1/30 in daylight
    ((40, 140, 10), (90, 255, 200)),  # GREEN
    # ((100, 150, 0), (140, 255, 255)),  # BLUE if needed
    #((25, 180, 100), (35, 255, 255)), # Yellow
    #((0, 80, 50), (10, 255, 255)),
    #((165, 80, 50), (180, 255, 255)), #RED
    
    
]

# -----------------------------------
# DETECTION PARAMETERS
# -----------------------------------
MIN_AREA = 300

# -----------------------------------
# STEERING VISUALIZATION PARAMETERS
# -----------------------------------
DEADZONE_FRAC = 0.15    # fraction of frame width
STEER_BAR_PIXELS = 250

# -----------------------------------
# CAMERA INIT
# -----------------------------------
picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(cfg)
picam2.start()

print("Starting steering visualization... Press 'q' to quit")

while True:
    frame = picam2.capture_array()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # -----------------------------------
    # BUILD MASK
    # -----------------------------------
    mask = None
    for low, high in HSV_RANGES:
        piece = cv2.inRange(hsv, low, high)
        mask = piece if mask is None else (mask | piece)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # -----------------------------------
    # STEERING REGIONS
    # -----------------------------------
    center_x = FRAME_WIDTH // 2
    deadzone_half = int(FRAME_WIDTH * DEADZONE_FRAC / 2)
    left_bound = center_x - deadzone_half
    right_bound = center_x + deadzone_half

    # Overlay regions
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0,0), (left_bound, FRAME_HEIGHT), (255,0,0), -1)
    cv2.rectangle(overlay, (left_bound,0), (right_bound, FRAME_HEIGHT), (0,255,0), -1)
    cv2.rectangle(overlay, (right_bound,0), (FRAME_WIDTH, FRAME_HEIGHT), (0,0,255), -1)
    bgr = cv2.addWeighted(overlay, 0.15, bgr, 0.85, 0)

    # Draw deadzone boundaries
    cv2.line(bgr, (left_bound, 0), (left_bound, FRAME_HEIGHT), (255,255,255), 2)
    cv2.line(bgr, (right_bound, 0), (right_bound, FRAME_HEIGHT), (255,255,255), 2)

    command = "SEARCHING"
    norm_error = 0.0

    # -----------------------------------
    # OBJECT SELECTION
    # -----------------------------------
    if contours:
        # pick largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            obj_cx = x + w // 2
            error = obj_cx - center_x
            norm_error = error / center_x
            norm_error = np.clip(norm_error, -1.0, 1.0)

            if obj_cx < left_bound:
                command = "STEER LEFT"
            elif obj_cx > right_bound:
                command = "STEER RIGHT"
            else:
                command = "GO STRAIGHT"

            # Draw bounding box + center + contour
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(bgr, (obj_cx, y + h // 2), 5, (0, 255, 255), -1)
            cv2.drawContours(bgr, [cnt], -1, (0, 255, 0), 2)

    # -----------------------------------
    # STEERING MAGNITUDE BAR
    # -----------------------------------
    bar_len = int(norm_error * STEER_BAR_PIXELS)
    cv2.line(
        bgr,
        (center_x, FRAME_HEIGHT - 40),
        (center_x + bar_len, FRAME_HEIGHT - 40),
        (0, 255, 255),
        6
    )

    # -----------------------------------
    # TEXT OVERLAY
    # -----------------------------------
    cv2.putText(
        bgr,
        command,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    cv2.imshow("Steering Debug View", bgr)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------------
# CLEANUP
# -----------------------------------
cv2.destroyAllWindows()
picam2.stop()
