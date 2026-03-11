import cv2
import numpy as np
import time
import sys
from collections import deque
from picamera2 import Picamera2

# -------------------------------
# IMPORT MATRIX LIDAR
# -------------------------------
sys.path.append("/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python")
from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c

I2C_ADDRESS = 0x33
WINDOW_SIZE = 5

tof = DFRobot_matrixLidar_i2c(I2C_ADDRESS)
history = deque(maxlen=WINDOW_SIZE)
center_indices = [35, 36, 43, 44]

DISTANCE_TRIGGER = 500  # mm

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
AR_MAX = 1.3

# -------------------------------
# STEERING VISUALIZATION
# -------------------------------
DEADZONE_FRAC = 0.15
STEER_BAR_PIXELS = 250

# -------------------------------
# WINDOW SETUP
# -------------------------------
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Steering Debug", cv2.WINDOW_NORMAL)

# -------------------------------
# DISTANCE FUNCTIONS
# -------------------------------
def get_center_pixels(raw_data):

    distances = [(raw_data[i+1]<<8)|raw_data[i] for i in range(0,len(raw_data),2)]

    return [distances[i] for i in center_indices]


def get_filtered_distance():

    data = tof.get_all_data()

    if not data:
        return None

    pixels = get_center_pixels(data)

    raw_avg = sum(pixels)/len(pixels)

    history.append(raw_avg)

    filt_avg = sum(history)/len(history)

    return filt_avg


# -------------------------------
# INITIALIZE DISTANCE SENSOR
# -------------------------------
print("Initializing TOF...")

while tof.begin()!=0:
    print("TOF init failed... retrying")
    time.sleep(1)

tof.set_Ranging_Mode(8)

print("TOF ready")

# -------------------------------
# TARGET LOCK VARIABLES
# -------------------------------
locked_target = None

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:

    prev_time = time.time()
    
    frame = picam2.capture_array()

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # -------------------------------
    # BLUE MASK
    # -------------------------------
    lower = np.array(BLUE_LOW)
    upper = np.array(BLUE_HIGH)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # -------------------------------
    # FIND CONTOURS
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
    # LOCK ONTO LARGEST OBJECT
    # -------------------------------
    if best_contour is not None:

        x,y,w,h = cv2.boundingRect(best_contour)

        ar = w/float(h)

        cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            bgr,
            f"AR={ar:.2f} A={int(best_area)}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        locked_target = (x,y,w,h)
        
        
        current_time = time.time()
        fps = 1/(current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            bgr,
            f"FPS: {fps:.1f}",
            (40,140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,0),
            3
)
# -------------------------------
# STEERING VISUALIZATION
# -------------------------------
    steer_frame = bgr.copy()

    center_x = FRAME_WIDTH // 2
    deadzone_half = int(FRAME_WIDTH * DEADZONE_FRAC / 2)

    left_bound = center_x - deadzone_half
    right_bound = center_x + deadzone_half

    # Region overlay
    overlay = steer_frame.copy()

    cv2.rectangle(overlay, (0,0), (left_bound, FRAME_HEIGHT), (255,0,0), -1)
    cv2.rectangle(overlay, (left_bound,0), (right_bound,FRAME_HEIGHT), (0,255,0), -1)
    cv2.rectangle(overlay, (right_bound,0), (FRAME_WIDTH,FRAME_HEIGHT), (0,0,255), -1)

    steer_frame = cv2.addWeighted(overlay,0.15,steer_frame,0.85,0)

    # Deadzone lines
    cv2.line(steer_frame,(left_bound,0),(left_bound,FRAME_HEIGHT),(255,255,255),2)
    cv2.line(steer_frame,(right_bound,0),(right_bound,FRAME_HEIGHT),(255,255,255),2)

    command = "SEARCHING"
    norm_error = 0.0

    if best_contour is not None:

        x,y,w,h = cv2.boundingRect(best_contour)

        obj_cx = x + w//2
        error = obj_cx - center_x

        norm_error = error / center_x
        norm_error = np.clip(norm_error,-1.0,1.0)

        if obj_cx < left_bound:
            command = "STEER LEFT"

        elif obj_cx > right_bound:
            command = "STEER RIGHT"

        else:
            command = "GO STRAIGHT"

        cv2.circle(steer_frame,(obj_cx,y+h//2),6,(0,255,255),-1)

    # steering magnitude bar
    bar_len = int(norm_error * STEER_BAR_PIXELS)

    cv2.line(
        steer_frame,
        (center_x,FRAME_HEIGHT-40),
        (center_x+bar_len,FRAME_HEIGHT-40),
        (0,255,255),
        6
    )

    cv2.putText(
        steer_frame,
        command,
        (40,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255,255,255),
        3
)
    # -------------------------------
    # READ DISTANCE SENSOR
    # -------------------------------
    distance = get_filtered_distance()

    if distance is not None:

        cv2.putText(
            bgr,
            f"Distance: {int(distance)} mm",
            (40,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            3
        )

        if distance < DISTANCE_TRIGGER:

            cv2.putText(
                bgr,
                "USE DISTANCE SENSOR MODE",
                (40,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

            print("DISTANCE < 500mm : SWITCH TO DISTANCE SENSOR")

    # -------------------------------
    # SHOW WINDOWS
    # -------------------------------
    cv2.imshow("Detection",bgr)
    cv2.imshow("Steering Debug", steer_frame)
    cv2.imshow("Mask",mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


picam2.stop()

cv2.destroyAllWindows()
