import sys
import time
import serial
import glob
import threading
import cv2
import numpy as np
from collections import deque
from picamera2 import Picamera2
from gpiozero import Button

# -----------------------------------------------
# SWITCH / COLOR SETUP
# -----------------------------------------------
SW1 = Button(17, pull_up=True)  # bit A
SW2 = Button(27, pull_up=True)  # bit B

COLOR_RANGES = {
    "RED": [
        ((  0,  80,  50), ( 10, 255, 255)),
        ((165,  80,  50), (180, 255, 255)),
    ],
    "YELLOW": [
        (( 25, 180, 100), ( 35, 255, 255)),
    ],
    "GREEN": [
        (( 40, 100,  10), ( 90, 255, 200)),
    ],
    "BLUE": [
        ((100, 150,   0), (140, 255, 255)),
    ],
}

def get_active_color():
    A = 0 if SW1.is_pressed else 1
    B = 0 if SW2.is_pressed else 1
    if   (A, B) == (0, 0): return "RED",    COLOR_RANGES["RED"]
    elif (A, B) == (0, 1): return "YELLOW", COLOR_RANGES["YELLOW"]
    elif (A, B) == (1, 0): return "GREEN",  COLOR_RANGES["GREEN"]
    else:                  return "BLUE",   COLOR_RANGES["BLUE"]

# -----------------------------------------------
# SENSOR SETUP (LiDAR)
# -----------------------------------------------
sys.path.append("/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python")
from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c

I2C_ADDRESS    = 0x33
WINDOW_SIZE    = 5
tof            = DFRobot_matrixLidar_i2c(I2C_ADDRESS)
history        = deque(maxlen=WINDOW_SIZE)
center_indices = [35, 36, 43, 44]

def get_center_pixels(raw_data):
    distances = [(raw_data[i+1] << 8) | raw_data[i] for i in range(0, len(raw_data), 2)]
    return [distances[i] for i in center_indices]

def get_filtered_distance():
    data = tof.get_all_data()
    if not data:
        return None
    pixels  = get_center_pixels(data)
    raw_avg = sum(pixels) / len(pixels)
    history.append(raw_avg)
    return sum(history) / len(history)

# -----------------------------------------------
# SHARED STATE
# -----------------------------------------------
shared_lock = threading.Lock()
shared = {
    "error":     0.0,
    "visible":   False,
    "command":   "SEARCHING",
    "tier":      "NONE",
    "area":      0,
    "state":     "STOPPED",
    "dist":      0.0,
    "left_pct":  0.0,
    "right_pct": 0.0,
    "color":     "BLUE",
}

# -----------------------------------------------
# MOTOR SETUP (Pico via Serial) — auto-detect port
# -----------------------------------------------
def find_pico_port():
    ports = sorted(glob.glob('/dev/ttyACM*'))
    if not ports:
        raise RuntimeError("No Pico found on any /dev/ttyACM* port")
    print(f"ACM ports found: {ports} — using {ports[0]}")
    return ports[0]

print("Connecting to Pico...")
pico = serial.Serial(find_pico_port(), 115200, timeout=10)
print("Waiting for Pico to boot...")
while True:
    line = pico.readline().decode('utf-8', errors='replace').strip()
    if line == "PICO_READY":
        print("Pico ready.")
        break
    elif line:
        print(f"  Pico boot msg: {line}")

# -----------------------------------------------
# TUNING
# -----------------------------------------------
MAX_TURN        = 4.0
BASE_THROTTLE   = 6.5
SEARCH_TURN     = 4.0
DRIVE_STOP_DIST = 300
DRIVE_SLOW_DIST = 400
DRIVE_SLOW_MULT = 0.6

last_left  = None
last_right = None

def drive(throttle, turn):
    global last_left, last_right
    left_speed  = max(-100.0, min(100.0, throttle + turn))
    right_speed = max(-100.0, min(100.0, throttle - turn)) * -1

    if (last_left is not None and
        abs(left_speed  - last_left)  < 0.5 and
        abs(right_speed - last_right) < 0.5):
        return

    pico.write(f"DRIVE {left_speed} {right_speed}\n".encode('utf-8'))
    pico.flush()
    last_left  = left_speed
    last_right = right_speed
    with shared_lock:
        shared["left_pct"]  = left_speed
        shared["right_pct"] = right_speed

def stop_motors():
    global last_left, last_right
    pico.write(b"STOP\n")
    pico.flush()
    last_left  = None
    last_right = None
    with shared_lock:
        shared["left_pct"]  = 0.0
        shared["right_pct"] = 0.0

# -----------------------------------------------
# CAMERA + BALL TRACKING CONFIG
# -----------------------------------------------
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
BALL_MIN_AREA  = 300
AR_MIN         = 0.7
AR_MAX         = 4.5
FAR_THRESH     =  3000
MED_THRESH     = 20000
CLOSE_THRESH   = 40000
DEADZONE_FAR   = 0.10
DEADZONE_MED   = 0.20
DEADZONE_CLOSE = 0.25

# -----------------------------------------------
# CAMERA THREAD
# -----------------------------------------------
def camera_thread():
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(cfg)
    picam2.start()
    center_x   = FRAME_WIDTH // 2
    last_color = None

    try:
        while True:
            color_name, color_ranges = get_active_color()
            if color_name != last_color:
                print(f"Color changed -> {color_name}")
                last_color = color_name
            with shared_lock:
                shared["color"] = color_name

            frame = picam2.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # OR together all ranges (handles red's hue wrap around 180->0)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (low, high) in color_ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(low), np.array(high)))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour = None
            best_area    = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < BALL_MIN_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if not (AR_MIN <= w / float(h) <= AR_MAX):
                    continue
                if area > best_area:
                    best_area    = area
                    best_contour = cnt

            norm_error    = 0.0
            ball_visible  = False
            command       = "SEARCHING"
            tier          = "NONE"
            deadzone_frac = DEADZONE_MED

            if best_contour is not None:
                ball_visible = True
                x, y, w, h  = cv2.boundingRect(best_contour)
                obj_cx       = x + w // 2

                if best_area < FAR_THRESH:
                    tier = "FAR";   deadzone_frac = DEADZONE_FAR
                elif best_area < MED_THRESH:
                    tier = "MED";   deadzone_frac = DEADZONE_MED
                elif best_area > CLOSE_THRESH:
                    tier = "CLOSE"; deadzone_frac = DEADZONE_CLOSE
                else:
                    tier = "MED";   deadzone_frac = DEADZONE_MED

                deadzone_half = int(FRAME_WIDTH * deadzone_frac / 2)
                left_bound    = center_x - deadzone_half
                right_bound   = center_x + deadzone_half

                norm_error = float(np.clip((obj_cx - center_x) / center_x, -1.0, 1.0))

                if obj_cx < left_bound:
                    command = "LEFT"
                elif obj_cx > right_bound:
                    command = "RIGHT"
                else:
                    command    = "STRAIGHT"
                    norm_error = 0.0

            with shared_lock:
                shared["error"]   = norm_error
                shared["visible"] = ball_visible
                shared["command"] = command
                shared["tier"]    = tier
                shared["area"]    = int(best_area)

            time.sleep(0.01)
    finally:
        picam2.stop()

# -----------------------------------------------
# INITIALISE LIDAR
# -----------------------------------------------
print("Initializing Matrix LiDAR...")
while tof.begin() != 0:
    time.sleep(1)
tof.set_Ranging_Mode(8)
print("Sensor ready.\n")

# -----------------------------------------------
# START CAMERA THREAD
# -----------------------------------------------
cam_thread = threading.Thread(target=camera_thread, daemon=True)
cam_thread.start()
print("Camera thread started.\n")

# -----------------------------------------------
# MAIN LOOP
# -----------------------------------------------
current_state = "STOPPED"
try:
    print("--- Ball Tracking Active (headless) ---")
    while True:
        dist = get_filtered_distance()

        with shared_lock:
            norm_error   = shared["error"]
            ball_visible = shared["visible"]

        if ball_visible:
            if current_state != "DRIVING":
                print("Ball acquired — tracking.")
            turn = norm_error * MAX_TURN

            if dist is not None and dist <= DRIVE_STOP_DIST:
                print(f"Too close! ({dist:.0f} mm) — stopping.")
                stop_motors()
                current_state = "STOPPED"
            elif dist is not None and dist <= DRIVE_SLOW_DIST:
                drive(throttle=BASE_THROTTLE * DRIVE_SLOW_MULT, turn=turn)
                current_state = "DRIVING"
            else:
                drive(throttle=BASE_THROTTLE, turn=turn)
                current_state = "DRIVING"

        else:
            if current_state == "DRIVING":
                print("Ball lost — stopping.")
                stop_motors()
                current_state = "STOPPED"
            elif current_state in ("STOPPED", "SEARCHING"):
                if dist is not None and dist <= 200:
                    if current_state != "STOPPED":
                        print(f"Obstacle! ({dist:.1f} mm) — braking.")
                    stop_motors()
                    current_state = "STOPPED"
                else:
                    if current_state != "SEARCHING":
                        print("Spinning to search...")
                    drive(throttle=0, turn=SEARCH_TURN)
                    current_state = "SEARCHING"

        with shared_lock:
            shared["state"] = current_state
            shared["dist"]  = dist if dist is not None else 0.0

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nInterrupted. Stopping robot.")
    stop_motors()
    pico.close()
