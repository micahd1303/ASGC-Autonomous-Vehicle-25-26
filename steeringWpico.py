import sys
import time
import serial
import threading
import cv2
import numpy as np
from collections import deque
from picamera2 import Picamera2

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
# MOTOR SETUP (Pico via Serial)
# -----------------------------------------------
print("Connecting to Pico...")
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=10)

print("Waiting for Pico to boot...")
while True:
    line = pico.readline().decode('utf-8', errors='replace').strip()
    if line == "PICO_READY":
        print("Pico ready.")
        break
    elif line:
        print(f"  Pico boot msg: {line}")

def drive(throttle, turn):
    left_speed  = max(-100.0, min(100.0, throttle + turn)) * -1  # mechanical flip
    right_speed = max(-100.0, min(100.0, throttle - turn))
    pico.write(f"DRIVE {left_speed} {right_speed}\n".encode('utf-8'))
    pico.flush()
    with shared_lock:
        shared["left_pct"]  = left_speed
        shared["right_pct"] = right_speed

def stop_motors():
    pico.write(b"STOP\n")
    pico.flush()
    with shared_lock:
        shared["left_pct"]  = 0.0
        shared["right_pct"] = 0.0

# -----------------------------------------------
# CAMERA + BALL TRACKING CONFIG
# -----------------------------------------------
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

BLUE_LOW  = (100, 150,   0)
BLUE_HIGH = (140, 255, 255)

BALL_MIN_AREA = 300
AR_MIN = 0.7
AR_MAX = 4.5

FAR_THRESH   = 3000
MED_THRESH   = 20000
CLOSE_THRESH = 40000

DEADZONE_FAR   = 0.10
DEADZONE_MED   = 0.20
DEADZONE_CLOSE = 0.35

# -----------------------------------------------
# TUNING
# -----------------------------------------------
MAX_TURN         = 10.0   # percent — gentle steering response
BASE_THROTTLE    = 5.0    # percent forward speed while tracking
SEARCH_TURN      = 10.0   # percent — spin speed when searching
                           # flip sign to prefer spinning left
STEER_BAR_PIXELS = 250

# -----------------------------------------------
# SHARED STATE
# camera thread writes: error, visible, command, tier, area, debug_frame
# main loop writes:     state, dist, left_pct, right_pct
# ALL imshow calls happen only in the main loop
# -----------------------------------------------
shared_lock = threading.Lock()
shared = {
    "error":       0.0,
    "visible":     False,
    "command":     "SEARCHING",
    "tier":        "NONE",
    "area":        0,
    "debug_frame": None,   # <-- camera thread puts the BGR frame here
    "state":       "STOPPED",
    "dist":        0.0,
    "left_pct":    0.0,
    "right_pct":   0.0,
}

# -----------------------------------------------
# CAMERA THREAD  (no imshow here — only frame prep)
# -----------------------------------------------
def camera_thread():
    picam2 = Picamera2()
    cfg    = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(cfg)
    picam2.start()

    center_x = FRAME_WIDTH // 2

    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr,   cv2.COLOR_BGR2HSV)

        # --- Mask (contours only, never displayed) ---
        mask   = cv2.inRange(hsv, np.array(BLUE_LOW), np.array(BLUE_HIGH))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- Contours ---
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

        # --- Error + debug frame ---
        debug         = bgr.copy()
        norm_error    = 0.0
        ball_visible  = False
        command       = "SEARCHING"
        tier          = "NONE"
        deadzone_frac = DEADZONE_MED

        if best_contour is not None:
            ball_visible = True
            x, y, w, h  = cv2.boundingRect(best_contour)
            obj_cx       = x + w // 2

            if   best_area < FAR_THRESH:   tier = "FAR";   deadzone_frac = DEADZONE_FAR
            elif best_area < MED_THRESH:   tier = "MED";   deadzone_frac = DEADZONE_MED
            elif best_area > CLOSE_THRESH: tier = "CLOSE"; deadzone_frac = DEADZONE_CLOSE
            else:                          tier = "MED";   deadzone_frac = DEADZONE_MED

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

            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug, (obj_cx, y + h // 2), 6, (0, 255, 255), -1)

        else:
            deadzone_half = int(FRAME_WIDTH * deadzone_frac / 2)
            left_bound    = center_x - deadzone_half
            right_bound   = center_x + deadzone_half

        # --- Zone overlay ---
        overlay = debug.copy()
        cv2.rectangle(overlay, (0, 0),           (left_bound,  FRAME_HEIGHT), (255,   0, 0), -1)
        cv2.rectangle(overlay, (left_bound, 0),  (right_bound, FRAME_HEIGHT), (0,   255, 0), -1)
        cv2.rectangle(overlay, (right_bound, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0,     0, 255), -1)
        debug = cv2.addWeighted(overlay, 0.15, debug, 0.85, 0)
        cv2.line(debug, (left_bound,  0), (left_bound,  FRAME_HEIGHT), (255, 255, 255), 2)
        cv2.line(debug, (right_bound, 0), (right_bound, FRAME_HEIGHT), (255, 255, 255), 2)

        # --- Steering bar ---
        bar_len = int(norm_error * STEER_BAR_PIXELS)
        cv2.line(debug,
                 (center_x, FRAME_HEIGHT - 40),
                 (center_x + bar_len, FRAME_HEIGHT - 40),
                 (0, 255, 255), 6)

        # --- Text ---
        cv2.putText(debug, f"Cmd: {command}",        (40,  60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(debug, f"Tier: {tier}",           (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,   255, 255), 2)
        cv2.putText(debug, f"Area: {int(best_area)}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,   255, 255), 2)
        cv2.putText(debug, f"Err: {norm_error:+.2f}", (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,   255, 255), 2)

        # --- Push everything to shared (including the finished frame) ---
        with shared_lock:
            shared["error"]       = norm_error
            shared["visible"]     = ball_visible
            shared["command"]     = command
            shared["tier"]        = tier
            shared["area"]        = int(best_area)
            shared["debug_frame"] = debug   # main loop will imshow this

    picam2.stop()

# -----------------------------------------------
# DASHBOARD DRAW HELPER
# -----------------------------------------------
STATE_COLORS = {
    "DRIVING":   (0,   200,   0),
    "SEARCHING": (0,   200, 255),
    "STOPPED":   (0,     0, 200),
}

def draw_dashboard(snap):
    W, H  = 500, 360
    dash  = np.zeros((H, W, 3), dtype=np.uint8)
    color = STATE_COLORS.get(snap["state"], (180, 180, 180))

    # state banner
    cv2.rectangle(dash, (0, 0), (W, 80), color, -1)
    cv2.putText(dash, snap["state"], (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

    # data rows
    rows = [
        ("Cam cmd",  snap["command"]),
        ("Tier",     snap["tier"]),
        ("Area",     str(snap["area"])),
        ("Err",      f'{snap["error"]:+.3f}'),
        ("LiDAR",    f'{snap["dist"]:.0f} mm'),
        ("L motor",  f'{snap["left_pct"]:+.1f}%'),
        ("R motor",  f'{snap["right_pct"]:+.1f}%'),
    ]
    for i, (label, value) in enumerate(rows):
        y = 120 + i * 34
        cv2.putText(dash, f"{label}:", (20,  y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1)
        cv2.putText(dash, value,       (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # motor bar
    bar_y = H - 30
    mid   = W // 2
    for pct, side in [(snap["left_pct"], -1), (snap["right_pct"], 1)]:
        blen = min(int(abs(pct) / 100.0 * (W // 2 - 20)), W // 2 - 20)
        bcol = (0, 220, 0) if pct > 0 else (0, 0, 220) if pct < 0 else (80, 80, 80)
        cv2.line(dash, (mid, bar_y), (mid + side * blen, bar_y), bcol, 12)
    cv2.line(dash, (mid, bar_y - 10), (mid, bar_y + 10), (200, 200, 200), 2)

    cv2.imshow("Robot Status", dash)

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

# Create both windows in the MAIN thread before the loop
cv2.namedWindow("Steering Debug", cv2.WINDOW_NORMAL)
cv2.namedWindow("Robot Status",   cv2.WINDOW_NORMAL)

# -----------------------------------------------
# MAIN LOOP  — all imshow calls live here
# -----------------------------------------------
current_state = "STOPPED"

try:
    print("--- Ball Tracking + Flinch Reflex Active ---")
    while True:
        dist = get_filtered_distance()

        with shared_lock:
            norm_error   = shared["error"]
            ball_visible = shared["visible"]
            debug_frame  = shared["debug_frame"]

        # ---- BALL VISIBLE ----
        if ball_visible:
            if current_state != "DRIVING":
                print("Ball acquired — tracking.")
            turn = norm_error * MAX_TURN
            drive(throttle=BASE_THROTTLE, turn=turn)
            current_state = "DRIVING"

        # ---- BALL LOST ----
        else:
            if current_state == "DRIVING":
                print("Ball lost — stopping.")
                stop_motors()
                current_state = "STOPPED"

            elif current_state in ("STOPPED", "SEARCHING"):
                if dist is not None and dist <= 200:
                    if current_state != "STOPPED":
                        print(f"Obstacle! ({dist:.1f} mm) — BRAKING.")
                        stop_motors()
                        current_state = "STOPPED"
                else:
                    if current_state != "SEARCHING":
                        print("Spinning to search...")
                    drive(throttle=0, turn=SEARCH_TURN)
                    current_state = "SEARCHING"

        # ---- Draw both windows from main thread ----
        with shared_lock:
            shared["state"] = current_state
            shared["dist"]  = dist if dist is not None else 0.0
            snap = dict(shared)

        if debug_frame is not None:
            cv2.imshow("Steering Debug", debug_frame)

        draw_dashboard(snap)
        cv2.waitKey(1)   # single pump for all windows

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterrupted. Stopping robot.")
    stop_motors()
    pico.close()
    cv2.destroyAllWindows()
