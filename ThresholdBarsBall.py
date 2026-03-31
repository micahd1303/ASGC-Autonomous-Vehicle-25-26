import cv2
import numpy as np
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
# WINDOW SETUP
# -------------------------------
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

def nothing(x):
    pass

# -------------------------------
# HSV TRACKBARS (GREEN DEFAULTS)
# -------------------------------
cv2.createTrackbar("H Min", "Controls", 40, 179, nothing)
cv2.createTrackbar("H Max", "Controls", 90, 179, nothing)

cv2.createTrackbar("S Min", "Controls", 150, 255, nothing)
cv2.createTrackbar("S Max", "Controls", 255, 255, nothing)

cv2.createTrackbar("V Min", "Controls", 30, 255, nothing)
cv2.createTrackbar("V Max", "Controls", 120, 255, nothing)

# -------------------------------
# BALL SHAPE TUNING
# -------------------------------
cv2.createTrackbar("Min Area", "Controls", 150, 5000, nothing)

cv2.createTrackbar("AR Min x100", "Controls", 85, 200, nothing)
cv2.createTrackbar("AR Max x100", "Controls", 115, 200, nothing)

print("HSV + Shape tuning started")
print("Press 'q' to quit and print final values")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Read HSV sliders
    h_min = cv2.getTrackbarPos("H Min", "Controls")
    h_max = cv2.getTrackbarPos("H Max", "Controls")
    s_min = cv2.getTrackbarPos("S Min", "Controls")
    s_max = cv2.getTrackbarPos("S Max", "Controls")
    v_min = cv2.getTrackbarPos("V Min", "Controls")
    v_max = cv2.getTrackbarPos("V Max", "Controls")

    # Read shape sliders
    min_area = cv2.getTrackbarPos("Min Area", "Controls")
    ar_min = cv2.getTrackbarPos("AR Min x100", "Controls") / 100.0
    ar_max = cv2.getTrackbarPos("AR Max x100", "Controls") / 100.0

    # Build mask
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Morph cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h)

        if not (ar_min <= ar <= ar_max):
            continue

        cv2.rectangle(bgr, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            bgr,
            f"AR={ar:.2f} A={int(area)}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # Show windows
    cv2.imshow("Detection", bgr)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# PRINT FINAL VALUES
# -------------------------------
print("\nFINAL TUNED VALUES:")
print(f"GREEN HSV LOW  = ({h_min}, {s_min}, {v_min})")
print(f"GREEN HSV HIGH = ({h_max}, {s_max}, {v_max})")
print(f"BALL MIN AREA = {min_area}")
print(f"BALL AR RANGE = ({ar_min}, {ar_max})")

cv2.destroyAllWindows()
picam2.stop()
