import cv2
import numpy as np
from picamera2 import Picamera2

# -----------------------------------
# CAMERA SETUP
# -----------------------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(cfg)
picam2.start()

# -----------------------------------
# TRACKBAR CALLBACK
# -----------------------------------
def nothing(x):
    pass

cv2.namedWindow("HSV Tuner")

for name, default in zip(
    ["H Min","S Min","V Min","H Max","S Max","V Max"],
    [0,0,0,180,255,255]
):
    cv2.createTrackbar(name, "HSV Tuner", default, 255, nothing)
    
# Adjust H Max separately to 180
cv2.setTrackbarMax("H Max", "HSV Tuner", 180)

print("Press ESC when done tuning to print the final HSV range.\n")

while True:
    frame = picam2.capture_array()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Get slider positions
    h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
    s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
    v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
    h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
    s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
    v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Build mask
    mask = cv2.inRange(hsv, lower, upper)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Combine original and mask for visualization
    combined = np.hstack([bgr, mask_bgr])

    cv2.imshow("HSV Tuner", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# Print the final tuned values
print("\nFinal HSV range:")
print(f"Lower = ({h_min}, {s_min}, {v_min})")
print(f"Upper = ({h_max}, {s_max}, {v_max})")

cv2.destroyAllWindows()
picam2.stop()
