import sys
import time
import cv2
import numpy as np
from picamera2 import Picamera2

sys.path.append("/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python")
from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c

I2C_ADDRESS = 0x33
tof = DFRobot_matrixLidar_i2c(I2C_ADDRESS)

# Camera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280,720)})
picam2.configure(config)

picam2.start()

GRID_SIZE = 8

def parse_distances(raw_data):
    distances = []
    for i in range(0, len(raw_data), 2):
        d = (raw_data[i+1] << 8) | raw_data[i]
        distances.append(d)
    return distances

while tof.begin() != 0:
    print("Waiting for TOF sensor...")
    time.sleep(1)

tof.set_Ranging_Mode(8)
print("System running. Press 's' to save a screenshot, 'Esc' to exit.")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        data = tof.get_all_data()

        if data:
            distances = parse_distances(data)

            h, w, _ = frame.shape
            grid_size = min(w, h)
            start_x = (w - grid_size) // 2
            start_y = (h - grid_size) // 2
            cell = grid_size // GRID_SIZE

            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    idx = row * GRID_SIZE + col
                    dist = distances[idx]

                    x1 = start_x + col*cell
                    y1 = start_y + row*cell
                    x2 = x1 + cell
                    y2 = y1 + cell

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                    cv2.putText(
                        frame,
                        str(dist),
                        (x1+5, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0,255,0),
                        1,
                        cv2.LINE_AA
                    )

        cv2.imshow("Camera + TOF Overlay", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc to exit
            break
        elif key == ord('s'):  # 's' to save screenshot
            filename = input("Enter filename for screenshot (include .png): ")
            if filename.strip() == "":
                print("Invalid filename, skipping.")
            else:
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")

finally:
    cv2.destroyAllWindows()
