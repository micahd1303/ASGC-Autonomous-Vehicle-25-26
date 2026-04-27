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

print("System running...")

try:
    while True:

        frame = picam2.capture_array()

        # flip camera 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        data = tof.get_all_data()

        if data:

            distances = parse_distances(data)

            h, w, _ = frame.shape
            cell_w = w // GRID_SIZE
            cell_h = h // GRID_SIZE

            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):

                    idx = row * GRID_SIZE + col
                    dist = distances[idx]

                    x1 = col * cell_w
                    y1 = row * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h

                    # draw grid box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

                    # draw distance text
                    text = str(dist)
                    cv2.putText(
                        frame,
                        text,
                        (x1+10, y1+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        1,
                        cv2.LINE_AA
                    )

        cv2.imshow("Camera + TOF Overlay", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cv2.destroyAllWindows()
