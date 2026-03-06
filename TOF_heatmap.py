import sys
import time
import cv2
import numpy as np
from picamera2 import Picamera2

sys.path.append("/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python")
from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c

# TOF setup
I2C_ADDRESS = 0x33
tof = DFRobot_matrixLidar_i2c(I2C_ADDRESS)
GRID_SIZE = 8

# Camera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280,720)})
picam2.configure(config)
picam2.start()

def parse_distances(raw_data):
    """Convert raw TOF data bytes to integer distances"""
    distances = []
    for i in range(0, len(raw_data), 2):
        d = (raw_data[i+1] << 8) | raw_data[i]
        distances.append(d)
    return distances

# Initialize TOF sensor
while tof.begin() != 0:
    print("Waiting for TOF sensor...")
    time.sleep(1)
tof.set_Ranging_Mode(8)
print("System running...")

try:
    while True:
        # Camera feed
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera Feed", frame)

        # TOF heatmap
        data = tof.get_all_data()
        if data:
            distances = parse_distances(data)
            # Convert to 8x8 array
            dist_array = np.array(distances, dtype=np.float32).reshape((GRID_SIZE, GRID_SIZE))

            # Normalize for colormap
            min_d = np.min(dist_array)
            max_d = np.max(dist_array)
            norm = ((dist_array - min_d) / (max_d - min_d + 1e-5) * 255).astype(np.uint8)

            # Apply color map (JET: blue = far, red = close)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

            # Resize to match better aspect ratio (sensor is wider than taller)
            heatmap_resized = cv2.resize(heatmap, (400, 200), interpolation=cv2.INTER_NEAREST)

            # Show heatmap
            cv2.imshow("TOF Heatmap", heatmap_resized)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cv2.destroyAllWindows()
