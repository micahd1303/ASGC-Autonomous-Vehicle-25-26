import sys
import time
from collections import deque
# Path to your DFRobot library
sys.path.append("/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python")
from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c

# Settings
WINDOW_SIZE = 5  # Average of last 5 frames (~0.25 seconds of data)
I2C_ADDRESS = 0x33
tof = DFRobot_matrixLidar_i2c(I2C_ADDRESS)

# This stores our sliding window of averages
history = deque(maxlen=WINDOW_SIZE)

def get_center_average(raw_data):
    distances = []
    for i in range(0, len(raw_data), 2):
        distances.append((raw_data[i+1] << 8) | raw_data[i])
    # Center 4 pixels of the 8x8 grid
    center_indices = [27, 28, 35, 36]
    return sum(distances[idx] for idx in center_indices) / 4.0

def run_mission():
    print("System Online. Filtering active.")
    try:
        while True:
            data = tof.get_all_data()
            if not data: continue

            # 1. Get current raw center average
            current_raw = get_center_average(data)
            
            # 2. Add to history (automatically pushes out oldest if size > 5)
            history.append(current_raw)
            
            # 3. Calculate filtered average
            filtered_avg = sum(history) / len(history)

            # Logic Triggers (Using Filtered Data)
            if len(history) == WINDOW_SIZE: # Wait until buffer is full
                if filtered_avg <= 200:
                    print(f"DEBUG: RAW={current_raw:.0f} | FILT={filtered_avg:.0f} >>> INITIATE CLAW <<<")
                elif filtered_avg <= 700:
                    print(f"DEBUG: RAW={current_raw:.0f} | FILT={filtered_avg:.0f} >>> BALL FOUND <<<")
                else:
                    print(f"Scanning... Clear ({filtered_avg:.0f}mm)   ", end='\r')
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    while tof.begin() != 0: time.sleep(1)
    tof.set_Ranging_Mode(8)
    run_mission()
