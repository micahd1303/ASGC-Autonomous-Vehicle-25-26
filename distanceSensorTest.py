# -*- coding: utf-8 -*-
import sys
import os
import time

# 1. Point Python to the DFRobot library location
LIBRARY_PATH = "/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python"
if LIBRARY_PATH not in sys.path:
    sys.path.append(LIBRARY_PATH)

# Now we can import the DFRobot modules
try:
    from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c
except ImportError:
    print(f"Error: Could not find DFRobot library at {LIBRARY_PATH}")
    sys.exit(1)

# Sensor configuration
I2C_ADDRESS = 0x33
tof = DFRobot_matrixLidar_i2c(I2C_ADDRESS)

def setup_sensor():
    print("Connecting to VL53L7CX...")
    while tof.begin() != 0:
        print("Sensor communication failed! Check wiring/address.")
        time.sleep(1)
    
    # Set to 8x8 mode (parameter 8 = 8x8, parameter 4 = 4x4)
    while tof.set_Ranging_Mode(8) != 0:
        print("Failed to set 8x8 mode.")
        time.sleep(1)
    print("Sensor Initialized Successfully!")

def run_test():
    try:
        while True:
            # Retrieve the raw byte data
            data = tof.get_all_data()
            
            # Process bytes into distance values (mm)
            distances = []
            for i in range(0, len(data) - 1, 2):
                low_byte = data[i]
                high_byte = data[i + 1]
                combined = (high_byte << 8) | low_byte
                distances.append(combined)

            # Clear terminal for a 'live' view effect
            print("\033c", end="") 
            print("--- VL53L7CX 8x8 Matrix (Distance in mm) ---")
            
            # Print as an 8x8 grid
            for row in range(8):
                row_vals = distances[row*8 : (row+1)*8]
                # Format each number to be 5 characters wide for alignment
                print(" ".join(f"{val:4}" for val in row_vals))
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nTesting stopped by user.")

if __name__ == "__main__":
    setup_sensor()
    run_test()
