import sys
import time
import csv
from collections import deque
sys.path.append("/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python")
from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c

I2C_ADDRESS = 0x33
WINDOW_SIZE = 10
LOG_FILE = "tof_calibration.csv"

tof = DFRobot_matrixLidar_i2c(I2C_ADDRESS)
history = deque(maxlen=WINDOW_SIZE)
center_indices = [35, 36, 43, 44]

def get_center_pixels(raw_data):
    distances = [(raw_data[i+1]<<8)|raw_data[i] for i in range(0,len(raw_data),2)]
    return [distances[i] for i in center_indices]

def run_calibration(actual_distance_mm):
    with open(LOG_FILE,"a",newline="") as f:
        writer = csv.writer(f)
        for _ in range(200):  # ~10s of data at 50ms intervals
            data = tof.get_all_data()
            if not data: 
                continue
            pixels = get_center_pixels(data)
            raw_avg = sum(pixels)/len(pixels)
            history.append(raw_avg)
            filt_avg = sum(history)/len(history)
            writer.writerow([actual_distance_mm,*pixels,raw_avg,filt_avg])
            time.sleep(0.05)

if __name__=="__main__":
    while tof.begin()!=0: time.sleep(1)
    tof.set_Ranging_Mode(8)
    # Write header
    with open(LOG_FILE,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual_mm","Pixel35","Pixel36","Pixel43","Pixel44","RawAvg","FiltAvg"])
    # Example distances to test
    for d in [100,200,300,400,500,600,700]:
        input(f"Place ball at {d} mm and press Enter...")
        run_calibration(d)
    print("Calibration complete!")
