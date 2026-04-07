import sys
import time
import serial
from collections import deque

# --- SENSOR SETUP ---
sys.path.append("/home/asgc/ASGC-Autonomous-Vehicle-25-26/DFRobot_MatrixLidar/python")
from raspberry.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c

I2C_ADDRESS = 0x33
WINDOW_SIZE = 5
tof = DFRobot_matrixLidar_i2c(I2C_ADDRESS)
history = deque(maxlen=WINDOW_SIZE)
center_indices = [35, 36, 43, 44]

def get_center_pixels(raw_data):
    distances = [(raw_data[i+1]<<8)|raw_data[i] for i in range(0,len(raw_data),2)]
    return [distances[i] for i in center_indices]

def get_filtered_distance():
    data = tof.get_all_data()
    if not data:
        return None
    pixels = get_center_pixels(data)
    raw_avg = sum(pixels) / len(pixels)
    history.append(raw_avg)
    return sum(history) / len(history)

# --- MOTOR SETUP ---
print("Connecting to Pico...")
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2) # Let Pico boot

def drive(throttle, turn):
    left_speed = max(-100.0, min(100.0, throttle + turn))
    right_speed = max(-100.0, min(100.0, (throttle - turn) * -1)) # Mechanical flip
    
    command_str = f"DRIVE {left_speed} {right_speed}\n"
    pico.write(command_str.encode('utf-8'))
    pico.flush()

def stop_motors():
    pico.write(b"STOP\n")
    pico.flush()

# --- INITIALIZATION ---
print("Initializing Matrix LiDAR...")
while tof.begin() != 0:
    time.sleep(1)
tof.set_Ranging_Mode(8)
print("Sensor ready.\n")

# --- MAIN LOOP ---
current_state = "STOPPED" # Track state so we don't spam the serial port

try:
    print("--- Flinch Reflex Active ---")
    while True:
        dist = get_filtered_distance()
        
        if dist is not None:
            # 1. THE TRIGGER: Obstacle detected within 200mm
            if dist <= 200 and current_state != "STOPPED":
                print(f"Obstacle Detected! ({dist:.1f} mm). BRAKING.")
                stop_motors()
                current_state = "STOPPED"
                
            # 2. THE ALL-CLEAR: Path is clear, resume driving at 10%
            elif dist > 200 and current_state != "DRIVING":
                print(f"Path Clear ({dist:.1f} mm). Driving Forward at 10%.")
                drive(throttle=10, turn=0)  # <-- UPDATED: Default throttle to 10%
                current_state = "DRIVING"
                
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nScript manually interrupted. Stopping robot.")
    stop_motors()
    pico.close()
