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

def get_distances():
    """Returns both the instant raw distance and the smoothed history distance."""
    data = tof.get_all_data()
    if not data:
        return None, None
        
    pixels = get_center_pixels(data)
    raw_avg = sum(pixels) / len(pixels) # The instant "Now"
    
    history.append(raw_avg)
    filt_avg = sum(history) / len(history) # The smoothed "Past"
    
    return raw_avg, filt_avg

# --- MOTOR SETUP ---
print("Connecting to Pico...")
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2) # Let Pico boot

def drive(throttle, turn):
    # Forward orientation fix: Left wheel is inverted, Right wheel is standard
    left_speed = max(-100.0, min(100.0, (throttle + turn) * -1)) 
    right_speed = max(-100.0, min(100.0, (throttle - turn)))     
    
    command_str = f"DRIVE {left_speed} {right_speed}\n"
    pico.write(command_str.encode('utf-8'))
    pico.flush()

def estop_motors():
    """Sends the zero-latency emergency brake command."""
    pico.write(b"ESTOP\n")
    pico.flush()

# --- INITIALIZATION ---
print("Initializing Matrix LiDAR...")
while tof.begin() != 0:
    time.sleep(1)
tof.set_Ranging_Mode(8)
print("Sensor ready.\n")

# --- MAIN LOOP ---
current_state = "STOPPED" 

try:
    print("--- 500mm Distance Driver Active ---")
    while True:
        raw_dist, filt_dist = get_distances()
        
        # --- NEW DEBUG LINE ---
        if raw_dist is None:
            print("DEBUG: Sensor returned None (No data)")
        else:
            print(f"DEBUG: Raw: {raw_dist:.1f}mm | Filtered: {filt_dist:.1f}mm | State: {current_state}")
        # ----------------------
        
        if raw_dist is not None:
            # 1. THE TRIGGER
            if raw_dist <= 200 and current_state != "STOPPED":
                print(f">>> Obstacle! Raw: {raw_dist:.1f}mm. SLAMMING BRAKES.")
                estop_motors()
                current_state = "STOPPED"
                
            # 2. THE ALL-CLEAR
            elif filt_dist > 200 and current_state != "DRIVING":
                print(f">>> Path Clear. Filtered: {filt_dist:.1f}mm. Driving Forward.")
                drive(throttle=10, turn=0)  
                current_state = "DRIVING"
                
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\nScript manually interrupted. Stopping robot.")
    estop_motors()
    pico.close()
