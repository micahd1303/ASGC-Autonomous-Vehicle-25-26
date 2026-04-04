import serial
import time

print("Connecting to Pico...")
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# Wait 2 seconds for the Pico to boot
time.sleep(2) 

def motorspeed(target_duty):
    """
    Sends a target duty cycle to the Pico. 
    The Pico will automatically ramp to this speed over 1 second.
    """
    command_str = f"SPEED {target_duty}\n"
    print(f"Commanding: {command_str.strip()}")
    
    # Send the string down the USB cable
    pico.write(command_str.encode('utf-8'))
    pico.flush()

try:
    print("\n--- Starting Acceleration Test ---")
    
    # Command the Pico to ramp up to 5100
    motorspeed(5100)
    
    # The Pi 5 waits 4 seconds total here. 
    # (1 second for the Pico to finish ramping, plus 3 seconds of driving).
    time.sleep(4)
    
    # Command the Pico to ramp gracefully back down to true neutral
    print("\n--- Ramping Down ---")
    motorspeed(4915)
    
    # Wait 2 seconds to let the deceleration finish before closing the script
    time.sleep(2)

finally:
    pico.close()
    print("Test complete.")
