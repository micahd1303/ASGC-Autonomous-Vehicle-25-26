import serial
import time

print("Connecting to Pico...")
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2) 

def drive(throttle, turn):
    """
    throttle: Positive = Forward, Negative = Reverse
    turn:     Positive = Right,   Negative = Left
    """
    # 1. Differential Steering Math
    left_speed = throttle + turn
    right_speed = throttle - turn
    
    # 2. Mechanical Flip (Invert right motor for chassis layout)
    right_speed = right_speed * -1
    
    # 3. Safety Clamp (Cap at +/- 100%)
    left_speed = max(-100.0, min(100.0, left_speed))
    right_speed = max(-100.0, min(100.0, right_speed))
    
    # 4. Fire the command down the wire
    command_str = f"DRIVE {left_speed} {right_speed}\n"
    print(f"\nCommanding: Throttle={throttle} Turn={turn}")
    print(f"-> Wheels: L={left_speed}% R={right_speed}%")
    
    pico.write(command_str.encode('utf-8'))
    pico.flush()
    
    # CRITICAL: Wait 1 second for the Pico to finish its ramping sequence
    time.sleep(1.05) 

try:
    print("\n--- Testing Steering + Ramping ---")
    
    print("\n1. Accelerating Straight Forward")
    drive(throttle=15, turn=0)
    time.sleep(2) # Drive for 2 seconds at target speed
    
    print("\n2. Transitioning to Sweeping Right Curve")
    # Left wheel stays at 15, right wheel ramps down to 5
    drive(throttle=10, turn=5)
    time.sleep(2)
    
    print("\n3. Transitioning to Spin in Place (Right)")
    # Left wheel ramps up to 15, right wheel ramps into reverse (-15)
    drive(throttle=0, turn=15)
    time.sleep(2)
    
    print("\n4. Ramping to Full Stop")
    drive(throttle=0, turn=0)

finally:
    pico.close()
    print("\nTest complete. Port closed.")
