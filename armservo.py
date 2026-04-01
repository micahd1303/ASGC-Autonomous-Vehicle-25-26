from gpiozero import PWMOutputDevice
from time import sleep

SERVO1_PIN = 17
SERVO2_PIN = 18

servo1 = PWMOutputDevice(SERVO1_PIN, frequency=200)
servo2 = PWMOutputDevice(SERVO2_PIN, frequency=200)

PERIOD_US = 5000  

def send_pulse(servo, micros):
    duty = micros / PERIOD_US
    servo.value = duty

# ==========================================================
# 🛑 CALIBRATION ZONE 🛑
# You must find these 4 exact numbers for your physical setup
# ==========================================================

# Servo 1 (e.g., Left Side)
S1_LOWER = 900    # Pulse for Servo 1 when arm is fully DOWN
S1_RAISE = 2100   # Pulse for Servo 1 when arm is fully UP

# Servo 2 (e.g., Right Side)
# These will likely NOT be exactly 3000 - S1_LOWER. 
# Adjust these until they physically match Servo 1 perfectly.
S2_LOWER = 2060   # Pulse for Servo 2 when arm is fully DOWN
S2_RAISE = 850    # Pulse for Servo 2 when arm is fully UP

# ==========================================================

def move_arm_sync(direction="UP", steps=100, delay=0.015):
    """
    Moves both servos smoothly by calculating their positions as a percentage 
    from 0.0 to 1.0. This guarantees they start and stop at the exact same time, 
    even if their total microsecond travel distances are different.
    """
    for i in range(steps + 1):
        # If going UP, percentage goes 0.0 -> 1.0. If DOWN, 1.0 -> 0.0
        percent = (i / steps) if direction == "UP" else (1.0 - (i / steps))
        
        # Calculate the exact microsecond for each servo at this percentage
        s1_current = S1_LOWER + ((S1_RAISE - S1_LOWER) * percent)
        s2_current = S2_LOWER + ((S2_RAISE - S2_LOWER) * percent)
        
        send_pulse(servo1, s1_current)
        send_pulse(servo2, s2_current)
        
        sleep(delay)

try:
    # Optional: Send them to the lower position immediately on startup
    move_arm_sync(direction="DOWN", steps=10, delay=0.05)
    sleep(1)

    while True:
        print("Synchronized Raise...")
        move_arm_sync(direction="UP", steps=100, delay=0.015)
        sleep(2)
        
        print("Synchronized Lower...")
        move_arm_sync(direction="DOWN", steps=100, delay=0.015)
        sleep(2)

finally:
    servo1.off()
    servo2.off()