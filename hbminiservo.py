from gpiozero import PWMOutputDevice
from time import sleep

# Define the GPIO pins for your 4 servos (Replaced 17 and 18 with 23 and 24)
SERVO_PINS = [23, 24, 27, 22] 

# Initialize PWM devices at 50 Hz (per MG90S datasheet)
servos = [PWMOutputDevice(pin, frequency=50) for pin in SERVO_PINS]

PERIOD_US = 20000  # 20 ms period for 50 Hz

# Pulse widths in microseconds based on the MG90S datasheet
POS_0_DEG = 1500  # Middle position (1.5 ms)
POS_90_DEG = 2000 # All the way to the right (~2.0 ms)

def set_angle(servo, micros):
    """Converts the microsecond pulse to a duty cycle percentage"""
    duty = micros / PERIOD_US
    servo.value = duty

# Variable to track the current state of the servos
is_toggled = False

try:
    print("Initializing servos to 0 degrees (middle)...")
    for servo in servos:
        set_angle(servo, POS_0_DEG)
    
    # Give the servos a moment to reach their starting position
    sleep(1) 

    print("\n--- Servo Toggle Control ---")
    print("Press 'Enter' to toggle positions. Press 'Ctrl+C' to quit.")

    while True:
        input() # Wait for the user to press Enter
        
        # Flip the toggle state
        is_toggled = not is_toggled

        # Determine the target pulse width based on the toggle state
        target_micros = POS_90_DEG if is_toggled else POS_0_DEG
        state_name = "+90 degrees" if is_toggled else "0 degrees"

        print(f"Moving all servos to {state_name} ({target_micros} µs)...")

        # Move all four servos simultaneously
        for servo in servos:
            set_angle(servo, target_micros)

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    # Safely power down the PWM signals
    for servo in servos:
        servo.off()
    print("Servos powered off safely.")