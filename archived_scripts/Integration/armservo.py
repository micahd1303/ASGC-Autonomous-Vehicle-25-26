from gpiozero import PWMOutputDevice
from time import sleep

# Frequency for DS3218 (50-330Hz supported)
FREQ = 200

# Initialize servos on GPIO 17 and GPIO 27
servo1 = PWMOutputDevice(17, frequency=FREQ)
servo2 = PWMOutputDevice(27, frequency=FREQ)

def set_servo_micros(servo_device, micros):
    """
    Sets the pulse width in microseconds for a specific servo.
    period_us = 1,000,000 / frequency
    """
    period_us = 1_000_000 / FREQ 
    duty_cycle = micros / period_us
    
    # Ensure duty cycle stays within 0.0 and 1.0
    servo_device.value = max(0, min(1, duty_cycle))
    print(f"GPIO {servo_device.pin.number}: Sending {micros}µs pulse")

try:
    # --- Example Sequence ---
	
    
    # Move Servo 1 to 2350µs and Servo 2 to 650µs
    set_servo_micros(servo1, 2350)
    set_servo_micros(servo2, 650) 
    sleep(3)
    
    # Move Servo 1 to 650µs and Servo 2 to 2350µs
    set_servo_micros(servo1, 650)
    set_servo_micros(servo2, 2350)
    sleep(3)
    
        # Move Servo 1 to 2350µs and Servo 2 to 650µs
    set_servo_micros(servo1, 2350)
    set_servo_micros(servo2, 650) 
    sleep(3)
    
    # Move Servo 1 to 650µs and Servo 2 to 2350µs
    set_servo_micros(servo1, 650)
    set_servo_micros(servo2, 2350)
    sleep(3)
    
  

finally:
    # Clean up: stop signals for both
    servo1.off()
    servo2.off()
    print("Servos detached.")
