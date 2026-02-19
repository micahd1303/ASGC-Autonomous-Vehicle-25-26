#from gpiozero import PWMOutputDevice
#from time import sleep

# 50 Hz servo signal on GPIO 17
#servo = PWMOutputDevice(17, frequency=50)

# 800 µs pulse width
#pulse_width_us = 2000
#period_us = 1_000_000 / 50  # 20,000 µs
#duty_cycle = pulse_width_us / period_us

#servo.value = duty_cycle  # send 800 µs
#sleep(2)                  # hold for .2 seconds
#servo.off()               # stop signal

from gpiozero import PWMOutputDevice
from time import sleep

# DS3218 supports 50-330Hz; 50Hz is standard 
servo = PWMOutputDevice(17, frequency=50)

def set_servo_micros(micros):
    period_us = 20_000  # 20ms period for 50Hz
    duty_cycle = micros / period_us
    servo.value = duty_cycle
    print(f"Sending {micros}µs pulse")

try:
    # 1. Close to the bottom (datasheet min is 500µs )
    set_servo_micros(700) 
    sleep(3)

    # 2. Middle / Neutral (datasheet neutral is 1500µs )
    set_servo_micros(1500)
    sleep(3)

    # 3. Close to the top (datasheet max is 2500µs )
    set_servo_micros(2300)
    sleep(3)

finally:
    servo.off() # Stop signal to let the motor rest
