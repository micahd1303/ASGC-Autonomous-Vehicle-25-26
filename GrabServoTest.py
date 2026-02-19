from gpiozero import PWMOutputDevice
from time import sleep

# 50 Hz servo signal on GPIO 17
servo = PWMOutputDevice(17, frequency=50)

# 1500 µs pulse width
pulse_width_us = 800
period_us = 1_000_000 / 50  # 20,000 µs
duty_cycle = pulse_width_us / period_us

servo.value = duty_cycle  # send 1500 µs
sleep(2)                  # hold for 2 seconds
servo.off()               # stop signal
