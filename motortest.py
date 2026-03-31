from gpiozero import Servo
from time import sleep

servo = Servo(18, min_pulse_width=0.001, max_pulse_width=0.002)

LOW_SPEED = 0.04   # ~2% power
NEUTRAL = 0.0      # adjust if needed later

print("Neutral...")
servo.value = NEUTRAL
sleep(3)

print("Very slow forward...")
servo.value = NEUTRAL + LOW_SPEED
sleep(4)

print("Back to neutral...")
servo.value = NEUTRAL
sleep(3)

print("Very slow reverse...")
servo.value = NEUTRAL - LOW_SPEED
sleep(4)

print("Back to neutral...")
servo.value = NEUTRAL
sleep(3)
