from gpiozero import PWMOutputDevice
from time import sleep

SERVO1_PIN = 17
SERVO2_PIN = 18

servo1 = PWMOutputDevice(SERVO1_PIN, frequency=200)
servo2 = PWMOutputDevice(SERVO2_PIN, frequency=200)

PERIOD_US = 5000  # 200 Hz PWM period
NEUTRAL = 1500

def send_pulse(servo, micros):
    duty = micros / PERIOD_US
    servo.value = duty

def set_servos(micros):
    mirrored = 3000 - micros

    send_pulse(servo1, micros)
    send_pulse(servo2, mirrored)

    print(f"Servo1: {micros} µs | Servo2: {mirrored} µs")


def move_servos(start, end, step=5, delay=0.01):
    if start < end:
        rng = range(start, end, step)
    else:
        rng = range(start, end, -step)

    for micros in rng:
        set_servos(micros)
        sleep(delay)

    set_servos(end)


try:

    LOWER = 1000
    RAISE = 1700

    while True:

        print("Raising arm")
        move_servos(LOWER, RAISE)

        sleep(2)

        print("Lowering arm")
        move_servos(RAISE, LOWER)

        sleep(2)

finally:
    servo1.off()
    servo2.off()
