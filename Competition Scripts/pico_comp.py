# This is the final script that was run on the pico during the competition. It recieves commands from the pi and executes them with the motors.
# Theoretically, it also would have driven the various servos within our system if all were connected.
# The drain was completely necessary for preventing the pico being overloaded with commands.
import sys
import time
import select
from machine import Pin, PWM

# --- Motor pin mapping ---
motorR = PWM(Pin(11))
motorL = PWM(Pin(12))
motorL.freq(50)
motorR.freq(50)

NEUTRAL    = 4915
RAMP_STEP  = 150
LOOP_DELAY = 0.005

current_duty_L = NEUTRAL
current_duty_R = NEUTRAL
target_duty_L  = NEUTRAL
target_duty_R  = NEUTRAL

def percent_to_duty(percentage):
    percentage = max(-100.0, min(100.0, float(percentage)))
    return int(4915 + (percentage * 16.38))

def set_duty_direct(dutyL, dutyR):
    motorL.duty_u16(dutyL)
    motorR.duty_u16(dutyR)

def step_toward(current, target, step):
    if current < target:
        return min(current + step, target)
    elif current > target:
        return max(current - step, target)
    return current

set_duty_direct(NEUTRAL, NEUTRAL)
print("PICO_READY")

while True:
    # --- Drain ALL waiting commands, keep only the last one ---
    latest_command = None
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            break
        line = sys.stdin.readline().strip()
        if line:
            latest_command = line

    # --- Process only the most recent command ---
    if latest_command is not None:
        if latest_command.startswith("DRIVE"):
            parts = latest_command.split()
            if len(parts) == 3:
                try:
                    target_duty_L = percent_to_duty(float(parts[1]))
                    target_duty_R = percent_to_duty(float(parts[2]))
                except ValueError as e:
                    print(f"ERR: {e}")
            else:
                print(f"ERR: DRIVE expects 2 args, got {len(parts)-1}")
        elif latest_command == "STOP":
            target_duty_L = NEUTRAL
            target_duty_R = NEUTRAL
        else:
            print(f"ERR: Unknown command '{latest_command}'")

    # --- Ramp toward target ---
    current_duty_L = step_toward(current_duty_L, target_duty_L, RAMP_STEP)
    current_duty_R = step_toward(current_duty_R, target_duty_R, RAMP_STEP)
    set_duty_direct(current_duty_L, current_duty_R)

    time.sleep(LOOP_DELAY)
