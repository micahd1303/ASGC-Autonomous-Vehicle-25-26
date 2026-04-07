import sys
import time
from machine import Pin, PWM

# Motor pin mapping
motorL = PWM(Pin(11)) 
motorR = PWM(Pin(12)) 
motorL.freq(50)
motorR.freq(50)

NEUTRAL = 4915
current_duty_L = NEUTRAL
current_duty_R = NEUTRAL

def percent_to_duty(percentage):
    """Safely converts a -100 to 100 percentage into a 16-bit PWM duty cycle."""
    percentage = max(-100.0, min(100.0, float(percentage)))
    return int(4915 + (percentage * 16.38))

def set_duty_direct(dutyL, dutyR):
    """Instantly pushes the 16-bit integer to the hardware pins."""
    motorL.duty_u16(dutyL)
    motorR.duty_u16(dutyR)

# Lock the brakes on boot
set_duty_direct(NEUTRAL, NEUTRAL)

def ramp_both(target_duty_L, target_duty_R):
    """Gradually changes both wheel speeds in 10 steps over 0.5 seconds."""
    global current_duty_L, current_duty_R
    steps = 10
    step_delay = 0.05  # <-- UPDATED: 0.05s per step = 0.5s total ramp time
    
    diff_L = target_duty_L - current_duty_L
    diff_R = target_duty_R - current_duty_R
    
    inc_L = diff_L / steps
    inc_R = diff_R / steps
    
    for i in range(1, steps + 1):
        new_L = int(current_duty_L + (inc_L * i))
        new_R = int(current_duty_R + (inc_R * i))
        set_duty_direct(new_L, new_R)
        time.sleep(step_delay)
        
    current_duty_L = target_duty_L
    current_duty_R = target_duty_R
    set_duty_direct(current_duty_L, current_duty_R)

# --- Main Listening Loop ---
while True:
    command = sys.stdin.readline().strip()
    
    if command.startswith("DRIVE"):
        try:
            parts = command.split()
            left_pct = float(parts[1])
            right_pct = float(parts[2])
            
            target_L = percent_to_duty(left_pct)
            target_R = percent_to_duty(right_pct)
            
            ramp_both(target_L, target_R)
            print(f"ACK: Ramped L={left_pct}% R={right_pct}%")
        except Exception as e:
            pass 
            
    elif command == "STOP":
        ramp_both(NEUTRAL, NEUTRAL)
        print("ACK: Stopping")
