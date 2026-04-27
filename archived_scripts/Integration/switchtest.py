import RPi.GPIO as GPIO
import time

# -----------------------------------
# GPIO SETUP
# -----------------------------------
SW1 = 17  # bit A
SW2 = 27  # bit B

GPIO.setmode(GPIO.BCM)
GPIO.setup(SW1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(SW2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# -----------------------------------
# COLOR DECODER
# -----------------------------------
def read_target_color():
    A = GPIO.input(SW1)
    B = GPIO.input(SW2)

    if   (A, B) == (0, 0): return "RED"
    elif (A, B) == (0, 1): return "YELLOW"
    elif (A, B) == (1, 0): return "GREEN"
    elif (A, B) == (1, 1): return "BLUE"

# -----------------------------------
# MAIN LOOP
# -----------------------------------
last_color = None

try:
    print("Flip switches to test color selection (Ctrl+C to quit)\n")

    while True:
        color = read_target_color()

        if color != last_color:
            A = GPIO.input(SW1)
            B = GPIO.input(SW2)
            print(f"A={A} B={B}  ->  {color}")
            last_color = color

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    GPIO.cleanup()
