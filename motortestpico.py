import serial
import time

print("Connecting to Pico...")
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2) 

def drive(percentage):
    """Tells the Pico to drive at a specific percentage (-100 to 100)."""
    command_str = f"DRIVE {percentage}\n"
    print(f"\nPi 5: Requesting {percentage}% power...")
    pico.write(command_str.encode('utf-8'))
    pico.flush()
    
    # Listen for the Pico's acknowledgment
    time.sleep(0.1)
    if pico.in_waiting > 0:
        reply = pico.readline().decode('utf-8').strip()
        print(f"Pico: {reply}")

def stop():
    """Tells the Pico to ramp down to 0%."""
    print("\nPi 5: Requesting Full Stop...")
    pico.write(b"STOP\n")
    pico.flush()
    
    time.sleep(0.1)
    if pico.in_waiting > 0:
        reply = pico.readline().decode('utf-8').strip()
        print(f"Pico: {reply}")

try:
    print("\n--- Testing Percentage Abstraction ---")
    
    # Drive forward at 15% power
    drive(5)
    time.sleep(4) # 1s ramp + 3s driving
    
    # Drive backward at 10% power
    drive(-5)
    time.sleep(4) 
    
    # Ramp back to exactly 0%
    stop()
    time.sleep(2)

finally:
    pico.close()
    print("\nTest complete.")
