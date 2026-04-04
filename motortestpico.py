import serial
import time

print("Opening port...")
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# Wait 2 seconds for the Pico to finish its boot sequence
time.sleep(2) 

print("Sending FORWARD command...")
pico.write(b'FORWARD\n')
pico.flush()
time.sleep(3) # Let it spin for 3 seconds

print("Sending NEUTRAL command...")
pico.write(b'NEUTRAL\n')
pico.flush()

pico.close()
print("Test Complete")
