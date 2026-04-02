import serial
import time

print("Connecting to Pico...")
# Open the serial port to the Pico
pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# CRITICAL: Wait 2 seconds for the Pico to boot up after opening the port
time.sleep(2) 

print("Commanding: NEUTRAL")
pico.write(b'NEUTRAL\n')
pico.flush()
time.sleep(3) # Wait 3 seconds with motor stopped

print("Commanding: FORWARD (1% Power)")
pico.write(b'FORWARD\n')
pico.flush()
time.sleep(3) # Motor should slowly spin for 3 seconds

print("Commanding: NEUTRAL")
pico.write(b'NEUTRAL\n')
pico.flush()
time.sleep(2) # Motor stops

pico.close()
print("Test complete.")
