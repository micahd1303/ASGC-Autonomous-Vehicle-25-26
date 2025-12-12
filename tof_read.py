import smbus2
import time
import numpy as np

I2C_ADDRESS = 0x33  # your i2c address
bus = smbus2.SMBus(1)

def read_matrix():
    # create a “read 68 bytes” message
    write = smbus2.i2c_msg.write(I2C_ADDRESS, [])
    read = smbus2.i2c_msg.read(I2C_ADDRESS, 68)
    bus.i2c_rdwr(write, read)

    data = list(read)
    # check the frame header
    if len(data) >= 4 and data[0] == 0x5A and data[1] == 0x5A:
        values = data[4:4+64]
        matrix = np.array(values).reshape((8, 8))
        return matrix
    else:
        return None

try:
    while True:
        frame = read_matrix()
        if frame is not None:
            print(frame)
        else:
            print("bad frame")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("done")
