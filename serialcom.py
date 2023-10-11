

import os

os.system("sudo chmod a+rw /dev/stich")


# Python code transmits a byte to Arduino /Microcontroller
import serial
import time
SerialObj = serial.Serial('/dev/stich') # COMxx  format on Windows
                  # ttyUSBx format on Linux
SerialObj.baudrate = 9600  # set Baud rate to 9600
while 1:
    SerialObj.write(b'u')    

    time.sleep(5)
    SerialObj.write(b'b') 
SerialObj.close()      