import numpy as np
import matplotlib.pyplot as plt
import serial
import time

# 🔌 Connect to Arduino (Change port if needed)
SERIAL_PORT = "/dev/cu.usbmodem14201"  # Linux/Mac Example: "/dev/ttyUSB0"
# SERIAL_PORT = "COM3"  # Windows Example
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Give the serial connection time to initialize

plt.ion()  # Enable interactive mode

# Live image streaming loop
while True:
    frame_data = []
    collecting = False
    
    while True:
        line = ser.readline().decode('utf-8').strip()
        
        if "📷 IMAGE_START" in line:
            frame_data = []
            collecting = True
            continue  # Start collecting pixel values
        
        if "📷 IMAGE_END" in line:
            collecting = False
            break  # Stop collecting when image ends
        
        if collecting:
            try:
                row_values = list(map(int, line.split(",")))
                if len(row_values) == 96:  # Ensure full row
                    frame_data.append(row_values)
            except ValueError:
                continue  # Skip bad data lines
    
    if len(frame_data) == 96:  # Ensure full frame captured
        image_data = np.array(frame_data, dtype=np.int8) + 128  # Convert to 0-255
        plt.imshow(image_data, cmap="gray")
        plt.axis("off")
        plt.title("Live Camera Feed")
        plt.draw()
        plt.pause(0.01)  # Update the plot

ser.close()



