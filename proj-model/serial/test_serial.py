import serial.tools.list_ports

SERIAL_PORT = "/dev/cu.usbmodem14201"
ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)

print(f"connected to {SERIAL_PORT}")

while True:
    line = ser.readline().decode('utf-8').strip()
    if line:
        print(line)