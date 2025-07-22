import numpy as np
import serial
import time

port = 'COM9'
loop_rate = 700  # Hz
dt = 1.0 / loop_rate  # 0.01 seconds
# --- Connect to Teensy ---
try:
    with serial.Serial(port, 500000, timeout=3) as arduino:
        time.sleep(1)
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()

        mode = 0b11
        size = 16
        cmd = (mode << 6) | size
        cmd_bytes = cmd.to_bytes(1, 'big')
        packet_len = 2*4 + (4*4*3 + 16*16*2) * 2  # 8 + 96 + 1024 = 1128

        print("Starting live stream. Press Ctrl+C to exit.")
        while True:
            arduino.write(cmd_bytes)
            start = time.time()

            # Accumulate bytes until full packet received or timeout
            buffer = bytearray()
            start_time = time.time()
            while len(buffer) < packet_len:
                if arduino.in_waiting > 0:
                    buffer += arduino.read(arduino.in_waiting)
                if time.time() - start_time > 1.0:
                    print("Timeout: Incomplete packet.")
                    break

            if len(buffer) < packet_len:
                continue  # Skip incomplete packet

            d = buffer[:packet_len]

            dt_acc = int.from_bytes(d[0:4], "little")
            dt_fsr = int.from_bytes(d[4:8], "little")
            acc = np.frombuffer(d[8:8 + 16*6], dtype=np.int16).copy().reshape(16, 3)
            # print(acc)
            fsr = np.frombuffer(d[8 + 16*6:], dtype=np.uint16).reshape(16, 2, 16)
            raw0 = fsr[:, 0, :].astype(np.float32)
            raw1 = fsr[:, 1, :].astype(np.float32)

            # Fix column bugs
            raw1[:, 6] = (raw1[:, 4] + raw1[:, 8]) / 2.0
            raw1[:, 7] = (raw1[:, 5] + raw1[:, 8]) / 2.0

            # Reshape accelerometer into 4x4 grid for each axis
            acc_x = acc[:, 0].reshape(4, 4)
            acc_y = acc[:, 1].reshape(4, 4)
            acc_z = acc[:, 2].reshape(4, 4)

            acc_z[3, 3] = (acc_z[3, 2] + acc_z[2, 3])/2
            elapsed = time.time() - start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
            print('Mean FSR 0: ', np.mean(raw0))
            print('Mean FSR 1: ', np.mean(raw1))
            squared_magnitude = acc_x ** 2 + acc_y ** 2 + acc_z ** 2
            print('Mean ACC: ', np.mean(squared_magnitude))

except KeyboardInterrupt:
    print("\nExited gracefully with Ctrl+C.")

except serial.SerialException as e:
    print("Serial error:", e)

except Exception as e:
    print("Unexpected error:", e)
