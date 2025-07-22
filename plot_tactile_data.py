import numpy as np
import serial
import time
import matplotlib.pyplot as plt
from collections import deque

port = 'COM9'
fsr_shape = (16, 16)
n_sensors = 16
window_size = 20  # 100 samples = 1 second at 100 Hz

# --- Initialize Plot ---
fig, axs = plt.subplots(2, 3, figsize=(10, 6))

# FSR heatmaps
im0 = axs[0, 0].imshow(np.zeros(fsr_shape), origin='lower', vmin=0, vmax=100, cmap='viridis')
im1 = axs[0, 1].imshow(np.zeros(fsr_shape), origin='lower', vmin=0, vmax=100, cmap='viridis')
im_diff = axs[0, 2].imshow(np.zeros(fsr_shape), origin='lower', vmin=-100, vmax=100, cmap='bwr')

axs[0, 0].set_title("FSR raw0 (channel 0)")
axs[0, 1].set_title("FSR raw1 (channel 1)")
axs[0, 2].set_title("FSR Difference (1 - 0)")

fig.colorbar(im0, ax=axs[0, 0])
fig.colorbar(im1, ax=axs[0, 1])
fig.colorbar(im_diff, ax=axs[0, 2])

# ACC line plots
acc_titles = ['Acc X', 'Acc Y', 'Acc Z']
acc_axes = axs[1, :3]
acc_lines = [[], [], []]
acc_buffers = [ [deque([0.0]*window_size, maxlen=window_size) for _ in range(n_sensors)] for _ in range(3)]

x_vals = np.arange(-window_size+1, 1)

for i, ax in enumerate(acc_axes):
    ax.set_title(acc_titles[i])
    ax.set_ylim(-2, 2)
    ax.set_xlim(-window_size + 1, 0)
    ax.set_ylabel("g")
    ax.set_xlabel("Samples")
    for _ in range(n_sensors):
        line, = ax.plot(x_vals, [0.0]*window_size)
        acc_lines[i].append(line)

plt.tight_layout()
plt.ion()
plt.show()

# --- Serial Communication and Plot Updating ---
try:
    with serial.Serial(port, 500000, timeout=3) as arduino:
        time.sleep(1)
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()

        mode = 0b11
        size = 16
        cmd = (mode << 6) | size
        cmd_bytes = cmd.to_bytes(1, 'big')
        packet_len = 2*4 + (4*4*3 + 16*16*2) * 2

        max_fsr0 = 100
        max_fsr1 = 100

        print("Starting live stream. Press Ctrl+C to exit.")

        while True:
            arduino.write(cmd_bytes)
            d = arduino.read(packet_len)
            if len(d) != packet_len:
                continue

            dt_acc = int.from_bytes(d[0:4], "little")
            dt_fsr = int.from_bytes(d[4:8], "little")

            acc = np.frombuffer(d[8:8 + 16*6], dtype=np.int16).copy().reshape(16, 3)
            acc_float = acc.astype(np.float32) / 16384.0

            lookupArr = [0, 7, 8, 9, 5, 6, 15, 10, 4, 2, 12, 11, 3, 1, 14, 13]
            acc_float = acc_float[lookupArr]
            acc_float[6, 2] = (acc_float[5, 2] + acc_float[7, 2]) / 2

            fsr = np.frombuffer(d[8 + 16 * 6:], dtype=np.uint16).reshape(16, 2, 16)
            raw0 = fsr[:, 0, :].astype(np.float32)
            raw1 = fsr[:, 1, :].astype(np.float32)

            raw1[:, 6] = (raw1[:, 4] + raw1[:, 8]) / 2.0
            raw1[:, 7] = (raw1[:, 5] + raw1[:, 8]) / 2.0

            # --- Update FSR Heatmaps ---
            im0.set_data(raw0.T)
            im1.set_data(raw1.T)
            if np.max(raw0) > max_fsr0:
                max_fsr0 = np.max(raw0)
                im0.set_clim(vmin=0, vmax=max_fsr0 + 1)
            if np.max(raw1) > max_fsr1:
                max_fsr1 = np.max(raw1)
                im1.set_clim(vmin=0, vmax=max_fsr1 + 1)

            # --- FSR Difference ---
            fsr_diff = raw1 - raw0
            im_diff.set_data(fsr_diff.T)
            # im_diff.set_clim(vmin=-np.max(np.abs(fsr_diff)), vmax=np.max(np.abs(fsr_diff)))

            # --- Update ACC Line Plots ---
            for axis in range(3):  # X, Y, Z
                for i in range(n_sensors):
                    acc_buffers[axis][i].append(acc_float[i, axis])
                    acc_lines[axis][i].set_ydata(acc_buffers[axis][i])

            plt.draw()
            plt.pause(0.001)

except KeyboardInterrupt:
    print("\nExited gracefully with Ctrl+C.")
except serial.SerialException as e:
    print("Serial error:", e)
except Exception as e:
    print("Unexpected error:", e)
