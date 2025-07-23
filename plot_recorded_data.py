"""
Code to plot the recorded data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def resample_filt(data, timestamps, new_timestamps):
    # This is mean filtering based resampling technique, can only work for downsampling,
    # check interpolate function in scikit-learn for other approaches
    resampled_data = [data[0]]
    prev_mean = data[0]

    # Iterate over each custom timestamp interval
    for i in range(1, len(new_timestamps)):
        # Define the start and end of the current interval
        start_time = new_timestamps[i - 1]
        end_time = new_timestamps[i]

        # Find indices where timestamps fall within the interval
        mask = (timestamps >= start_time) & (timestamps < end_time)

        # Extract values within the interval
        interval_values = data[mask]
        if len(interval_values) > 0:
            mean_value = np.mean(interval_values, axis=0)
            prev_mean = mean_value
        else:
            mean_value = prev_mean
        # Store the result
        resampled_data.append(mean_value)
    return np.array(resampled_data)


def visualize_data(npy_file, show_plot=False):
    data = np.load(npy_file, allow_pickle=True).item()

    tactile_data = data["tactile_data"]
    kinesthetic_data = data["kinesthetic_data"]
    rates = data["sampling_rates"]
    label = data["label"]

    # --- Extract and prepare tactile ---
    ts_tactile = np.array([t[0] - tactile_data[0][0] for t in tactile_data])
    acc = np.stack([t[1] for t in tactile_data])
    fsr0 = np.stack([t[2] for t in tactile_data])
    fsr1 = np.stack([t[3] for t in tactile_data])
    fsr_diff = fsr1 - fsr0

    # Mean values
    acc_mean = acc.mean(axis=1)
    fsr0_mean = fsr0.mean(axis=(1, 2))
    fsr1_mean = fsr1.mean(axis=(1, 2))
    fsr_diff_mean = fsr_diff.mean(axis=(1, 2))

    # --- Extract and prepare kinesthetic ---
    ts_kin = [k[0] - kinesthetic_data[0][0] for k in kinesthetic_data]
    poses = np.stack([k[1] for k in kinesthetic_data])
    positions = poses[:, :3, 3]
    # Compute roll, pitch, yaw from rotation matrices
    rotations = poses[:, :3, :3]
    rpy = R.from_matrix(rotations).as_euler('xyz', degrees=True)  # shape (N, 3)

    # TODO: Use the resample function if required
    ts_target = np.arange(0, 10, 1 / 30)
    fsr0_ds = resample_filt(fsr0, ts_tactile, ts_target)

    # TODO: Evaluate different normalization techniques and see the effect

    # --- Plotting ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(ts_tactile, fsr0_mean, label='FSR Layer 1')
    axs[0].plot(ts_tactile, fsr1_mean, label='FSR Layer 2')
    axs[0].plot(ts_tactile, fsr_diff_mean, label='FSR diff (1 - 0)', linestyle='--')
    axs[0].set_ylabel("FSR (a.u.)")
    axs[0].set_title("FSR Pressure Readings")
    axs[0].legend()

    axs[1].plot(ts_tactile, acc_mean[:, 0], label='Acc X')
    axs[1].plot(ts_tactile, acc_mean[:, 1], label='Acc Y')
    axs[1].plot(ts_tactile, acc_mean[:, 2], label='Acc Z')
    axs[1].set_ylabel("Acceleration (g)")
    axs[1].set_title("Accelerometer Mean Readings")
    axs[1].legend()

    axs[2].plot(ts_kin, positions[:, 0], label='X')
    axs[2].plot(ts_kin, positions[:, 1], label='Y')
    axs[2].plot(ts_kin, positions[:, 2], label='Z')
    axs[2].set_ylabel("Position (m)")
    axs[2].set_title("Sensor Translation in World Frame")
    axs[2].legend()

    axs[3].plot(ts_kin, rpy[:, 0], label='Roll (°)')
    axs[3].plot(ts_kin, rpy[:, 1], label='Pitch (°)')
    axs[3].plot(ts_kin, rpy[:, 2], label='Yaw (°)')
    axs[3].set_ylabel("Angle (°)")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_title("Sensor Orientation (Roll, Pitch, Yaw)")
    axs[3].legend()

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        filename = f"dump/{label[0]}_{label[1]}_{label[2]}_{label[3]}_{label[4]}.png"
        plt.savefig(filename)



if __name__ == "__main__":
    import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument("file", help="Path to .npy file recorded by RecordHapticData")
    #args = parser.parse_args()

    #visualize_data(args.file)

    visualize_data('dataset/pilot_study/D_soft_flat_1.npy', show_plot=True)