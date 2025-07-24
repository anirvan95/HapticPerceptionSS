import numpy as np
import os
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

dataset_dir = os.path.join('dataset', 'letters_dataset')
combined_data = {
    "tactile_data": [],
    "kinesthetic_data": [],
    "labels": []
}
fsr0_all = []
fsr1_all = []
acc_all = []
ts_tactile_all = []
ts_kinesthetic_all = []
kinesthetic_all = []

target_ts_tactile = np.arange(2, 19.5, 1 / 635)
target_ts_kin = np.arange(2, 19.5, 1 / 29)

# Iterate through all .npy files in the directory
for fname in os.listdir(dataset_dir):
    if fname.endswith(".npy"):
        fpath = os.path.join(dataset_dir, fname)
        try:
            data = np.load(fpath, allow_pickle=True).item()

            tactile_data = data["tactile_data"]
            kinesthetic_data = data["kinesthetic_data"]
            rates = data["sampling_rates"]
            label = data["label"]
            print(rates)

            if tactile_data and kinesthetic_data and label is not None:
                # --- Extract and prepare tactile ---
                ts_tactile = np.array([t[0] - tactile_data[0][0] for t in tactile_data])
                # print(ts_tactile[-1])
                acc = np.stack([t[1] for t in tactile_data])
                fsr0 = np.stack([t[2] for t in tactile_data])
                fsr1 = np.stack([t[3] for t in tactile_data])
                # Extract baseline by computing mean value in first 2 second and remove bias -
                baseline_mask = ts_tactile < 2.0

                acc_baseline = acc[baseline_mask].mean(axis=0)
                fsr0_baseline = fsr0[baseline_mask].mean(axis=0)
                fsr1_baseline = fsr1[baseline_mask].mean(axis=0)

                acc_corrected = acc - acc_baseline  # shape (T, 16, 3)
                fsr0_corrected = fsr0 - fsr0_baseline
                fsr1_corrected = fsr1 - fsr1_baseline

                acc_ds = resample_filt(acc_corrected, ts_tactile, target_ts_tactile)
                fsr0_ds = resample_filt(fsr0_corrected, ts_tactile, target_ts_tactile)
                fsr1_ds = resample_filt(fsr1_corrected, ts_tactile, target_ts_tactile)
                ts_tactile_all.append(ts_tactile)
                acc_all.append(acc_ds)
                fsr0_all.append(fsr0_ds)
                fsr1_all.append(fsr1_ds)

                # --- Extract and prepare kinesthetic ---
                ts_kin = np.array([k[0] - kinesthetic_data[0][0] for k in kinesthetic_data])
                interaction_mask = (ts_kin >= 2.0) & (ts_kin <= 19.5)
                poses = np.stack([k[1] for k in kinesthetic_data])
                positions = poses[:, :3, 3]
                # Compute roll, pitch, yaw from rotation matrices
                rotations = poses[:, :3, :3]
                rpy = R.from_matrix(rotations).as_euler('xyz', degrees=True)  # shape (N, 3)
                precession = np.concatenate((positions[:, 2:3], rpy[:, 2:3]), axis=-1)
                precession_ds = resample_filt(precession, ts_kin, target_ts_kin)
                print('Resampled Length', len(acc_ds), len(fsr0_ds), len(fsr1_ds), len(precession_ds))
                kinesthetic_all.append(precession_ds)
                ts_kinesthetic_all.append(ts_kin)
            else:
                print(f"[⚠️] Missing data or label in {fname}, skipping.")

        except Exception as e:
            print(f"[❌] Error loading {fname}: {e}")

# Find the min and max sequence, normalise, and find features.
print('test')