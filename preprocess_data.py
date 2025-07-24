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
fsr0_all = []
fsr1_all = []
acc_all = []
ts_tactile_all = []
ts_kinesthetic_all = []
kinesthetic_all = []
label_all = []
target_ts_tactile = np.arange(2, 19.5, 1 / 635)
target_ts_kin = np.arange(2, 19.5, 1 / 29)

# Iterate through all .npy files in the directory
for fname in os.listdir(dataset_dir):
    print('Processing', fname)
    if fname.endswith(".npy"):
        fpath = os.path.join(dataset_dir, fname)
        try:
            data = np.load(fpath, allow_pickle=True).item()

            tactile_data = data["tactile_data"]
            kinesthetic_data = data["kinesthetic_data"]
            rates = data["sampling_rates"]
            label = data["label"]

            if tactile_data and kinesthetic_data and label is not None:
                # --- Extract and prepare tactile ---
                ts_tactile = np.array([t[0] - tactile_data[0][0] for t in tactile_data])
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
                label_all.append(label)

                # --- Extract and prepare kinesthetic ---
                ts_kin = np.array([k[0] - kinesthetic_data[0][0] for k in kinesthetic_data])
                interaction_mask = (ts_kin >= 2.0) & (ts_kin <= 19.5)
                poses = np.stack([k[1] for k in kinesthetic_data])
                positions = poses[:, :3, 3]
                # Compute roll, pitch, yaw from rotation matrices
                rotations = poses[:, :3, :3]
                rpy = R.from_matrix(rotations).as_euler('xyz', degrees=True)  # shape (N, 3)
                precession = np.concatenate((positions[:, 0:1], positions[:, 2:3], rpy[:, 2:3]), axis=-1)
                precession_ds = resample_filt(precession, ts_kin, target_ts_kin)
                # print('Resampled Length', len(acc_ds), len(fsr0_ds), len(fsr1_ds), len(precession_ds))
                kinesthetic_all.append(precession_ds)
                ts_kinesthetic_all.append(ts_kin)
            else:
                print(f"[⚠️] Missing data or label in {fname}, skipping.")

        except Exception as e:
            print(f"[❌] Error loading {fname}: {e}")

# Convert all lists to arrays
fsr0_all = np.array(fsr0_all)            # (N, T, 16, 16)
fsr1_all = np.array(fsr1_all)
acc_all = np.array(acc_all)              # (N, T, 16, 3)
kinesthetic_all = np.array(kinesthetic_all)  # (N, T_kin, 3)

# Make sure all have same length before chunking
splits = 15
min_len = int(fsr0_all.shape[1]/splits)*splits
fsr0_all = fsr0_all[:, :min_len]
fsr1_all = fsr1_all[:, :min_len]
acc_all = acc_all[:, :min_len]
min_len = int(kinesthetic_all.shape[1]/splits)*splits
kinesthetic_all = kinesthetic_all[:, :min_len]

# --- Chunking each into 10 equal time segments along time axis ---
fsr0_chunks = np.array_split(fsr0_all, splits, axis=1)  # → 10 chunks, shape (N, ~T/10, 16, 16)
fsr1_chunks = np.array_split(fsr1_all, splits, axis=1)
acc_chunks = np.array_split(acc_all, splits, axis=1)
kinesthetic_chunks = np.array_split(kinesthetic_all, splits, axis=1)
labels_chunked = np.repeat(np.array(label_all), repeats=splits, axis=0)   # shape: (10 * N,)

# Combine each chunk along batch axis
fsr0_final = np.concatenate(fsr0_chunks, axis=0)        # shape: (10*N, T/10, 16, 16)
fsr1_final = np.concatenate(fsr1_chunks, axis=0)
acc_final = np.concatenate(acc_chunks, axis=0)
kinesthetic_final = np.concatenate(kinesthetic_chunks, axis=0)

# Work on the label part - duplication -

# Optional: Print shapes to confirm
print("[INFO] Final shapes:")
print("  FSR0:", fsr0_final.shape)
print("  FSR1:", fsr1_final.shape)
print("  ACC:", acc_final.shape)
print("  KIN:", kinesthetic_final.shape)
print(" Labels:", labels_chunked.shape)

out_path = os.path.join('dataset', 'processed_data_15.npz')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(out_path,
                    fsr0=fsr0_final,
                    fsr1=fsr1_final,
                    acc=acc_final,
                    kinesthetic=kinesthetic_final,
                    labels=labels_chunked)

print(f"[✅] Combined and chunked dataset saved to: {out_path}")

