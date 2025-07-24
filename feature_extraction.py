import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.ndimage import laplace
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter
from scipy.signal import spectrogram
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_safe_window_length(T, fraction=0.1):
    """Return a valid odd window length for savgol_filter"""
    win = int(fraction * T)
    win = max(5, min(win, T - 1))      # not too small, not bigger than T
    if win % 2 == 0:
        win -= 1  # make odd
    return win


def compute_fft_bands(fsr, fs=635):
    N, T, H, W = fsr.shape
    fft_vals = np.fft.rfft(fsr, axis=1)  # (N, F, H, W)
    fft_power = np.abs(fft_vals) ** 2  # (N, F, H, W)

    # Frequency bins
    freqs = np.fft.rfftfreq(T, d=1 / fs)
    band_edges = [0, 10, 50, fs // 2]  # adjust as needed
    band_feats = []

    for i in range(len(band_edges) - 1):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        band_energy = fft_power[:, mask, :, :].mean(axis=(1, 2, 3))
        band_feats.append(band_energy)

    return np.stack(band_feats, axis=1)  # shape (N, 3)

def compute_energy_entropy_acc(acc, fs=635):
    """
    Compute mean spectral energy from acc magnitude signal
    acc: (N, T, 16, 3)
    Returns: (N,) vector of mean spectral energy
    """
    acc_mag = np.linalg.norm(acc, axis=-1)  # (N, T, 16)
    N, T, S = acc_mag.shape
    energy = []
    entropy_vals = []
    for i in range(N):
        fft_power = np.abs(np.fft.rfft(acc_mag[i], axis=0)) ** 2 + 1e-12  # (F, 16)
        norm_power = fft_power / fft_power.sum(axis=0, keepdims=True)
        ent = entropy(norm_power, axis=0).mean()  # Mean across sensors
        entropy_vals.append(ent)
        total_energy = 0
        for s in range(S):
            f, t, Sxx = spectrogram(acc_mag[i, :, s], fs=fs, nperseg=64, noverlap=32)
            total_energy += Sxx.mean()
        energy.append(total_energy / S)

    return np.array(energy), np.array(entropy_vals)

def compute_spectral_centroid(fsr, fs=635):
    N, T, H, W = fsr.shape
    fft_vals = np.fft.rfft(fsr, axis=1)
    fft_power = np.abs(fft_vals) ** 2
    energy = fft_power.mean(axis=(1, 2, 3))  # (N,)

    freqs = np.fft.rfftfreq(T, d=1/fs)

    num = (fft_power * freqs[:, None, None]).sum(axis=1)  # (N, H, W)
    den = fft_power.sum(axis=1) + 1e-8
    centroid = (num / den).mean(axis=(1, 2))  # (N,)
    return centroid, energy

def compute_spectral_flatness(fsr):
    N, T, H, W = fsr.shape
    fft_vals = np.fft.rfft(fsr, axis=1)
    fft_power = np.abs(fft_vals) ** 2 + 1e-8

    geo_mean = np.exp(np.log(fft_power).mean(axis=1))  # (N, H, W)
    arith_mean = fft_power.mean(axis=1)
    flatness = (geo_mean / arith_mean).mean(axis=(1, 2))  # (N,)
    return flatness

from scipy.stats import entropy

def compute_spectral_entropy(fsr):
    N, T, H, W = fsr.shape
    fft_vals = np.fft.rfft(fsr, axis=1)
    fft_power = np.abs(fft_vals)**2 + 1e-12
    power_norm = fft_power / fft_power.sum(axis=1, keepdims=True)  # normalize
    ent = entropy(power_norm, axis=1).mean(axis=(1, 2))  # (N,)
    return ent

import numpy as np

def compute_fsr_kin_features(fsr0, fsr1, kin):
    """
    Compute 12 features per sample:
    - 4 quadrant mean differences (fsr1 - fsr0)
    - 4 quadrant (max - min) differences
    - 4 stiffness estimates: (Δmax-min) / Z-displacement

    Args:
        fsr0: np.ndarray, shape (N, T, 16, 16)
        fsr1: np.ndarray, shape (N, T, 16, 16)
        kin:  np.ndarray, shape (N, T_kin, 3)

    Returns:
        features: np.ndarray, shape (N, 12)
    """

    N = fsr0.shape[0]
    diff_features = np.zeros((N, 12))

    # Define quadrants as slices (row_start, row_end, col_start, col_end)
    quadrants = [
        (0, 8, 0, 8),   # upper-left
        (0, 8, 8, 16),  # upper-right
        (8, 16, 0, 8),  # lower-left
        (8, 16, 8, 16)  # lower-right
    ]

    for i in range(N):
        fsr0_trial = fsr0[i]  # (T, 16, 16)
        fsr1_trial = fsr1[i]
        kin_trial = kin[i]    # (T_kin, 3)

        # Mean and extrema across time
        fsr0_mean = fsr0_trial.mean(axis=0)
        fsr1_mean = fsr1_trial.mean(axis=0)
        fsr0_max = fsr0_trial.max(axis=0)
        fsr0_min = fsr0_trial.min(axis=0)
        fsr1_max = fsr1_trial.max(axis=0)
        fsr1_min = fsr1_trial.min(axis=0)

        # Displacement range along Z-axis
        disp_x = kin_trial[:, 0]
        disp_z = kin_trial[:, 1]
        disp_range = np.sqrt((disp_x.max() - disp_x.min())**2 + (disp_z.max() - disp_z.min())**2)

        for q, (r1, r2, c1, c2) in enumerate(quadrants):
            # 1. Mean difference (fsr1 - fsr0)
            mean0 = fsr0_mean[r1:r2, c1:c2].mean()
            mean1 = fsr1_mean[r1:r2, c1:c2].mean()
            diff_features[i, q] = mean1 - mean0

            # 2. Max-min difference
            delta0 = fsr0_max[r1:r2, c1:c2].mean() - fsr0_min[r1:r2, c1:c2].mean()
            delta1 = fsr1_max[r1:r2, c1:c2].mean() - fsr1_min[r1:r2, c1:c2].mean()
            diff_features[i, q+4] = delta1 - delta0

            # 3. Stiffness = (delta1 - delta0) / displacement range
            diff_features[i, q+8] = (delta1 - delta0) / disp_range

    return diff_features


def compute_sav_with_savgol(fsr, poly=3):
    """
    Compute SAV (spatial variance per time step),
    then smooth it using Savitzky-Golay filter and compute gradient
    """
    N, T, H, W = fsr.shape
    sav_signal = fsr.var(axis=(2, 3))  # (N, T)
    window = get_safe_window_length(T, fraction=0.1)

    # Smooth signal
    smoothed_sav = savgol_filter(sav_signal, window_length=window, polyorder=poly, axis=1)

    # Compute derivative of smoothed SAV
    sav_gradient = savgol_filter(sav_signal, window_length=window, polyorder=poly, deriv=1, axis=1)

    # Aggregate over time
    sav_mean = smoothed_sav.mean(axis=1)
    sav_slope = sav_gradient.mean(axis=1)

    return sav_mean, sav_slope


def compute_global_stats(fsr):
    """
    Compute global statistical features over flattened FSR array:
    mean, variance, skewness, kurtosis, and max.

    Parameters:
        fsr (np.ndarray): shape (N, T, 16, 16)

    Returns:
        stats (np.ndarray): shape (N, 5), each row = [mean, var, skew, kurt, max]
    """
    fsr_flat = fsr.reshape(fsr.shape[0], -1)  # (N, T*16*16)

    fsr_mean = fsr_flat.mean(axis=1)
    fsr_var = fsr_flat.var(axis=1)
    fsr_skew = skew(fsr_flat, axis=1)
    fsr_kurt = kurtosis(fsr_flat, axis=1)
    fsr_max = fsr_flat.max(axis=1)

    return np.stack([fsr_mean, fsr_var, fsr_skew, fsr_kurt, fsr_max], axis=1)  # (N, 5)

def compute_quadrant_stats(fsr):
    """
    Compute per-quadrant stats from FSR data of shape (N, T, 16, 16).
    For each quadrant: mean, std, skew, kurtosis, max → 5 values
    Total: 4 quadrants × 5 = 20 features per sample
    Returns:
        (N, 20) array of features
    """
    N = fsr.shape[0]
    features = []

    # Define quadrant slices
    quadrants = [
        (slice(None), slice(None), slice(0, 8), slice(0, 8)),   # upper-left
        (slice(None), slice(None), slice(0, 8), slice(8, 16)),  # upper-right
        (slice(None), slice(None), slice(8, 16), slice(0, 8)),  # lower-left
        (slice(None), slice(None), slice(8, 16), slice(8, 16))  # lower-right
    ]

    for q in quadrants:
        q_data = fsr[q]              # shape: (N, T, 8, 8)
        q_flat = q_data.reshape(N, -1)  # shape: (N, T*8*8)

        q_mean = q_flat.mean(axis=1)
        q_std = q_flat.var(axis=1)
        q_skew = skew(q_flat, axis=1)
        q_kurt = kurtosis(q_flat, axis=1)
        q_max = q_flat.max(axis=1)

        features.extend([q_mean, q_std, q_skew, q_kurt, q_max])  # list of (N,)

    return np.stack(features, axis=1)  # shape: (N, 20)

# ################################# FEATURE EXTRACTION CODE ##########################################################
data = np.load(os.path.join('dataset', 'processed_data.npz'))
fsr0 = data['fsr0']            # shape: (N, T, 16, 16)
fsr1 = data['fsr1']            # shape: (N, T, 16, 16)
acc = data['acc']              # shape: (N, T, 16, 3)
kin = data['kinesthetic']      # shape: (N, T_kin, 3)
labels = data['labels']

object_keys = [tuple(label[:3]) for label in labels]  # [('C', 'hard', 'convex'), ...]
# Create unique map for combinations of (letter, stiffness, shape)
unique_object_keys = sorted(set(object_keys))
object_map = {key: idx for idx, key in enumerate(unique_object_keys)}  # maps to 0–15
object_labels = np.array([object_map[tuple(label[:3])] for label in labels])  # shape (450,)
action_labels = np.array([int(float(label[3]) * 10) for label in labels])  # shape (450,)

# ##################################### FSR 0 and FSR 1 Features ##########################################
print('Computing fsr features...')
# Mean, Sum Feature # Covers letter shape
gstat_fsr0 = compute_global_stats(fsr0)  # shape (N, 20)
gstat_fsr1 = compute_global_stats(fsr1)  # shape (N, 20)
# Mean, Sum Feature # Covers letter shape
qstat_fsr0 = compute_quadrant_stats(fsr0)  # shape (N, 20)
qstat_fsr1 = compute_quadrant_stats(fsr1)  # shape (N, 20)
# Laplacian magnitude
N, T, H, W = fsr0.shape
lap_fsr0 = np.array([laplace(f) for f in fsr0.reshape(-1, H, W)]).reshape(N, T, H, W)
lap_feature0 = np.abs(lap_fsr0).mean(axis=(1, 2, 3))
lap_fsr1 = np.array([laplace(f) for f in fsr1.reshape(-1, H, W)]).reshape(N, T, H, W)
lap_feature1 = np.abs(lap_fsr1).mean(axis=(1, 2, 3))
# Gradient magnitude
sav0_mean, sav0_slope = compute_sav_with_savgol(fsr0)
sav1_mean, sav1_slope = compute_sav_with_savgol(fsr1)
dy, dx = np.gradient(fsr0, axis=[2, 3])
grad_mag = np.sqrt(dx**2 + dy**2).mean(axis=(1, 2, 3))
fft_bands_diff = compute_fft_bands(fsr0-fsr1)             # (N, 3)
fft_centroid_diff,  fft_energy_diff = compute_spectral_centroid(fsr0-fsr1)  # (N,)
fft_flatness_diff = compute_spectral_flatness(fsr0-fsr1)  # (N,)
fft_entropy_diff = compute_spectral_entropy(fsr0-fsr1)    # (N,)

# ##################################### Kinesthetic Features ##########################################
print('Computing kinesthetic features...')
kin_var = kin.var(axis=1)
kin_skew = skew(kin, axis=1)
kin_kurt = kurtosis(kin, axis=1)

features_fsr_kin = compute_fsr_kin_features(fsr0, fsr1, kin)  # shape: (N, 12) # Haptic actually

print('Computing acc features...')
# ##################################### Acc Features ##########################################
acc_mag = np.linalg.norm(acc, axis=-1)  # (N, T, 16)
acc_mean = acc_mag.mean(axis=(1, 2))          # Mean over time and sensors
acc_std = acc_mag.std(axis=(1, 2))            # Standard deviation
acc_min = acc_mag.min(axis=(1, 2))            # Minimum value
acc_max = acc_mag.max(axis=(1, 2))            # Maximum value
acc_range = acc_max - acc_min                 # Dynamic range
acc_skew = skew(acc_mag.reshape(N, -1), axis=1)
acc_kurt = kurtosis(acc_mag.reshape(N, -1), axis=1)

acc_mean_t = acc_mag.mean(axis=1)      # (N, 16)
acc_std_t = acc_mag.std(axis=1)        # (N, 16)

mean_over_sensors = acc_mean_t.mean(axis=1)  # shape (N,)
std_over_sensors = acc_std_t.mean(axis=1)    # shape (N,)
acc_energy, acc_entropy = compute_energy_entropy_acc(acc)

print('Save all the features in a dictionary...')

features_dict = {
    # FSR 0 and FSR 1 global stats
    'gstat_fsr0': gstat_fsr0,  # (N, 5)
    'gstat_fsr1': gstat_fsr1,

    # Per-quadrant stats
    'qstat_fsr0': qstat_fsr0,  # (N, 20)
    'qstat_fsr1': qstat_fsr1,

    # Laplacian
    'lap_feature0': lap_feature0,  # (N,)
    'lap_feature1': lap_feature1,

    # SAV + gradient
    'sav0_mean': sav0_mean,
    'sav0_slope': sav0_slope,
    'sav1_mean': sav1_mean,
    'sav1_slope': sav1_slope,
    'grad_mag': grad_mag,

    # FFT features
    'fft_bands_diff': fft_bands_diff,          # (N, 3)
    'fft_centroid_diff': fft_centroid_diff,    # (N,)
    'fft_energy_diff': fft_energy_diff,
    'fft_flatness_diff': fft_flatness_diff,
    'fft_entropy_diff': fft_entropy_diff,

    # Kinesthetic
    'kin_var': kin_var,                        # (N, 3)
    'kin_skew': kin_skew,
    'kin_kurt': kin_kurt,

    # Haptic interaction features
    'features_fsr_kin': features_fsr_kin,      # (N, 12)

    # Accelerometer
    'acc_mean': acc_mean,
    'acc_std': acc_std,
    'acc_min': acc_min,
    'acc_max': acc_max,
    'acc_range': acc_range,
    'acc_skew': acc_skew,
    'acc_kurt': acc_kurt,
    'mean_over_sensors': mean_over_sensors,
    'std_over_sensors': std_over_sensors,
    'acc_energy': acc_energy,
    'acc_entropy': acc_entropy,

    # Labels
    'object_labels': object_labels,
    'action_labels': action_labels
}

np.savez('features_all_modalities.npz', **features_dict)
print("✅ Features saved to 'features_all_modalities.npz'")
