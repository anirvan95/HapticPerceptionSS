"""
Analyze the structure of haptic data files to understand their format
"""

import os

import numpy as np


def analyze_data_file(filepath):
    """Analyze a single .npy file"""
    data = np.load(filepath, allow_pickle=True)
    print(f"File: {os.path.basename(filepath)}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")

    # Handle object array (dictionary-like structure)
    if data.dtype == object and data.shape == ():
        content = data.item()
        print(f"Content type: {type(content)}")
        if isinstance(content, dict):
            print(f"Keys: {list(content.keys())}")
            for key, value in content.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    if value.size > 0:
                        print(f"    Range: {value.min():.3f} to {value.max():.3f}")
                else:
                    print(f"  {key}: {type(value)} = {value}")
    else:
        print(
            f"First few values:\n{data[:5] if len(data.shape) == 1 else data[:5, :5]}"
        )
        print(f"Data range: {data.min():.3f} to {data.max():.3f}")

    print("-" * 50)
    return data.shape, data.dtype


# Analyze a few sample files
dataset_dir = "letters_dataset"
sample_files = [
    "C_hard_convex_0.5_1.npy",
    "D_med_flat_1_1.npy",
    "Q_soft_flat_1.5_1.npy",
]

for filename in sample_files:
    filepath = os.path.join(dataset_dir, filename)
    if os.path.exists(filepath):
        analyze_data_file(filepath)

