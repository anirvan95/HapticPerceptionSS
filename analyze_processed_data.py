"""
Analyze the structure of the new processed_data.npz file
"""

import numpy as np
import os

def analyze_npz_file(filepath):
    """Analyze the structure of the .npz file"""
    print(f"Analyzing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Load the .npz file
    data = np.load(filepath)
    
    print(f"Keys in .npz file: {list(data.keys())}")
    print("-" * 50)
    
    for key in data.keys():
        array = data[key]
        print(f"Key: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        
        if array.size > 0:
            if array.dtype in [np.float32, np.float64]:
                print(f"  Range: {array.min():.3f} to {array.max():.3f}")
            else:
                print(f"  Unique values: {np.unique(array)[:10]}")  # Show first 10 unique values
        
        # Show first few samples for small arrays
        if len(array.shape) <= 2 and array.shape[0] <= 10:
            print(f"  First few samples:\n{array[:5]}")
        elif len(array.shape) == 2:
            print(f"  First few samples (first 5 rows, first 5 cols):\n{array[:5, :5]}")
        
        print("-" * 30)

# Analyze the processed data
filepath = "dataset/processed_data.npz"
analyze_npz_file(filepath)