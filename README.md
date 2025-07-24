# HapticPerceptionSS

Code base for the Haptic Perception Workshop.

> ⚙️ **Python Requirement:**  
> Python version **3.9.13** is recommended. Please update requirements.txt if another version is installed.

---

## 🧪 Test Scripts

- `test_eskin.py`  
  Checks connectivity and prints mean FSR and accelerometer values.

- `test_realsense.py`  
  Checks RealSense camera connectivity and plots RGB and depth channels.

- `test_aruco_detection.py`  
  Detects ArUco markers in the RGB feed. Can be used to determine the camera's pose or location.

---

## 📊 Plot Scripts

- `plot_kinesthetic_data.py`  
  Plots the relative 6D pose (translation + rotation) of the e-skin/tactile sensor.

- `plot_tactile_data.py`  
  Real-time visualization of FSR and accelerometer values.

- `plot_tactile_data_3D.py`  
  Efficient 3D visualization of tactile data — intuitive for spatiotemporal inspection.

---

## 🧵 Main Scripts

- `record_haptic_data.py`  
  Records both **tactile** and **kinesthetic** data streams (at different sampling rates)  
  and saves them as `.npy` files for further analysis.

- `plot_recorded_data.py`  
  Visualizes recorded tactile data (mean FSR, mean ACC) and kinesthetic data (position).  
  Useful for evaluating experiment protocol, filtering steps, and pilot study consistency.

- `preprocess.py` 
  Processes the collected data, splits

- `feature_extraction.py`
  Extracts features

- `evaluate_classifier.py`
  Performs ML Classification

- `plot_bar_plot.py`
  Plots main result
---

