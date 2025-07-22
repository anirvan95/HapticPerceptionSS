# HapticPerceptionSS

Code base for Haptic Perception Workshop

Requirements are for Python version 3.9.13, please update if other versions are present. 

# Test Scripts
test_eskin.py - Checks connectivity and prints mean fsr and acc values
test_realsense.py - Checks camera connectivity and plots rgb and depth channels
test_aruco_detection.py - Detects the aruco markers in rgb channel, can be used to set camera location

# Plot Scripts
plot_kinesthetic_data.py - Plots the relative 6D pose of the eskin/tactile sensor
plot_tactile_data.py - Real time plot of the FSR and Accelerometer
plot_tactile_data_3D.py - Efficient and intuitive plot of the tactile data

# Main scripts
record_haptic_data.py - Records tactile and kinesthetic at different sampling rate and saves in .npy format
plot_recorded_data.py - Plots the tactile (mean fsr, mean acc) and kinesthetic (position) information, 
                        useful to evaluate experiment protocol, filtering, and pilot study 

haptic_discrimination.py (TODO) - Processes the data, computes features, and perform ML based classification

