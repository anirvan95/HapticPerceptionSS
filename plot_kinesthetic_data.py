import os
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque

# === Parameters ===
window_size = 10
sensor_transforms = deque(maxlen=window_size)
marker_length = 0.05  # meters
ref_id = 0
target_id = 20

# === Load camera calibration ===
calib = np.load("camera_calibration_rs.npz")
camera_matrix = calib['camera_matrix']
dist_coeffs = calib['dist_coeffs']

# === ArUco dictionary ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()

# === Setup RealSense pipeline ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# === Matplotlib setup ===
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
plt.ion()

# === Utility functions (same as your code, unchanged) ===
def get_marker_pose(id_query, ids, rvecs, tvecs):
    for i, marker_id in enumerate(ids):
        if marker_id == id_query:
            return rvecs[i], tvecs[i]
    return None, None

def make_cube_points(size=0.01):
    s = size / 2.0
    return np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ])

def make_cube_faces(points):
    return [
        [points[j] for j in [0, 1, 2, 3]],
        [points[j] for j in [4, 5, 6, 7]],
        [points[j] for j in [0, 1, 5, 4]],
        [points[j] for j in [2, 3, 7, 6]],
        [points[j] for j in [1, 2, 6, 5]],
        [points[j] for j in [0, 3, 7, 4]]
    ]

def draw_axes(origin, R, length=0.02):
    x_axis = origin + R[:, 0] * length
    y_axis = origin + R[:, 1] * length
    z_axis = origin + R[:, 2] * length
    ax.plot(*zip(origin, x_axis), color='r')
    ax.plot(*zip(origin, y_axis), color='g')
    ax.plot(*zip(origin, z_axis), color='b')

def remap_coords(t): return np.array([t[0], -t[2], t[1]])

def remap_rotation(R):
    M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    return M @ R @ M.T

def rotate_points(points, R): return (R @ points.T).T

def make_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def rot_x(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0], [0, c, -s], [0, s,  c]])

def draw_axes_matplotlib_frame(origin, R, length=0.01):
    ax.plot(*zip(origin, origin + R[:, 0]*length), color='r')
    ax.plot(*zip(origin, origin + R[:, 1]*length), color='g')
    ax.plot(*zip(origin, origin + R[:, 2]*length), color='b')

# === Main Loop ===
print("🟢 RealSense camera running... Press ESC to quit.")
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        ids = ids.flatten()

        rvec_ref, tvec_ref = get_marker_pose(ref_id, ids, rvecs, tvecs)
        rvec_tar, tvec_tar = get_marker_pose(target_id, ids, rvecs, tvecs)

        if rvec_ref is not None and rvec_tar is not None:
            T_marker0_camera = make_transform(rvec_ref, tvec_ref)
            T_marker20_camera = make_transform(rvec_tar, tvec_tar)

            T_world_camera = T_marker0_camera.copy()
            T_world_camera[:3, :3] = T_world_camera[:3, :3] @ rot_x(-90)

            T_sensor_camera = T_marker20_camera.copy()
            T_sensor_marker20 = np.eye(4)
            T_sensor_marker20[:3, :3] = rot_x(-90)
            T_sensor_marker20[:3, 3] = [0.0, -0.0775, -0.095]

            T_sensor_camera = T_sensor_camera @ T_sensor_marker20
            T_camera_world = np.linalg.inv(T_world_camera)
            T_sensor_world = T_camera_world @ T_sensor_camera
            T_marker20_world = T_camera_world @ T_marker20_camera
            sensor_transforms.append(T_sensor_camera.copy())
            avg_translation = np.mean([T[:3, 3] for T in sensor_transforms], axis=0)
            avg_rotation = np.mean([T[:3, :3] for T in sensor_transforms], axis=0)
            U, _, Vt = np.linalg.svd(avg_rotation)
            avg_rotation = U @ Vt

            sensor_pos = avg_translation
            sensor_rot = avg_rotation

            # === Plotting ===
            ax.clear()
            ax.set_xlim([-0.25, 0.25])
            ax.set_ylim([-0.25, 0.25])
            ax.set_zlim([-0.0, 0.25])
            ax.set_xlabel("X (right)")
            ax.set_ylabel("Y (down)")
            ax.set_zlabel("Z (up)")
            ax.set_title("Sensor Frame (F) in World Frame")

            draw_axes_matplotlib_frame(np.zeros(3), np.eye(3), length=0.05)
            draw_axes_matplotlib_frame(T_marker20_world[:3, 3], T_marker20_world[:3, :3])
            draw_axes_matplotlib_frame(T_sensor_world[:3, 3], T_sensor_world[:3, :3], length=0.035)

            plt.draw()
            plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

pipeline.stop()
plt.ioff()
plt.show()
