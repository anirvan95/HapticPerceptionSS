"""
Code to record haptic data
"""
import os
import serial
import time
import pyrealsense2 as rs
import cv2
import numpy as np
from threading import Thread, Event
from queue import Queue


class RecordHapticData:
    # Global variable definitions
    def __init__(self):
        self.interaction_time = 10  # seconds
        self.port = 'COM9'
        self.marker_length = 0.05
        self.world_frame_id = 0
        self.sensor_frame_id = 20

        calib = np.load("camera_calibration_rs.npz")
        self.camera_matrix = calib['camera_matrix']
        self.dist_coeffs = calib['dist_coeffs']
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.running = Event()
        self.running.set()

        self.tactile_queue = Queue()
        self.kinesthetic_queue = Queue()
        self.timestamps = {
            'tactile': [],
            'kinesthetic': []
        }
    # Utility
    def get_marker_pose(self, id_query, ids, rvecs, tvecs):
        for i, marker_id in enumerate(ids):
            if marker_id == id_query:
                return rvecs[i], tvecs[i]
        return None, None

    def make_transform(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        return T

    def rot_x(self, theta_deg):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def tactile_data_subscriber(self):
        try:
            with serial.Serial(self.port, 500000, timeout=3) as arduino:
                time.sleep(1)
                arduino.reset_input_buffer()
                arduino.reset_output_buffer()

                mode = 0b11
                size = 16
                cmd = (mode << 6) | size
                cmd_bytes = cmd.to_bytes(1, 'big')
                packet_len = 2 * 4 + (4 * 4 * 3 + 16 * 16 * 2) * 2

                print("[Tactile] Streaming started")
                while self.running.is_set():
                    arduino.write(cmd_bytes)
                    d = arduino.read(packet_len)
                    if len(d) != packet_len:
                        continue

                    ts = time.time()
                    # Reformat accelerometer
                    acc = np.frombuffer(d[8:8 + 16 * 6], dtype=np.int16).reshape(16, 3)
                    acc_float = acc.astype(np.float32) / 16384.0
                    lookupArr = [0, 7, 8, 9, 5, 6, 15, 10, 4, 2, 12, 11, 3, 1, 14, 13]
                    acc_float = acc_float[lookupArr]
                    acc_float[6, 2] = (acc_float[5, 2] + acc_float[7, 2]) / 2

                    # Reformat FSR
                    fsr = np.frombuffer(d[8 + 16 * 6:], dtype=np.uint16).reshape(16, 2, 16)
                    fsr0 = fsr[:, 0, :].astype(np.float32)
                    fsr1 = fsr[:, 1, :].astype(np.float32)
                    fsr1[:, 6] = (fsr1[:, 4] + fsr1[:, 8]) / 2.0
                    fsr1[:, 7] = (fsr1[:, 5] + fsr1[:, 8]) / 2.0

                    # TODO: Add checks for accelerometer, fsr data in case it is corrupted to redo the interaction

                    self.tactile_queue.put((ts, acc_float, fsr0, fsr1))
                    self.timestamps['tactile'].append(ts)



        except Exception as e:
            print(f"[Tactile] Error: {e}")
            self.running.clear()

    def kinesis_data_subscriber(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)

        print("[Kinesthetic] Streaming started")

        try:
            while self.running.is_set():
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

                if ids is not None:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
                    ids = ids.flatten()

                    rvec_ref, tvec_ref = self.get_marker_pose(self.world_frame_id, ids, rvecs, tvecs)
                    rvec_tar, tvec_tar = self.get_marker_pose(self.sensor_frame_id, ids, rvecs, tvecs)

                    if rvec_ref is not None and rvec_tar is not None:
                        ts = time.time()

                        trans_markerRef_camera = self.make_transform(rvec_ref, tvec_ref)
                        trans_markerSens_camera = self.make_transform(rvec_tar, tvec_tar)

                        trans_world_camera = trans_markerRef_camera.copy()
                        trans_world_camera[:3, :3] = trans_world_camera[:3, :3] @ self.rot_x(-90)

                        trans_sensor_markerSens = np.eye(4)
                        trans_sensor_markerSens[:3, :3] = self.rot_x(-90)
                        trans_sensor_markerSens[:3, 3] = [0.0, -0.0775, -0.095]

                        trans_sensor_camera = trans_sensor_markerSens @ trans_markerSens_camera
                        trans_camera_world = np.linalg.inv(trans_world_camera)
                        trans_sensor_world = trans_camera_world @ trans_sensor_camera

                        #TODO: Add checks for kinesthetic data in case it is not detected/out of frame during interaction

                        self.kinesthetic_queue.put((ts, trans_sensor_world.copy()))
                        self.timestamps['kinesthetic'].append(ts)
        finally:
            pipeline.stop()

    def record_data(self):
        input("[SYSTEM] Press ENTER to start recording...")
        print(f"[SYSTEM] Recording for {self.interaction_time} seconds...")

        thread1 = Thread(target=self.tactile_data_subscriber)
        thread2 = Thread(target=self.kinesis_data_subscriber)

        thread1.start()
        thread2.start()

        time.sleep(self.interaction_time)
        self.running.clear()

        thread1.join()
        thread2.join()

        # Convert queues to lists
        tactile_data = list(self.tactile_queue.queue)
        kinesthetic_data = list(self.kinesthetic_queue.queue)

        # Estimate sampling rates
        def estimate_rate(t_list):
            if len(t_list) < 2:
                return 0
            diffs = np.diff(t_list)
            return round(1.0 / np.mean(diffs), 2)

        rates = {
            "tactile_Hz": estimate_rate(self.timestamps['tactile']),
            "kinesthetic_Hz": estimate_rate(self.timestamps['kinesthetic'])
        }

        # Save to file
        out_data = {
            "tactile_data": tactile_data,
            "kinesthetic_data": kinesthetic_data,
            "sampling_rates": rates
        }
        # TODO: rename and save data based on experimental protocol, add object label if required

        out_name = f"recorded_data_{int(time.time())}.npy"
        np.save(out_name, out_data)
        print(f"[✅] Saved data to {out_name}")
        print(f"Sampling rates: {rates}")


if __name__ == "__main__":
    recorder = RecordHapticData()
    recorder.record_data()
