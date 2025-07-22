import pyrealsense2 as rs
import numpy as np
import cv2

# Setup ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable only the color stream (no depth)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None:
            # Draw detected markers and print info
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in zip(ids, corners):
                print(f"Detected marker ID: {i[0]} at corners: {corner.reshape(-1, 2)}")

        # Show result
        cv2.imshow("Aruco Marker Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
