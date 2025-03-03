'''This script is for calculating camera pose with respect to a chessboard.

1. Specify the size and the square width of the chessboard.
2. Specify the file name of camera parameters.
3. Press 'q' to quit.
'''

import cv2
import numpy as np
import yaml
import time
from scipy.spatial.transform import Rotation as rot

# Specify camera input and chessboard configuration
camera = cv2.VideoCapture(0)
CHESSBOARD_CORNER_NUM_X = 9
CHESSBOARD_CORNER_NUM_Y = 6
SQUARE_WIDTH = 26.24  # Square width in mm
CAMERA_PARAMETERS_INPUT_FILE = r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\cam1.yaml"

# Get frame dimensions
frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the chessboard
objp = np.zeros((CHESSBOARD_CORNER_NUM_X * CHESSBOARD_CORNER_NUM_Y, 3), np.float32)
for i in range(CHESSBOARD_CORNER_NUM_Y):
    for j in range(CHESSBOARD_CORNER_NUM_X):
        objp[i * CHESSBOARD_CORNER_NUM_X + j, 0] = j * SQUARE_WIDTH
        objp[i * CHESSBOARD_CORNER_NUM_X + j, 1] = i * SQUARE_WIDTH

# Define axis for visualization (x: red, y: green, z: blue)
axis = np.float32([[SQUARE_WIDTH, 0, 0], [0, SQUARE_WIDTH, 0], [0, 0, -SQUARE_WIDTH]]).reshape(-1, 3)

# Load camera intrinsic parameters
with open(CAMERA_PARAMETERS_INPUT_FILE, 'r') as f:
    loadeddict = yaml.load(f, Loader=yaml.SafeLoader)  # Use SafeLoader for secure loading
    mtx = np.array(loadeddict.get('camera_matrix'))
    dist = np.array(loadeddict.get('dist_coeff'))
    mtx_inv = np.linalg.inv(mtx)  # Inverse of camera matrix

# Main loop for camera pose estimation
while True:
    ret, img = camera.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, (CHESSBOARD_CORNER_NUM_X, CHESSBOARD_CORNER_NUM_Y), flags=cv2.CALIB_CB_FAST_CHECK
    )
    
    if ret:
        # Refine corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find rotation and translation vectors using PnP
        t0 = time.time()
        _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        t1 = time.time()
        rotation_euler = rot.from_rotvec(rvec.T).as_euler('xyz', degrees=True)

        # Display results
        cv2.putText(
            img,
            f'PnP Pose (Time: {1000*(t1-t0):.3f}ms):',
            (20, int(frame_height - 210)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 255, 255),
        )
        cv2.putText(
            img,
            f'Rotation (Euler): X: {rotation_euler[0][0]:.2f} Y: {rotation_euler[0][1]:.2f} Z: {rotation_euler[0][2]:.2f}',
            (20, int(frame_height) - 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )
        cv2.putText(
            img,
            f'Translation (mm): X: {tvec[0][0]:.2f} Y: {tvec[1][0]:.2f} Z: {tvec[2][0]:.2f}',
            (20, int(frame_height) - 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )

        # Visualize pose
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        for i, point in enumerate(imgpts):
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i]  # x: red, y: green, z: blue
            cv2.line(img, tuple(corners2[0].ravel().astype(int)), tuple(point.ravel().astype(int)), color, 3)

    # Display the image
    cv2.imshow('img', img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
