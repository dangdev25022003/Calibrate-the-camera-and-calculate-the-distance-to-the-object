import cv2
import numpy as np
import os

# Thông số của camera 1 và camera 2 (Đã cập nhật)
camera_matrix_1 = np.array([[718.91823876, 0., 523.81819565],
                            [0., 729.96806991, 285.67619216],
                            [0., 0., 1.]])
dist_coeffs_1 = np.array([3.10648813e-02, -1.79709590e-01, 8.74315051e-04, 1.77034087e-04, 2.13728774e-01])

camera_matrix_2 = np.array([[722.059916, 0., 513.69591401],
                            [0., 732.73497174, 292.73939187],
                            [0., 0., 1.]])
dist_coeffs_2 = np.array([0.01106403, -0.1368946, -0.0014158, 0.001299, 0.20229815])

# Ma trận quay (R) và vector dịch chuyển (T)
R = np.array([[0.99936536, 0.00766219, -0.0347876],
              [-0.00858505, 0.99961309, -0.0264568],
              [0.03457142, 0.02673866, 0.99904447]])
T = np.array([[6.32313335], [-0.14376617], [0.13653523]])

# Đọc ảnh stereo (có hai camera)
left_image = cv2.imread(r'c:\Users\LAPTOP24H\Downloads\l.png')
right_image = cv2.imread(r'c:\Users\LAPTOP24H\Downloads\r.png')

# Hiệu chỉnh ảnh (Undistortion)
h, w = left_image.shape[:2]
new_camera_matrix_1, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_1, dist_coeffs_1, (w, h), 1, (w, h))
new_camera_matrix_2, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_2, dist_coeffs_2, (w, h), 1, (w, h))

undistorted_left = cv2.undistort(left_image, camera_matrix_1, dist_coeffs_1, None, new_camera_matrix_1)
undistorted_right = cv2.undistort(right_image, camera_matrix_2, dist_coeffs_2, None, new_camera_matrix_2)

# Tính ma trận rectification và map
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2,
                                                     (w, h), R, T)

# Tạo ma trận warp cho hai camera
map1_x, map1_y = cv2.initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, (w, h), cv2.CV_32F)
map2_x, map2_y = cv2.initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, (w, h), cv2.CV_32F)

# Rectify các ảnh
rectified_left = cv2.remap(undistorted_left, map1_x, map1_y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(undistorted_right, map2_x, map2_y, cv2.INTER_LINEAR)

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_left", exist_ok=True)
os.makedirs(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_right", exist_ok=True)

# Lưu ảnh đã được rectified vào các thư mục tương ứng
cv2.imwrite(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_left\rectified_left.png", rectified_left)
cv2.imwrite(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_right\rectified_right.png", rectified_right)

print("Ảnh đã được rectified và lưu vào các thư mục rectified_left và rectified_right.")
