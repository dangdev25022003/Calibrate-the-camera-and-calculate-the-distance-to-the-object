import cv2
import numpy as np
import os

# Đọc ảnh stereo đã được rectified
left_rectified = cv2.imread(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_left\rectified_left.jpg")
right_rectified = cv2.imread(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_right\rectified_right.jpg")

# Chuyển ảnh sang grayscale
gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

# Tạo đối tượng StereoBM hoặc StereoSGBM để tính Disparity Map
# Dưới đây là sử dụng StereoSGBM (Semi Global Block Matching)
window_size = 5
min_disp = 0
num_disp = 16*5  # Số bước chênh lệch, thường là bội số của 16
block_size = 5

# Khởi tạo StereoSGBM
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=block_size,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2,
                               disp12MaxDiff=1,
                               preFilterCap=63,
                               uniquenessRatio=15,
                               speckleWindowSize=0,
                               speckleRange=2)

# Tính Disparity Map
disparity = stereo.compute(gray_left, gray_right)

# Chuyển Disparity Map sang dạng 8-bit để dễ dàng hiển thị
disparity = (disparity / 16).astype(np.uint8)

# Hiển thị Disparity Map
cv2.imshow("Disparity Map", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu Disparity Map
os.makedirs(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\disparity_map", exist_ok=True)
cv2.imwrite(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\disparity_map\disparity_map.jpg", disparity)

print("Disparity Map đã được tạo và lưu.")
