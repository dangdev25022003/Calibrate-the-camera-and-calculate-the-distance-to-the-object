import cv2
import numpy as np
import os

# Ma trận nội của camera (camera 1)
camera_matrix_1 = np.array([[718.91823876, 0., 523.81819565],
                            [0., 729.96806991, 285.67619216],
                            [0., 0., 1.]])
focal_length_x = camera_matrix_1[0, 0]  # f_x
c_x = camera_matrix_1[0, 2]             # c_x
c_y = camera_matrix_1[1, 2]             # c_y

# Khoảng cách giữa hai camera (baseline)
baseline = 0.54  # Đơn vị: mét (ví dụ: 54 cm)

# Đọc ảnh disparity
disparity = cv2.imread(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\left_eye_50cm.jpg", cv2.IMREAD_GRAYSCALE)

# Chuyển disparity về dạng float32 và tránh chia cho 0
disparity = disparity.astype(np.float32)
disparity[disparity == 0] = 0.1

# Tính tọa độ Z (depth)
Z = (focal_length_x * baseline) / disparity

# Tạo lưới tọa độ ảnh (u, v)
h, w = disparity.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Tính tọa độ X và Y
X = ((u - c_x) * Z) / focal_length_x
Y = ((v - c_y) * Z) / focal_length_x

# Tính khoảng cách thực tế D
distance = np.sqrt(X**2 + Y**2 + Z**2)

# Chuyển đổi về uint8 để hiển thị
distance_normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Hiển thị khoảng cách
cv2.imshow("Distance Map", distance_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu khoảng cách vào file ảnh và file số liệu
os.makedirs(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\depth_map", exist_ok=True)
cv2.imwrite(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\depth_map\distance_map.jpg", distance_normalized)

# Lưu khoảng cách thực tế ra file
distance_file_path = r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\depth_map\distance_values.txt"
np.savetxt(distance_file_path, distance, fmt='%0.2f')

print(f"Distance map và dữ liệu đã được lưu tại {distance_file_path}")
