# import cv2
# import numpy as np
# import os

# # Kích thước bàn cờ (số ô trong chiều rộng, chiều cao)
# CHECKERBOARD = (7, 6)  # Thay đổi tùy theo bàn cờ bạn sử dụng

# # Tạo điểm 3D cho bàn cờ
# objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# # Danh sách để lưu các điểm 3D và 2D
# objpoints = []  # Điểm 3D trong không gian thế giới
# imgpoints = []  # Điểm 2D của camera

# # Khởi động webcam trái và phải
# cap_left = cv2.VideoCapture(0)  # Camera trái
# cap_right = cv2.VideoCapture(1)  # Camera phải

# # Hướng dẫn người dùng
# print("Nhấn 's' để chụp ảnh bàn cờ. Nhấn 'q' để thoát.")

# while True:
#     # Đọc frame từ cả hai camera
#     ret_left, frame_left = cap_left.read()
#     ret_right, frame_right = cap_right.read()

#     if not ret_left or not ret_right:
#         break

#     gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
#     gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

#     # Tìm các điểm góc bàn cờ
#     ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
#     ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

#     if ret_left and ret_right:
#         # Vẽ các điểm góc
#         cv2.drawChessboardCorners(frame_left, CHECKERBOARD, corners_left, ret_left)
#         cv2.drawChessboardCorners(frame_right, CHECKERBOARD, corners_right, ret_right)

#     # Hiển thị khung hình
#     cv2.imshow('Camera trái', frame_left)
#     cv2.imshow('Camera phải', frame_right)

#     # Chụp ảnh nếu nhấn phím 's'
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         if ret_left and ret_right:
#             objpoints.append(objp)
#             imgpoints.append(corners_left)
#             imgpoints.append(corners_right)

#             # Lưu ảnh đã chụp
#             img_count = len(objpoints)
#             cv2.imwrite(f'calibration_images_left/image_{img_count}.png', frame_left)
#             cv2.imwrite(f'calibration_images_right/image_{img_count}.png', frame_right)
#             print(f"Đã chụp ảnh thứ {img_count}.")

#             # Kiểm tra xem đã đủ số lượng góc cần thiết chưa
#             if img_count >= 10:  # Bạn có thể thay đổi giá trị này
#                 print("Đã đủ ảnh. Tiến hành tính toán thông số camera.")
#                 break
#         else:
#             print("Không tìm thấy bàn cờ! Vui lòng điều chỉnh góc máy ảnh.")

#     # Thoát nếu nhấn phím 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Tính toán các thông số camera cho từng camera
# ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray_left.shape[::-1], None, None)
# ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray_right.shape[::-1], None, None)

# if ret_left and ret_right:
#     # Lưu các tham số camera vào file
#     np.savez('camera_parameters_left.npz', mtx=mtx_left, dist=dist_left)
#     np.savez('camera_parameters_right.npz', mtx=mtx_right, dist=dist_right)

#     print("Đã tính toán xong các thông số camera.")
# else:
#     print("Không thể tính toán thông số camera.")

# # Giải phóng tài nguyên
# cap_left.release()
# cap_right.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import os

# Kích thước bàn cờ (số ô trong chiều rộng, chiều cao)
CHECKERBOARD = (7, 6)  # Thay đổi tùy theo bàn cờ bạn sử dụng

# Tạo điểm 3D cho bàn cờ
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Danh sách để lưu các điểm 3D và 2D
objpoints = []  # Điểm 3D trong không gian thế giới
imgpoints = []  # Điểm 2D của camera

# Khởi động webcam
cap = cv2.VideoCapture(0)  # Camera đơn

# Hướng dẫn người dùng
print("Nhấn 's' để chụp ảnh bàn cờ. Nhấn 'q' để thoát.")

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tìm các điểm góc bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Vẽ các điểm góc
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)

    # Hiển thị khung hình
    cv2.imshow('Camera', frame)

    # Chụp ảnh nếu nhấn phím 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Lưu ảnh đã chụp
            img_count = len(objpoints)
            if not os.path.exists('calibration_images'):
                os.makedirs('calibration_images')
            cv2.imwrite(f'calibration_images/image_{img_count}.png', frame)
            print(f"Đã chụp ảnh thứ {img_count}.")

            # Kiểm tra xem đã đủ số lượng ảnh cần thiết chưa
            if img_count >= 10:  # Bạn có thể thay đổi giá trị này
                print("Đã đủ ảnh. Tiến hành tính toán thông số camera.")
                break
        else:
            print("Không tìm thấy bàn cờ! Vui lòng điều chỉnh góc máy ảnh.")

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tính toán các thông số camera
# ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# if ret:
#     # Lưu các tham số camera vào file
#     np.savez('camera_parameters.npz', mtx=mtx, dist=dist)

#     print("Đã tính toán xong các thông số camera.")
# else:
#     print("Không thể tính toán thông số camera.")

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
