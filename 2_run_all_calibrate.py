import cv2
import numpy as np
from ultralytics import YOLO

# Thông số camera 1 và camera 2 (Đã cập nhật)
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

# Mở video trái và phải
video_left_path = r"c:\Users\LAPTOP24H\Downloads\Screencast from 06-12-2024 14_52_03.webm"
video_right_path = r"c:\Users\LAPTOP24H\Downloads\Screencast from 06-12-2024 14_53_04.webm"
cap_left = cv2.VideoCapture(video_left_path)
cap_right = cv2.VideoCapture(video_right_path)

# Thiết lập các thông số disparity
window_size = 5
min_disp = 0
num_disp = 16 * 5
block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
)

# Thông số camera stereo
baseline = 0.20  # Đơn vị: mét
focal_length_x = 718.91823876  # Lấy từ camera_matrix_1

# Load mô hình YOLO từ Ultralytics
model = YOLO(r"c:\Users\LAPTOP24H\Downloads\3_12.pt")  # Đường dẫn tới mô hình YOLOv5

# Xử lý từng frame từ hai video
while cap_left.isOpened() and cap_right.isOpened():
    ret_left, left_frame = cap_left.read()
    ret_right, right_frame = cap_right.read()

    if not ret_left or not ret_right:
        break

    # Hiệu chỉnh ảnh (Undistortion)
    h, w = left_frame.shape[:2]
    new_camera_matrix_1, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_1, dist_coeffs_1, (w, h), 1, (w, h))
    new_camera_matrix_2, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_2, dist_coeffs_2, (w, h), 1, (w, h))

    undistorted_left = cv2.undistort(left_frame, camera_matrix_1, dist_coeffs_1, None, new_camera_matrix_1)
    undistorted_right = cv2.undistort(right_frame, camera_matrix_2, dist_coeffs_2, None, new_camera_matrix_2)

    # Tính ma trận rectification và map
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2,
                                                         (w, h), R, T)

    # Tạo ma trận warp cho hai camera
    map1_x, map1_y = cv2.initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, (w, h), cv2.CV_32F)
    map2_x, map2_y = cv2.initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, (w, h), cv2.CV_32F)

    # Rectify các ảnh
    rectified_left = cv2.remap(undistorted_left, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(undistorted_right, map2_x, map2_y, cv2.INTER_LINEAR)

    # Chuyển ảnh sang grayscale
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # Tính Disparity Map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16

    # Tránh chia cho 0, đặt giá trị nhỏ nhất là 0.1
    disparity[disparity <= 0] = 0.1

    # Tính toán khoảng cách từ disparity
    Z = (focal_length_x * baseline) / disparity

    # Chạy YOLO để phát hiện đối tượng
    results = model(rectified_left)

    # Xử lý kết quả phát hiện
    h, w = rectified_left.shape[:2]
    for result in results:
        for i in range(len(result.boxes)):
            # Lấy thông tin từ bounding box
            box = result.boxes.xyxy[i].cpu().numpy()  # Tọa độ bounding box
            conf = result.boxes.conf[i].item()       # Độ tin cậy
            class_id = int(result.boxes.cls[i].item())  # Lớp đối tượng

            # Kiểm tra ngưỡng confidence
            if conf > 0.3:
                x1, y1, x2, y2 = map(int, box)  # Tọa độ các góc bounding box

                # Lấy vùng disparity trong bounding box
                disparity_region = disparity[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                # Tính khoảng cách trung bình trong vùng disparity
                Z_region = Z[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                average_distance = np.mean(Z_region[Z_region > 0])  # Tránh giá trị 0

                # Hiển thị kết quả với lớp đối tượng và khoảng cách
                cv2.rectangle(rectified_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rectified_left, f"Class: {class_id} Distance: {average_distance:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Giảm kích thước khung hình xuống 50%
    small_frame = cv2.resize(rectified_left, (0, 0), fx=0.5, fy=0.5)

    # Hiển thị video kết quả
    cv2.imshow("Detected Objects with Distance", small_frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
