# # import cv2
# # import numpy as np
# # from ultralytics import YOLO

# # # Load mô hình YOLO từ Ultralytics
# # model = YOLO(r"c:\Users\LAPTOP24H\Downloads\model_200ep_27_11_3_label.pt")  # Đường dẫn tới mô hình YOLOv5

# # # Đọc ảnh stereo đã rectified
# # left_rectified = cv2.imread(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_left\rectified_left.png")
# # right_rectified = cv2.imread(r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\rectified_right\rectified_right.png")

# # # Chuyển ảnh sang grayscale
# # gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
# # gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

# # # Tính Disparity Map
# # window_size = 5
# # min_disp = 0
# # num_disp = 16 * 5
# # block_size = 5

# # stereo = cv2.StereoSGBM_create(
# #     minDisparity=min_disp,
# #     numDisparities=num_disp,
# #     blockSize=block_size,
# #     P1=8 * 3 * window_size ** 2,
# #     P2=32 * 3 * window_size ** 2,
# #     disp12MaxDiff=1,
# #     preFilterCap=63,
# #     uniquenessRatio=15,
# #     speckleWindowSize=0,
# #     speckleRange=2,
# # )
# # disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16

# # # Tránh chia cho 0
# # disparity[disparity <= 0] = 0.1

# # # Tính toán khoảng cách từ disparity
# # baseline = 0.54  # Đơn vị: mét
# # focal_length_x = 718.91823876  # Lấy từ camera_matrix_1
# # Z = (focal_length_x * baseline) / disparity

# # # Chạy YOLO để phát hiện đối tượng
# # results = model(left_rectified)  # Dự đoán đối tượng trên ảnh trái

# # # Xử lý kết quả phát hiện
# # h, w = left_rectified.shape[:2]
# # for result in results:
# #     for i in range(len(result.boxes)):
# #         # Lấy thông tin từ bounding box
# #         box = result.boxes.xyxy[i].cpu().numpy()  # Tọa độ bounding box
# #         conf = result.boxes.conf[i].item()       # Độ tin cậy
# #         class_id = int(result.boxes.cls[i].item())  # Lớp đối tượng

# #         # Kiểm tra ngưỡng confidence
# #         if conf > 0.6:
# #             x1, y1, x2, y2 = map(int, box)  # Tọa độ các góc bounding box

# #             # Lấy vùng disparity trong bounding box
# #             disparity_region = disparity[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

# #             # Tính khoảng cách trung bình
# #             Z_region = Z[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
# #             average_distance = np.mean(Z_region[Z_region > 0])

# #             # Hiển thị kết quả
# #             cv2.rectangle(left_rectified, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(left_rectified, f"Distance: {average_distance:.2f}m", (x1, y1 - 10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # # Hiển thị ảnh kết quả
# # cv2.imshow("Detected Objects with Distance", left_rectified)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load mô hình YOLO từ Ultralytics
# model = YOLO(r"c:\Users\LAPTOP24H\Downloads\model_200ep_27_11_3_label.pt")  # Đường dẫn tới mô hình YOLOv5

# # Đường dẫn tới video trái và phải
# video_left_path = r"c:\Users\LAPTOP24H\Downloads\20241128_095323_EF.mp4"
# video_right_path = r"c:\Users\LAPTOP24H\Downloads\20241128_095323_ER.mp4"

# # Mở video trái và phải
# cap_left = cv2.VideoCapture(video_left_path)
# cap_right = cv2.VideoCapture(video_right_path)

# # Thiết lập các thông số disparity
# window_size = 5
# min_disp = 0
# num_disp = 16 * 5
# block_size = 5

# stereo = cv2.StereoSGBM_create(
#     minDisparity=min_disp,
#     numDisparities=num_disp,
#     blockSize=block_size,
#     P1=8 * 3 * window_size ** 2,
#     P2=32 * 3 * window_size ** 2,
#     disp12MaxDiff=1,
#     preFilterCap=63,
#     uniquenessRatio=15,
#     speckleWindowSize=0,
#     speckleRange=2,
# )

# # Thông số camera stereo
# baseline = 0.54  # Đơn vị: mét
# focal_length_x = 718.91823876  # Lấy từ camera_matrix_1

# # Xử lý từng frame từ hai video
# while cap_left.isOpened() and cap_right.isOpened():
#     ret_left, left_frame = cap_left.read()
#     ret_right, right_frame = cap_right.read()

#     if not ret_left or not ret_right:
#         break

#     # Chuyển ảnh sang grayscale
#     gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
#     gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

#     # Tính Disparity Map
#     disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16

#     # Tránh chia cho 0, đặt giá trị nhỏ nhất là 0.1
#     disparity[disparity <= 0] = 0.1

#     # Tính toán khoảng cách từ disparity
#     Z = (focal_length_x * baseline) / disparity

#     # Chạy YOLO để phát hiện đối tượng
#     results = model(left_frame)

#     # Xử lý kết quả phát hiện
#     h, w = left_frame.shape[:2]
#     for result in results:
#         for i in range(len(result.boxes)):
#             # Lấy thông tin từ bounding box
#             box = result.boxes.xyxy[i].cpu().numpy()  # Tọa độ bounding box
#             conf = result.boxes.conf[i].item()       # Độ tin cậy
#             class_id = int(result.boxes.cls[i].item())  # Lớp đối tượng

#             # Kiểm tra ngưỡng confidence
#             if conf > 0.6:
#                 x1, y1, x2, y2 = map(int, box)  # Tọa độ các góc bounding box

#                 # Lấy vùng disparity trong bounding box
#                 disparity_region = disparity[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

#                 # Tính khoảng cách trung bình trong vùng disparity
#                 Z_region = Z[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
#                 average_distance = np.mean(Z_region[Z_region > 0])  # Tránh giá trị 0

#                 # Hiển thị kết quả với lớp đối tượng và khoảng cách
#                 cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(left_frame, f"Class: {class_id} Distance: {average_distance:.2f}m", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Hiển thị video kết quả
#     cv2.imshow("Detected Objects with Distance", left_frame)

#     # Thoát nếu nhấn phím 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Giải phóng tài nguyên
# cap_left.release()
# cap_right.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO

# Load mô hình YOLO từ Ultralytics
model = YOLO(r"c:\Users\LAPTOP24H\Downloads\3_12.pt")  # Đường dẫn tới mô hình YOLO

# Đường dẫn tới video trái và phải
video_left_path = r"c:\Users\LAPTOP24H\Downloads\20241128_100645_EF.mp4"
video_right_path = r"c:\Users\LAPTOP24H\Downloads\20241128_100645_ER.mp4"

# Mở video trái và phải
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
baseline = 0.54  # Đơn vị: mét
focal_length_x = 718.91823876  # Lấy từ camera_matrix_1

# Kích thước nhỏ cho cửa sổ hiển thị
resize_width = 640
resize_height = 360

# Xử lý từng frame từ hai video
while cap_left.isOpened() and cap_right.isOpened():
    ret_left, left_frame = cap_left.read()
    ret_right, right_frame = cap_right.read()

    if not ret_left or not ret_right:
        break

    # Chuyển ảnh sang grayscale
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Tính Disparity Map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16

    # Tránh chia cho 0, đặt giá trị nhỏ nhất là 0.1
    disparity[disparity <= 0] = 0.1

    # Tính toán khoảng cách từ disparity
    Z = (focal_length_x * baseline) / disparity

    # Chạy YOLO để phát hiện đối tượng
    results = model(left_frame)

    # Xử lý kết quả phát hiện
    h, w = left_frame.shape[:2]
    for result in results:
        for i in range(len(result.boxes)):
            # Lấy thông tin từ bounding box
            box = result.boxes.xyxy[i].cpu().numpy()  # Tọa độ bounding box
            conf = result.boxes.conf[i].item()       # Độ tin cậy
            class_id = int(result.boxes.cls[i].item())  # Lớp đối tượng

            # Kiểm tra ngưỡng confidence
            if conf > 0.6:
                x1, y1, x2, y2 = map(int, box)  # Tọa độ các góc bounding box

                # Lấy vùng disparity trong bounding box
                disparity_region = disparity[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                # Tính khoảng cách trung bình trong vùng disparity
                Z_region = Z[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                average_distance = np.mean(Z_region[Z_region > 0])  # Tránh giá trị 0

                # Hiển thị kết quả với lớp đối tượng và khoảng cách
                cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(left_frame, f"Class: {class_id} Distance: {average_distance:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize video và disparity map
    resized_left_frame = cv2.resize(left_frame, (resize_width, resize_height))
    resized_disparity_map = cv2.resize(disparity, (resize_width, resize_height))

    # Hiển thị video trái, phát hiện đối tượng và disparity map
    cv2.imshow("Left Video with YOLO Detection", resized_left_frame)
    cv2.imshow("Disparity Map", resized_disparity_map)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
    