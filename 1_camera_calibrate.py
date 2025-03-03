import numpy as np
import cv2
import glob


class StereoCalibration(object):
    def __init__(self, images_left_path, images_right_path):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((11*7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:11, 0:7].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.read_images(images_left_path, images_right_path)

    def read_images(self, images_left_path, images_right_path):
        images_left = glob.glob(images_left_path + '/*.png')
        images_right = glob.glob(images_right_path + '/*.png')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (11, 7), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (11, 7), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l:
                cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (11, 7), corners_l, ret_l)
                cv2.imshow(f"Left Image {i+1}", img_l)
                cv2.waitKey(500)

            if ret_r:
                cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (11, 7), corners_r, ret_r)
                cv2.imshow(f"Right Image {i+1}", img_r)
                cv2.waitKey(500)

            img_shape = gray_l.shape[::-1]

        # Calibrate each camera
        _, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        _, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_ZERO_TANGENT_DIST
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        camera_model = {
            'M1': M1, 'M2': M2,
            'dist1': d1, 'dist2': d2,
            'R': R, 'T': T, 'E': E, 'F': F
        }

        cv2.destroyAllWindows()
        return camera_model


if __name__ == '__main__':
    images_left_path = r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\left"  # Thay bằng đường dẫn thư mục chứa ảnh trái
    images_right_path = r"C:\Users\LAPTOP24H\Downloads\streamlit\camera_stereo\camera-calibration-using-opencv-python\right"  # Thay bằng đường dẫn thư mục chứa ảnh phải

    cal_data = StereoCalibration(images_left_path, images_right_path)
