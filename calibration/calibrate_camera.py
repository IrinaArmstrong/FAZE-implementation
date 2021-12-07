import pickle

import cv2
from typing import Union
from pathlib import Path

import numpy as np

import logging_handler
logger = logging_handler.get_logger(__name__)


class CameraCalibrator:

    def __init__(self, camera_idx: int = 0):
        """
        Initialize camera, selected by index
        :param camera_idx: 0 or 1 means camera's or webcam's index
            (important if there are more then single one).
        """
        self._camera_idx = camera_idx
        self._root_dir = Path(__file__).resolve().parent.parent
        # Camera matrix
        # include focal length (fx,fy) and optical centers (cx, cy).
        # fx  0  cx
        # 0  fy  cy
        # 0   0  1
        self._camera_matrix = np.eye(3)
        # Distortion coefficients as (k1, k2, p1, p2, k3)
        self._distortion_coeffs = np.zeros((1, 5))
        self._capture = None

    def calibrate(self, image_fn: Union[str, Path],
                  save_fn: str = None):
        """
        Calibrate camera and save its parameters to local file
        """
        logger.info(f"Initializing capture stream from camera # {self._camera_idx}...")
        self._capture = cv2.VideoCapture(self._camera_idx)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info(f"Calibrate camera once and use saved parameters later.")
        logger.info(f"Print pattern.png, paste on a clipboard, show to camera, capture non-blurry images in which points are detected well.")
        logger.info(f"Press S to save frame, C to continue to next frame and Q to quit collecting data and proceed to calibration.")
        logger.info("Starting...")

        # self.__show_calibration_pattern(image_fn)
        self.__mirror_calibration()
        self._capture.release()

        # Save
        if save_fn is None:
            save_fn = f'camera_{self._camera_idx}_calibration_params.pkl'
        with open(save_fn, 'wb') as f:
            pickle.dump({'camera_matrix': self._camera_matrix,
                         'distortion_coeffs': self._distortion_coeffs}, f)
        logger.info(f"Successfully saved to: {save_fn}")

        logger.info(f"Finished.")

    def __show_calibration_pattern(self, image_fn: Union[str, Path]):
        # Check dataset folder
        if type(image_fn) == str:
            image_fn = Path(image_fn).resolve()
        if not image_fn.exists():
            logger.error(f"Calibration pattern image: {image_fn} do not exists!")
            raise FileNotFoundError(f"Calibration pattern image: {image_fn} do not exists!")
        try:
            calib_img = cv2.imread(image_fn)
            cv2.imshow("Calibration pattern image", calib_img)
            cv2.waitKey(0)
        except Exception as ex:
            logger.error("Error occured during calibration pattern image translation!")
            logger.error(f"Try to open it yourself!")

    def __mirror_calibration(self):
        """
        Provides an implementation of our mirror-based camera calibration algorithm.
        """
        # termination criteria
        # cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
        # cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        pts = np.zeros((6 * 9, 3), np.float32)
        pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # capture calibration frames
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.
        frames = []

        while True:
            retval, frame = self._capture.read()
            frame_copy = frame.copy()

            corners = []
            # retval = false if no frames has been grabbed
            if retval:
                # frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Finds the approximate positions of internal corners of the chessboard
                # (9, 6) - number of inner corners per a chessboard row and column
                ret_corners, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                # ret_corners = false if no corners has been found
                if ret_corners:
                    # The function iterates to find the sub-pixel accurate location of corners
                    # (11, 11) - Half of the side length of the search window
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    # Draw and display the corners
                    cv2.drawChessboardCorners(frame_copy, (9, 6), corners, retval)
                    cv2.imshow('Points detected', frame_copy)
                    # User menu
                    # S to save
                    if cv2.waitKey(0) & 0xFF == ord('s'):
                        img_points.append(corners)
                        obj_points.append(pts)
                        frames.append(frame)
                        logger.info(f"Frame & corners saved.")
                        cv2.destroyAllWindows()
                    # C to continue,
                    elif cv2.waitKey(0) & 0xFF == ord('c'):
                        logger.info(f"Continue...")
                        cv2.destroyAllWindows()
                        continue
                    # Q to quit
                    elif cv2.waitKey(0) & 0xFF == ord('q'):
                        logger.info("Quite.")
                        cv2.destroyAllWindows()
                        break

        # Compute calibration matrices
        logger.info("Calibrating camera...")
        ret_calib, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,
                                                                                 frames[0].shape[0:2], None, None)

        # Check re-projection error
        error = 0.0
        for i in range(len(frames)):
            proj_img_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, distortion)
            error += (cv2.norm(img_points[i], proj_img_points, cv2.NORM_L2) / len(proj_img_points))
        logger.info(f"Camera calibrated successfully, total re-projection error: {error / len(frames)}")

        self._camera_matrix = camera_matrix
        self._distortion_coeffs = distortion
        logger.info(f"Calculated camera parameters:")
        logger.info(f"Camera matrix:\n{self._camera_matrix}")
        logger.info(f"Distortion coefficients:\n{self._distortion_coeffs}")


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent
    camera_params_fn = root_path / "settings" / "asus_laptop_camera_calibration_params.pkl"
    calib_pattern_params_fn = root_path / "calibration" / "pattern.png"
    calibrator = CameraCalibrator(camera_idx=0)
    calibrator.calibrate(image_fn=calib_pattern_params_fn,
                         save_fn=str(camera_params_fn))




