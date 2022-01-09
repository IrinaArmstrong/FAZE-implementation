# Basic
import cv2
import time
import pickle
import random
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

import logging_handler
logger = logging_handler.get_logger(__name__)

from preprocessing.undistortion import Undistorter
from extensions.landmarks import LandmarksDetector
from extensions.headpose_estimation import EOSHeadPoseEstimator
from preprocessing.gaze_mormalization import RuntimeGazeNormalizer


class Preprocessor:

    def __init__(self, camera_calib_fn: Union[Path, str], **kwargs):
        """
        Data collection & organization for final Person-specific Adaptation of Gaze Network.
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self.__init_camera_calibration(camera_calib_fn)
        self.__undistorter = Undistorter()
        # Use the solutions from original paper & reposotory
        # todo: embed MediaPipe for face & landmarks detection, head pose estimation -> all in one lib
        self.__lmks_detector = LandmarksDetector()
        self.__headpose_estimator = EOSHeadPoseEstimator()
        self.__gaze_normalizer = RuntimeGazeNormalizer(self.__headpose_estimator.get_anchor_landmarks_3d())

    def __init_camera_calibration(self, camera_calib_fn: Union[Path, str]):
        """
        Initialize camera matrix and distortion parameters from pre-saved calibration file.
        """
        if type(camera_calib_fn) == str:
            camera_calib_fn = Path(camera_calib_fn).resolve()
        if not camera_calib_fn.exists():
            logger.error(f"Camera calibration parameters file not found. Check file name and try again.")
            logger.error(f"""In case you haven't calibrated the camera yet, 
            run `calibrate_camera.py` and save parameters. Then try again.""")
            raise FileNotFoundError(f"Camera calibration parameters file not found. Check file name and try again.")

        try:
            with open(camera_calib_fn, 'rb') as f:
                camera_calib_params = pickle.load(f)
        except Exception as ex:
            logger.error(f"Error occurred during camera calibration parameters reading: {ex}")
            camera_calib_params = {'camera_matrix': np.eye(3),
                                   'distortion_coeffs': np.zeros((1, 5))}

        self.__camera_matrix = camera_calib_params.get('camera_matrix', np.eye(3))
        self.__distortion_coeffs = camera_calib_params.get('distortion_coeffs', np.zeros((1, 5)))

    def process_frame(self, frame: np.ndarray, gaze_target_2d: Tuple[int, int] = None,
                      gaze_target_3d: Tuple[float, float, float] = None,
                      visualize: bool = False):
        """
        Process single frame from data stream with following pipeline of operations:
            - undistort raw frame
            - detect face location (if it was not successfully found, skip current frame and exit function)
            - filter location with Kalman filters
            - detect landmarks and filter them with Kalman filters
            - estimate current frame head pose R and T
            - normalize gaze vector using head pose (in the way that was introduced in
            "Revisiting Data Normalization for Appearance-Based Gaze Estimation" X. Zhang 2018)
            (see code: https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/)
            -

        """
        # Undistort frame
        frame = self.__undistorter(frame, self.__camera_matrix, self.__distortion_coeffs)

        # Detect face & landmarks
        success_flg, lmks, face_location = self.__lmks_detector.detect(frame, return_face_location=True)
        if not success_flg:
            return None

        # Estimate head pose
        rvec, tvec = self.__headpose_estimator.estimate_headpose(frame, landmarks_2d=lmks,
                                                                 camera_matrix=self.__camera_matrix,
                                                                 distortion_coeffs=self.__distortion_coeffs,
                                                                 visualize=visualize)

        # Create normalized: eye patch, gaze vector and head pose (R, t),
        # if the ground truth point of regard (PoR) is given.
        # * The line of sight (LoS) is a ray that originates from the center of the eye and follows the eyesight.
        # * The point of regard (PoR) is the intersection of the LoS ray with the stimulus plane
        head_pose = (rvec, tvec)
        por = None
        # will be defined as 3d vector in pixels of the screen
        if gaze_target_2d is not None:
            por = np.zeros((3, 1))
            por[:2] = gaze_target_2d  # (with z coord set to zero)

        normalized_frame_attr = self.__gaze_normalizer.normalize_frame(frame=frame, rvec=rvec, tvec=tvec,
                                                                       camera_matrix=self.__camera_matrix,
                                                                       gaze_target_3d_pix=por)
        # Normalized data returned as key-value pairs.
        # Available keys: 'norm_frame', 'gaze_direction', 'gaze_origin', 'gaze_target',
        # 'head_pose', 'normalization_matrix', 'normalized_gaze_direction', 'normalized_head_pose'




