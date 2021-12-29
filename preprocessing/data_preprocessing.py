# Basic
import cv2
import time
import pickle
import random
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union

import logging_handler
logger = logging_handler.get_logger(__name__)

from preprocessing.undistortion import Undistorter
from extensions.landmarks import LandmarksDetector


class Preprocessor:

    def __init__(self, camera_calib_fn: Union[Path, str], **kwargs):
        """
        Data collection & organization for final Person-specific Adaptation of Gaze Network.
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self.__init_camera_calibration(camera_calib_fn)
        self.__undistorter = Undistorter()
        self.__lmks_detector = LandmarksDetector()

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

