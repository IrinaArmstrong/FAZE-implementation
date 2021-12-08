import pickle

import cv2
from typing import Union
from pathlib import Path

import numpy as np

import logging_handler
logger = logging_handler.get_logger(__name__)


class PersonCalibrator:

    def __init__(self):
        """
        Data collection & organization for final Person-specific Adaptation of Gaze Network.
        """
        self._root_dir = Path(__file__).resolve().parent.parent

    def calibrate(self, num_point_calibrate: int = 9):
        """
        Calibration with desired number of samples (default is 9, enough for good precision).
        """
        pass

    def collect_data(self, num_points_show: int = 20):
        """
        Collect data: frames and gaze targets for fine tuning network
        for person specific evaluation.
        """
        pass
