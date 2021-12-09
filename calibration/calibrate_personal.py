# Basic
import cv2
import pickle
import random
import numpy as np
from typing import Union
from pathlib import Path

from monitor import Monitor

import logging_handler
logger = logging_handler.get_logger(__name__)


class PersonCalibrator:

    def __init__(self, make_grid: bool = False, grid_size: int = 16):
        """
        Data collection & organization for final Person-specific Adaptation of Gaze Network.
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self._monitor = Monitor()
        self._monitor_w = self._monitor.get_width_pixels()
        self._monitor_h = self._monitor.get_height_pixels()

        self._has_grid = make_grid
        self._grid_size = grid_size

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

    def create_calibration_image(self, point_num: int):
        """
        Generate stimulus image with random point for calibration.
        :return: image to show and 3D gaze target
        """
        if self._has_grid:
            if self._grid_size == 9:
                row = point_num % 3
                col = int(point_num / 3)
                x = int((0.02 + 0.48 * row) * self._monitor_w)
                y = int((0.02 + 0.48 * col) * self._monitor_h)
            elif self._grid_size == 16:
                row = point_num % 4
                col = int(point_num / 4)
                x = int((0.05 + 0.3 * row) * self._monitor_w)
                y = int((0.05 + 0.3 * col) * self._monitor_h)
        else:
            x = int(random.uniform(0, 1) * self._monitor_w)
            y = int(random.uniform(0, 1) * self._monitor_h)

        # Compute the ground truth point of regard
        x_cam, y_cam, z_cam = self._monitor.monitor_to_camera(x, y)
        g_target = (x_cam, y_cam)


