# Basic
import cv2
import pickle
import random
import numpy as np
from typing import Union
from pathlib import Path

from monitor import Monitor
from io_utils import VideoRecorder

import logging_handler
logger = logging_handler.get_logger(__name__)


class PersonCalibrator:

    available_directions = ['l', 'r', 'u', 'd']

    def __init__(self, make_grid: bool = False, grid_size: int = 16):
        """
        Data collection & organization for final Person-specific Adaptation of Gaze Network.
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self._monitor = Monitor()
        self._video_recorder = VideoRecorder()
        self._monitor_w = self._monitor.get_width_pixels()
        self._monitor_h = self._monitor.get_height_pixels()

        self._has_grid = make_grid
        self._grid_size = grid_size

    def calibrate(self, num_point_calibrate: int = 9):
        """
        Calibration with desired number of samples (default is 9, enough for good precision).
        """
        # Create a window as a placeholder for images
        cv2.namedWindow("Calibration_image", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        calibration_data = {'frames': [],
                            'g_targets': []}
        point_num = 0
        while point_num < num_point_calibrate:
            self._video_recorder.start()
            direction = random.choice(self.available_directions)
            image_to_show, g_target = self.create_calibration_image(point_num=point_num,
                                                                    direction=direction)
            cv2.imshow('Calibration image', image_to_show)
            key_press = cv2.waitKey(0)
            if key_press == ord(direction):
                logger.info(f"Right direction pressed, left {num_point_calibrate - point_num} calibration points.")
                self._video_recorder.stop()
                calibration_data['frames'].append(self._video_recorder.get_frames())
                calibration_data['g_t'].append(g_target)
                point_num += 1
            # Q to quit
            elif key_press & 0xFF == ord('q'):
                logger.info("Stopping calibration...")
                self._video_recorder.stop()
                cv2.destroyAllWindows()
                break
            else:
                logger.info(f"Wrong direction pressed, try again.")
                self._video_recorder.stop()

    def collect_data(self, num_points_show: int = 20):
        """
        Collect data: frames and gaze targets for fine tuning network
        for person specific evaluation.
        """
        pass

    def create_calibration_image(self, point_num: int, direction: str):
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

        # normal size sans-serif font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # BGR color space
        color = [0, 0, 0]
        # Image to show
        image = np.ones((self._monitor_h, self._monitor_w, 3), np.float32)
        # Flip horizontally
        if direction == 'r' or direction == 'l':
            if direction == 'r':
                cv2.putText(image, "R", (x, y), font, 0.5, color, 2, cv2.LINE_AA)
            elif direction == 'l':
                cv2.putText(image, "L", (self._monitor_w - x, y), font, 0.5, color, 2, cv2.LINE_AA)
                image = cv2.flip(image, 1)
        #  Flip vertically - ???
        elif direction == 'u' or direction == 'd':
            image_trf = np.ones((self._monitor_w, self._monitor_h, 3), np.float32)
            if direction == 'd':
                cv2.putText(image_trf, "D", (y, x), font, 0.5, color, 2, cv2.LINE_AA)
            elif direction == 'u':
                cv2.putText(image_trf, "U", (self._monitor_h - y, x), font, 0.5, color, 2, cv2.LINE_AA)
                image_trf = cv2.flip(image_trf, 1)
            image = image_trf.transpose((1, 0, 2))

        return image, g_target


