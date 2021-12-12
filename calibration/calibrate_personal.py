# Basic
import cv2
import time
import pickle
import random
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union

from monitor import Monitor
from io_utils import VideoRecorder

import logging_handler
logger = logging_handler.get_logger(__name__)


class PersonCalibrator:

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

    def calibrate_with_points(self, num_point_calibrate: int = 9, to_save: bool = True,
                              save_fn: Union[str, Path] = None):
        """
        Calibration with desired number of samples (default is 9, enough for good precision).
        """
        logger.info(f"INSTRUCTIONS:")
        logger.info(f"Press SPACE key when you look straight at the point,")
        logger.info(f"After the key pressed, next point will appear.")
        logger.info(f"Press Q to force quit collecting data for calibration.")

        # Create a window as a placeholder for images
        cv2.namedWindow("Calibration_image", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        calibration_data = {'frames': [],  # as list of lists of arrays [480x640x3]
                            # 3D gaze targets on the screen
                            'g_targets_3D_mm': [],  # as list of lists of tuples [float, float, float]
                            # 2D gaze targets on the screen in pixels
                            'calib_points_2D_pixels': [],  # as list of lists of tuples [float, float]
                            }
        point_num = 0
        while point_num < num_point_calibrate:
            logger.info(f"New point #{point_num}")

            image_to_show, g_target, point_coords = self.create_simple_calibration_image(point_num=point_num)
            cv2.imshow('Calibration image', image_to_show)
            key_press = cv2.waitKey(0)
            # Space button pressed
            if key_press & 0xFF == 32:
                # Record few frames when person is looking straight at point
                self._video_recorder.start()
                time.sleep(1)  # wait for 1 second
                self._video_recorder.stop()
                logger.info(f"{num_point_calibrate - point_num} calibration points are left")
                calibration_data['frames'].append(self._video_recorder.get_frames())
                calibration_data['g_targets_3D_mm'].append(g_target)
                calibration_data['calib_points_2D_pixels'].append(point_coords)
                point_num += 1
            # Q to quit
            elif key_press & 0xFF == ord('q'):
                logger.info("Stopping calibration...")
                self._video_recorder.stop()
                cv2.destroyAllWindows()
                break
            else:
                logger.info(f"Wrong direction pressed, try again.")

        cv2.destroyAllWindows()
        logger.info("Calibration finished.")
        if to_save:
            self.save_calibration_data(calibration_data, save_fn)
        return calibration_data

    def collect_data(self, num_points_show: int = 20):
        """
        Collect data: frames and gaze targets for fine tuning network
        for person specific evaluation.
        Use specific randomized stimulus drawing.
        """
        pass

    def create_simple_calibration_image(self, point_num: int):
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
            eps = 1e-4  # for not placing point on the edges of screen
            x = int(random.uniform(0 + eps, 1 - eps) * self._monitor_w)
            y = int(random.uniform(0 + eps, 1 - eps) * self._monitor_h)

        # Compute the ground truth point of regard
        x_cam, y_cam, z_cam = self._monitor.monitor_to_camera(x, y)
        g_target = (x_cam, y_cam, z_cam)

        # BGR color space
        color = [0, 0, 0]
        # Radius of circle
        radius = 10  # in pixels
        # Line thickness of 2 px
        thickness = -1
        # Image to show
        image = np.ones((self._monitor_h, self._monitor_w, 3), np.float32)
        image = cv2.circle(image, (x, y), radius, color, thickness)
        return image, g_target, (x, y)

    def save_calibration_data(self, calibration_data: Dict[str, List[Any]],
                              save_fn: Union[str, Path] = None):
        """
        Save data collected during calibration procedure for future use.
        """
        if save_fn is None:
            save_fn = f'person_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_calibration_data.pkl'
        save_fn = self._root_dir / "outputs" / save_fn
        logger.info(f"Saving calibration data to: {str(save_fn)}")
        with open(str(save_fn), 'wb') as f:
            pickle.dump(calibration_data, f)
        logger.info(f"Successfully saved to: {str(save_fn)}")


if __name__ == "__main__":
    calibrator = PersonCalibrator(make_grid=False,
                                  grid_size=9)
    calib_data = calibrator.calibrate_with_points(num_point_calibrate=10, to_save=True)
    print(f"Calibration finished!")
