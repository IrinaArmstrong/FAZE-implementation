import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import logging_handler
logger = logging_handler.get_logger(__name__)


class Undistorter:
    """
    Removing the distortion using OpenCV utils.
    The class computes the joint undistortion and rectification transformation
    and represents the result in the form of maps. Then remap image.
    The undistorted image looks like original, as if it is captured
    with a camera using the camera matrix=newCameraMatrix and zero distortion.
    """

    def __init__(self):
        self._map = None
        self._previous_parameters = None

    def __call__(self, image: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray,
                 is_gazecapture=False):
        """
        Arguments:
            image - frame as np.ndarray marix of pixels values.
            camera_matrix - Input camera matrix A of shape 3x3.
            distortion - Input vector of distortion coefficients of 4, 5, 8, 12 or 14 elements:
                        (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]).
        """
        # todo: make checking types of arguments!
        h, w, _ = image.shape
        all_parameters = np.concatenate([camera_matrix.flatten(),
                                         distortion.flatten(),
                                         [h, w]])
        if (self._previous_parameters is None
                or len(self._previous_parameters) != len(all_parameters)
                or not np.allclose(all_parameters, self._previous_parameters)):
            logger.info('Distortion map parameters updated.')
            # The function builds the maps for the inverse mapping algorithm that is used by remap
            # return map_x, map_y
            self._map = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                R=None,  # optional rectification transformation in the object space
                newCameraMatrix=camera_matrix if is_gazecapture else None,
                size=(w, h),  # Undistorted image size
                m1type=cv2.CV_32FC1)  # Type of the first output map

            self._previous_parameters = np.copy(all_parameters)
        # Apply
        return cv2.remap(image, self._map[0], self._map[1], cv2.INTER_LINEAR)