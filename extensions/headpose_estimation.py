# Basic
import sys
import cv2
import numpy as np
from itertools import chain
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Tuple

# Landmarks detection
import eos
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

import logging_handler
logger = logging_handler.get_logger(__name__)

from extensions.landmarks import (hr_landmarks2ids_mapping, hr_ibug_landmarks2ids_mapping,
                                  LandmarksDetector)

# Mapping based on additional/BlazeFace_mesh_map.jpg
# Assuming: left ---> right (on image/photo!)
mp_landmarks2ids_mapping = {
    "nose bridge": [168, 197, 195, 5],
    "nose base": [102, 2, 294],
    "mouth corners": [61, 291],
    "left-eye corners": [33, 133],  # the most left eye on image
    "right-eye corners": [362, 263]  # the most right eye on image
}


class MPHeadPoseEstimator:

    def __init__(self,  **kwargs):
        """
        Head pose estimation class, that calculates the relative orientation
        of the viewer's head with respect to a camera as solution of as a Perspective-n-Point problem.
        * Uses Google's MediaPipe framework - a set of DL/ML solutions for live and streaming media
        (Face Detection, Face Mesh Creation, Iris Detection, Hands Detection, Pose Estimation, etc.).
        In this class we use Face Mesh Model for deriving 3D World coordinates of landmark points.
        * And Calibration camera matrix, which consist of intrinsic parameters.
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self.__model = mp.solutions.face_mesh
        # Mediapipe’s face landmarks detection model for video stream
        self.__mp_face_mesh = mp.solutions.face_mesh
        self.__face_mesh = self.__mp_face_mesh.FaceMesh(static_image_mode=False,
                                                        min_detection_confidence=0.5,
                                                        min_tracking_confidence=0.5)

    def estimate_headpose(self, frame: np.ndarray, landmarks_2d: np.ndarray,
                          camera_matrix: np.ndarray, distortion_coeffs: np.ndarray,
                          visualize: bool = False) ->  Tuple[Any, Any]:
        """
        Calculate the rotational (R) and the translational (T) matrix of head pose
        by solving standart Perspective-n-Point (PnP) equation.
        For this requires three components, such as:
            - The 2D coordinates in the image space (take as input argument);
            - The 3D coordinates in the world space (calculate inside);
            - The camera parameters, such as the focal points,
            the center coordinate, and the skew parameter (take as input argument);
        :return - the translational vector (Tvec) and the rotational vector (Rvec).
        for more info see: https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600
        """
        if camera_matrix.shape != (3, 3):
            logger.error(f"Invalid `camera_matrix` shape: {camera_matrix.shape}, required [3 x 3] matrix.")
            return [None, None]

        if landmarks_2d.shape != (68, 2):
            logger.error(f"Invalid `landmarks_2d` shape: {landmarks_2d.shape}, required [68 x 2] matrix.")
            return [None, None]

        # Select 'anchor' landmarks for PnP problem solution
        anchor_landmarks_2d = dict.fromkeys(hr_landmarks2ids_mapping.keys())
        for lmks_key, lmks_ids in hr_landmarks2ids_mapping.items():
            anchor_landmarks_2d[lmks_key] = landmarks_2d[lmks_ids, :]
        # As Nx2 1-channel array, where N is the number of points.
        anchor_landmarks_2d = np.vstack(list(anchor_landmarks_2d.values()))

        # Estimate 3d landmarks
        anchor_landmarks_3d = self.estimate_3d_landmarks(frame, visualize=visualize)
        # As Nx3 1-channel array, where N is the number of points.
        anchor_landmarks_3d = np.vstack(list(anchor_landmarks_3d.values()))

        # Initial fit, use of RANSAC makes the solution resistant to outliers
        # cv.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs
        # [, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError
        # [, confidence[, inliers[, flags]]]]]]]]	) ->	retval, rvec, tvec, inliers
        success, rvec, tvec, inliers = cv2.solvePnPRansac(anchor_landmarks_3d, anchor_landmarks_2d,
                                                          camera_matrix, distortion_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if success:
            logger.debug(f"""PnPRansac finished successfully with 
            {len(inliers)} inliers from {len(anchor_landmarks_3d)} anchor points.""")
        else:
            logger.warning(f"PnPRansac finished without success!")

        # Second fit for higher accuracy
        # cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs
        # [, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) → retval, rvec, tvec
        success, rvec, tvec = cv2.solvePnP(anchor_landmarks_3d, anchor_landmarks_2d,
                                           camera_matrix, distortion_coeffs,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        # cv::SOLVEPNP_ITERATIVE Iterative method is based on a Levenberg-Marquardt optimization
        if success:
            logger.info(f"""PnP finished successfully.""")
        else:
            logger.warning(f"PnP finished without success!")

        # Converts a rotation matrix to a rotation vector
        rmat, jac = cv2.Rodrigues(rvec)
        return rvec, tvec

    def estimate_3d_landmarks(self, frame: np.ndarray, visualize: bool = False) \
            -> Dict[str, List[Tuple[float]]]:
        """
        Performs the 3D landmarks detection. Requires a pass of the image (in RGB format),
        then runs detection  pipeline and gets a list of 468 facial landmarks for each detected face in the image.
        Each landmark will have:
            x – It is the landmark x-coordinate normalized to [0.0, 1.0] by the image width.
            y – It is the landmark y-coordinate normalized to [0.0, 1.0] by the image height.
            z – It is the landmark z-coordinate normalized to roughly the same scale as x.
            It represents the landmark depth with the center of the head being the origin,
            and the smaller the value is, the closer the landmark is to the camera.
        """
        # To RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get image proportions
        frame_h, frame_w, _ = frame_rgb.shape

        # To improve performance - lock the data, making it read-only
        frame_rgb.flags.writeable = False
        # Detect
        face_mesh_results = self.__face_mesh.process(frame_rgb)
        # Unlock
        frame_rgb.flags.writeable = True

        # Construct output format
        landmarks_res = dict.fromkeys(mp_landmarks2ids_mapping.keys())

        # If any face detected on frame
        if face_mesh_results.multi_face_landmarks:
            # Get first found face landmarks
            all_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
            for lmks_key, lmks_ids in mp_landmarks2ids_mapping.items():
                landmarks_res[lmks_key] = [(all_landmarks[i].x * frame_w,
                                            all_landmarks[i].y * frame_h,
                                            all_landmarks[i].z * frame_w)
                                           for i in lmks_ids]
        else:
            logger.error(f"No facial landmarks found with MediaPipe on frame.")

        if visualize:
            self.visualize_landmarks(frame=frame_rgb, landmarks=all_landmarks)
        return landmarks_res

    @staticmethod
    def visualize_landmarks(frame: np.ndarray, landmarks: Any,
                            frame_size: List[int] = (960, 720),
                            figsize: List[int] = (20, 17)):
        # Create a copy of the sample image in RGB format to draw the found facial landmarks on.
        frame_copy = frame.copy()
        frame_copy = cv2.resize(frame_copy, frame_size)

        # Prepare DrawingSpec for drawing the face landmarks later.
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=-1, circle_radius=2)
        contours_style = mp.solutions.drawing_styles.get_default_face_mesh_contours_style()

        for lmks_key, lmks_ids in mp_landmarks2ids_mapping.items():
            # NormalizedLandmarkList format
            lmks = landmark_pb2.NormalizedLandmarkList(landmark=[landmarks[i] for i in lmks_ids])
            mp.solutions.drawing_utils.draw_landmarks(image=frame_copy,
                                                      landmark_list=lmks,
                                                      landmark_drawing_spec=drawing_spec,
                                                      connection_drawing_spec=contours_style)
        # Display the resultant image with the face mesh drawn.
        fig = plt.figure(figsize=figsize)
        plt.title("Selected landmarks")
        plt.axis('off')
        plt.imshow(frame_copy)
        plt.show()


class EOSHeadPoseEstimator:

    def __init__(self,  **kwargs):
        """
        Head pose estimation class, that calculates the relative orientation
        of the viewer's head with respect to a camera as solution of as a Perspective-n-Point problem.
        * Uses EOS - a lightweight 3D Morphable Face Model fitting library (C++11/14)
        for deriving 3D World coordinates of landmark points;
        * And Calibration camera matrix, which consist of intrinsic parameters.
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self._lib_dir = Path(__file__).resolve().parent / "eos"
        # Load SFM model's weights
        self.__model = eos.morphablemodel.load_model(str(self._lib_dir / 'share' / 'sfm_shape_3448.bin'))
        # Get shape PCA model
        self.__shape_model = self.__model.get_shape_model()
        # Mapping from the 68-point ibug annotations to the Surrey Face Model (SFM) mesh vertex indices.
        # see: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
        self.__landmark_mapper = eos.core.LandmarkMapper(str(self._lib_dir / 'share' / 'ibug_to_sfm.txt'))

        # For all 68 landmarks find mean coords values (if they are included in mapping)
        self.__sfm_points_ibug_subset = np.array([
            self.__shape_model.get_mean_at_point(int(self.__landmark_mapper.convert(str(d))))
            for d in range(1, 69) if self.__landmark_mapper.convert(str(d)) is not None])

        # Get mean coords for selected anchor landmarks [13 x 3]
        self.__sfm_points_for_pnp = np.array([
            self.__shape_model.get_mean_at_point(int(self.__landmark_mapper.convert(str(d))))
            for d in list(chain.from_iterable(hr_landmarks2ids_mapping.values()))])

        # Rotate face around
        rotate_mat = np.asarray([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]], dtype=np.float64)
        self.__sfm_points_ibug_subset = np.matmul(self.__sfm_points_ibug_subset.reshape(-1, 3), rotate_mat)
        self.__sfm_points_for_pnp = np.matmul(self.__sfm_points_for_pnp.reshape(-1, 3), rotate_mat)

        # Center on mean point between eye corners
        between_eye_point = np.mean(self.__sfm_points_for_pnp[-4:, :], axis=0)
        self.__sfm_points_ibug_subset -= between_eye_point.reshape(1, 3)
        self.__sfm_points_for_pnp -= between_eye_point.reshape(1, 3)

    def estimate_headpose(self, frame: np.ndarray, landmarks_2d: np.ndarray,
                          camera_matrix: np.ndarray, distortion_coeffs: np.ndarray,
                          visualize: bool = False) -> Tuple[Any, Any]:
        """
        Calculate the rotational (R) and the translational (T) matrix of head pose
        by solving standart Perspective-n-Point (PnP) equation.
        For this requires three components, such as:
            - The 2D coordinates in the image space (take as input argument);
            - The 3D coordinates in the world space (calculate inside);
            - The camera parameters, such as the focal points,
            the center coordinate, and the skew parameter (take as input argument);
        :return - the translational vector (Tvec) and the rotational vector (Rvec).
        for more info see: https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600
        """
        if camera_matrix.shape != (3, 3):
            logger.error(f"Invalid `camera_matrix` shape: {camera_matrix.shape}, required [3 x 3] matrix.")
            return [None, None]

        if landmarks_2d.shape != (68, 2):
            logger.error(f"Invalid `landmarks_2d` shape: {landmarks_2d.shape}, required [68 x 2] matrix.")
            return [None, None]

        # Select 'anchor' landmarks for PnP problem solution
        anchor_landmarks_2d = np.array([landmarks_2d[i - 1, :]
                                        for i in list(chain.from_iterable(hr_ibug_landmarks2ids_mapping.values()))],
                                       dtype=np.float64)
        # As Nx2 1-channel array, where N is the number of points.
        # anchor_landmarks_2d = np.vstack(list(anchor_landmarks_2d.values()))

        # As Nx3 1-channel array, where N is the number of points.
        anchor_landmarks_3d = self.__sfm_points_for_pnp

        # Initial fit, use of RANSAC makes the solution resistant to outliers
        # cv.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs
        # [, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError
        # [, confidence[, inliers[, flags]]]]]]]]	) ->	retval, rvec, tvec, inliers
        success, rvec, tvec, inliers = cv2.solvePnPRansac(anchor_landmarks_3d, anchor_landmarks_2d,
                                                          camera_matrix, distortion_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if success:
            logger.debug(f"""PnPRansac finished successfully with 
                    {len(inliers)} inliers from {len(anchor_landmarks_3d)} anchor points.""")
        else:
            logger.warning(f"PnPRansac finished without success!")

        # Second fit for higher accuracy
        # cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs
        # [, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) → retval, rvec, tvec
        success, rvec, tvec = cv2.solvePnP(anchor_landmarks_3d, anchor_landmarks_2d,
                                           camera_matrix, distortion_coeffs,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        # cv::SOLVEPNP_ITERATIVE Iterative method is based on a Levenberg-Marquardt optimization
        if success:
            logger.info(f"""PnP finished successfully.""")
        else:
            logger.warning(f"PnP finished without success!")

        if visualize:
            self.visualize_headpose(frame, rvec, tvec, camera_matrix, distortion_coeffs)

        # Converts a rotation matrix to a rotation vector
        # rmat, jac = cv2.Rodrigues(rvec)
        return rvec, tvec

    def project_points(self, rvec: np.ndarray, tvec: np.ndarray,
                       camera_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> np.ndarray:
        """
        Compute projection of 3D points in World Coordinates System (WCS)
        to Image Coordinate System (ICS).
        :param rvec: The rotation vector of shape (3, 1).
        :param tvec: The translation vector of shape (3, 1).
        :param camera_parameters: Camera metrix of shape (3, 3).
        :return:
        """
        if rvec.shape != (3, 1):
            if len(rvec.flatten()) == 3:
                rvec = rvec.reshape(3, 1)
        else:
            logger.error(f"Invalid `rvec` shape: {rvec.shape}, required [3 x 1] vector.")
            return None

        if tvec.shape != (3, 1):
            if len(tvec.flatten()) == 3:
                tvec = tvec.reshape(3, 1)
        else:
            logger.error(f"Invalid `tvec` shape: {tvec.shape}, required [3 x 1] vector.")
            return None

        if camera_matrix.shape != (3, 3):
            logger.error(f"Invalid `camera_matrix` shape: {camera_matrix.shape}, required [3 x 3] matrix.")
            return None

        # Projects 3D points to an image plane
        points, _ = cv2.projectPoints(self.__sfm_points_for_pnp, rvec, tvec,
                                      camera_matrix, distortion_coeffs)
        return points

    def visualize_headpose(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                 camera_matrix: np.ndarray, distortion_coeffs: np.ndarray,
                 frame_size: List[int] = (960, 720)):

        model_axes = np.array([
            np.array([0., -20., 0.]).reshape(1, 3),
            np.array([50., -20., 0.]).reshape(1, 3),
            np.array([0., -70., 0.]).reshape(1, 3),
            np.array([0., -20., -50.]).reshape(1, 3)
        ])

        # Create a copy of the sample image in RGB format to draw the found facial landmarks on.
        frame_copy = frame.copy()

        # Project axes to ICS
        proj_axes, _ = cv2.projectPoints(model_axes, rvec, tvec, camera_matrix, distortion_coeffs)

        cv2.line(frame_copy,
                 (int(proj_axes[0, 0, 0]), int(proj_axes[0, 0, 1])),
                 (int(proj_axes[1, 0, 0]), int(proj_axes[1, 0, 1])),
                 (0, 255, 255), 2)
        cv2.line(frame_copy,
                 (int(proj_axes[0, 0, 0]), int(proj_axes[0, 0, 1])),
                 (int(proj_axes[2, 0, 0]), int(proj_axes[2, 0, 1])),
                 (255, 0, 255), 2)
        cv2.line(frame_copy,
                 (int(proj_axes[0, 0, 0]), int(proj_axes[0, 0, 1])),
                 (int(proj_axes[3, 0, 0]), int(proj_axes[3, 0, 1])),
                 (255, 255, 0), 2)

        frame_copy = cv2.resize(frame_copy, frame_size)
        cv2.imshow("Estimated head pose", frame_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_anchor_landmarks_3d(self) -> np.ndarray:
        """
        Getter for selected reference landmarks in 3D World coordinates
        of Surray Face Model.
        """
        return self.__sfm_points_for_pnp


if __name__ == "__main__":
    from calibration.io_utils import VideoReader
    import pickle

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    reader = VideoReader()
    frames = reader.read(output_dir / "2021-12-12_20-04-04_calibration.avi")
    print(f"Video-file read: {len(frames)} frames")

    # detect facial points
    lmk_detector = LandmarksDetector()

    settings_dir = Path(__file__).resolve().parent.parent / 'settings'
    camera_params_fn = settings_dir / "asus_laptop_camera_calibration_params.pkl"
    try:
        with open(camera_params_fn, 'rb') as f:
            camera_calib_params = pickle.load(f)
    except Exception as ex:
        print(f"Error occurred during camera calibration parameters reading: {ex}")
        camera_calib_params = {'camera_matrix': np.eye(3),
                               'distortion_coeffs': np.zeros((1, 5))}
    camera_matrix = camera_calib_params['camera_matrix']
    distortion_coeffs = camera_calib_params['distortion_coeffs']

    # headpose_estimator = MPHeadPoseEstimator()
    # frames_landmarks = []
    # for frame_i, frame in enumerate(frames):
    #     frame_lmks = headpose_estimator.estimate_3d_landmarks(frame, visualize=True)
    #     frames_landmarks.append(frame_lmks)
    #     if frame_i > 1:
    #         break

    headpose_estimator = EOSHeadPoseEstimator()
    anchor_points_coords = headpose_estimator.get_anchor_landmarks_3d()

    frames_rvecs = []
    frames_tvecs = []
    for frame_i, frame in enumerate(frames):
        _, lmks = lmk_detector.detect(frame, return_face_location=False)
        rvec, tvec = headpose_estimator.estimate_headpose(frame, landmarks_2d=lmks,
                                                          camera_matrix=camera_matrix,
                                                          distortion_coeffs=distortion_coeffs,
                                                          visualize=True)
        frames_rvecs.append(rvec)
        frames_tvecs.append(tvec)
        if frame_i > 1:
            break


