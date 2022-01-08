import os
import cv2
import h5py
import timeit
import numpy as np
from tqdm import tqdm
from pathlib import Path

from collections import defaultdict
from typing import (List, Dict, Any, Tuple, Union)

import warnings
warnings.filterwarnings('ignore')

import logging_handler
logger = logging_handler.get_logger(__name__)

from utils import (vector_to_pitch_yaw, batches,
                   clean_folder, unzip_archive)
from helpers import read_json
from preprocessing.undistortion import Undistorter


class DatasetGazeNormalizer:

    def __init__(self, dataset_dir: Union[str, Path], meta_fn: Union[str, Path]):
        """
        Revisiting Data Normalization for Appearance-Based Gaze Estimation
        Xucong Zhang, Yusuke Sugano, Andreas Bulling
        in Proc. International Symposium on Eye Tracking Research and Applications (ETRA), 2018
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        # Check dataset folder
        if type(dataset_dir) == str:
            dataset_dir = Path(dataset_dir).resolve()
        if not dataset_dir.exists():
            print(f"Dataset folder: {dataset_dir} do not exists!")
            raise FileNotFoundError(f"Dataset folder: {dataset_dir} do not exists!")
        self._dataset_dir = dataset_dir

        # Check meta-file path
        if type(meta_fn) == str:
            meta_fn = Path(meta_fn).resolve()
        if not meta_fn.exists():
            print(f"Meta-file: {meta_fn} do not exists!")
            raise FileNotFoundError(f"Meta-file: {meta_fn} do not exists!")
        self._meta_fn = meta_fn

        self._meta_group = None
        self.__init_normalized_camera()
        self.__init_3d_face_model()
        self._undistort = Undistorter()

    def __init_3d_face_model(self):
        params_fn = self._root_dir / "settings" / "sfm_face_coordinates.npy"
        if not params_fn.exists():
            logger.error(f"3D face model file do not exists!")
            raise FileNotFoundError(f"3D face model file do not exists!")

        self.__face_model_3d_coordinates = np.load(str(params_fn))
        logger.info(f"Face model coordinates loaded")

    def __init_normalized_camera(self):
        params_fn = self._root_dir / "settings" / "normalized_camera.json"
        if not params_fn.exists():
            logger.error(f"Normalized camera file do not exists!")
            raise FileNotFoundError(f"Normalized camera file do not exists!")

        self.__normalized_camera_params = dict(read_json(str(params_fn)))
        self.__norm_camera_matrix = np.array(
            [
                [self.__normalized_camera_params['focal_length'], 0, 0.5 * self.__normalized_camera_params['size'][0]],
                [0, self.__normalized_camera_params['focal_length'], 0.5 * self.__normalized_camera_params['size'][1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        logger.info(f"Normalized camera loaded")

    def normalize_dataset(self, sess_names: List[str],
                          output_fn: Union[str, Path]):
        """
        Normalize dataset of sessions.
        """
        total_to_write = defaultdict(dict)
        for sess_name in tqdm(sess_names):
            try:
                norm_session = self.normalize_session(str(sess_name))
                total_to_write[sess_name] = norm_session
            except Exception as ex:
                logger.error(f"For session #{sess_name} error occured: {ex}")

        # Write to output HDF5 file
        if type(output_fn) == str:
            output_fn = Path(output_fn).resolve()

        # Append to file, if it already exists
        if output_fn.exists():
            mode = 'a'
        else:
            mode = 'w'

        with h5py.File(str(output_fn), mode) as f:
            logger.info(f"Started write to output file: {output_fn} in mode: `{mode}`")
            logger.info(f"Total to write {len(total_to_write)} items.")

            for sess_name, data_to_write in total_to_write.items():
                logger.info(f"Write session #{sess_name} with {len(data_to_write)} frames")
                # Delete, if session already exists in output file
                if sess_name in f.keys():
                    logger.warning(f"Session #{sess_name} already exists in output file, re-writing...")
                    del f[sess_name]
                # Create group == session
                group = f.create_group(sess_name)
                # Create dataset == frame
                for key, values in data_to_write.items():
                    group.create_dataset(
                        key, data=values,
                        chunks=(
                            tuple([1] + list(values.shape[1:]))
                            if isinstance(values, np.ndarray)
                            else None
                        ),
                        compression='gzip',  # lzf or set gzip, can be set compression_opts=1..9, default=4
                        compression_opts=9
                    )

    def normalize_session(self, sess_name: str) -> Dict[str, Any]:
        """
        Normalize session.
        """
        # Prepare processed data for storing in .h5 file
        to_write = defaultdict(list)

        with h5py.File(str(self._meta_fn), 'r') as f:

            if sess_name not in f.keys():
                print(f"Meta file does not contain session #{sess_name}!")
                return None

            group = f[sess_name]

            # Iterate through group (frames)
            num_frames = next(iter(group.values())).shape[0]
            print(f"Session #{sess_name} contains {num_frames} frames")

            for i in range(num_frames):
                # Perform data normalization
                processed_frame = self._normalize_frame(group, i)

                # Gather all of the person's data
                to_write['frames'].append(processed_frame['norm_frame'])
                to_write['gaze_labels'].append(np.concatenate([
                    processed_frame['normalized_gaze_direction'],
                    processed_frame['normalized_head_pose'],
                ]))
                to_write['subject_labels'].append(processed_frame['subject_id'])

        if len(to_write) == 0:
            logger.warning(f"Session normalization output is empty!")

        # Cast to numpy arrays
        for key, values in to_write.items():
            to_write[key] = np.asarray(values)
            logger.debug(f'{key}: {to_write[key].shape}')

        return to_write

    def _normalize_frame(self, group: h5py.Group, frame_id: int) -> Dict[str, np.ndarray]:
        """
        Normalize single frame.
        :param group - should have keys: 3d_gaze_target,
                        camera_parameters, distortion_parameters,
                        file_name, head_pose.
        :frame_id - id of frame from current session (should be in frames folder).
        """
        # Form original camera matrix
        fx, fy, cx, cy = group['camera_parameters'][frame_id, :]
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]],
                                 dtype=np.float64)
        # Grab image
        file_name = self._dataset_dir / group['file_name'][frame_id].decode('utf-8')
        logger.info(f"Frame path: {str(file_name)}")
        frame = cv2.imread(str(file_name), cv2.IMREAD_COLOR)
        # Undistort frame
        frame = self._undistort(frame, camera_matrix,
                                group['distortion_parameters'][frame_id, :],  # 4 params: k1,k2,p1,p2
                                is_gazecapture=True)
        # Convert BGR and RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate rotation matrix and euler angles
        rvec = group['head_pose'][frame_id, :3].reshape(3, 1)
        tvec = group['head_pose'][frame_id, 3:].reshape(3, 1)
        rotate_mat, _ = cv2.Rodrigues(rvec)

        # Take mean face model landmarks and get transformed 3D positions
        # Landmarks: 14
        # 11-12 - between 2 eyes
        landmarks_3d = np.matmul(rotate_mat, self.__face_model_3d_coordinates.T).T
        landmarks_3d += tvec.T

        # Gaze-origin (g_o) and target (g_t)
        # Gaze-origin (g_o)
        g_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
        g_o = g_o.reshape(3, 1)
        # Gaze target (g_t)
        g_t = group['3d_gaze_target'][frame_id, :].reshape(3, 1)
        # Gaze
        g = g_t - g_o
        g /= np.linalg.norm(g)

        # Code below is an adaptation of code by Xucong Zhang
        # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

        # actual distance between gaze origin and original camera
        distance = np.linalg.norm(g_o)
        z_scale = self.__normalized_camera_params['distance'] / distance

        S = np.eye(3, dtype=np.float64)
        S[2, 2] = z_scale

        hRx = rotate_mat[:, 0]   # [0, rz, −ry]
        forward = (g_o / distance).reshape(3)  # z_c / ||z_c||

        down = np.cross(forward, hRx)  # x_c
        down /= np.linalg.norm(down)  # x_c / ||x_c||

        right = np.cross(down, forward)  # y_c
        right /= np.linalg.norm(right)  # y_c / ||y_c||

        # Rotation matrix as: R = [x_c, y_c, z_c]
        R = np.c_[right, down, forward].T  # rotation matrix R

        # transformation matrix is defined as M=SR
        M = np.dot(self.__norm_camera_matrix, S)

        # Cn is the camera projection matrix defined for the normalized camera
        # Cr is the original camera projection matrix obtained from camera calibration

        # W - transformation matrix for perspective image warping
        W = np.dot(M, np.dot(R, np.linalg.inv(camera_matrix)))

        # Output size - size of cropped eye image
        ow, oh = self.__normalized_camera_params['size']
        norm_frame = cv2.warpPerspective(frame, W, (ow, oh))  # image normalization

        R = np.asmatrix(R)

        # Correct head pose
        h = np.array([np.arcsin(rotate_mat[1, 2]),
                      np.arctan2(rotate_mat[0, 2], rotate_mat[2, 2])])
        head_mat = R * rotate_mat
        n_h = np.array([np.arcsin(head_mat[1, 2]),
                        np.arctan2(head_mat[0, 2], head_mat[2, 2])])

        # Correct gaze
        n_g = R * g
        n_g /= np.linalg.norm(n_g)
        n_g = vector_to_pitch_yaw(-n_g.T).flatten()

        return {
            # Subject ID == Person ID, as integer number
            'subject_id': np.asarray(int(group['file_name'][frame_id].decode('utf-8').split("/")[0])).astype(np.uint8),
            'norm_frame': norm_frame.astype(np.uint8),  # Image with eyes region, normalized
            'gaze_direction': g.astype(np.float32),  # As difference between target and origin, 3D unit vector
            'gaze_origin': g_o.astype(np.float32),  # Gaze-origin as 3D vector
            'gaze_target': g_t.astype(np.float32),  # Gaze-target as 3D vector (actually, 2D place on the screen with Z coordinate set to 0)
            'head_pose': h.astype(np.float32),  # Head pose as pitch and yaw angles (azimuth and elevation), 2D vector
            'normalization_matrix': np.transpose(R).astype(np.float32),  # Rotation [3x3] matrix
            'normalized_gaze_direction': n_g.astype(np.float32),  # Gaze vector as pitch and yaw angles, 2D vector
            'normalized_head_pose': n_h.astype(np.float32), # as pitch and yaw angles (azimuth and elevation), 2D vector
        }


if __name__ == "__main__":
    # Batching parameters
    batch_size = 20

    # Paths
    dataset_direction = Path("E:/DATA/Eye Data/GazeCapture/GazeCapture3")
    meta_filename = Path(__file__).parent.parent.parent / "FAZE Few Shot Gaze" / "faze_preprocess" / "GazeCapture_supplementary.h5"
    output_folder = Path("E:/DATA/Eye Data/GazeCapture/GazeCapture_processed")

    if not meta_filename.exists():
        logger.info(f"Meta-file name: {meta_filename} do not exists!")

    if not dataset_direction.exists():
        logger.info(f"Dataset folder: {dataset_direction} do not exists!")

    normalizer = DatasetGazeNormalizer(dataset_dir=dataset_direction, meta_fn=meta_filename)

    # Process all available archives by batches
    archives = list(dataset_direction.glob("*.tar.gz"))
    logger.info(f"Found {len(archives)} archives in dataset folder.")

    start_time = timeit.default_timer()
    for batch_i, batch in enumerate(batches(archives, batch_size)):
        logger.info(f"Batch #{batch_i} with {len(batch)} elements.")
        batch_folders = []
        for arch_i, arch in enumerate(batch):
            try:
                logger.info(f"Unzipping #{arch_i} archive")
                out_arch_folder = unzip_archive(arch, output_folder=dataset_direction)
                batch_folders.append(out_arch_folder)
            except Exception as ex:
                logger.error(f"Exception occurred during {arch} unzipping (#{arch_i}), skipping...")

        sessions = [Path(fn).stem for fn in batch_folders]
        logger.info(f"Found {len(sessions)} unzipped sessions")
        try:
            normalizer.normalize_dataset(sess_names=sessions,
                                         output_fn=(output_folder / "test_gaze_capture_processed.h5"))
        except Exception as ex:
            logger.error(f"Exception occurred during normalization on batch #{batch_i}, skipping...")
        finally:
            logger.info(f"Cleaning folders...")
            for fn in batch_folders:
                clean_folder(fn)
            logger.info(f"Folders cleaned for batch #{batch_i}.")
    elapsed_time = timeit.default_timer() - start_time
    logger.info(f"Finished in: {elapsed_time} seconds.")

    # Check results
    if (output_folder / "train_gaze_capture_processed.h5").exists():
        logger.info(f"Output file exists!")
        with h5py.File(str(output_folder / "train_gaze_capture_processed.h5"), 'r') as f:
            for group_name, group in f.items():
                logger.info(f"Group {group_name}:")
                for k, val in group.items():
                    logger.info(f"key: {k} - value shape: {val.shape}")
