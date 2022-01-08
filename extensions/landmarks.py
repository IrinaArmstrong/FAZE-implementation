# Basic
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from collections import OrderedDict
from typing import Tuple, List, Union, Optional

# Torch
import torch
import torch.backends.cudnn as cudnn

import logging_handler
logger = logging_handler.get_logger(__name__)

sys.path.insert(0, "HRNet-Facial-Landmark-Detection/")
from lib.config import config
import lib.models as models
from lib.core import evaluation
from lib.utils import transforms

from face_detection import FaceDetector, show_bboxes
from models.models_utils import check_cuda_available, acquire_device
from preprocessing.kalman_filters import KalmanFilter1D

# Mapping based on additional/300W_lmks_annotation.png
# Assuming: left ---> right (on image/photo!)
hr_landmarks2ids_mapping = {
    "nose bridge": [27, 28, 29, 30],
    "nose base": [31, 33, 35],
    "mouth corners": [48, 54],
    "left-eye corners": [36, 39],  # the most left eye on image
    "right-eye corners": [42, 45],  # the most right eye on image

}
# Enumerate from 1 ... 68
hr_ibug_landmarks2ids_mapping = {
    "nose bridge": [28, 29, 30, 31],
    "nose base": [32, 33, 34, 35, 36],
    "left-eye corners": [37, 40],  # the most left eye on image
    "right-eye corners": [43, 46]  # the most right eye on image

}


class LandmarksDetector:

    def __init__(self, addit_config_fn: str = None,
                 **kwargs):
        """
        Uses the official code of High-resolution networks (HRNets) for facial landmark detection.
        Adopted: HRNetV2-W18 (#Params=9.3M, GFLOPs=4.3G) for facial landmark detection,
        trained and evaluated on WFLW (Wider Facial Landmarks in the Wild) dataset.
        ---
        The WFLW database contains 10000 faces with 98 annotated landmarks.
        This database also features rich attribute annotations in terms of occlusion, head pose,
        make-up, illumination, blur and expressions.
        Official page: https://wywu.github.io/projects/LAB/WFLW.html
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self._lib_dir = Path(__file__).resolve().parent / "HRNet-Facial-Landmark-Detection"
        self.__config = config

        if addit_config_fn is None:
            addit_config_fn = str(self._lib_dir / "experiments" / "wflw" / "face_alignment_wflw_hrnet_w18.yaml")
            logger.info(f"Update defaults HRNet model config with WFLW w18 config file.")

        if not Path(addit_config_fn).exists():
            logger.error(f"Config file for HRNet model was not found!")
            raise FileNotFoundError(f"Config file for HRNet model was not found!")

        # Update config
        self.__config.defrost()
        self.__config.merge_from_file(addit_config_fn)
        self.__config.MODEL.INIT_WEIGHTS = False  # will load later
        self.__config.freeze()

        # Cudnn related params:
        self.__device = acquire_device('cpu')
        if check_cuda_available():
            cudnn.enabled = config.CUDNN.ENABLED  # enable cuDNN
            # True - causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
            cudnn.benchmark = config.CUDNN.BENCHMARK
            # True - causes cuDNN to only use deterministic convolution algorithms
            cudnn.determinstic = config.CUDNN.DETERMINISTIC
            self.__device = acquire_device('cuda')
        else:
            cudnn.enabled = False

        # Initialize MTCNN model for face bounding box detection
        self.__face_detector = FaceDetector()
        # Initialize 1D Kalman filters for different coordinates
        # will be used for face detection over a single object
        self.__face_loc_filters = [KalmanFilter1D(dim_x=10, R_=1e-4) for _ in range(2)]

        # Init HRNet for facial landmarks detection
        lmk_model_fn = self._lib_dir / 'hrnetv2_pretrained' / 'HR18-WFLW.pth'
        if not lmk_model_fn.exists():
            logger.error(f"Pre-trained weights file for HRNet model was not found: {str(lmk_model_fn)}")
            raise FileNotFoundError(f"Pre-trained weights file for HRNet model was not found: {str(lmk_model_fn)}")
        self.__lmk_model = models.get_face_alignment_net(config)
        try:
            state_dict = torch.load(str(lmk_model_fn), map_location=torch.device(self.__device.type))
            self.__lmk_model.load_state_dict(state_dict, strict=False)
            logger.info(f"HRNet model successfully initialized.")
        except Exception as ex:
            logger.error(f"Error occurred during HRNet model initialization: {ex}")
            raise ex

        # Initialize 68 1D Kalman filters for different coordinates
        # will be used to smooth landmarks over the face for a single face tracking
        self.__lmks_filters = [KalmanFilter1D(dim_x=10, R_=2.5e-05) for _ in range(68)]

        # Initialize 1D Kalman filter for the on-screen gaze point-of regard
        self.__gaze_filter = KalmanFilter1D(dim_x=10, R_=1e-4)

    def detect(self, frame: np.ndarray, **kwargs) -> Tuple[bool, np.ndarray, Optional[np.ndarray]]:
        """
        Detect 68 landmarks on given frame.
        :return: landmarks as 2-dim array with shape [68 x 2] of float coordinates.
        """
        logger.info(f"Landmarks detection got frame with shape: {frame.shape}, starting...")
        # as two points with (x, y) coords
        face_location, suc_flg = self.__face_detector.detect(frame=frame, scale=kwargs.get('scale', 1.0),
                                                             policy=kwargs.get('policy', 'size'))
        if not suc_flg:
            logger.error(f"Failed to detect face on frame, no landmarks will be detected.")
            if kwargs.get('return_face_location', False):
                return suc_flg, np.zeros((68, 2), dtype=np.float32), np.zeros((4,), dtype=np.float32)
            return suc_flg, np.zeros((68, 2), dtype=np.float32)

        # Use two 1D Kalman filters to smooth bounding box position coordinates
        # (upper left corner and lower right corner)
        # Assuming them as two complex numbers: a = x0 +j*y0 and b = x1 +j*y1
        output_tracked = self.__face_loc_filters[0].update(face_location[0] + 1j * face_location[1])
        face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
        output_tracked = self.__face_loc_filters[1].update(face_location[2] + 1j * face_location[3])
        face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

        # unpack location
        x_min = face_location[0]
        y_min = face_location[1]
        x_max = face_location[2]
        y_max = face_location[3]

        width = x_max - x_min
        height = y_max - y_min

        # ???
        scale = max(width, height) / 200
        scale *= 1.25

        # face location approx center
        center = torch.Tensor([(x_min + x_max) / 2,
                               (y_min + y_max) / 2])

        # using complex cropping function from HRNet utils + RGB
        frame_cropped = transforms.crop(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                                 dtype=np.float32),
                                        center, scale=scale, output_size=[256, 256], rot=0)
        frame_cropped = frame_cropped.astype(np.float32)
        # Z - normalize
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_cropped = (frame_cropped / 255.0 - img_mean) / img_std
        # to C-H-W order of dims
        frame_cropped = frame_cropped.transpose([2, 0, 1])
        # insert dummy batch dim = 1
        frame_cropped = torch.Tensor(np.expand_dims(frame_cropped, axis=0))

        self.__lmk_model.eval()
        output = self.__lmk_model(frame_cropped)
        score_map = output.data.cpu()
        center = np.expand_dims(np.array(center, dtype=np.float32), axis=0)
        scale = np.expand_dims(np.array(scale, dtype=np.float32), axis=0)
        # Decoding lmks probabilities
        predicted_lmks = evaluation.decode_preds(score_map, center, scale, res=[64, 64])
        predicted_lmks = np.squeeze(predicted_lmks.numpy(), axis=0)

        # convert 96 WFLW lmks to 68 from 300VW dataset lmks
        idx_300vw = self.__map_landmarks_to_300vw()
        predicted_lmks_300vw = predicted_lmks[idx_300vw, :]

        # Using 68 1D Kalman filter on landmarks to smooth their movements
        for i in range(68):
            lmk_f = self.__lmks_filters[i].update(predicted_lmks_300vw[i, 0] + 1j * predicted_lmks_300vw[i, 1])
            predicted_lmks_300vw[i, 0] = np.real(lmk_f)
            predicted_lmks_300vw[i, 1] = np.imag(lmk_f)

        if kwargs.get('return_face_location', False):
            return suc_flg, predicted_lmks_300vw, face_location

        return suc_flg, predicted_lmks_300vw

    def __map_landmarks_to_300vw(self) -> List[int]:
        """
        Function for converting 96 WFLW dataset landmarks ids
        to 68 from 300VW dataset landmarks ids.
        (see schemes in `additional` folder)
        :return: list of indexes for corresponding points selection from output of model
                that predicted WFLW dataset landmarks.
        """
        lmks_68_to_96_mapping = OrderedDict()
        lmks_68_to_96_mapping.update(dict(zip(range(0, 17), range(0, 34, 2))))  # jaw | 17 pts
        lmks_68_to_96_mapping.update(
            dict(zip(range(17, 22), range(33, 38))))  # left upper eyebrow points | 5 pts
        lmks_68_to_96_mapping.update(
            dict(zip(range(22, 27), range(42, 47))))  # right upper eyebrow points | 5 pts
        lmks_68_to_96_mapping.update(dict(zip(range(27, 36), range(51, 60))))  # nose points | 9 pts
        lmks_68_to_96_mapping.update({36: 60})  # left eye points | 6 pts
        lmks_68_to_96_mapping.update({37: 61})
        lmks_68_to_96_mapping.update({38: 63})
        lmks_68_to_96_mapping.update({39: 64})
        lmks_68_to_96_mapping.update({40: 65})
        lmks_68_to_96_mapping.update({41: 67})
        lmks_68_to_96_mapping.update({42: 68})  # right eye | 6 pts
        lmks_68_to_96_mapping.update({43: 69})
        lmks_68_to_96_mapping.update({44: 71})
        lmks_68_to_96_mapping.update({45: 72})
        lmks_68_to_96_mapping.update({46: 73})
        lmks_68_to_96_mapping.update({47: 75})
        lmks_68_to_96_mapping.update(dict(zip(range(48, 68), range(76, 96))))  # mouth points | 20 pts
        lmks_96_to_68_mapping = {k: v for k, v in lmks_68_to_96_mapping.items()}

        return list(lmks_96_to_68_mapping.values())

    def plot_markers(self, image: np.ndarray,
                     face_location: np.ndarray,
                     facial_landmarks: np.ndarray,
                     color: Union[List[int], str] = (0, 0, 255),
                     radius: int = 2, drawline: bool = False):
        """
        Plot detected landmark points on frame.
        """
        # show face location bounding box
        image = cv2.rectangle(image,
                              (int(face_location[0]), int(face_location[1])),
                              (int(face_location[2]), int(face_location[3])),
                              (255, 0, 0), 2)

        # plot only 68 points, others (if any???) skipping
        for idx, p in enumerate(facial_landmarks):
            if idx > 68:
                logger.warning(f"Number of given landmark more than 68!")
                continue

            if idx in hr_landmarks2ids_mapping['left-eye corners']:
                color = (0, 255, 0)  # green
            elif idx in hr_landmarks2ids_mapping['right-eye corners']:
                color = (255, 0, 0)  # blue
            else:
                color = (0, 0, 255)  # red

            image = cv2.circle(image, center=(int(p[0]), int(p[1])),
                               radius=radius, color=color, thickness=-1)
            cv2.putText(image, text=f"{idx}", org=(int(p[0]), int(p[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.2, color=(0, 0, 0), thickness=1,
                        lineType=cv2.LINE_AA)

        if drawline:
            # 0-16
            image = cv2.polylines(image, facial_landmarks[0:17].astype('int32').reshape(-1, 1, 2),
                                  isClosed=False, color=color, thickness=1)
            # 17-21
            image = cv2.polylines(image, facial_landmarks[17:22].astype('int32').reshape(-1, 1, 2),
                                  isClosed=False, color=color, thickness=1)
            # 22-26
            image = cv2.polylines(image, facial_landmarks[22:27].astype('int32').reshape(-1, 1, 2),
                                  isClosed=False, color=color, thickness=1)
            # 27-35
            image = cv2.polylines(image, facial_landmarks[27:36].astype('int32').reshape(-1, 1, 2),
                                  isClosed=False, color=color, thickness=1)
            # 36-41
            image = cv2.polylines(image, facial_landmarks[36:41].astype('int32').reshape(-1, 1, 2),
                                  isClosed=False, color=color, thickness=1)
            # 42-47
            image = cv2.polylines(image, facial_landmarks[42:48].astype('int32').reshape(-1, 1, 2),
                                  isClosed=False, color=color, thickness=1)
            # 48-67
            image = cv2.polylines(image, facial_landmarks[48:68].astype('int32').reshape(-1, 1, 2),
                                  isClosed=False, color=color, thickness=1)

        return image


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    test_img_dir = Path(__file__).resolve().parent.parent / "additional" / "test_samples"
    # 'elon_musk.jpg' 'empty.jpg' 'test_image.png'
    image = Image.open(str(test_img_dir / 'empty.jpg'))
    image = image.resize((640, 480), Image.ANTIALIAS)

    # detect facial points
    lmk_detector = LandmarksDetector()
    suc_flg, lmks, face_location = lmk_detector.detect(np.array(image), return_face_location=True)
    if not suc_flg:
        print(f"Failed to detect face on frame, no landmarks will be detected.")
        sys.exit(-1)
    # Save
    # np.save(str(output_dir / 'test_image_face_loc.npy'), face_location)
    # np.save(str(output_dir / 'test_image_HRNet_lmks.npy'), lmks)

    # Show
    draw_image = lmk_detector.plot_markers(np.array(image),
                                           np.asarray(face_location),
                                           lmks, drawline=False)
    draw_image = cv2.resize(draw_image, (960, 720))
    cv2.imshow('Landmarks detected', draw_image)
    cv2.waitKey(0)
    # Save
    # cv2.imwrite(str(output_dir / "lmks_detected.png"), draw_image)

