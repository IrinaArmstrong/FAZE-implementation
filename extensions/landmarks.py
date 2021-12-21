# Basic
import sys
import cv2
import time
import pickle
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union

# Torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import logging_handler
logger = logging_handler.get_logger(__name__)

sys.path.insert(0, "HRNet-Facial-Landmark-Detection/")
from lib.config import config
import lib.models as models
from lib.datasets import get_dataset
from lib.core import evaluation
from lib.utils import transforms

from face_detection import FaceDetector
from models.models_utils import check_cuda_available, acquire_device


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

        # Init MTCNN model for face bounding box detection
        self.__face_detector = FaceDetector()

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

    def detect(self, frame: np.ndarray, **kwargs):
        """

        :param frame:
        :return:
        """
        logger.info(f"Landmarks detection got frame with shape: {frame.shape}, starting...")
        # as two points with (x, y) coords
        face_location = self.__face_detector.detect(frame=frame, scale=kwargs.get('scale', 1.0),
                                                    policy=kwargs.get('policy', 'size'))
        # todo: here filter face location with Kalman filter
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
        #
        preds = evaluation.decode_preds(score_map, center, scale, res=[64, 64])
        preds = np.squeeze(preds.numpy(), axis=0)

        # get the 68 300 VW points:
        # idx_300vw = self.map_to_300vw()
        # preds = preds[idx_300vw, :]

        return preds



