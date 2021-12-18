# Basic
import sys
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

# from face import face
from models.models_utils import check_cuda_available

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
        if check_cuda_available():
            cudnn.enabled = config.CUDNN.ENABLED  # enable cuDNN
            # True - causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
            cudnn.benchmark = config.CUDNN.BENCHMARK
            # True - causes cuDNN to only use deterministic convolution algorithms
            cudnn.determinstic = config.CUDNN.DETERMINISTIC
        else:
            cudnn.enabled = False

        lmk_model_fn = self._lib_dir / 'hrnetv2_pretrained' / 'HR18-WFLW.pth'
        if not lmk_model_fn.exists():
            logger.error(f"Pre-trained weights file for HRNet model was not found: {str(lmk_model_fn)}")
            raise FileNotFoundError(f"Pre-trained weights file for HRNet model was not found: {str(lmk_model_fn)}")
        self.__lmk_model = models.get_face_alignment_net(config)
        try:
            state_dict = torch.load(str(lmk_model_fn))
            self.__lmk_model.load_state_dict(state_dict, strict=False)
            logger.info(f"HRNet model successfully initialized.")
        except Exception as ex:
            logger.error(f"Error occurred during HRNet model initialization: {ex}")
            raise ex
