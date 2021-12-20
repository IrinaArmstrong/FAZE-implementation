# Basic
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List

import logging_handler
logger = logging_handler.get_logger(__name__)

sys.path.insert(0, "mtcnn-pytorch/")
from src.detector import detect_faces
from src.visualization_utils import show_bboxes


class FaceDetector:

    def __init__(self):
        """
        Use PyTorch implementation of face detection algorithm MTCNN described in
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.
        """
        self._root_dir = Path(__file__).resolve().parent.parent
        self._lib_dir = Path(__file__).resolve().parent / "mtcnn-pytorch"

    def detect(self, frame: np.ndarray, scale: float = 1.0,
               policy: str = 'size') -> np.ndarray:
        """
        Detect face(s) on a frame.
        :param scale: frame resize factor;
        :param policy: which face to select - by size or by maximal confidence score?
                       so `size` or `score` respectively.
        :return: ???
        """
        # detect face
        frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # BBoxes as 5 numbers: left top (x and y), height, width and score from final ONet.
        # LMKs as 10 numbers (x, y): left eye, right eye, nose, left mouth corner, right mouth corner.
        bounding_boxes, landmarks = detect_faces(pil_frame, min_face_size=30.0)
        if not len(bounding_boxes):
            logger.error(f"No face detected! Return full frame.")
            return np.asarray([0, 0, frame.shape[0], frame.shape[1]])

        scores = [x[4] for x in bounding_boxes]
        bounding_boxes = [x[:4] for x in bounding_boxes]
        face_location = self.__select_best(bounding_boxes, scores, policy)

        if not len(face_location):
            logger.error(f"No face detected! Return full frame.")
            return np.asarray([0, 0, frame.shape[0], frame.shape[1]])
        # rescale back to initial sizes
        face_location = face_location * (1 / scale)
        logger.info(f"Face successfully detected.")
        return face_location

    @staticmethod
    def __select_best(bounding_boxes: List[np.ndarray], scores: List[float],
                      policy: str) -> np.ndarray:
        """
        Select single face location based on provided policy of selection.
        :param bounding_boxes: bounding boxes as 4 numbers: left top (x and y), height, width,
        :param scores: probability scores of face location, produced by ONet of MTCNN model,
        :param policy: which face to select - by size or by maximal confidence score?
                       so `size` or `score` respectively.
        :return: most probable face bounding box
        """
        if policy not in ['size', 'score']:
            logger.error(f"Provided policy is unknown, use default: `score`")

        face_location = []
        if len(bounding_boxes) > 0:
            best_property = 0
            best_id = -1
            for i, d in enumerate(bounding_boxes):
                if policy == 'score':
                    property = scores[i]
                elif policy == 'size':
                    property = abs(d[2] - d[0]) * abs(d[3] - d[1])
                # new best bbox found
                if best_property < property:
                    best_property = property
                    best_id = i
            if policy == 'score':
                if best_property > -0.5:
                    face_location = bounding_boxes[best_id]
            else:
                face_location = bounding_boxes[best_id]
            return face_location

