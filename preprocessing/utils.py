import cv2
import shutil
import zipfile
import tarfile
import numpy as np
from pathlib import Path
from typing import (List, Dict, Any, Tuple, Union)

import warnings
warnings.filterwarnings('ignore')

import logging_handler
logger = logging_handler.get_logger(__name__)


def vector_to_pitch_yaw(vectors: np.ndarray) -> np.ndarray:
    """
    Convert given gaze vectors to pitch (theta) and yaw (phi) angles.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def draw_gaze(image_in: np.ndarray, eye_pos: Tuple[float, float],
              pitchyaw: np.ndarray, length: float = 40.0,
              thickness: int = 2, color=(0, 0, 255)) -> np.ndarray:
    """
    Draw gaze angle on given image with a given eye positions.
    """
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                    tuple(np.round([eye_pos[0] + dx,
                                    eye_pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out


def unzip_archive(filename: Union[Path, str], output_folder: Union[Path, str]) -> str:
    """
    Reads and extracts gzip, bz2, lzma and zip compressed archives.
    :param filename: absolute path to file with extension (.tar.gz, .tar.bz2, .tar.xz, .zip);
    :param output_folder: path (relative or absolute) for extracting file;
    """
    # Check file path
    if type(filename) == str:
        filename = Path(filename).resolve()
    if not filename.exists():
        logger.error(f"File: {filename} do not exists!")
        raise FileNotFoundError(f"File: {filename} do not exists!")
    # Check output folder path
    if type(output_folder) == str:
        filename = Path(output_folder).resolve()
    if not output_folder.exists():
        logger.error(f"Output folder: {output_folder} do not exists, creating...")
        output_folder.mkdir()

    if filename.suffix == "zip":
        with zipfile.ZipFile(str(filename)) as f:
            logger.info(f"Opened: {str(filename)}")
            f.extractall(str(output_folder))
    elif filename.name.endswith("tar.gz"):
        with tarfile.open(str(filename), "r") as tar:
            logger.info(f"Opened: {str(filename)}")
            tar.extractall(str(output_folder))
    else:
        logger.error(f"Unknown file extension, should be one of the following: .zip or .tar.gz")
        return None

    f_output_folder = output_folder / filename.stem.split(".")[0]
    logger.info(f"Extracted to: {f_output_folder}")
    return str(f_output_folder)


def clean_folder(folder: Union[Path, str]):
    """
    Try to remove not empty folder;
    if failed show an error using try...except on screen
    """
    # Check output folder path
    if type(folder) == str:
        folder = Path(folder).resolve()
    if not folder.exists():
        logger.error(f"Output folder: {folder} already do not exists, skipping...")
        return
    try:
        shutil.rmtree(folder)
    except OSError as e:
        logger.error(f"Error: {e.filename} - {e.strerror}.")


def batches(sequence: Any, batch_size: int = 20) -> Any:
    for x in range(0, len(sequence), batch_size):
        yield sequence[x:x + batch_size]