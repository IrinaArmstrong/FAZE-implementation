import cv2
import pickle
import numpy as np
from threading import Thread
from typing import Union, Tuple, List
from pathlib import Path

import logging_handler
logger = logging_handler.get_logger(__name__)


class VideoRecorder:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread from web-camera.
    """

    def __init__(self, camera_idx: int = 0):
        self.__camera_idx = camera_idx
        self.__capture_thread = None
        self.__capture = cv2.VideoCapture(self.__camera_idx)
        self.__frames = []
        self.__retval = True
        self.__stopped = True

    def start(self):
        """
        Start new point recording.
        """
        self.reset()
        logger.info(f"Starting VideoCapture thread...")
        self.__stopped = False
        # Run thread
        self.__capture_thread = Thread(target=self.__get_frame, args=())
        self.__capture_thread.start()
        logger.info(f"VideoCapture thread running...")
        return self

    def __get_frame(self):
        """
        Thread target function, which reads frames from VideoCapture and stores them.
        """
        # stop, if camera closed/disconnected/not available
        # or user stopped
        while not self.__stopped:
            # retval = false if no frames has been grabbed
            (self.__retval, frame) = self.__capture.read()
            if self.__retval:
                # logger.info(f"Got frame #{len(self.__frames)}!")
                self.__frames.append(frame)
            else:
                logger.warning(f"Returned value from VideoCapture is false, stopping...")
                self.stop()

    def stop(self):
        """
        Stops capturing frames for a while.
        """
        # todo: use .isAlive() method
        if self.__stopped:
            logger.warning("Already stopped.")
        self.__stopped = True

    def reset(self):
        """
        Ends recording and resets inner variables, including frames.
        """
        logger.info(f"Reset VideoCapture stream")
        if self.__capture_thread is not None:
            logger.info("Stopping thread...")
            self.__capture_thread = None
        self.__frames = []

    def get_frames(self):
        """
        Get recorded frames.
        """
        logger.info(f"Recorded {len(self.__frames)} frames")
        return self.__frames


class VideoWriter:
    """
    Class that holds functionality of writing video streams to files
    in different modes and formats.
    """

    def __init__(self, output_dir: Union[str, Path],
                 fps: float, fourcc: str, frame_size: Tuple[int, int] = (640, 480)):
        """
        :type output_dir: direction name (absolute) to save videos;
        :param fps: frame-per-second of the created video stream;
        :param fourcc: 4-character code of codec used to compress the frames.
                        For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec,
                        VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc.
        :type frame_size: Size of the video frames.
        """
        self._fps = fps
        self._frame_size = frame_size
        if type(output_dir) == str:
            output_dir = Path(output_dir).resolve()
        self._output_dir = output_dir
        logger.info(f"Videos will be saved to: {str(output_dir)}")
        if len(fourcc) != 4:
            logger.error(f"Wrong codec FOURCC passed: {fourcc}!")
            raise ValueError(f"Wrong codec FOURCC passed: {fourcc}!")
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)

    def save_calibration_data(self, subject_id: str,
                              frames: List[List[np.ndarray]],
                              targets: List[List[Tuple[float, float, float]]]):
        video_fn = f'{subject_id}_calibration.avi'
        target_fn = f'{subject_id}_calibration_target.pkl'
        video_writer = cv2.VideoWriter(str(self._output_dir / video_fn),
                                              self._fourcc,
                                              self._fps, self._frame_size)
        save_targets = []
        logger.info(f"Data contains {len(frames)} calibration points.")
        for point_i, (point_frames, target) in enumerate(zip(frames, targets)):
            logger.info(f"Calibration point #{point_i} contains {len(point_frames)} frames.")
            for frame_i , frame in enumerate(point_frames):
                # Write frame
                video_writer.write(frame)
                # Show for debugging
                cv2.putText(frame, f"P#{point_i}F#{frame_i}", (20,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv2.LINE_AA)
                cv2.imshow('Frames', frame)
                cv2.waitKey(10)
        # Close everything
        cv2.destroyAllWindows()
        video_writer.release()
        logger.info(f"Saving calibration target to: {str(target_fn)}")
        with open(str(self._output_dir / target_fn), 'wb') as f:
            pickle.dump(target, f)


