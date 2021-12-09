import cv2
from threading import Thread

import logging_handler
logger = logging_handler.get_logger(__name__)


class VideoRecorder:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread from web-camera.
    """

    def __init__(self, camera_idx: int = 0):
        self.__capture = cv2.VideoCapture(camera_idx)
        self.__capture_thread = None
        self.__frames = []
        self.__retval = True
        self.__stopped = False

    def start(self):
        logger.info(f"Starting VideoCapture thread...")
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
                self.__frames.append(frame)
            else:
                self.stop()

    def stop(self):
        """
        Stops everything: VideoCapture, thread.
        """
        self.__stopped = True
        self.__capture.release()
        self.__capture_thread.join()

    def get_frames(self):
        """
        Get recorded frames.
        """
        logger.info(f"Frames: {len(self.__frames)}")
        return self.__frames