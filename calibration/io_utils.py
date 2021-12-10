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
        self.__camera_idx = camera_idx
        self.__capture_thread = None
        self.__capture = None
        self.__frames = []
        self.__retval = True
        self.__stopped = False

    def start(self):
        if not self.__stopped:
            logger.error(f"Previous VideoCapture thread is still running!")
            logger.error("Stopping...")
            self.stop()

        logger.info(f"Starting VideoCapture thread...")
        # Reset variables
        self.__capture = cv2.VideoCapture(self.__camera_idx)
        self.__stopped = False
        self.__frames = []
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
                self.__frames.append(frame)
            else:
                self.stop()

    def stop(self):
        """
        Stops everything: VideoCapture, thread.
        """
        if self.__stopped:
            logger.warning("Already stopped.")
        self.__stopped = True
        if self.__capture is not None:
            logger.info("Stopping VideoCapture...")
            self.__capture.release()
        if self.__capture_thread is not None:
            logger.info("Stopping thread...")
            self.__capture_thread.join()
            self.__capture_thread = None

    def get_frames(self):
        """
        Get recorded frames.
        """
        logger.info(f"Frames: {len(self.__frames)}")
        return self.__frames