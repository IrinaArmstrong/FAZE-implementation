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