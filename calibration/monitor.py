import tkinter as tk
import numpy as np
from typing import Tuple

class Monitor:

    def __init__(self):
        screen = tk.Tk()
        # height & width of the screen, in mm.
        self._h_mm = screen.winfo_screenmmheight()
        self._w_mm = screen.winfo_screenmmwidth()
        # height & width of the screen, in pixels
        self._h_pixels = screen.winfo_screenheight()
        self._w_pixels = screen.winfo_screenwidth()

    def monitor_to_camera(self, x_pixel: int, y_pixel: int) -> Tuple[float, float, float]:
        """
        Convert pixels of target point on the screen (in 2D Screen Coordinate System)
        in to the camera-relative 3D coordinates.
        Assumes in-build laptop camera, located centered and 10 mm above display.
        You update this function for you camera and monitor
        using: https://github.com/computer-vision/takahashi2012cvpr
        """
        x_cam_mm = ((int(self._w_pixels / 2) - x_pixel) / self._w_pixels) * self._w_mm
        y_cam_mm = 10.0 + (y_pixel / self._h_pixels) * self._h_mm
        z_cam_mm = 0.0
        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm: float, y_cam_mm: float) -> Tuple[int, int]:
        """
        Convert camera-relative 3D coordinates into
        pixels of target point on the screen (in 2D Screen Coordinate System).
        Assumes in-build laptop camera, located centered and 10 mm above display.
        You update this function for you camera and monitor
        using: https://github.com/computer-vision/takahashi2012cvpr
        """
        x_mon_pixel = np.ceil(int(self._w_pixels / 2) - x_cam_mm * self._w_pixels / self._w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self._h_pixels / self._h_mm)
        return x_mon_pixel, y_mon_pixel

    def get_width_pixels(self):
        return self._w_pixels

    def get_height_pixels(self):
        return self._h_pixels

    def get_width_mm(self):
        return self._w_mm

    def get_height_mm(self):
        return self._h_mm