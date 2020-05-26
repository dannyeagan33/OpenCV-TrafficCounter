# test sectiosn of big script
import logging
import logging.handlers
import os
import time
import sys

import cv2
import numpy as np

from vehicle_counter import VehicleCounter


IMAGE_DIR = "images"
IMAGE_FILENAME_FORMAT = IMAGE_DIR + "/frame_%04d.png"


log = logging.getLogger("main")

log.debug("Creating background subtractor...")
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

log.debug("Pre-training the background subtractor...")
default_bg = cv2.imread(IMAGE_FILENAME_FORMAT % 119)
bg_subtractor.apply(default_bg, None, 1.0)

car_counter = None # Will be created after first frame is captured