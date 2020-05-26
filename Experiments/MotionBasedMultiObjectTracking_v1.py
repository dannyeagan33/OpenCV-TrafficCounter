import sys
import cv2
from random import randint
import os
import numpy as np


class systemObject(object):
   pass

obj = systemObject()

def setupSystemObjects(vidPath):
    obj.reader = cv2.VideoCapture(vidPath)

    mask = np.zeros(obj.reader.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)

    obj.detector = cv2.grabCut(obj.reader,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    return obj




def MotionBasedMultiObjectTracking():

    folderInputName = 'Test Videos Mac'
    saveLoc = 'Output without Masking'

    location = 'test'

    yellowBoxToggle = 1

    testVideo = '00000017.ASF'

    # Initialize Video Folder Input
    
    sys.path.append(folderInputName)### no addpath command in Python addpath(folderInputName), python works in whatever workspace code is in?
    videos = sorted(os.listdir(os.path.abspath(folderInputName)))
    count = 0
    oldFrame = []

    # Masking -- Toggle and Implementation -- do this later!!!

    for k in range(0, len(videos)):
        n = 1
        vidName = videos[k]
        vidPath = os.path.abspath(folderInputName) + '/' + vidName
    
    setupSystemObjects(vidPath)


MotionBasedMultiObjectTracking()
    