import numpy as np
import cv2

class BackGroundSegmentor(object):
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=450, varThreshold=50, detectShadows=True)
        self.fgbg.setNMixtures(3)
        self.vbg = None

    def set_varThreshold(self, varThreshold):
        self.fgbg.setVarThreshold(varThreshold)
            def get_varThreshold(self):
        return self.fgbg.getVarThreshold()

    def set_history(self, history):
        self.fgbg.setHistory(history)

    def get_history(self):
        return self.fgbg.getHistory()

    def set_mixtures(self, mixtures):
        self.fgbg.setNMixtures(mixtures)

    def get_mixtures(self):
        return self.fgbg.getNMixtures()

    def mask(self, image, mask):
        if np.sum(mask) == 0:
            self.vbg = image.copy()
        else:
            image[mask, :] = self.vbg[mask, :]

        fgmask = self.fgbg.apply(image, learningRate=-1)
        fgmask[fgmask < 255] = 0
        return fgmask




cap = cv2.VideoCapture('00002522.ASF')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = cap.read() #underscore because we don't need/want that return value
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # REDUCE NOISE IN ORIGINAL FRAME

    blur = cv2.GaussianBlur(gray, (15,15), 0) # Gaussian Blur
   
    # APPLY FOREGROUND MASK
    fgmask = fgbg.apply(blur) # Less processing power to apply mask to gray image over original?



    cv2.imshow('Original',frame)
    cv2.imshow('Foreground Objects', fgmask)
    cv2.imshow('GaussBlur',blur)

    k = cv2.waitKey(5) & 0xFF
    if k == 27: #esc key to stop
        break

cv2.destroyAllWindows()
cap.release() # this lets the webcam go