import numpy as np
import cv2


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read() #underscore because we don't need/want that return value
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)

    k = cv2.waitKey(5) & 0xFF
    if k == 27: #esc key to stop
        break

cv2.destroyAllWindows()
cap.release() # this lets the webcam go