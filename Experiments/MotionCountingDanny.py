from imutils.video import VideoStream
import datetime
import math
import cv2
import numpy as np
import argparse
import imutils
import time
import smtplib


# Global Variables
width = 0
height = 0
entranceCounter = 0
exitCounter = 0
areaContourMinLimit = 10000
thresholdBinarization = 127
offsetLinesRef = 30
timeout = time.time() + 60*1

# Verifying object is entering monitored zone
def testIntersectionEntrance(y,yCoordinateEntranceLine, yCoordinateExitLine):
    absDiff = abs(y - yCoordinateEntranceLine)

    if ((absDiff <= 2)) and (y < yCoordinateExitLine):
        return 1
    else:
        return 0

# Verifying object is leaving monitored zone
def testInterestionExit(y, yCoordinateEntranceLine, yCoordinateExitLine):
    absDiff = abs(y - yCoordinateExitLine)

    if ((absDiff <= 2) and (y > yCoordinateEntranceLine)):
        return 1
    else:
        return 0

# need help understanding this section
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
        help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args()) 

camera = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

firstFrame = None

# Doing some frame reading before analysis, because some cameras
# take longer to adjust to ambient light when turning on, capturing 
# consecutive frames with lots of light variation. To not process these
# effects, consecutive captures are outside the image processing giving 
# the camera some time to adjust to lighting conditions.

for i in range(0,20):
    (grabbed, frame) = camera.read(), camera.read()

while True:
    # Reading first frame and determining size
    (grabbed, frame) = camera.read(), camera.read()
    height = np.size(frame,0)
    width = np.size(frame,1)

    #Converting frame to grayscale and applying blur effect to highlight shapes
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv2.GaussianBlur(frameGray, (21, 21), 0)

    # Comparison is made between two image. If first frame is null, it's initialized
    if firstFrame is None:
        firstFrame = frameGray
        continue

    # Absolute difference between initial frame and actual frame (background subtraction)
    # Also creates binarization of frame and subtracted background
    frameDelta = cv2.absdiff(firstFrame,frameGray)
    frameThresh = cv2.threshold(frameDelta, thresholdBinarization, 255, cv2.THRESH_BINARY) [1]

    # Creates the dilatation of binary frame to eliminate holes, white zones inside found shapes
    # so that detected objects will be considered black mass, also finds shapes after dilatation
    frameThresh = cv2.dilate(frameThresh, None, iterations=2)
    cnts, hierarchy = cv2.findContours(frameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # what is format of above line? left side of equation

    shapeCounts = 0

    # Draw reference lines
    yCoordinateEntranceLine = (height / 2) - offsetLinesRef
    yCoordinateExitLine = (height / 2) + offsetLinesRef
    cv2.line(frame, (0, int(yCoordinateEntranceLine)), (int(width), int(yCoordinateEntranceLine)), (255, 0, 0), 2)
    cv2.line(frame, (0, int(yCoordinateExitLine)), (int(width), int(yCoordinateExitLine)), (0, 0, 255), 2)

    # Wiping found shapes
    for c in cnts:
        # Ignore small shapes
        if cv2.contourArea(c) < areaContourMinLimit:
            continue # back to while loop functionality
        # Count number of found shapes for debugging
        shapeCounts = shapeCounts + 1

        # Gets shapes coordinates (a rectangle that encapsulates object), highlighting its shape
        (x, y, w, h) = cv2.boundingRect(c) #x e y: coordinates of upper left vertex
                                        # width and height respectively
        # does above line do anything?

        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        # Determines centroid of shape and circles
        xCenterContour = int((x+x+w)/2)
        yCenterContour = int((y+y+h)/2)
        centroidContour = (xCenterContour,yCenterContour)
        cv2.circle(frame, centroidContour, 1, (0, 0, 0), 5)

        # Tests intersection of centers from the shapes and the reference lines. 
        # So it can count which shapes cross lines.
        if (testIntersectionEntrance(yCenterContour, yCoordinateEntranceLine, yCoordinateExitLine)):
            entranceCounter += 1

        if (testInterestionExit(yCenterContour, yCoordinateEntranceLine, yCoordinateExitLine)):
            exitCounter += 1

        #If needed, uncomment these lines to show framerate
        # cv2.imshow("binaryFrame", frameThresh)
        # cv2.waitKey(1);
        # cv2.imshow("Frame with background subtraction", frameDelta)
        # cv2.waitKey(1);

    print("Contours found: "+ str(shapeCounts))

    # Writes on screen number of people who enter or leave watched area
    cv2.putText(frame, "Entrances: {}".format(str(entranceCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(frame, "Exits: {}".format(str(exitCounter)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Original", frame)
    key = cv2.waitKey(1) & 0xFF

    #If you want to exit the program from a keystroke, uncomment these lines and comment the next ones,
    #which makes the program exit by itself after a certain ammount of time.    
    # if the `q` key was pressed, break from the loop
    

    # test = 0
    # if test == 5 or time.time() > timeout:
    #     break
    # test = test - 1

# Clean up the camera and close any open windows

# Program writes the count to a file.
f = open( 'contagem.txt', 'w' ) #File path and it's name with extension, to write to.
f.write( 'entranceCounter = ' + repr(entranceCounter) + '\n' ) #Variables to write
f.write( 'exitCounter = ' + repr(exitCounter) + '\n' ) #Variables to write
f.close()

cv2.destroyAllWindows()
camera.stop()