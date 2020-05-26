# most code taken from https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515
import logging
import logging.handlers
import os
import time
import sys

import numpy as np
import cv2

from vehicle_counter import VehicleCounter

def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        #
        cv2.imshow('first frame',image)
        cv2.imwrite('firstframe.jpg', image)

#================================================================================================

imageDir = "images"
imageFilenameFormat =imageDir+ "/frame_%04d.png" #don't understand what this does

# To support either video or individual image file
captureFromVideo = True
if captureFromVideo:
    imageSource = "00002517.ASF" # Video filename
else:
    imageSource = imageFilenameFormat # Image sequence

# Set time to wait between frames, 0 = forever
waitTime = 1 # 250 # ms

logToFile = True

# Colors for drawing on processed frames
dividerColor = (255, 255, 0)
boundingBoxColor = (255, 0, 0)
centroidColor = (0, 0, 255)

#================================================================================================

# Initialize Logging Function
def init_logging(): # what does this do? what do I need to know how to change?
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if logToFile:
        handler_file = logging.handlers.RotatingFileHandler("debug.log"
            , maxBytes = 2**24
            , backupCount = 10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger


# ============================================================================

#Save Frame Function, saves with input parameters
def save_frame(file_name_format, frame_number, frame, label_format):
    file_name = file_name_format % frame_number
    label = label_format % frame_number

    log.debug("Saving %s as '%s'", label, file_name)
    cv2.imwrite(file_name, frame)


# ============================================================================

# Function to calculate centroid given first corner and height/width
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

# ============================================================================

# Detect vehicles, parameters for adjusting, called after applying background removal filter
def detect_vehicles(fg_mask):
    log = logging.getLogger("detect_vehicles")

    MIN_CONTOUR_WIDTH = 21
    MIN_CONTOUR_HEIGHT = 21

    # Find the contours of any vehicles in the image
    contours, hierarchy = cv2.findContours(fg_mask
        , cv2.RETR_EXTERNAL
        , cv2.CHAIN_APPROX_SIMPLE)

    log.debug("Found %d vehicle contours.", len(contours))

    matches = []
    for (i, contour) in enumerate(contours): # loops through every found contour
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT) # checks contour against min values

        log.debug("Contour #%d: pos=(x=%d, y=%d) size=(w=%d, h=%d) valid=%s"
            , i, x, y, w, h, contour_valid) # logs info about each found contour

        if not contour_valid:
            continue # if doesn't meet size requirements, goes to next contour

        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid)) # adds valid contour to matches array with box corners and center

    return matches

# ============================================================================

#can't we also do these same dilation/opening/etc. transformations with commands like in tutorials?
def filter_mask(fg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # kinda like adjusting size of eraser on paint program, used for filtering

    # Fill any small holes
    closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations = 2)

    return dilation

# ============================================================================

# Processes individual video frame -- draws counter line, background removal, applies mask, saves frame (don't need), looks for vehicle matches, separates match into contour/centroid, gets corner vals
# draws bounding rectangle around vehicle and circle around centroid, updates count
def process_frame(frame_number, frame, bg_subtractor, car_counter):
    log = logging.getLogger("process_frame")

    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Draw dividing line -- we count cars as they cross this line.
    cv2.line(processed, (0, car_counter.divider), (frame.shape[1], car_counter.divider), DIVIDER_COLOUR, 1)

    # Remove the background
    fg_mask = bg_subtractor.apply(frame, None, 0.01)
    fg_mask = filter_mask(fg_mask)

    save_frame(IMAGE_DIR + "/mask_%04d.png"
        , frame_number, fg_mask, "foreground mask for frame #%d")

    matches = detect_vehicles(fg_mask) # calls vehicle detection function with fg masked image

    log.debug("Found %d valid vehicle contours.", len(matches))
    for (i, match) in enumerate(matches):
        contour, centroid = match

        log.debug("Valid vehicle contour #%d: centroid=%s, bounding_box=%s", i, centroid, contour)

        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)

    log.debug("Updating vehicle count...")
    car_counter.update_count(matches, processed)

    return processed

# ============================================================================

# Main function calling and combining above functions, complete with logging
def main():
    log = logging.getLogger("main")

    log.debug("Creating background subtractor...")
    bg_subtractor = cv2.BackgroundSubtractorMOG2()

    log.debug("Pre-training the background subtractor...")
    #firstFrame = getFirstFrame('00002517.ASF')
    default_bg = cv2.imread(imageFilenameFormat % 119)
    bg_subtractor.apply(default_bg, None, 1.0)

    car_counter = None # Will be created after first frame is captured

    # Set up image source
    log.debug("Initializing video capture device #%s...", imageSource)
    cap = cv2.VideoCapture(IMAGE_SOURCE)

    frame_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    log.debug("Video capture frame size=(w=%d, h=%d)", frame_width, frame_height)

    log.debug("Starting capture loop...")
    frame_number = -1
    while True:
        frame_number += 1
        log.debug("Capturing frame #%d...", frame_number)
        ret, frame = cap.read()
        if not ret:
            log.error("Frame capture failed, stopping...")
            break

        log.debug("Got frame #%d: shape=%s", frame_number, frame.shape)

        if car_counter is None:
            # We do this here, so that we can initialize with actual frame size
            log.debug("Creating vehicle counter...")
            car_counter = VehicleCounter(frame.shape[:2], frame.shape[0] / 2)

        # Archive raw frames from video to disk for later inspection/testing
        if captureFromVideo:
            save_frame(imageFilenameFormat
                , frame_number, frame, "source frame #%d")

        log.debug("Processing frame #%d...", frame_number)
        processed = process_frame(frame_number, frame, bg_subtractor, car_counter)

        save_frame(imageDir + "/processed_%04d.png"
            , frame_number, processed, "processed frame #%d")

        cv2.imshow('Source Image', frame)
        cv2.imshow('Processed Image', processed)

        log.debug("Frame #%d processed.", frame_number)

        c = cv2.waitKey(waitTime)
        if c == 27:
            log.debug("ESC detected, stopping...")
            break

    log.debug("Closing video capture device...")
    cap.release()
    cv2.destroyAllWindows()
    log.debug("Done.")

# ============================================================================

if __name__ == "__main__":
    log = init_logging()

    if not os.path.exists(imageDir):
        log.debug("Creating image directory `%s`...", imageDir)
        os.makedirs(imageDir)

    main()
















### Create capture and masks
##cap = cv2.VideoCapture('00002517.ASF')
##fgbg = cv2.createBackgroundSubtractorMOG2()
##
### Initialize Variables
##
##counter = 0
##
##
##while True:
##    ret, frame = cap.read()
##    fgmask = fgbg.apply(frame, None, 0.01) #apply background subtractor to frame, look up args
##
##    # Create exit line on frame
##    cv2.line(frame, (170, 130), (236, 232), (255,255,0), 2)
##
##    # Find Contours of Moving Objects
##    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##
##    # Define hierarchy
##    try:
##        hierarchy = hierarchy[0]
##
##    except:
##        hierarchy = []
##
##    for contour, hier in zip(contours, hierarchy):
##        (x,y,w,h) = cv2.boundingRect(contour)
##
##        # set parameters for minimum contour size to analyze
##        if w > 200 and h > 200:
##            
##            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2) # draw bounding rectangle
##
##        # Find centroid of moving object
##        x1 = w/2
##        y1 = h/2
##
##        cx = x+x1
##        cy = y+y1
##
##        centroid = (cx, cy)
##
##        # Draw Circle at Centroid
##        cv2.circle(frame, (int(cx), int(cy)), 2, (0, 0, 255), -1)
##
##        # Make sure object crosses the line
##        if centroid > (27, 38) and centroid < (134, 108):
##            #if (cx <= 132) and (cx >= 20):
##            counter +=1
##            print(counter)
##
##            # if cy > 10 and cy < 160:
##            # to put counter text on screen:
##        #cv2.putText(frame, str(counter), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
##
##    # Show original image
##    cv2.imshow('Output', frame)
##    #cv2.imshow('fgmask', fgmask)
##
##    # Use Esc key to exit
##    key = cv2.waitKey(30)
##    if key == 27:
##        break
##
##cap.release()
##cv2.destroyAllWindows()
##            
