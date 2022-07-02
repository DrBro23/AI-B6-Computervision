# Using GeneratePictures.py
# Type in command line:
# python GeneratePictures.py --output Pictues\\<name>

# source:
# https://github.com/jasmcaus/opencv-course/tree/master/Section%20#3%20-%20Faces
# not reachable

# Repo caffe model
# https://gist.github.com/GilLevi/c9e99062283c719c03de

# Import required modules
import cv2 as cv
import time
import argparse
import os


# Function for detection and annotation
def getFaceBox(net, frame, conf_threshold):
    # Frame shape parameters
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    
    bboxes = []
    
    for i in range(detections.shape[2]):
        # Annotation of the confidences
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            w = x2 - x1
            h = y2 - y1
            
            # Annotation of the frame in color
            cv.rectangle(frameOpencvDnn, (x1,y1), (x1+w,y1+h), (colorcode), thickness=1)
            lineWidth = min(int((x2-x1)*0.2), int((y2-y1)*0.2))
            cv.line(frameOpencvDnn, (x1,y1), (x1+lineWidth, y1), (colorcode), thickness=5)
            cv.line(frameOpencvDnn, (x1,y1), (x1, y1+lineWidth), (colorcode), thickness=5)
            cv.line(frameOpencvDnn, (x2,y1), (x2-lineWidth, y1), (colorcode), thickness=5)
            cv.line(frameOpencvDnn, (x2,y1), (x2,y1+lineWidth), (colorcode), thickness=5)
            cv.line(frameOpencvDnn, (x1,y2), (x1+lineWidth, y2), (colorcode), thickness=5)
            cv.line(frameOpencvDnn, (x1,y2), (x1, y2-lineWidth), (colorcode), thickness=5)
            cv.line(frameOpencvDnn, (x2,y2), (x2-lineWidth, y2), (colorcode), thickness=5)
            cv.line(frameOpencvDnn, (x2,y2), (x2,y2-lineWidth), (colorcode), thickness=5)
    return frameOpencvDnn, bboxes

# Color code for annotation
colorcode = (0,129,255)

# Picturecounter
counter = 0
total = 1

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
args = vars(ap.parse_args())

# Face detector
faceProto = "Source/opencv_face_detector.pbtxt"
faceModel = "Source/opencv_face_detector_uint8.pb"

# Load network
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Starting the camera stream
cap = cv.VideoCapture((0), cv.CAP_DSHOW)
padding = 20
# Minumum of confidence to start working
conf_threshold = 0.7

# Record the time of the last processed frame
prev_frame_time = 0
# Record the time of the current processed frame
new_frame_time = 0


# Check existing pictures in folder
picRaw = os.listdir(args["output"])

if not picRaw:
    print("\n--- No pictures available yet! ---\n")
    print("Happy about new pictures!")
else:
	picName = (picRaw.pop()).split('.')
	total = (int(picName[0])) + 1
	print("\n--- Adding new pictures to existing ones! ---\n")
	print("Current pictures in folder:", (total - 1))
	print("---------------------------")
 
 
 
while True:
    # Read frame
    t = time.time()
    new_frame_time = time.time()
    
	# Calculating the fps
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
	# Converting the fps
    fps = int(fps)
    fps = str(fps)

    # Get frame and check if valid frame exist
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    frame_flipped = cv.flip(frame,1)

    # Call detection function and check if frame is detected
    frameFace, bboxes = getFaceBox(faceNet, frame_flipped, conf_threshold)
    if not bboxes:
        print("No face Detected, checking next frame!")
        continue
    
    # Pictures adjustment with padding
    for bbox in bboxes:
        face = frame_flipped[max(0,bbox[1]-padding):min(bbox[3]+padding,frame_flipped.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame_flipped.shape[1]-1)]
        # Displaying camera stream on screen
        cv.imshow("Generate Pictures", frameFace)

    # Interaction parameter
    key = cv.waitKey(1) & 0xFF
    # Pressing k saves a picture
    # Pressing q quits while loop
    if key == ord("k"):
        p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
        print("{}.png".format(str(total).zfill(5)))
        cv.imwrite(p, face)
        total += 1
        counter += 1
    
    elif key == ord("q"):
        break
   
   

# Print the number of total and saved faces
print("---------------------------")
print("Taken face images:", counter)
print("Total face images: {}".format(total - 1))
print("\nCleaning up...")

# Destroys camera stream and OpenCV windows
cap.release()
cv.destroyAllWindows()