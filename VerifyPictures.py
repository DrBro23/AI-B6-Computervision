# Using VerifyPictures.py
# Type in command line:
# python VerifyPictures.py

# source:
# Caffe models 
# Source: https://gist.github.com/GilLevi/c9e99062283c719c03de

# Import required modules
import cv2 as cv
import numpy as np
import time
import argparse


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


# Color code for Annotation
colorcode = (0,129,255)

# Defined people and recognition file
people = [ -- names -- ] ## Add names TODO
face_recognizer = cv.face.LBPHFaceRecognizer_create() 
face_recognizer.read('Source/face_trained.yml')

# Defined emotions and expression model
emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
expression_net = cv.dnn.readNetFromONNX('Source/emotion-ferplus-8.onnx')

# Argument parser for input and used device
parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="gpu", help="Device to inference on")
args = parser.parse_args()
args = parser.parse_args()

# Face detector model
faceProto = "Source/opencv_face_detector.pbtxt"
faceModel = "Source/opencv_face_detector_uint8.pb"
# Age recognition model
ageProto = "Source/age_deploy.prototxt"
ageModel = "Source/age_net.caffemodel"
ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
# Gender recognition model
genderProto = "Source/gender_deploy.prototxt"
genderModel = "Source/gender_net.caffemodel"
genderList = ['Male', 'Female']

# Defined mean values for blop
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load network
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

# Argument for using CPU or GPU (CUDA installation required)
if args.device == "cpu":
    faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")

elif args.device == "gpu":
    faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Starting the camera stream
cap = cv.VideoCapture((args.input if args.input else 0), cv.CAP_DSHOW)
padding = 20
# Minumum of confidence to start working
conf_threshold = 0.7

# Starting recording
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# out = cv.VideoWriter('Video/video.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Record the time of the last processed frame
prev_frame_time = 0
# Record the time of the current processed frame
new_frame_time = 0



while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    new_frame_time = time.time()
    
	# Calculating the fps
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
	# Converting the datatype of the fps
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
        
        # Convert RGB to BGR
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        rec_label, rec_confidence = face_recognizer.predict(gray)
        
        # Resize Image
        bleed_resized_face = cv.resize(gray, (64,64))
        bleed_processed_face = bleed_resized_face.reshape(1,1,64,64)
        
        # Recognize expression
        expression_net.setInput(bleed_processed_face)
        Output = expression_net.forward()
        
        # Compute softmax values for each sets of scores  
        expanded = np.exp(Output - np.max(Output))
        probablities =  expanded / expanded.sum()
        
        # Get the final probablities by getting rid of any extra dimensions 
        prob = np.squeeze(probablities)
        # Get the predicted emotion
        predicted_emotion = emotions[prob.argmax()]

        # Image preprocessing for network
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # Recognize age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        # Recognize gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Annotation of the frame
        cv.putText(frameFace, ("{} ({:.1f})".format(people[rec_label], rec_confidence)), (bbox[0], bbox[1]-61), cv.FONT_HERSHEY_SIMPLEX, 0.6, (colorcode), 2, cv.LINE_AA)
        cv.putText(frameFace, ("{} ({:.1f})".format(age, agePreds[0].max() * 100)), (bbox[0], bbox[1]-44), cv.FONT_HERSHEY_SIMPLEX, 0.6, (colorcode), 2, cv.LINE_AA)
        cv.putText(frameFace, ("{} ({:.1f})".format(gender, genderPreds[0].max() * 100)), (bbox[0], bbox[1]-27), cv.FONT_HERSHEY_SIMPLEX, 0.6, (colorcode), 2, cv.LINE_AA)
        cv.putText(frameFace,'{}'.format(predicted_emotion), (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (colorcode), 2, cv.LINE_AA)
        cv.putText(frameFace, fps, (7, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (colorcode), 2, cv.LINE_AA)

        cv.namedWindow("output", cv.WINDOW_NORMAL) 
        # resizing = cv.resize(frameFace, (1152,864), cv.INTER_AREA)
        resizing = cv.resize(frameFace, (1024,768), cv.INTER_AREA)
        # Displaying camera stream on screen
        cv.imshow("Verify Pictures", resizing)
        # Write camera stream to file
        # out.write(frameFace)
    
    # Interaction parameter
    key = cv.waitKey(1) & 0xFF
    # Pressing q quits while loop
    if key == ord("q"):
        break
    
# Destroys camera stream and OpenCV windows
cap.release()
cv.destroyAllWindows()