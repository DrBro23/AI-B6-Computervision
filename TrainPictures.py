# Using TrainPictures.py
# Type in command line:
# python TrainPictures.py

# source:
# https://github.com/jasmcaus/opencv-course/tree/master/Section%20#3%20-%20Faces
# pylint:disable=no-member

# Import required modules
import cv2 as cv
import numpy as np
import os


# Function for labeling pictures
def label_training_images(DIR, list_of_people):
    label_list =[]
    img_list = []
    
    for person in list_of_people:
            path = os.path.join(DIR, person)
            label = list_of_people.index(person)

            for img in os.listdir(path):
                img_path = os.path.join(path,img)
                img_array = cv.imread(img_path)
                
                # Convert RGB to BGR
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
                if gray is None:
                    continue 
                
                label_list.append(label)
                img_list.append(gray)
                
    return label_list, img_list
    
    
# Defined people and path
people = [ -- names -- ] ## Add names TODO
DIR = r'Pictures'

print("----- Preprocessing start -----")

# Call labeling function     
label_list, img_list = label_training_images(DIR, people)
features = np.array(img_list, dtype='object')
labels = np.array(label_list)

print("----- Preprocessing done -----")

# Train the recognizer on the features list and the labels list
print("----- Training start -----")
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

# Save trained model to file
face_recognizer.save('Source/face_trained.yml')
np.save('Source/features.npy', features)
np.save('Source/labels.npy', labels)

print("----- Training done -----")
print("----- Model saved -----")