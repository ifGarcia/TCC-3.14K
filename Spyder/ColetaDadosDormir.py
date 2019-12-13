# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:13:05 2019

@author: Bruno

Coleta de dados.

Dormir.

"""

import cv2
import dlib


video_capture = cv2.VideoCapture(0)
#Change Frame Rate
video_capture.set(cv2.CAP_PROP_FPS, 30)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

while True:
    a, frame = video_capture.read()
    frame = cv2.flip(frame,180) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) #Detect the faces in the image
    
    if detections != None:
        for k,d in enumerate(detections): #For each detected face  
            shape = predictor(clahe_image, d) #Get coordinates
                
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
        
cv2.destroyAllWindows()

video_capture.release()