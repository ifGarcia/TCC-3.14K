#!/usr/bin/env python
# coding: utf-8

from math import sqrt
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
            for i in range(1,68): #There are 68 landmark points on each face
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=-1) #For each point, draw a red circle with thickness2 on the original frame
        for x in range(1,17):#queixo
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        for x in range(18,22):#Somb d
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        for x in range(23,27):#Somb e
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        for x in range(37,42):#olho e
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        cv2.line(frame, (shape.part(36).x, shape.part(36).y), (shape.part(41).x, shape.part(41).y), (0,255,0), 1)
        for x in range(43,48):#olho d
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        cv2.line(frame, (shape.part(42).x, shape.part(42).y), (shape.part(47).x, shape.part(47).y), (0,255,0), 1)    
        for x in range(28,36):#nariz
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        for x in range(30,36):#nariz ponta
            cv2.line(frame, (shape.part(30).x, shape.part(30).y), (shape.part(x).x, shape.part(x).y), (0,255,0), 1)
        for x in range(49,60):#boca fora
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        cv2.line(frame, (shape.part(48).x, shape.part(48).y), (shape.part(59).x, shape.part(59).y), (0,255,0), 1)
        for x in range(61,68):#boca dentro
            cv2.line(frame, (shape.part(x).x, shape.part(x).y), (shape.part(x-1).x, shape.part(x-1).y), (0,255,0), 1)
        cv2.line(frame, (shape.part(60).x, shape.part(60).y), (shape.part(67).x, shape.part(67).y), (0,255,0), 1)

        cv2.putText(frame, "Olho d: " + str(sqrt((shape.part(45).x-shape.part(47).x)**2) + ((shape.part(45).y-shape.part(47).y)**2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "Olho e: " + str(sqrt((shape.part(39).x-shape.part(41).x)**2) + ((shape.part(39).y-shape.part(41).y)**2)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "Boca  : " + str(sqrt((shape.part(63).x-shape.part(67).x)**2) + ((shape.part(63).y-shape.part(67).y)**2)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imshow("image", frame) #Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
        
cv2.destroyAllWindows()

video_capture.release()




