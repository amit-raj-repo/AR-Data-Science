# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:54:46 2020

@author: amit.sanghvi
"""
import cv2
import pickle


cap = cv2.VideoCapture(0);

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

with open("labels.pickel", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

while cap.isOpened():
    ret, frame = cap.read()
    
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 4)
    
    for (x, y, w, h) in faces:
        roi_gray = grey[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        
        if conf>=50: #and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
        
    cv2.imshow('face_detect', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
