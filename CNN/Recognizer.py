# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:14:17 2020

@author: amit.sanghvi
"""
import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath('D:\ImRec'))
image_dir = os.path.join('D:\ImRec', "images")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower()
            #print(label, path)
            
            #getting labels and assigning them an Id
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            #print(label_ids)
            
            pil_image = Image.open(path).convert("L") # converting into grayscale
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, 1.1, 4)
            
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickel", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
