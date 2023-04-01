import cv2,os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
#path = 'C:\\New folder\\dataset'

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        if(nbr_predicted == 31 and confidence >= 35):
            nbr_predicted = "Kundan"
        elif(nbr_predicted == 32 and confidence >= 35):
            nbr_predicted = "Sachin"
        elif(nbr_predicted == 33 and confidence >= 35):
            nbr_predicted = "Himanshu"    
        else:
            nbr_predicted = "Unknown"
        #cv2.cv.PutText(cv2.cv.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
        cv2.putText(im,nbr_predicted,(x,y-10),font,0.55,(0,255,0),1)

        cv2.imshow('im',im)
        cv2.waitKey(10)
