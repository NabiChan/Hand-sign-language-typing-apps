import cv2
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import cvzone
import string

bbox_simplemodel = load_model("detect_hand_signs.h5")
translation_table=str.maketrans("", "", "JZ")
alphabelt_list= list(string.ascii_uppercase.translate(translation_table))

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success,img=cap.read()
    inputimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inputimg=cv2.resize(inputimg, (28, 28))
    inputimg= inputimg/255
    inputimg = np.expand_dims(inputimg, axis=-1)
    inputimg = np.expand_dims(inputimg, axis=0)

    alphabelt,box=bbox_simplemodel.predict(inputimg)
    detecton=np.empty((0,4))
    for result in box:
        x1, y1, x2, y2 = box[0]
        x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,255),1)
        w, h = x2-x1,y2-y1
        alphabelt_name=np.argmax(alphabelt, axis=1)
        current_alphabelt=alphabelt_list[alphabelt_name[0]]

        cvzone.cornerRect(img,(x1,y1,w,h),l=9)
        cvzone.putTextRect(img,f'{current_alphabelt}',(max(0,x1),max(35,y1)),
                        scale=1, thickness=1,offset=3)
        currentArray = np.array ([x1,x1,y2,y2])
        detecton=np.vstack((detecton,currentArray))

    cv2.imshow("Hahaha i like ur mom", img)
    cv2.waitKey(1)