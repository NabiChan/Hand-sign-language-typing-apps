import streamlit as st
import cv2
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import cvzone
import string

translation_table=str.maketrans("", "", "JZ")
alphabelt_list= list(string.ascii_uppercase.translate(translation_table))

st.title('Hand Sign Langugae Typing App')
selected=st.selectbox("chose your model",("Simple CNN model (faster)","Alexnet model (have error with accuracy still fixing so chose it still like chose simple cnn)"))

if selected=="Simple CNN model (faster)":
    model = load_model("detect_hand_signs.h5")
elif selected=="Alexnet model (have error with accuracy still fixing so chose it still like chose simple cnn)":
    model=load_model("detect_hand_signs.h5")

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

video_holder=st.empty()
stop_button= st.button('Stop')
enter_button= st.button('Enter')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

while True:
    success,img=cap.read()
    showed_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    inputimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inputimg=cv2.resize(inputimg, (28, 28))
    inputimg= inputimg/255
    inputimg = np.expand_dims(inputimg, axis=-1)
    inputimg = np.expand_dims(inputimg, axis=0)

    alphabelt,box=model.predict(inputimg)
    detecton=np.empty((0,4))
    for result in box:
        x1, y1, x2, y2 = box[0]
        x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
        cv2.rectangle(showed_img, (x1,y1),(x2,y2), (255,0,255),1)
        w, h = x2-x1,y2-y1
        alphabelt_name=np.argmax(alphabelt, axis=1)
        current_alphabelt=alphabelt_list[alphabelt_name[0]]

        cvzone.cornerRect(showed_img,(x1,y1,w,h),l=9)
        cvzone.putTextRect(showed_img,f'{current_alphabelt}',(max(0,x1),max(35,y1)),
                        scale=1, thickness=1,offset=3)
        currentArray = np.array ([x1,x1,y2,y2])
        detecton=np.vstack((detecton,currentArray))
        
        if enter_button:
            st.chat_message("user").markdown(current_alphabelt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": current_alphabelt})

    video_holder.image(showed_img, channels='RGB')
    cv2.waitKey(1)
    if stop_button:
         break
    
cap.release()
cv2.destroyAllWindows()