import tensorflow as tf
import keras
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import string
import random
import cv2

train=pd.read_csv('archive/sign_mnist_test.csv')
test=pd.read_csv('archive/sign_mnist_test.csv')

tungtungtung=tf.convert_to_tensor(train.drop('label',axis=1))
balerinacapuchina=tf.convert_to_tensor(test.drop('label',axis=1))


trainlabel= keras.utils.to_categorical(train['label'])
testlabel= keras.utils.to_categorical(test['label'])
print(testlabel)

def show_img(ahihidongoc):
    ahihidongoc=tf.reshape(ahihidongoc,[-1,28,28])
    plt.imshow(ahihidongoc[0])
    plt.show()

show_img(tungtungtung)

minmax=MinMaxScaler()
tungtungtung=minmax.fit_transform(tungtungtung)
tungtungtung= tf.reshape(tungtungtung,[-1,28,28,1])
balerinacapuchina=minmax.fit_transform(balerinacapuchina)
balerinacapuchina= tf.reshape(balerinacapuchina,[-1,28,28,1])

X_train, X_val, y_train, y_val=train_test_split(tungtungtung.numpy(), trainlabel, test_size=0.33)
X_train=tf.convert_to_tensor(X_train)
X_val=tf.convert_to_tensor(X_val)

simplemodel=keras.Sequential([
    layers.InputLayer(shape=(28,28,1)),
    layers.Conv2D(64,kernel_size=(3,3), strides=(2,2),activation='relu'),
    layers.MaxPool2D(pool_size=3),
    layers.Conv2D(128,kernel_size=(5,5), strides=(1,1),padding='same',activation='relu'),
    layers.Conv2D(256,kernel_size=(5,5), strides=(1,1),padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=3,strides=(2,2)),
    layers.Flatten(),
    layers.Dense(32,activation='relu')
])

input_layer=keras.Input(shape=(28,28,1))

ahaha=simplemodel(input_layer)

alphabelts= layers.Dense(25, activation='softmax', name='alphabelt')(ahaha)

box= layers.Dense(4, activation='sigmoid', name='box')(ahaha)

bbox_simplemodel= keras.Model(inputs=input_layer, outputs=[alphabelts, box])

def get_box(tensor):
    img = tensor.numpy()
    img = img.squeeze()
    #check normalized or not
    if img.max()<=1.0:
        img= (img * 255).astype('uint8')
    else:
        img =img.astype('uint8')
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    return np.array([x/28, y/28, (x+w)/28, (y+h)/28], dtype='float32')

y_train_box= tf.convert_to_tensor([get_box(img) for img in X_train])
y_val_box= tf.convert_to_tensor([get_box(img) for img in X_val])
y_test_box=tf.convert_to_tensor([get_box(img) for img in balerinacapuchina])

bbox_simplemodel.summary()
bbox_simplemodel.compile(optimizer ='adam', loss ={'alphabelt':'categorical_crossentropy','box':keras.losses.Huber()}, 
                         metrics ={'alphabelt':'accuracy','box':'mae'})
bbox_simplemodel.fit(X_train, {'alphabelt':y_train,'box':y_train_box}, batch_size=64, 
                     epochs=5,validation_data=(X_val,{'alphabelt':y_val,'box':y_val_box}))

predicted,box= bbox_simplemodel.predict(balerinacapuchina)
predicted=np.argmax(predicted, axis=1)
testlabel=np.argmax(testlabel, axis=1)

translation_table=str.maketrans("", "", "JZ")
alphabelt_list= list(string.ascii_uppercase.translate(translation_table))

print(classification_report(testlabel,predicted,target_names=alphabelt_list))
cm = confusion_matrix(testlabel, predicted)
cmp = ConfusionMatrixDisplay(cm, display_labels=alphabelt_list)
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(ax=ax)
plt.show()

def show_randomresult_img(ahihidongoc,model):
    ahihidongoc=tf.reshape(ahihidongoc,[-1,28,28,1])
    Gojo_satoru=ahihidongoc[random.randint(0,len(ahihidongoc)-1)]
    Gojo_satoru=tf.expand_dims(Gojo_satoru, axis=0)
    pred,box=model.predict(Gojo_satoru)
    
    img=tf.squeeze(Gojo_satoru).numpy()
    img = img * 255
    img = img.astype(np.uint8)
    img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    x1, y1, x2, y2 = box[0]
    x1*= 28
    y1*= 28
    x2*= 28
    y2*= 28
    x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
    x1= max(0, min(x1, 27))
    y1= max(0, min(y1, 27))
    x2= max(0, min(x2, 27))
    y2= max(0, min(y2, 27))
    
    cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,255),1)
    pred=np.argmax(pred, axis=1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Class:{alphabelt_list[pred[0]]}')
    plt.show()

show_randomresult_img(balerinacapuchina,bbox_simplemodel)

bbox_simplemodel.save('detect_hand_signs.h5')



Alexnet=keras.Sequential([
    layers.InputLayer(shape=(28,28,1)),
    layers.Resizing(224,224),
    layers.Conv2D(64,kernel_size=(11,11), strides=(2,2),padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=3,strides=(2,2)),
    layers.Conv2D(128,kernel_size=(5,5), strides=(1,1),padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=3,strides=(2,2)),
    layers.Conv2D(256,kernel_size=(3,3), strides=(1,1),padding='same',activation='relu'),
    layers.Conv2D(128,kernel_size=(3,3), strides=(1,1),padding='same',activation='relu'),
    layers.Conv2D(64,kernel_size=(3,3), strides=(1,1),padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=3,strides=(2,2)),
    layers.Flatten(),
    layers.Dense(4096,activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(4096,activation='relu'),
    layers.Dropout(0.25)
])

inputs=layers.Input(shape=(28,28,1))

ahaha=Alexnet(inputs)

alphabelt_complex= layers.Dense(25, activation='softmax',name='alphabelt_complex')(ahaha)

box_complex= layers.Dense(4, activation='sigmoid',name='box_complex')(ahaha)

bbox_alexnet_model= keras.Model(inputs=inputs, outputs=[alphabelt_complex, box_complex])

bbox_alexnet_model.compile(optimizer ='adam', loss ={'alphabelt_complex':'categorical_crossentropy','box_complex':keras.losses.Huber()}, 
                         metrics ={'alphabelt_complex':'accuracy','box_complex':'mae'})
bbox_alexnet_model.fit(X_train, {'alphabelt_complex':y_train,'box_complex':y_train_box}, batch_size=64, 
                     epochs=5,validation_data=(X_val,{'alphabelt_complex':y_val,'box_complex':y_val_box}))

predicted= bbox_alexnet_model.predict(balerinacapuchina)
predicted=np.argmax(predicted, axis=1)

bbox_alexnet_model.save('detect_hand_signs_alexnet.h5')