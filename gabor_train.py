import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from  keras.utils import to_categorical
image_directory='dataset/'

no_tumor = os.listdir(image_directory+ 'gabor-no/')
yes_tumor = os.listdir(image_directory+ 'gabor-yes/')

dataset=[]
label=[] 

input_size=64
for i , image_name in enumerate(no_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'gabor-no/'+image_name)
        image=Image.fromarray(image,'RGB') 
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
for i , image_name in enumerate(yes_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'gabor-yes/'+image_name)
        image=Image.fromarray(image,'RGB') 
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
        
dataset=np.array(dataset)
label=np.array(label)

x_train, x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=0)

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)



model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,
batch_size=16,verbose=1,epochs=10,
validation_data=(x_test,y_test ),
shuffle=False)
model.save('gabormodel.h5')

#saving accuracy value in n ote pad

accuracy = model.evaluate(x_test, y_test, verbose=1)[1]
accuracy_var = accuracy

print(accuracy_var)
prn=str(accuracy_var)
f = open("accu.txt", "a")
f.write(""+prn+"")
f.close()