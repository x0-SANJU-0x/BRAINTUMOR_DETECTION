import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array

# ...

dataset = []
label = []

target_size = (64, 64)


dataset = np.array(dataset)
label = np.array(label)

folders = ['gabor1', 'gabor2', 'gabor3', 'gabor4', 'gabor5', 'gabor6']

best_accuracy = 0.0
best_model = None
def update_location_eq3(accuracy,best_accuracy):
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
    pass


for folder_name in folders:
    image_directory = 'dataset/' + folder_name + '/'

    no_tumor = os.listdir(image_directory + 'gabor-no/')
    yes_tumor = os.listdir(image_directory + 'gabor-yes/')

    dataset = []
    label = []
    G = 3 
    PD = 10  
    SD = 20 
    
    R2 = np.random.rand()  # Initialize R2 randomly
    n = 100  # Number of accuracy values
    Xbest = None  # Global optimal solution, initialize as None
    input_size = 64
    for i, image_name in enumerate(no_tumor):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'gabor-no/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize(target_size)  # Resize the image
            image = img_to_array(image)  # Convert to a NumPy array
            dataset.append(image)
            label.append(0)
    
    for i, image_name in enumerate(yes_tumor):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'gabor-yes/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize(target_size)  # Resize the image
            image = img_to_array(image)  # Convert to a NumPy array
            dataset.append(image)
            label.append(1)

    dataset = np.array(dataset)
    label = np.array(label)

    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

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

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)

    accuracy = model.evaluate(x_test, y_test, verbose=1)[1]

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        
best_model.save('best_no_model.h5')
#start
accuracy_values = [np.random.uniform(0, 1) for _ in range(n)]
t = 0
while t < G:
    accuracy_values.sort()
    current_best = accuracy_values[0]  
    
    if Xbest is None or current_best > Xbest:
        Xbest = current_best
    
    R2 = np.random.rand()
    
    for i in range(PD):
        update_location_eq3(accuracy,best_accuracy)

    t += 1
best_accuracyy = Xbest
print("Best Accuracy Value:", best_accuracy)
#end
accuracy_var = best_accuracy
with open("best_accuracy.txt", "w") as f:
    f.write(str(accuracy_var))
