import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np

# LBP' model
best_no_model = load_model('gabormodel.h5')

# TAMURA & GABOR
best_tamura_model = load_model('best_tamura.h5')
best_lbp_model = load_model('best_lbp.h5')
image = cv2.imread('C:\\Users\\sanch\\braintumor\\dataset\\pred\\pred51.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)

result_no = best_no_model.predict(input_img)
result_tamura = best_tamura_model.predict(input_img)
result_lbp = best_lbp_model.predict(input_img)
classes_x_no = np.argmax(result_no, axis=1)
classes_x_tamura = np.argmax(result_tamura, axis=1)
classes_x_lbp = np.argmax(result_lbp, axis=1)
Flg=0
flg=0
if(classes_x_no==1):
    ff=1
else:
    ff=0
if classes_x_no == 0:
    flg=flg+1
    Flg=1

if classes_x_tamura == 0:
    flg=flg+1

if classes_x_lbp == 0:
    flg=flg+1

if ff==0:
    print(": NO BRAIN TUMOR DETECTED")
elif ff==1:
    print(": BRAIN TUMOR DETECTED")
