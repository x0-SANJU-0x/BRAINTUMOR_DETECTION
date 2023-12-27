import numpy as np
import cv2
import pandas as pd

def compute_coarseness(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #coarseness 
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    coarseness_image = cv2.filter2D(gray_image, -1, kernel)

    return coarseness_image

for i in range (0,1500):
        x=-1
        j=str(i)
        image = cv2.imread('C:\\Users\\sanch\\braintumor\\dataset\\noo\\no'+j+'.jpg')

    
        num = 1 
        count=1
        x +=1
        y=str(x)
  
        coarseness_image = compute_coarseness(image)
        temp=str(count)
        cv2.imwrite('C:\\Users\\sanch\\braintumor\\dataset\\tamura2\\tamura-no\\no'+j+'.jpg', coarseness_image)
        count +=1
        num += 1  
                    
