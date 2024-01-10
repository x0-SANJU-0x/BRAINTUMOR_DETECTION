import numpy as np
import cv2
import pandas as pd
 
for i in range (0,1500):
    x=-1
    j=str(i)
    img = cv2.imread('C:\\Users\\sanch\\braintumor\\dataset\\noo\\no'+j+'.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img2 = img.reshape(-1)
    df = pd.DataFrame()
    df['Original Image'] = img2

#Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []  #Create empty list to hold all kernels that we will generate in a loop
    sigma= 5
    lamda= 1.5707963267948966 
    gamma= 0.5 
    gabor_label = 'Gabor' + str(num)  
#                print(gabor_label)
    ksize=5  #Try 15 for hidden image. Or 9 for others
    phi = 0  #0.8 for hidden image. Otherwise leave it to 0
    count=1
    x +=1
    y=str(x)
    for theta in (2.3844981,5.497787143782138, 7,0.7854,0.7854 ,0.7854):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)    
        kernels.append(kernel)
#Now filter the image and add values to a new column 
        fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)                
        filtered_img = fimg.reshape(-1)
        temp=str(count)

        cv2.imwrite('C:\\Users\\sanch\\braintumor\\dataset\\gabor'+temp+'-no\\y'+j+'.jpg', filtered_img.reshape(img.shape))
        df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
        print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
        count +=1
        num += 1  #Increment for gabor column label
                    
print(df.head()) 
