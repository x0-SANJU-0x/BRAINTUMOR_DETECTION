import cv2
import numpy as np
import csv

file = open('Book1.xlsx', 'w')
file =  csv.writer(file)

image=cv2.imread('C:\\Users\\sanch\\braintumor\\dataset\\pred\\pred0.jpg', cv2.IMREAD_GRAYSCALE)
 # Apply Gaussian blur to the image
blurred = cv2.GaussianBlur(image, (5, 5), 0)

gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# Calculate coarseness
coarseness = np.mean(np.abs(gradient_x) + np.abs(gradient_y))

# Calculate contrast
contrast = np.std(image)

# Calculate directionality
gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
directionality = np.arctan2(np.mean(gradient_y), np.mean(gradient_x))

# Convert directionality from radians to degrees
directionality_degrees = np.degrees(directionality)
print(f'Coarseness: {coarseness}')
print(f'Contrast: {contrast}')
print(f'Directionality (degrees): {directionality_degrees}')
file.writerow([coarseness, contrast, directionality])
