import cv2
import numpy as np
import os

# Function to calculate LBP pattern of a pixel
def get_pixel(img, center, x, y):
    try:
        if img[x][y] >= center:
            return 1
        else:
            return 0
    except:
        return 0

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []

    # Define the 8 neighbors' positions
    neighbors = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                 (x, y + 1), (x + 1, y + 1), (x + 1, y),
                 (x + 1, y - 1), (x, y - 1)]

    for neighbor in neighbors:
        val_ar.append(get_pixel(img, center, *neighbor))

    # Convert the binary values to a decimal value
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * (2 ** i)

    return val

# Specify the input directory containing the images
input_directory = 'C:/Users/sanch/braintumor/dataset/yess'  # Replace with your input directory path

# Specify the output directory where LBP-filtered images will be saved
output_directory = 'C:/Users/sanch/braintumor/dataset/lbp/lbp-yes'  # Replace with your output directory path

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through all image files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_directory, filename)

        # Load the image in color format (BGR)
        img_bgr = cv2.imread(input_path, 1)

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Create an empty NumPy array for the LBP image
        img_lbp = np.zeros_like(img_gray)

        # Iterate through all pixels and compute LBP values
        for i in range(1, img_gray.shape[0] - 1):
            for j in range(1, img_gray.shape[1] - 1):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

        # Save the LBP image in the output directory with the same filename as the original image
        lbp_output_path = os.path.join(output_directory, f'LBP_{filename}')
        cv2.imwrite(lbp_output_path, img_lbp)
        print(f"LBP image saved: {lbp_output_path}")

print("LBP processing is finished")
