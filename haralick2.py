import cv2
import os
import mahotas.features.texture as mht
import numpy as np

# Define the input and output directories
input_dir = 'C:/Users/sanch/braintumor/dataset/noo'  # Replace with the path to your input folder containing images
output_dir = 'C:/Users/sanch/braintumor/dataset/haralic_no'  # Replace with the path to your output folder

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all image files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read the image in grayscale
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Apply Haralick texture features
        textures = mht.haralick(img)

        # Calculate the mean of the Haralick features
        mean_textures = np.mean(textures, axis=0)

        # Save the processed image (you can also save the original image if needed)
        cv2.imwrite(output_path, img)

        # Print the mean Haralick texture features
        print(f'Image: {filename}')
        print('Mean Haralick Features:')
        print(mean_textures)
        print()

print("Processing completed. Images saved in the output folder.")
