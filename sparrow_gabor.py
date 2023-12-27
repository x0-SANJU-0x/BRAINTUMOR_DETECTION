import cv2
import os
import numpy as np
import pandas as pd

# Function to extract flattened number array from an image
def extract_numbers_from_image(image_path):
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Perform any necessary preprocessing on the image
        # For example, you can resize the image, apply filters, etc.
        # For simplicity, let's assume the image is already preprocessed
        
        # Convert the image to a flattened NumPy array
        number_array = image.ravel()
        
        return number_array
    except Exception as e:
        print(f"Error extracting numbers from {image_path}: {str(e)}")
        return None

# Directories for input (resized images) and output (CSV file)
input_directory = 'C:/Users/sanch/braintumor/dataset/no'
output_csv_filename = 'C:/Users/sanch/braintumor/values_no.csv'

# List to store extracted number arrays
number_arrays = []

# Iterate through all image files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_directory, filename)
        number_array = extract_numbers_from_image(image_path)
        
        if number_array is not None:
            number_arrays.append(number_array)

# Convert the list of number arrays to a Pandas DataFrame
df = pd.DataFrame(number_arrays)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_filename, index=False)

print(f"Number arrays saved to {output_csv_filename}")

# Select the best 500 rows (you can adjust the criteria for selecting the best)
top_500_df = df.head(500 )

# Save the selected 500 rows to a new CSV file
selected_csv_filename = 'C:/Users/sanch/braintumor/selected_values.csv'
top_500_df.to_csv(selected_csv_filename, index=False)

print(f"Top 500 rows saved to {selected_csv_filename}")
