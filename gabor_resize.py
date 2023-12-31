import os
import cv2


input_directory =  'C:/Users/sanch/braintumor/dataset/no'  
output_directory =  'C:/Users/sanch/braintumor/dataset/noo' 
input_file='C:/Users/sanch/braintumor/dataset/no'
target_dimensions = (200, 200)  #dimensions 
os.makedirs(output_directory, exist_ok=True)


for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        
        # Read the image using OpenCV
        image = cv2.imread(input_path)
        
        if image is not None:
            
            resized_image = cv2.resize(image, target_dimensions, interpolation=cv2.INTER_AREA)
            
            
            cv2.imwrite(output_path, resized_image)
            print(f"Resized and saved: {output_path}")
        else:
            print(f"Error reading image: {input_path}")
