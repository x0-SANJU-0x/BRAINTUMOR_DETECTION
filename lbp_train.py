import os
import cv2
import numpy as np
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Specify the directory containing "yes" and "no" image categories
image_directory = 'dataset/'

no_tumor = os.listdir(os.path.join(image_directory, 'noo'))
yes_tumor = os.listdir(os.path.join(image_directory, 'yess'))

# Lists to store LBP features and labels
lbp_features = []
labels = []

# Extract LBP features from images
for category in ["yes", "no"]:
    category_dir = os.path.join(image_directory, category)
    label = 1 if category == "yes" else 0  # Map "yes" to 1 and "no" to 0
    for filename in os.listdir(category_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(category_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            lbp = feature.local_binary_pattern(image, 8, 1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
            lbp_features.append(hist)
            labels.append(label)

# Convert lists to NumPy arrays
X = np.array(lbp_features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define a simple neural network model
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model as an H5 file
model.save('lbp_model.h5')
