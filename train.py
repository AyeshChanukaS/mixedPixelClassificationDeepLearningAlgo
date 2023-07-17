import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set the paths for the original and masked image folders
original_images_path = 'E:/Project Files/research project/02/project stuff/dataset_low/train/satellite/'
masked_images_path = 'E:/Project Files/research project/02/project stuff/dataset_low/train/mask/'

# Define the desired output shape
output_height, output_width = 118, 118

# Load the original and masked images into arrays
original_images = []
masked_images = []

# Iterate over the images in the folder
for filename in os.listdir(original_images_path):
    if filename.endswith("_sat.jpg"):  # Filter for original image files
        original_image = Image.open(os.path.join(original_images_path, filename))
        original_image = original_image.resize((output_width, output_height))
        original_images.append(np.array(original_image))
        print("Appended original image:", filename)

for filename in os.listdir(masked_images_path):
    if filename.endswith("_mask.png"):  # Filter for masked image files
        masked_image = Image.open(os.path.join(masked_images_path, filename))
        masked_image = masked_image.resize((output_width, output_height))
        masked_images.append(np.array(masked_image))
        print("Appended masked image:", filename)

# Convert the lists to numpy arrays
original_images = np.array(original_images)
masked_images = np.array(masked_images)

# Normalize the images
original_images = original_images / 255.0
masked_images = masked_images / 255.0

# Define your model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(output_height, output_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(output_height * output_width * 3, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(original_images, masked_images, epochs=100, batch_size=32, validation_split=0.2)

# Save the trained model in the Keras format (.h5)
model.save('my_model.keras')
