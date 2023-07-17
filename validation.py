import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set the paths for the original and masked image folders for validation
val_original_images_path = 'E:/Project Files/research project/02/project stuff/dataset_low/valid/satellite/'
val_masked_images_path = 'E:/Project Files/research project/02/project stuff/dataset_low/valid/mask/'

# Define the desired output shape
output_height, output_width = 118, 118

# Load the validation original and masked images into arrays
val_original_images = []
val_masked_images = []

# Iterate over the images in the validation folder
for filename in os.listdir(val_original_images_path):
    if filename.endswith("_sat.jpg"):  # Filter for original image files
        original_image = Image.open(os.path.join(val_original_images_path, filename))
        original_image = original_image.resize((output_width, output_height))
        val_original_images.append(np.array(original_image))

for filename in os.listdir(val_masked_images_path):
    if filename.endswith("_mask.png"):  # Filter for masked image files
        masked_image = Image.open(os.path.join(val_masked_images_path, filename))
        masked_image = masked_image.resize((output_width, output_height))
        val_masked_images.append(np.array(masked_image))

# Convert the lists to numpy arrays
val_original_images = np.array(val_original_images)
val_masked_images = np.array(val_masked_images)

# Normalize the validation images
val_original_images = val_original_images / 255.0
val_masked_images = val_masked_images / 255.0

# Load the trained model
model = keras.models.load_model('my_model.keras')

# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(val_original_images, val_masked_images.reshape(-1, output_height * output_width * 3))
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)
