import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('my_model.keras')

# Set the paths for the original and masked image folders
original_images_path = 'E:/Project Files/research project/02/project stuff/dataset/valid/'
masked_images_path = 'E:/Project Files/research project/02/project stuff/dataset_low/valid/masked_identified/'

# Define the desired output shape
output_height, output_width = 118, 118

# Create the masked output folder if it doesn't exist
if not os.path.exists(masked_images_path):
    os.makedirs(masked_images_path)

# Function to create a red mask for mixed pixels
def create_red_mask(mask):
    red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    red_mask[mask == 1] = [255, 0, 0]  # Set mixed pixels to red color
    return red_mask

# Function to create a white mask for pure pixels
def create_white_mask(mask):
    white_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    white_mask[mask == 0] = [255, 255, 255]  # Set pure pixels to white color
    return white_mask

# Iterate over the images in the folder
for filename in os.listdir(original_images_path):
    if filename.endswith(".jpg"):  # Filter for original image files
        original_image = Image.open(os.path.join(original_images_path, filename))
        original_image = original_image.resize((output_width, output_height))
        original_image_array = np.array(original_image) / 255.0

        # Predict the mask using the trained model
        mask_pred = model.predict(np.expand_dims(original_image_array, axis=0))
        mask_pred = mask_pred.reshape((output_height, output_width, 3))

        # Convert the mask to binary format
        mask = np.zeros((output_height, output_width), dtype=np.uint8)
        mask[mask_pred[:, :, 0] > 0.5] = 1

        # Create the red mask for mixed pixels
        red_mask = create_red_mask(mask)

        # Create the white mask for pure pixels
        white_mask = create_white_mask(mask)

        # Combine the red and white masks
        combined_mask = red_mask + white_mask

        # Save the combined mask as an image
        combined_mask_image = Image.fromarray(combined_mask)
        combined_mask_image.save(os.path.join(masked_images_path, filename))
