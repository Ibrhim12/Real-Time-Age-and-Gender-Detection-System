import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tensorflow.keras.losses import MeanAbsoluteError

# Define the path to the saved model
model_path = r"trained_model.h5"

# Load the saved model with custom objects
model = load_model(model_path, custom_objects={'mae': MeanAbsoluteError()})

# Define a function to preprocess a single image
def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    return img

# Function to predict age and gender
def predict_age_gender(model, image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    pred_gender = 1 if predictions[0][0][0] > 0.5 else 0  # Convert sigmoid output to binary prediction
    pred_age = round(predictions[1][0][0])
    return pred_gender, pred_age

# User-given image path
user_image_path = r"C:\Users\raoib\Downloads\profile-pic (2).png"  # Replace with the actual path to the user image

# Predict age and gender for the user-given image
pred_gender, pred_age = predict_age_gender(model, user_image_path)

# Map gender prediction to labels
gender_dict = {1: 'Female', 0: 'Male'}

# Print the predictions
print("Predicted Gender:", gender_dict[pred_gender], "Predicted Age:", pred_age)

# Display the image
img = load_img(user_image_path, color_mode='grayscale', target_size=(128, 128))
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.title(f"Predicted Gender: {gender_dict[pred_gender]}, Predicted Age: {pred_age}")
plt.show()
