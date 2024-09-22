import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import MeanAbsoluteError

# Define the path to the saved model
model_path = r"trained_model.h5"

# Load the saved model with custom objects
model = load_model(model_path, custom_objects={'mae': MeanAbsoluteError()})

# Define a function to preprocess a single image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    return img

# Function to predict age and gender
def predict_age_gender(model, image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    pred_gender = 1 if predictions[0][0][0] > 0.5 else 0  # Convert sigmoid output to binary prediction
    pred_age = round(predictions[1][0][0])
    return pred_gender, pred_age

# Map gender prediction to labels
gender_dict = {1: 'Female', 0: 'Male'}

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Predict age and gender
    pred_gender, pred_age = predict_age_gender(model, frame)

    # Display the predictions on the frame
    label = f"Gender: {gender_dict[pred_gender]}, Age: {pred_age}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Age and Gender Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
