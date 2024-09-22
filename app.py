from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import MeanAbsoluteError
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = r"trained_model.h5"
model = load_model(model_path, custom_objects={'mae': MeanAbsoluteError()})

# Dictionary for gender classification
gender_dict = {1: 'Female', 0: 'Male'}

# Preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Predict age and gender
def predict_age_gender(model, image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    pred_gender = 1 if predictions[0][0][0] > 0.5 else 0
    pred_age = round(predictions[1][0][0])
    return gender_dict[pred_gender], pred_age

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        gender, age = predict_age_gender(model, frame)
        result = f'Gender: {gender}, Age: {age}'
        return render_template('index.html', result=result)
    return render_template('index.html', result="Failed to capture image.")

if __name__ == "__main__":
    app.run(debug=True)