# Real-Time Age and Gender Detection System

This project is designed primarily for educational and research purposes. It demonstrates the use of machine learning techniques to detect the age and gender of a person in real time. The system leverages a trained deep learning model to make predictions through a simple and efficient interface.

## Features
- **Real-time detection**: Detects age and gender from live camera feed.
- **Image-based detection**: Detects age and gender from a provided image file.
- **Custom model training**: Offers the flexibility to train the model on your own dataset.
- **Flask app integration**: Runs through a Flask web application for ease of deployment.

## How It Works
1. **Real-Time Detection with Flask App**  
   Run the `app.py` script to start the Flask application. This app uses your webcam feed to detect and display the gender and estimated age of the person in real time.

2. **Image-Based Detection**  
   Use the `detect_image.py` script to detect age and gender in a provided image file. Simply specify the image path in the script.

3. **Real-Time Detection without Flask**  
   If you prefer not to use the Flask app, the `realtime.py` script can run real-time detection directly from your terminal or command line.

4. **Custom Model Training**  
   The `model_training.py` script allows you to train the model on your own dataset. Make sure to specify the dataset path within the script before running the training process.

## Requirements
To run the project, you will need the following dependencies:

- Python 3.x
- Flask
- OpenCV
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Usage Instructions

### Running the Flask App
To start the Flask application for real-time age and gender detection:
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000/` in your browser to access the web interface.

### Detecting from an Image
To detect age and gender from an image, modify the path in `detect_image.py` and run:
```bash
python detect_image.py
```

### Real-Time Detection without Flask
For real-time detection without using the Flask app, simply run:
```bash
python realtime.py
```

### Model Training
To train the model on your custom dataset, provide the correct dataset path in `model_training.py` and run the script:
```bash
python model_training.py
```

## Important Notes
- Ensure that the path to the pre-trained model (`trained_model.h5`) is accurate in all scripts to avoid errors.
- The model provided is a generic one, and accuracy may vary. Custom model training is recommended for higher accuracy.
  
## Contributing
Contributions are welcome! If you'd like to improve the project or add features, feel free to fork the repository and submit a pull request.

## License
This project is for educational purposes and is open to contributions. Feel free to explore and improve it!
