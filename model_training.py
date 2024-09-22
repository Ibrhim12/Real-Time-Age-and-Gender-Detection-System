import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Define base directory
Base_Dir=(r"put here training dataset path")

# Initialize lists for image paths, gender labels, and age labels
image_paths = []
gender_labels = []
age_labels = []

# Load data from directory
for filename in tqdm(os.listdir(Base_Dir)):
    image_path = os.path.join(Base_Dir, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

# Create DataFrame
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels

# Example function to extract features
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale', target_size=(128, 128))
        img = np.array(img)
        features.append(img)
    return np.array(features)

# Extract features from images
X = extract_features(df['image'])

# Normalize features
X = X / 255.0

# Prepare labels
y_gender = np.array(df['gender'])

# Prepare age labels (no need to convert to categorical for regression)
y_age = np.array(df['age'])

# Define the input shape
input_shape = (128, 128, 1)  # Example input shape, adjust as needed

# Define the model architecture
inputs = Input(shape=input_shape)

# Convolutional Layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

# Fully Connected Layers
dense_1 = Dense(256, activation='relu')(flatten)
dropout_1 = Dropout(0.3)(dense_1)

dense_2 = Dense(256, activation='relu')(dropout_1)
dropout_2 = Dropout(0.3)(dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_2)  # Adjust output for binary classification
output_2 = Dense(1, activation='linear', name='age_out')(dropout_2)

# Create the model
model = Model(inputs=inputs, outputs=[output_1, output_2])

# Compile the model with metrics for each output
model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics={'gender_out': 'accuracy', 'age_out': 'mae'})

# Print the model summary
model.summary()

# Plot the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# Train the model
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=30, validation_split=0.2)

# Save the trained model
model.save('trained_model.h5')


# Access metrics from history.history
gender_accuracy = history.history.get('gender_out_accuracy')
val_gender_accuracy = history.history.get('val_gender_out_accuracy')
gender_loss = history.history.get('gender_out_loss')
val_gender_loss = history.history.get('val_gender_out_loss')

age_mae = history.history.get('age_out_mae')
val_age_mae = history.history.get('val_age_out_mae')

# Plot results for gender accuracy
if gender_accuracy and val_gender_accuracy:
    plt.plot(gender_accuracy, label='Training Accuracy')
    plt.plot(val_gender_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Gender Accuracy')
    plt.legend()
    plt.show()

# Plot results for gender loss
if gender_loss and val_gender_loss:
    plt.plot(gender_loss, label='Training Loss')
    plt.plot(val_gender_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Gender Loss')
    plt.legend()
    plt.show()

# Plot results for age MAE
if age_mae and val_age_mae:
    plt.plot(age_mae, label='Training MAE')
    plt.plot(val_age_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Age MAE')
    plt.legend()
    plt.show()

# Prediction for random image
image_index = 35
print("Original Gender:", y_gender[image_index], "Original Age:", y_age[image_index])

# Predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = 1 if pred[0][0][0] > 0.5 else 0  # Convert sigmoid output to binary prediction
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)

# Display the image
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray')
plt.show()
