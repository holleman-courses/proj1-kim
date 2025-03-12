import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load trained model
model = tf.keras.models.load_model("dog_detector.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Set image size
IMG_SIZE = 224

# Load and preprocess an image for testing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image file {image_path}. Skipping...")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  
    return img

# Define test dataset folder
test_folder = "dataset/test/"

# Store true labels and predictions
y_true = []
y_pred = []

# Iterate through the test folder
for subdir, _, files in os.walk(test_folder):
    class_name = os.path.basename(subdir).lower()  # Get folder name
    if class_name not in ["dog", "food"]:  
        continue

    label = 1 if class_name == "dog" else 0 

    for file in files:
        file_path = os.path.join(subdir, file)

        
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        
        image = preprocess_image(file_path)
        if image is None:
            continue  

        prediction = model.predict(image)[0][0]  # Get prediction score

  
        predicted_label = 1 if prediction > 0.5 else 0
        y_true.append(label)
        y_pred.append(predicted_label)

# Check if there are valid test images
if len(y_true) == 0:
    print("No valid images found in the test set.")
else:
    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    # Print results
    print(f"Total test images processed: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
