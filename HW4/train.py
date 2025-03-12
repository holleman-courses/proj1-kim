import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size and batch size
IMG_SIZE = 224  # MobileNetV2 input size
BATCH_SIZE = 2  

# Paths 
TRAIN_DIR = "dataset/train/"
VALID_DIR = "dataset/valid/"

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,      # Normalize pixel values
    rotation_range=10,  
    zoom_range=0.2,      
    horizontal_flip=True 
)

valid_datagen = ImageDataGenerator(rescale=1./255)  

# Load dataset
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Binary classification (Dog vs No Dog)
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load pre-trained MobileNetV2 model (without top layers)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model layers (keep pretrained weights)
base_model.trainable = False

# Add custom layers for object detection
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)  
x = Dense(64, activation="relu")(x)
output_layer = Dense(1, activation="sigmoid")(x)  # Binary classification

# Create model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_generator, validation_data=valid_generator, epochs=5)

# Save model in .h5 format
model.save("dog_detector.h5")

print(model.summary())

print("Training complete. Model saved as dog_detector.h5")
