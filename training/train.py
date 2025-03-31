
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2


# %%
# Paths
DATASET_PATH = "datasets/processed/"
LABELS_CSV = "datasets/processed_labels.csv"
MODEL_SAVE_PATH = "Model_Fix.h5"

# %%

# Load dataset labels
df = pd.read_csv(LABELS_CSV)
df["filename"] = df["filename"].apply(lambda x: os.path.join(DATASET_PATH, x))
print(df['label'].value_counts())  

# Perform train-validation split
train_files, val_files, train_labels, val_labels = train_test_split(
    df["filename"], df["label"], test_size=0.2, random_state=42
)

# Check dataset split
print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")


# %%
from sklearn.model_selection import train_test_split

# First split: Train (80%) + Temp (20%) → Temp will be further split into Val and Test
train_files, temp_files, train_labels, temp_labels = train_test_split(
    df["filename"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Second split: Split Temp (20%) into Validation (10%) and Test (10%)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Print dataset sizes
print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")
print(f"Test samples: {len(test_files)}")


# %%
test_images = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in test_files])
test_labels = np.array(test_labels)

# Expand dimensions to match model input shape (96, 96, 1)
test_images = np.expand_dims(test_images, axis=-1)

# Normalize pixel values
test_images = test_images.astype(np.float32) / 255.0

# %%


# Load images directly since they are already preprocessed
train_images = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in train_files])
val_images = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in val_files])
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# Expand dimensions to match model input shape (96, 96, 1)
train_images = np.expand_dims(train_images, axis=-1)
val_images = np.expand_dims(val_images, axis=-1)

# Ensure pixel values are in float32 and normalized to [0,1]
train_images = train_images.astype(np.float32) / 255.0
val_images = val_images.astype(np.float32) / 255.0


# %%
# Check the number of samples in train & validation sets
print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# Check class balance in both sets
train_df = pd.DataFrame({"filename": train_files, "label": train_labels})
val_df = pd.DataFrame({"filename": val_files, "label": val_labels})

print("\nTraining Set Distribution:")
print(train_df["label"].value_counts())

print("\nValidation Set Distribution:")
print(val_df["label"].value_counts())

# %%
print(f"Training data shape: {train_images.shape}")  # Should be (batch_size, 96, 96, 1)
print(f"Validation data shape: {val_images.shape}")  # Shou

# %%
IMAGE_SIZE = 96
BATCH_SIZE = 32


batch_size = 50
validation_split = 0.1

epochs = 20,10,20
lrates = .001, .0005, .00025

color_mode = 'grayscale'
if color_mode == 'grayscale':
  n_color_chans = 1
elif color_mode == 'rgb':
  n_color_chans = 3
else:
  raise ValueError("color_mode should be either 'rgb' or 'grayscale'")



def tinyml_mobilenet():
    input_shape = (96, 96, 1)  # Ensure grayscale input
    num_classes = 1  # Binary classification: Dog vs Not-Dog
    num_filters = 8  # Reduced filters for TinyML

    inputs = Input(shape=input_shape)
    

    # 1st Layer: Standard Convolution
    x = Conv2D(num_filters, (3,3), strides=2, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    # 2nd Layer: Depthwise Separable Convolution
    x = DepthwiseConv2D((3,3), strides=1, padding='same', 
                         depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2  # Increase filters
    x = Conv2D(num_filters, (1,1), strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3rd Layer: Depthwise Separable Convolution
    x = DepthwiseConv2D((3,3), strides=2, padding='same', 
                         depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2  # Increase filters
    x = Conv2D(num_filters, (1,1), strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 4th Layer: Depthwise Separable Convolution
    x = DepthwiseConv2D((3,3), strides=1, padding='same', 
                         depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 5th Layer: Depthwise Separable Convolution
    x = DepthwiseConv2D((3,3), strides=2, padding='same', 
                         depthwise_initializer='he_normal', depthwise_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_filters *= 2  # Increase filters
    x = Conv2D(num_filters, (1,1), strides=1, padding='same', 
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)  # Prevent overfitting

    # Output Layer
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification (Dog/Not-Dog)

    # Define Model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Compile the Model
model = tinyml_mobilenet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Display Model Summary
model.summary()


# %%



# Image size and batch size
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 30  # Will stop early if needed
# 🔥 Dataset Summary
label_counts = {
    0: 14256,  # Not-Dog
    1: 29976   # Dog
}
TOTAL_SAMPLES = sum(label_counts.values())
# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Compute class weights based on dataset size
class_counts = np.array([label_counts[0], label_counts[1]])  # [Not-Dog, Dog]
classes = np.array([0, 1])  # Labels: Not-Dog (0), Dog (1)

class_weights = compute_class_weight("balanced", classes=classes, y=np.repeat(classes, class_counts))
class_weights = {i: w for i, w in enumerate(class_weights)}

# Print computed class weights
print(f"Computed Class Weights: {class_weights}")

# Define Model
model = tinyml_mobilenet()  # Call your updated model function

# Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,  # Apply class balancing
    callbacks=[early_stopping]
)

MODEL_SAVE_PATH = "Final_Model_3.h5"
model.save(MODEL_SAVE_PATH)
print(f"✅ Model training complete! Saved as {MODEL_SAVE_PATH}")


# %%



# Plot training & validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy Over Epochs')
plt.legend()
plt.show()



# %%


# Paths
TEST_IMG_FOLDER = "datasets/test_img/"
# Load trained model
#model = tf.keras.models.load_model(MODEL_PATH)
#model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Image preprocessing (same as training)
IMG_SIZE = (96, 96)  # Ensure same size used in training


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        print(f"⚠️ Warning: Could not read {image_path}. Skipping...")
        return None

    img = cv2.resize(img, IMG_SIZE)  # Resize
    img = img.astype(np.float32) / 255.0  # Normalize

    img = np.expand_dims(img, axis=-1)  # Add grayscale channel dimension (96,96,1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1,96,96,1)
    return img

# Function to run predictions on all images in the test folder
def run_predictions():
    test_images = sorted(os.listdir(TEST_IMG_FOLDER))  # Sort files for consistent display
    predictions = []
    images_to_show = []

    for file in test_images:
        if file.endswith((".jpg", ".png")):
            image_path = os.path.join(TEST_IMG_FOLDER, file)
            img = preprocess_image(image_path)
            
            if img is not None:
                prediction = model.predict(img)[0][0]  # Get prediction score
                label = "Dog" if prediction > 0.5 else "Not Dog"
                confidence = round(prediction, 4)

                print(f"{file}: {label} (Confidence: {confidence})")

                # Append image and prediction for visualization
                images_to_show.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))  # Read in grayscale
                predictions.append(f"{label}\n{confidence:.4f}")

    # Display images in a matrix
    display_results(images_to_show, predictions)

# Function to display test images and predictions in a grid
def display_results(images, predictions, cols=4):
    rows = (len(images) + cols - 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle("Test Image Predictions", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap="gray")  # Display image in grayscale
            ax.set_title(predictions[i])  # Show prediction
            ax.axis("off")
        else:
            ax.axis("off")  # Hide extra subplots

    plt.show()

# Run the predictions
run_predictions()






