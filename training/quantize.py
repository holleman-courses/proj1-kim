
# %%
import tensorflow as tf
import numpy as np

# Load the trained Keras model
#MODEL_SAVE_PATH = "tinyml_mobilenet_dog_detector.h5"
model = tf.keras.models.load_model(model)

# Create a representative dataset function for quantization
def representative_dataset():
    for _ in range(100):
        sample = np.random.rand(1, 96, 96, 1).astype(np.float32)  # Fake grayscale images
        yield [sample]

# Convert the model to a fully quantized TensorFlow Lite model (int8)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Force INT8
converter.inference_input_type = tf.int8  # Force int8 input
converter.inference_output_type = tf.int8  # Force int8 output

tflite_model = converter.convert()

# Save the quantized TFLite model
INT8_TFLITE_MODEL_PATH = "80kparam_model.tflite"
with open(INT8_TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ INT8 Quantized Model Saved as {INT8_TFLITE_MODEL_PATH}")


# %%
import tensorflow as tf

# Load the new INT8 model
interpreter = tf.lite.Interpreter(model_path="80kparam_model.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ INT8 Model Loaded Successfully!")
print("📌 Input Type:", input_details[0]['dtype'])  # Should print int8
print("📌 Output Type:", output_details[0]['dtype'])  # Should print int8


# %%
import tensorflow.lite as tflite

MODEL_PATH = "80kparam_model.tflite"
with open(MODEL_PATH, "rb") as f:
    model_data = f.read()

c_array = ", ".join(str(b) for b in model_data)

header_content = f"""#ifndef DOG_DETECT_MODEL_DATA_H_
#define DOG_DETECT_MODEL_DATA_H_

extern const unsigned char g_dog_detect_model_data[];
extern const int g_dog_detect_model_data_len;

#endif  // DOG_DETECT_MODEL_DATA_H_
"""

cpp_content = f"""#include "dog_detect_model_data.h"

const unsigned char g_dog_detect_model_data[] = {{
    {c_array}
}};

const int g_dog_detect_model_data_len = {len(model_data)};
"""

with open("dog_detect_model_data_2.h", "w") as f:
    f.write(header_content)

with open("dog_detect_model_data.cpp_2", "w") as f:
    f.write(cpp_content)

print("✅ Model converted successfully!")