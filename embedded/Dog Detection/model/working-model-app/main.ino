#include <Arduino.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "dog_detect_model_data.h"
#include "image_provider.h"  // ✅ Include image capture function


// TensorFlow Lite objects
tflite::MicroErrorReporter error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// Tensor Arena
constexpr int kTensorArenaSize = 140 * 1024;  // Adjust if needed
uint8_t tensor_arena[kTensorArenaSize];


void setup() {
    Serial.begin(115200);
    while (!Serial);  // Wait for Serial Monitor


    Serial.println("🚀 Camera + Dog Detection Test");


    // Load the model
    model = tflite::GetModel(g_dog_detect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("🚨 Model version mismatch!");
        return;
    }
    Serial.println("✅ Model Loaded Successfully");


    // Register TensorFlow Lite operators
    static tflite::MicroMutableOpResolver<8> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddLogistic();


    // Create the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    interpreter = &static_interpreter;


    // Allocate memory for tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("🚨 Tensor allocation failed!");
        return;
    }
    Serial.println("✅ Tensor Allocation Succeeded!");


    // Get input & output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);


    Serial.println("✅ Setup Complete! Ready to capture images.");
}


void loop() {
    Serial.println("🔍 Capturing Image...");


    // Capture and crop the image using GetImage()
    if (kTfLiteOk != GetImage(&error_reporter, 96, 96, 1, input->data.int8)) {
        Serial.println("🚨 Image capture failed!");
        return;
    }


    Serial.println("✅ Image Captured! Running Inference...");


    // Debug: Print first pixel value to check preprocessing
    Serial.print("📷 First Pixel Value: ");
    Serial.println(input->data.int8[0]);  // Should be between -128 and 127


    // Debug: Print first 10 pixel values
    Serial.print("📷 First 10 Pixels: ");
    for (int i = 0; i < 10; i++) {
        Serial.print(input->data.int8[i]);
        Serial.print(" ");
    }
    Serial.println();


    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("🚨 Inference failed!");
        return;
    }


    // Read model output
    int8_t raw_output = output->data.int8[0];


    // Debug: Print raw model output
    Serial.print("🔢 Raw Model Output: ");
    Serial.println(raw_output);  // Should be between -128 and 127


    // Convert output to probability
    float probability = (raw_output - output->params.zero_point) * output->params.scale;


    // Debug: Print final probability
    Serial.print("🐶 Dog Detection Probability: ");
    Serial.println(probability * 100.0);


    delay(2000);  // Run every 2 seconds
}










