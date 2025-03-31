#include <Arduino.h>
#include <TensorFlowLite.h>
#include "image_provider.h"
#include "model_settings.h"
#include "dog_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


// TensorFlow Lite variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena
constexpr int kTensorArenaSize = 140 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(5000);
    Serial.println("🚀 Dog Detection Model Starting...");

    // Debug: Print Model Size
    Serial.print("📦 Model Size: ");
    Serial.println(g_dog_detect_model_data_len);

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Load the model
    model = tflite::GetModel(g_dog_detect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("🚨 Model version mismatch!");
        return;
    }
    Serial.println("✅ Model Loaded Successfully");

    // Register TensorFlow Lite operations
    static tflite::MicroMutableOpResolver<9> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddAveragePool2D();  // Needed for Sigmoid output
    micro_op_resolver.AddMean();
    micro_op_resolver.AddLogistic();

    // Initialize the TensorFlow Lite interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory for tensors
    Serial.println("🚀 Allocating Tensors...");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("🚨 Tensor allocation failed!");
        return;
    }
    Serial.println("✅ Tensor Allocation Succeeded!");

    input = interpreter->input(0);
}

void loop() {
    Serial.println("🔍 Capturing Image...");

    if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, input->data.int8)) {
        Serial.println("🚨 Image capture failed!");
        return;
    }

    Serial.println("✅ Image Captured! Running Inference...");

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("🚨 Inference failed!");
        return;
    }

    // Get output tensor
    output = interpreter->output(0);

    // Debug: Print raw model output
    Serial.print("🔢 Raw Model Output: ");
    Serial.println(output->data.int8[0]);

    // Convert output to probability
    float probability = (output->data.int8[0] - output->params.zero_point) * output->params.scale;

    Serial.print("🐶 Dog Detection Probability: ");
    Serial.println(probability * 100.0);

    delay(2000);
}


