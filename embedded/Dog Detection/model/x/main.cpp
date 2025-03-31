#include <Arduino.h>
#include <stdint.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "dog_detect_model_data.h"


// Global TensorFlow Lite variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena (adjust size if needed)
constexpr int kTensorArenaSize = 100 * 1024;  // 80 KB
static uint8_t tensor_arena[kTensorArenaSize];

// Function to print available RAM
extern "C" char *sbrk(int i);
void printMemoryUsage(const char* label) {
   char top;
   Serial.print("📢 Available RAM ");
   Serial.print(label);
   Serial.print(": ");
   Serial.println(&top - reinterpret_cast<char*>(sbrk(0)));
}

void setup() {
   Serial.begin(115200);

   Serial.println("🚀 Dog Detection Model Starting...");
   printMemoryUsage("AT STARTUP");
   // Print Model Input Details
    Serial.println("📌 Model Input Tensor Details:");
    Serial.print("Shape: ");
    for (int i = 0; i < input->dims->size; i++) {
        Serial.print(input->dims->data[i]);
        Serial.print(" ");
    }
    Serial.println();
    Serial.print("Data Type: ");
    Serial.println(input->type);  // Should print 0 (uint8)
    Serial.print("Quantization Scale: ");
    Serial.println(input->params.scale);
    Serial.print("Quantization Zero Point: ");
    Serial.println(input->params.zero_point);

    // Print Model Output Details
    Serial.println("📌 Model Output Tensor Details:");
    Serial.print("Shape: ");
    for (int i = 0; i < output->dims->size; i++) {
        Serial.print(output->dims->data[i]);
        Serial.print(" ");
    }
    Serial.println();
    Serial.print("Data Type: ");
    Serial.println(output->type);  // Should print 0 (uint8)
    Serial.print("Quantization Scale: ");
    Serial.println(output->params.scale);
    Serial.print("Quantization Zero Point: ");
    Serial.println(output->params.zero_point);





   // Set up logging
   static tflite::MicroErrorReporter micro_error_reporter;
   error_reporter = &micro_error_reporter;

   // Load the model
   model = tflite::GetModel(g_dog_detect_model_data);
   if (model->version() != TFLITE_SCHEMA_VERSION) {
       Serial.println("🚨 Model schema version mismatch!");
       return;
   }
   Serial.println("✅ Model version is correct.");

   // Register only required TensorFlow Lite operations
   static tflite::MicroMutableOpResolver<7> micro_op_resolver;
   micro_op_resolver.AddConv2D();
   micro_op_resolver.AddDepthwiseConv2D();
   micro_op_resolver.AddMaxPool2D();
   micro_op_resolver.AddReshape();
   micro_op_resolver.AddFullyConnected();
   micro_op_resolver.AddSoftmax();
   micro_op_resolver.AddAveragePool2D();  // ADDED for MobileNet compatibility
   

   // Initialize TensorFlow Lite interpreter
   static tflite::MicroInterpreter static_interpreter(
       model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
   interpreter = &static_interpreter;

   // Allocate memory for model tensors
   Serial.println("🚀 Allocating Tensors...");
   TfLiteStatus allocate_status = interpreter->AllocateTensors();
   if (allocate_status != kTfLiteOk) {
       Serial.println("🚨 AllocateTensors() failed!");
       return;
   }
   Serial.println("✅ AllocateTensors() succeeded!");

   // Get input tensor
   input = interpreter->input(0);
   output = interpreter->output(0);

   // Print input tensor details
   Serial.println("📌 Model Input Tensor Details:");
   Serial.print("Dimensions: ");
   for (int i = 0; i < input->dims->size; i++) {
       Serial.print(input->dims->data[i]);
       Serial.print(" ");
   }
   Serial.println();
   Serial.print("Tensor Type: ");
   Serial.println(input->type);
}

void run_inference() {
   Serial.println("🔍 Running Inference...");

   // Simulate grayscale image input (96x96)
   for (int i = 0; i < 96 * 96; i++) {
       input->data.uint8[i] = static_cast<uint8_t>(random(0, 256));  // Fake grayscale image
   }

   // Invoke the model
   if (interpreter->Invoke() != kTfLiteOk) {
       Serial.println("🚨 Inference failed!");
       return;
   }

   // Read output
   float dog_probability = output->data.uint8[0] / 255.0;  // Convert uint8 to float

   // Display results
   Serial.print("🐶 Dog Detection Probability: ");
   Serial.println(dog_probability);

   // Light up LED if a dog is detected with high confidence
   if (dog_probability > 0.75) {
       Serial.println("✅ DOG DETECTED!");
   } else {
       Serial.println("❌ No Dog Detected.");
   }
}

void loop() {
   run_inference();
   delay(2000);
}

