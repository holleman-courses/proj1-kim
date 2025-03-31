/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include "image_provider.h"
#include "Arduino.h"
#include <TinyMLShield.h>

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
  int image_height, int channels, int8_t* image_data) {
static bool g_is_camera_initialized = false;


// Initialize camera if not already done
if (!g_is_camera_initialized) {
if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {  // QCIF = 176x144
TF_LITE_REPORT_ERROR(error_reporter, "🚨 Camera initialization failed!");
return kTfLiteError;
}
g_is_camera_initialized = true;
}


// Capture a full QCIF (176x144) grayscale image
byte camera_frame[176 * 144];  
Camera.readFrame(camera_frame);


int min_x = (176 - 96) / 2;  // Crop width center
int min_y = (144 - 96) / 2;  // Crop height center
int index = 0;


// Crop 96x96 from center
for (int y = min_y; y < min_y + 96; y++) {
for (int x = min_x; x < min_x + 96; x++) {
// Convert uint8_t (0-255) to int8_t (-128 to 127)
image_data[index++] = static_cast<int8_t>(camera_frame[(y * 176) + x] - 128);
}
}


return kTfLiteOk;
}



