#ifndef MODEL_SETTINGS_H_
#define MODEL_SETTINGS_H_

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;

// constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

//constexpr int kCategoryCount = 2;
//constexpr int kPersonIndex = 1;
//constexpr int kNotAPersonIndex = 0;
//extern const char* kCategoryLabels[kCategoryCount];

constexpr int kDogIndex = 1;
constexpr int kNotADogIndex = 0;

#endif 