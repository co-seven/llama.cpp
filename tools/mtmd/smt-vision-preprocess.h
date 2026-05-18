#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct smt_vision_preprocess_result {
    bool                 was_image       = false;
    int32_t              target_w        = 0;
    int32_t              target_h        = 0;
    bool                 normalize_to_01 = false;
    std::vector<uint8_t> tensor_bytes;
};

struct smt_vision_preprocess_config {
    float rescale_factor       = 1.0f;
    float image_mean[3]        = { 0.0f, 0.0f, 0.0f };
    float image_std[3]         = { 1.0f, 1.0f, 1.0f };
    bool  has_normalize_config = false;
};

// If input bytes decode as image (jpg/png/webp/...), preprocess them into
// float32 NCHW bytes for SMT vision ONNX input. Otherwise returns was_image=false.
smt_vision_preprocess_result smt_vision_preprocess_if_image(const std::vector<uint8_t> &         input,
                                                            const std::string &                  architecture,
                                                            int32_t                              input_width  = 0,
                                                            int32_t                              input_height = 0,
                                                            const smt_vision_preprocess_config * config = nullptr);
