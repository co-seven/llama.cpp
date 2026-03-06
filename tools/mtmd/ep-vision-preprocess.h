#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ep_vision_preprocess_result {
    bool was_image = false;
    int32_t target_w = 0;
    int32_t target_h = 0;
    bool normalize_to_01 = false;
    std::vector<uint8_t> tensor_bytes;
};

// If input bytes decode as image (jpg/png/webp/...), preprocess them into
// float32 NCHW bytes for EP vision ONNX input. Otherwise returns was_image=false.
ep_vision_preprocess_result ep_vision_preprocess_if_image(
        const std::vector<uint8_t> & input,
        const std::string & architecture);
