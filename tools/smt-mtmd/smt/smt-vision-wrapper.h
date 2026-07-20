// SMT vision wrapper for llama-mtmd-cli.
// Replaces CLIP/GGUF vision encoding with the SpacemiT SMT ONNX vision engine.

#pragma once

#include "smt-vision-preprocess.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct smt_vision_context {
    smt_vision_context(const smt_vision_context &)             = delete;
    smt_vision_context & operator=(const smt_vision_context &) = delete;
    ~smt_vision_context();

    // Initialize from SMT config directory (containing config.json)
    static std::unique_ptr<smt_vision_context> create(const std::string & config_dir, bool warmup = true);

    // Encode a preprocessed image binary file using ONNX vision engine
    // Returns image embedding vector (n_tokens * hidden_size floats)
    std::vector<float> encode_image(const std::string & binary_path);

    std::vector<float> encode_image_mem(const uint8_t * data, size_t len);

    // Get the hidden size (embedding dimension)
    int64_t hidden_size() const;

    // Get the vocab size
    int64_t vocab_size() const;

    // Get the token embedding file path
    const std::string & token_embedding_path() const;

    // Get the model architecture name
    const std::string & architecture() const;

    // Get the fixed ONNX input size expected by the SMT vision model.
    int32_t input_width() const;
    int32_t input_height() const;

    // Get image preprocess config for external SMT preprocessing.
    const smt_vision_preprocess_config & preprocess_config() const;

  private:
    smt_vision_context() = default;
    struct impl;
    std::unique_ptr<impl> pimpl_;
};
