// EP Vision Wrapper - Spacemit EP ONNX vision engine adapter for llama-mtmd-cli
// Replaces CLIP/GGUF vision encoding with EP's ONNX vision engine

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

struct ep_vision_context {
    ep_vision_context(const ep_vision_context &) = delete;
    ep_vision_context & operator=(const ep_vision_context &) = delete;
    ~ep_vision_context();

    // Initialize from EP config directory (containing config.json)
    static std::unique_ptr<ep_vision_context> create(const std::string & config_dir);

    // Encode a preprocessed image binary file using ONNX vision engine
    // Returns image embedding vector (n_tokens * hidden_size floats)
    std::vector<float> encode_image(const std::string & binary_path);

    // Get the hidden size (embedding dimension)
    int64_t hidden_size() const;

    // Get the vocab size
    int64_t vocab_size() const;

    // Get the token embedding file path
    const std::string & token_embedding_path() const;

    // Get the model architecture name
    const std::string & architecture() const;

private:
    ep_vision_context() = default;
    struct impl;
    std::unique_ptr<impl> pimpl_;
};
