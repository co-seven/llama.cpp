// EP Vision Wrapper - Spacemit EP ONNX vision engine adapter
// Wraps SpineVisionModelEngine and SpineLLMArgParser for use in mtmd-cli-ep

#include "ep-vision-wrapper.h"

#include "spine_llm_argparser.h"
#include "spine_vision_engine.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace onnxruntime {
const OrtApi * g_ort = NULL;
}  // namespace onnxruntime

struct ep_vision_context::impl {
    onnxruntime::spacemit::SpineModelConfig                        config;
    std::unique_ptr<onnxruntime::spacemit::SpineVisionModelEngine> vision_engine;
    std::string                                                    arch_name;
};

ep_vision_context::~ep_vision_context() = default;

std::unique_ptr<ep_vision_context> ep_vision_context::create(const std::string & config_dir) {
    auto ctx    = std::unique_ptr<ep_vision_context>(new ep_vision_context());
    ctx->pimpl_ = std::make_unique<impl>();
    auto & d    = *ctx->pimpl_;

    // 1. Load config from directory
    if (!onnxruntime::spacemit::SpineLLMArgParser::LoadConfigFromDir(config_dir, d.config)) {
        throw std::runtime_error("Failed to load EP config from: " + config_dir);
    }

    if (!d.config.architectures.empty()) {
        d.arch_name = d.config.architectures[0];
    }

    // 2. Initialize ORT API
    onnxruntime::g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // 3. Create vision engine and session
    d.vision_engine = std::make_unique<onnxruntime::spacemit::SpineVisionModelEngine>(d.config.vision_model_path);
    d.vision_engine->CreateVisionModelSession();

    return ctx;
}

std::vector<float> ep_vision_context::encode_image(const std::string & binary_path) {
    auto & d = *pimpl_;
    std::string  path_copy    = binary_path;
    Ort::Value & input_tensor = d.vision_engine->SetInputTensor(path_copy);
    return d.vision_engine->RunSession(input_tensor);
}

int64_t ep_vision_context::hidden_size() const {
    return pimpl_->config.hidden_size;
}

int64_t ep_vision_context::vocab_size() const {
    return pimpl_->config.vocab_size;
}

const std::string & ep_vision_context::token_embedding_path() const {
    return pimpl_->config.token_embedding_path;
}

const std::string & ep_vision_context::architecture() const {
    return pimpl_->arch_name;
}
