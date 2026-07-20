// SMT vision wrapper for llama.cpp SpacemiT integration.

#include "smt-vision-wrapper.h"

#include "smt-media-common.h"
#include "smt-profile.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {

using smt_media::extract_double_value;
using smt_media::extract_int64_value;
using smt_media::extract_number_array;
using smt_media::extract_string_array;
using smt_media::extract_string_map;
using smt_media::extract_string_value;
using smt_media::find_closing_brace;
using smt_media::get_ep_thread_num;
using smt_media::get_io_names;
using smt_media::has_spacemit_ep_affinity;
using smt_media::init_spacemit_execution_provider;
using smt_media::make_name_ptrs;
using smt_media::make_tensor_f32;
using smt_media::normalize_path;
using smt_media::read_file_to_string;
using smt_media::trim_ascii;

class smt_ort_vision_engine {
  public:
    smt_ort_vision_engine(std::string                                  model_path,
                          std::unordered_map<std::string, std::string> ep_config,
                          std::string                                  arch_name) :
        model_path_(std::move(model_path)),
        ep_config_(std::move(ep_config)),
        arch_name_(std::move(arch_name)),
        env_(ORT_LOGGING_LEVEL_WARNING, "smt-vision") {}

    Ort::Session & create_session() {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        provider_options_ = smt_media::make_provider_options(ep_config_, 1, 1);

        const int intra_thread_num = get_ep_thread_num(provider_options_, "SPACEMIT_EP_INTRA_THREAD_NUM", 1);
        const int inter_thread_num = get_ep_thread_num(provider_options_, "SPACEMIT_EP_INTER_THREAD_NUM", 1);
        if (!has_spacemit_ep_affinity(provider_options_)) {
            session_options_.SetIntraOpNumThreads(intra_thread_num);
            session_options_.SetInterOpNumThreads(inter_thread_num);
        } else {
            std::cerr << "[SMT][vision] detected SPACEMIT_EP_INTRA_THREAD_AFFINITY, skip ORT session thread pinning"
                      << " to avoid conflicting with EP-managed affinity\n";
        }

        std::string error_message;
        if (!init_spacemit_execution_provider(session_options_, provider_options_, error_message)) {
            throw std::runtime_error("[SMT][vision] failed to initialize Spacemit EP: " + error_message);
        }

        std::cerr << "[SMT][vision] Spacemit EP enabled (";
        for (const auto & pair : provider_options_) {
            std::cerr << ", " << pair.first << "=" << pair.second;
        }
        std::cerr << ")\n";

        session_          = Ort::Session(env_, model_path_.c_str(), session_options_);
        input_names_      = get_io_names(session_, true);
        output_names_     = get_io_names(session_, false);
        input_names_raw_  = make_name_ptrs(input_names_);
        output_names_raw_ = make_name_ptrs(output_names_);

        if (input_names_raw_.size() != 1 || output_names_raw_.empty()) {
            throw std::runtime_error("Unexpected SMT vision ONNX IO signature");
        }

        return session_;
    }

    void query_input_layout(std::vector<int64_t> & input_shape, size_t & expected_bytes) {
        auto type_info   = session_.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape      = tensor_info.GetShape();

        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("SMT vision expects float32 input tensor");
        }

        size_t input_size = 1;
        for (const int64_t dim : input_shape) {
            if (dim <= 0) {
                throw std::runtime_error("SMT vision input tensor must have static positive shape");
            }
            if (input_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
                throw std::runtime_error("SMT vision input tensor is too large");
            }
            input_size *= static_cast<size_t>(dim);
        }

        if (input_size > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::runtime_error("SMT vision input tensor is too large");
        }

        expected_bytes = input_size * sizeof(float);
    }

    Ort::Value & set_input_tensor_from_memory(const uint8_t * data, size_t len) {
        std::vector<int64_t> input_shape;
        size_t               expected_bytes = 0;
        query_input_layout(input_shape, expected_bytes);

        if (data == nullptr || len != expected_bytes) {
            throw std::runtime_error("SMT vision input size mismatch: expected " + std::to_string(expected_bytes) +
                                     ", actual " + std::to_string(len));
        }

        input_data_.resize(expected_bytes / sizeof(float));
        std::memcpy(input_data_.data(), data, expected_bytes);

        input_tensor_ = make_tensor_f32(input_shape, input_data_);
        return input_tensor_;
    }

    Ort::Value & set_input_tensor(const std::string & input_binary_path) {
        std::vector<int64_t> input_shape;
        size_t               expected_bytes = 0;
        query_input_layout(input_shape, expected_bytes);

        std::ifstream file(input_binary_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open SMT vision input binary: " + input_binary_path);
        }

        const std::streamoff actual_bytes = file.tellg();
        if (actual_bytes < 0 || static_cast<size_t>(actual_bytes) != expected_bytes) {
            throw std::runtime_error("SMT vision input binary size mismatch: expected " +
                                     std::to_string(expected_bytes) + ", actual " +
                                     std::to_string(actual_bytes < 0 ? 0 : static_cast<size_t>(actual_bytes)));
        }

        input_data_.resize(expected_bytes / sizeof(float));
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char *>(input_data_.data()), static_cast<std::streamsize>(expected_bytes));
        if (!file) {
            throw std::runtime_error("failed to read SMT vision input binary: " + input_binary_path);
        }

        input_tensor_ = make_tensor_f32(input_shape, input_data_);
        return input_tensor_;
    }

    Ort::Value make_zero_input_tensor() {
        auto                       type_info   = session_.GetInputTypeInfo(0);
        auto                       tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> input_shape = tensor_info.GetShape();

        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("SMT vision warmup expects float32 input tensor");
        }

        size_t input_size = 1;
        for (const int64_t dim : input_shape) {
            if (dim <= 0) {
                throw std::runtime_error("SMT vision warmup requires a static positive input shape");
            }
            if (input_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
                throw std::runtime_error("SMT vision warmup input tensor is too large");
            }
            input_size *= static_cast<size_t>(dim);
        }

        warmup_data_.assign(input_size, 0.0f);
        return make_tensor_f32(input_shape, warmup_data_);
    }

    static std::vector<int64_t> normalize_output_shape(std::vector<int64_t> shape) {
        if (shape.size() == 3 && shape[0] == 1) {
            shape = { shape[1], shape[2] };
        }
        return shape;
    }

    static std::vector<float> output_to_vector(const Ort::Value & output) {
        auto tensor_info = output.GetTensorTypeAndShapeInfo();
        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Expected float32 output from SMT vision model");
        }
        std::vector<int64_t> shape = normalize_output_shape(tensor_info.GetShape());
        if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
            throw std::runtime_error("Unexpected output shape from SMT vision encoder");
        }
        const size_t  total = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);
        const float * data  = output.GetTensorData<float>();
        return std::vector<float>(data, data + total);
    }

    std::vector<float> run_session(Ort::Value & input_tensor) {
        std::vector<Ort::Value> output_tensors =
            session_.Run(Ort::RunOptions{ nullptr }, input_names_raw_.data(), &input_tensor, input_names_raw_.size(),
                         output_names_raw_.data(), output_names_raw_.size());

        if (output_tensors.empty()) {
            throw std::runtime_error("SMT vision ONNX returned no outputs");
        }

        const size_t n_outputs = output_tensors.size();

        if (n_outputs == 1) {
            return output_to_vector(output_tensors[0]);
        }

        if (arch_name_ != "Qwen3VL") {
            std::cerr << "[SMT][vision] warning: ONNX model has " << n_outputs << " outputs but architecture '"
                      << arch_name_ << "' is not deepstack-aware; only the first output is used\n";
            return output_to_vector(output_tensors[0]);
        }

        int64_t                    n_tokens = 0;
        int64_t                    n_embd   = 0;
        std::vector<const float *> out_data(n_outputs, nullptr);

        for (size_t i = 0; i < n_outputs; ++i) {
            auto tensor_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                throw std::runtime_error("Expected float32 output from SMT vision model");
            }

            std::vector<int64_t> shape = normalize_output_shape(tensor_info.GetShape());
            if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
                throw std::runtime_error("Unexpected output shape from SMT vision encoder");
            }

            if (i == 0) {
                n_tokens = shape[0];
                n_embd   = shape[1];
            } else if (shape[0] != n_tokens || shape[1] != n_embd) {
                throw std::runtime_error("SMT vision outputs have mismatched shapes");
            }

            out_data[i] = output_tensors[i].GetTensorData<float>();
        }

        const size_t tokens   = static_cast<size_t>(n_tokens);
        const size_t embd     = static_cast<size_t>(n_embd);
        const size_t row_size = embd * n_outputs;

        std::vector<float> result(tokens * row_size);
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t o = 0; o < n_outputs; ++o) {
                std::memcpy(result.data() + t * row_size + o * embd, out_data[o] + t * embd, embd * sizeof(float));
            }
        }
        return result;
    }

  private:
    std::string                                  model_path_;
    std::unordered_map<std::string, std::string> ep_config_;
    std::string                                  arch_name_;
    std::unordered_map<std::string, std::string> provider_options_;
    Ort::Env                                     env_;
    Ort::SessionOptions                          session_options_;
    Ort::Session                                 session_{ nullptr };
    std::vector<std::string>                     input_names_;
    std::vector<std::string>                     output_names_;
    std::vector<const char *>                    input_names_raw_;
    std::vector<const char *>                    output_names_raw_;
    std::vector<float>                           input_data_;
    std::vector<float>                           warmup_data_;
    Ort::Value                                   input_tensor_{ nullptr };
};

struct smt_vision_config {
    std::vector<std::string>                     architectures;
    std::string                                  vision_model_path;
    std::unordered_map<std::string, std::string> ep_config;
    int64_t                                      hidden_size  = 0;
    int32_t                                      input_width  = 0;
    int32_t                                      input_height = 0;
    smt_vision_preprocess_config                 preprocess_config;
};

static std::string canonicalize_vision_architecture(std::string arch) {
    const std::string trimmed = trim_ascii(arch);
    if (trimmed == "Qwen3_5ForConditionalGeneration") {
        return "Qwen3VL";
    }
    return trimmed;
}

static void warn_legacy_spacemit_ep_config_if_needed(const std::string & text, const char * section_name) {
    smt_media::warn_legacy_spacemit_ep_config_if_needed(text, "vision", section_name);
}

static void apply_legacy_spacemit_ep_config(const std::string & text, smt_vision_config & config) {
    int32_t intra_thread_num = config.ep_config.count("SPACEMIT_EP_INTRA_THREAD_NUM") ?
                                   (int32_t) std::stoll(config.ep_config.at("SPACEMIT_EP_INTRA_THREAD_NUM")) :
                                   4;
    int32_t inter_thread_num = config.ep_config.count("SPACEMIT_EP_INTER_THREAD_NUM") ?
                                   (int32_t) std::stoll(config.ep_config.at("SPACEMIT_EP_INTER_THREAD_NUM")) :
                                   1;
    smt_media::apply_legacy_spacemit_ep_config(text, config.ep_config, intra_thread_num, inter_thread_num);
}

static bool load_smt_vision_config(const std::string & config_dir, smt_vision_config & config) {
    const std::string config_path = config_dir + "/config.json";
    const std::string content     = read_file_to_string(config_path);
    if (content.empty()) {
        std::cerr << "Error: Failed to read config file: " << config_path << "\n";
        return false;
    }

    const size_t vision_start = content.find("\"vision_model\":");
    if (vision_start == std::string::npos) {
        return false;
    }
    const size_t vision_block_start = content.find('{', vision_start);
    const size_t vision_block_end   = find_closing_brace(content, vision_block_start);
    if (vision_block_start == std::string::npos || vision_block_end == std::string::npos) {
        std::cerr << "Error: Invalid 'vision_model' block.\n";
        return false;
    }
    const std::string vision_block = content.substr(vision_block_start, vision_block_end - vision_block_start + 1);

    const size_t text_start = content.find("\"text_model\":");
    if (text_start == std::string::npos) {
        std::cerr << "Error: 'text_model' section not found.\n";
        return false;
    }
    const size_t text_block_start = content.find('{', text_start);
    const size_t text_block_end   = find_closing_brace(content, text_block_start);
    if (text_block_start == std::string::npos || text_block_end == std::string::npos) {
        std::cerr << "Error: Invalid 'text_model' block.\n";
        return false;
    }
    const std::string text_block = content.substr(text_block_start, text_block_end - text_block_start + 1);

    std::string  preprocess_block;
    const size_t preprocess_start = content.find("\"vision_preprocess\":");
    if (preprocess_start != std::string::npos) {
        const size_t preprocess_block_start = content.find('{', preprocess_start);
        const size_t preprocess_block_end   = find_closing_brace(content, preprocess_block_start);
        if (preprocess_block_start == std::string::npos || preprocess_block_end == std::string::npos) {
            std::cerr << "Error: Invalid 'vision_preprocess' block.\n";
            return false;
        }
        preprocess_block = content.substr(preprocess_block_start, preprocess_block_end - preprocess_block_start + 1);
    }

    warn_legacy_spacemit_ep_config_if_needed(vision_block, "vision_model");
    warn_legacy_spacemit_ep_config_if_needed(content, "top-level config");

    config.vision_model_path = normalize_path(config_dir, extract_string_value(vision_block, "model_path"));
    config.hidden_size       = extract_int64_value(text_block, "hidden_size", 0);
    config.architectures     = extract_string_array(content, "architectures");
    const int64_t input_size = extract_int64_value(vision_block, "input_size", 0);
    config.input_width       = (int32_t) extract_int64_value(vision_block, "input_width", input_size);
    config.input_height      = (int32_t) extract_int64_value(vision_block, "input_height", input_size);
    config.ep_config         = extract_string_map(vision_block, "ep_config");
    if (!preprocess_block.empty()) {
        config.preprocess_config.rescale_factor = (float) extract_double_value(preprocess_block, "rescale_factor", 1.0);
        const auto image_mean                   = extract_number_array(preprocess_block, "image_mean");
        const auto image_std                    = extract_number_array(preprocess_block, "image_std");
        if (image_mean.size() == 3 && image_std.size() == 3) {
            for (size_t i = 0; i < 3; ++i) {
                config.preprocess_config.image_mean[i] = (float) image_mean[i];
                config.preprocess_config.image_std[i]  = (float) image_std[i];
            }
            config.preprocess_config.has_normalize_config = true;
        }
    }
    apply_legacy_spacemit_ep_config(vision_block, config);

    // 从顶层配置读取 ep_config（如果 vision_model 块中没有设置）
    auto top_ep_config = extract_string_map(content, "ep_config");
    for (const auto & pair : top_ep_config) {
        if (config.ep_config.find(pair.first) == config.ep_config.end()) {
            config.ep_config[pair.first] = pair.second;
        }
    }
    apply_legacy_spacemit_ep_config(content, config);

    if (config.vision_model_path.empty()) {
        std::cerr << "Error: Missing required key 'vision_model.model_path'.\n";
        return false;
    }
    if (config.hidden_size <= 0) {
        std::cerr << "Error: Missing or invalid required key 'text_model.hidden_size'.\n";
        return false;
    }
    if (config.architectures.empty()) {
        std::cerr << "Error: Missing required key 'architectures'.\n";
        return false;
    }
    if ((config.input_width < 0 || config.input_height < 0) ||
        ((config.input_width == 0) != (config.input_height == 0))) {
        std::cerr << "Error: vision_model.input_width/input_height must both be set and positive.\n";
        return false;
    }

    return true;
}

}  // namespace

struct smt_vision_context::impl {
    smt_vision_config                      config;
    std::unique_ptr<smt_ort_vision_engine> vision_engine;
    std::string                            arch_name;
};

namespace {

static void warmup_vision_engine(smt_ort_vision_engine & vision_engine, const std::string & arch_name) {
    std::cerr << "[SMT][vision] warmup ONNX session";
    if (!arch_name.empty()) {
        std::cerr << " for " << arch_name;
    }
    std::cerr << "\n";

    Ort::Value input_tensor = vision_engine.make_zero_input_tensor();
    (void) vision_engine.run_session(input_tensor);
}

}  // namespace

smt_vision_context::~smt_vision_context() = default;

std::unique_ptr<smt_vision_context> smt_vision_context::create(const std::string & config_dir, bool warmup) {
    auto ctx    = std::unique_ptr<smt_vision_context>(new smt_vision_context());
    ctx->pimpl_ = std::make_unique<impl>();
    auto & d    = *ctx->pimpl_;

    // 1. Load config from directory
    if (!load_smt_vision_config(config_dir, d.config)) {
        throw std::runtime_error("Failed to load SMT config from: " + config_dir);
    }

    if (!d.config.architectures.empty()) {
        d.arch_name = canonicalize_vision_architecture(d.config.architectures[0]);
    }

    // 2. Initialize ORT API
    onnxruntime::g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // 3. Create vision engine and session
    d.vision_engine =
        std::make_unique<smt_ort_vision_engine>(d.config.vision_model_path, d.config.ep_config, d.arch_name);
    (void) d.vision_engine->create_session();
    if (warmup) {
        warmup_vision_engine(*d.vision_engine, d.arch_name);
    }

    return ctx;
}

std::vector<float> smt_vision_context::encode_image(const std::string & binary_path) {
    auto & d = *pimpl_;

    ggml_trace_log_begin("encode_image", "Vision", NULL);

    ggml_trace_log_begin("set_input_tensor", "Vision", NULL);
    Ort::Value & input_tensor = d.vision_engine->set_input_tensor(binary_path);
    ggml_trace_log_end("set_input_tensor", "Vision", NULL);

    ggml_trace_log_begin("vision_session_run", "Vision", NULL);
    std::vector<float> result = d.vision_engine->run_session(input_tensor);
    ggml_trace_log_end("vision_session_run", "Vision", NULL);

    ggml_trace_log_end("encode_image", "Vision", NULL);
    ggml_profile_flush_tls();
    return result;
}

std::vector<float> smt_vision_context::encode_image_mem(const uint8_t * data, size_t len) {
    auto & d = *pimpl_;

    ggml_trace_log_begin("encode_image", "Vision", NULL);

    ggml_trace_log_begin("set_input_tensor", "Vision", NULL);
    Ort::Value & input_tensor = d.vision_engine->set_input_tensor_from_memory(data, len);
    ggml_trace_log_end("set_input_tensor", "Vision", NULL);

    ggml_trace_log_begin("vision_session_run", "Vision", NULL);
    std::vector<float> result = d.vision_engine->run_session(input_tensor);
    ggml_trace_log_end("vision_session_run", "Vision", NULL);

    ggml_trace_log_end("encode_image", "Vision", NULL);
    ggml_profile_flush_tls();
    return result;
}

int64_t smt_vision_context::hidden_size() const {
    return pimpl_->config.hidden_size;
}

int32_t smt_vision_context::input_width() const {
    return pimpl_->config.input_width;
}

int32_t smt_vision_context::input_height() const {
    return pimpl_->config.input_height;
}

int64_t smt_vision_context::vocab_size() const {
    return 0;
}

const std::string & smt_vision_context::token_embedding_path() const {
    static const std::string empty;
    return empty;
}

const std::string & smt_vision_context::architecture() const {
    return pimpl_->arch_name;
}

const smt_vision_preprocess_config & smt_vision_context::preprocess_config() const {
    return pimpl_->config.preprocess_config;
}
