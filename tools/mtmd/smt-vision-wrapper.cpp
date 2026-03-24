// SMT vision wrapper for llama.cpp SpacemiT integration.

#include "smt-vision-wrapper.h"

#include "spine_vision_engine.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace onnxruntime {
const OrtApi * g_ort = NULL;
}  // namespace onnxruntime

namespace {

struct smt_vision_config {
    std::vector<std::string> architectures;
    std::string              vision_model_path;
    int64_t                  hidden_size = 0;
};

static std::string read_file_to_string(const std::string & path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {};
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static size_t find_closing_brace(const std::string & text, size_t start_pos) {
    int depth = 0;
    for (size_t index = start_pos; index < text.size(); ++index) {
        if (text[index] == '{') {
            ++depth;
        } else if (text[index] == '}') {
            --depth;
            if (depth == 0) {
                return index;
            }
        }
    }
    return std::string::npos;
}

static std::string trim_ascii(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

static std::string extract_string_value(const std::string & text, const std::string & key) {
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return {};
    }

    const size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return {};
    }

    const size_t first_quote = text.find('"', colon_pos + 1);
    if (first_quote == std::string::npos) {
        return {};
    }

    const size_t second_quote = text.find('"', first_quote + 1);
    if (second_quote == std::string::npos) {
        return {};
    }

    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

static int64_t extract_int64_value(const std::string & text, const std::string & key, int64_t default_value) {
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return default_value;
    }

    const size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return default_value;
    }

    size_t value_start = colon_pos + 1;
    while (value_start < text.size() && std::isspace(static_cast<unsigned char>(text[value_start]))) {
        ++value_start;
    }

    size_t value_end = value_start;
    if (value_end < text.size() && (text[value_end] == '-' || text[value_end] == '+')) {
        ++value_end;
    }
    while (value_end < text.size() && std::isdigit(static_cast<unsigned char>(text[value_end]))) {
        ++value_end;
    }

    if (value_end == value_start) {
        return default_value;
    }

    try {
        return std::stoll(text.substr(value_start, value_end - value_start));
    } catch (...) {
        return default_value;
    }
}

static std::vector<std::string> extract_string_array(const std::string & text, const std::string & key) {
    std::vector<std::string> values;

    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return values;
    }

    const size_t bracket_start = text.find('[', key_pos + marker.size());
    const size_t bracket_end = text.find(']', bracket_start == std::string::npos ? key_pos : bracket_start + 1);
    if (bracket_start == std::string::npos || bracket_end == std::string::npos || bracket_end <= bracket_start) {
        return values;
    }

    std::string content = text.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    size_t pos = 0;
    while (pos < content.size()) {
        const size_t first_quote = content.find('"', pos);
        if (first_quote == std::string::npos) {
            break;
        }
        const size_t second_quote = content.find('"', first_quote + 1);
        if (second_quote == std::string::npos) {
            break;
        }
        values.push_back(content.substr(first_quote + 1, second_quote - first_quote - 1));
        pos = second_quote + 1;
    }

    return values;
}

static std::string normalize_path(const std::string & base_dir, const std::string & path) {
    const std::string trimmed = trim_ascii(path);
    if (trimmed.empty()) {
        return {};
    }
    if (!trimmed.empty() && trimmed.front() == '/') {
        return trimmed;
    }
    return base_dir + "/" + trimmed;
}

static bool load_smt_vision_config(const std::string & config_dir, smt_vision_config & config) {
    const std::string config_path = config_dir + "/config.json";
    const std::string content = read_file_to_string(config_path);
    if (content.empty()) {
        std::cerr << "Error: Failed to read config file: " << config_path << "\n";
        return false;
    }

    const size_t vision_start = content.find("\"vision_model\":");
    if (vision_start == std::string::npos) {
        return false;
    }
    const size_t vision_block_start = content.find('{', vision_start);
    const size_t vision_block_end = find_closing_brace(content, vision_block_start);
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
    const size_t text_block_end = find_closing_brace(content, text_block_start);
    if (text_block_start == std::string::npos || text_block_end == std::string::npos) {
        std::cerr << "Error: Invalid 'text_model' block.\n";
        return false;
    }
    const std::string text_block = content.substr(text_block_start, text_block_end - text_block_start + 1);

    config.vision_model_path = normalize_path(config_dir, extract_string_value(vision_block, "model_path"));
    config.hidden_size = extract_int64_value(text_block, "hidden_size", 0);
    config.architectures = extract_string_array(content, "architectures");

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

    return true;
}

} // namespace

struct smt_vision_context::impl {
    smt_vision_config                                             config;
    std::unique_ptr<onnxruntime::spacemit::SpineVisionModelEngine> vision_engine;
    std::string                                                    arch_name;
};

smt_vision_context::~smt_vision_context() = default;

std::unique_ptr<smt_vision_context> smt_vision_context::create(const std::string & config_dir) {
    auto ctx    = std::unique_ptr<smt_vision_context>(new smt_vision_context());
    ctx->pimpl_ = std::make_unique<impl>();
    auto & d    = *ctx->pimpl_;

    // 1. Load config from directory
    if (!load_smt_vision_config(config_dir, d.config)) {
        throw std::runtime_error("Failed to load SMT config from: " + config_dir);
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

std::vector<float> smt_vision_context::encode_image(const std::string & binary_path) {
    auto & d = *pimpl_;
    std::string  path_copy    = binary_path;
    Ort::Value & input_tensor = d.vision_engine->SetInputTensor(path_copy);
    return d.vision_engine->RunSession(input_tensor);
}

int64_t smt_vision_context::hidden_size() const {
    return pimpl_->config.hidden_size;
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
