// SMT vision wrapper for llama.cpp SpacemiT integration.

#include "smt-vision-wrapper.h"

#include "ggml-profile.h"
#include "spine_vision_engine.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(_WIN32)
#    include <io.h>
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <unistd.h>
#endif

namespace onnxruntime {
const OrtApi * g_ort = NULL;
}  // namespace onnxruntime

namespace {

template <typename...> using void_t = void;

template <typename T, typename = void> struct has_ep_config_vision_ctor : std::false_type {};

template <typename T>
struct has_ep_config_vision_ctor<
    T,
    void_t<decltype(T(std::declval<std::string &>(),
                      std::declval<const std::string &>(),
                      std::declval<const std::unordered_map<std::string, std::string> &>()))>> : std::true_type {};

template <typename T, typename = void> struct has_legacy_affinity_vision_ctor : std::false_type {};

template <typename T>
struct has_legacy_affinity_vision_ctor<
    T,
    void_t<decltype(T(std::declval<std::string &>(),
                      std::declval<int>(),
                      std::declval<int>(),
                      std::declval<const std::string &>()))>> : std::true_type {};

static std::unique_ptr<onnxruntime::spacemit::SpineVisionModelEngine> create_spine_vision_model_engine(
    std::string &                                        model_path,
    const std::string &                                  architecture,
    const std::unordered_map<std::string, std::string> & ep_config) {
    int         intra_thread_num = 4;
    int         inter_thread_num = 1;
    std::string intra_thread_affinity;

    if (ep_config.count("SPACEMIT_EP_INTRA_THREAD_NUM")) {
        intra_thread_num = std::stoi(ep_config.at("SPACEMIT_EP_INTRA_THREAD_NUM"));
    }
    if (ep_config.count("SPACEMIT_EP_INTER_THREAD_NUM")) {
        inter_thread_num = std::stoi(ep_config.at("SPACEMIT_EP_INTER_THREAD_NUM"));
    }
    if (ep_config.count("SPACEMIT_EP_INTRA_THREAD_AFFINITY")) {
        intra_thread_affinity = ep_config.at("SPACEMIT_EP_INTRA_THREAD_AFFINITY");
    }

    if constexpr (has_ep_config_vision_ctor<onnxruntime::spacemit::SpineVisionModelEngine>::value) {
        return std::make_unique<onnxruntime::spacemit::SpineVisionModelEngine>(model_path, architecture, ep_config);
    } else if constexpr (has_legacy_affinity_vision_ctor<onnxruntime::spacemit::SpineVisionModelEngine>::value) {
        return std::make_unique<onnxruntime::spacemit::SpineVisionModelEngine>(model_path, intra_thread_num,
                                                                               inter_thread_num,
                                                                               intra_thread_affinity);
    } else {
        if (!intra_thread_affinity.empty()) {
            std::cerr << "[SMT][vision] warning: SPACEMIT_EP_INTRA_THREAD_AFFINITY is ignored by this "
                         "SpineVisionModelEngine version\n";
        }
        return std::make_unique<onnxruntime::spacemit::SpineVisionModelEngine>(model_path, intra_thread_num,
                                                                               inter_thread_num);
    }
}

struct smt_vision_config {
    std::vector<std::string>                     architectures;
    std::string                                  vision_model_path;
    std::unordered_map<std::string, std::string> ep_config;
    int64_t                                      hidden_size  = 0;
    int32_t                                      input_width  = 0;
    int32_t                                      input_height = 0;
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
    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
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
    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
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

static std::unordered_map<std::string, std::string> extract_string_map(const std::string & text,
                                                                       const std::string & key) {
    std::unordered_map<std::string, std::string> values;

    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return values;
    }

    const size_t brace_start = text.find('{', key_pos + marker.size());
    const size_t brace_end   = find_closing_brace(text, brace_start);
    if (brace_start == std::string::npos || brace_end == std::string::npos || brace_end <= brace_start) {
        return values;
    }

    std::string content = text.substr(brace_start + 1, brace_end - brace_start - 1);
    size_t      pos     = 0;
    while (pos < content.size()) {
        // Skip whitespace
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos >= content.size()) {
            break;
        }

        // Find key
        if (content[pos] != '"') {
            break;
        }
        const size_t key_start = pos + 1;
        const size_t key_end   = content.find('"', key_start);
        if (key_end == std::string::npos) {
            break;
        }
        std::string map_key = content.substr(key_start, key_end - key_start);
        pos                 = key_end + 1;

        // Skip :
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos >= content.size() || content[pos] != ':') {
            break;
        }
        ++pos;

        // Skip whitespace
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }

        // Find value
        if (content[pos] != '"') {
            break;
        }
        const size_t value_start = pos + 1;
        const size_t value_end   = content.find('"', value_start);
        if (value_end == std::string::npos) {
            break;
        }
        std::string map_value = content.substr(value_start, value_end - value_start);
        pos                   = value_end + 1;

        values[map_key] = map_value;

        // Skip comma or end
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos < content.size() && content[pos] == ',') {
            ++pos;
        }
    }

    return values;
}

static std::vector<std::string> extract_string_array(const std::string & text, const std::string & key) {
    std::vector<std::string> values;

    const std::string marker  = "\"" + key + "\"";
    const size_t      key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return values;
    }

    const size_t bracket_start = text.find('[', key_pos + marker.size());
    const size_t bracket_end   = text.find(']', bracket_start == std::string::npos ? key_pos : bracket_start + 1);
    if (bracket_start == std::string::npos || bracket_end == std::string::npos || bracket_end <= bracket_start) {
        return values;
    }

    std::string content = text.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    size_t      pos     = 0;
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

static std::string canonicalize_vision_architecture(std::string arch) {
    const std::string trimmed = trim_ascii(arch);
    if (trimmed == "Qwen3_5ForConditionalGeneration") {
        return "Qwen3VL";
    }
    return trimmed;
}

static bool contains_legacy_spacemit_ep_config(const std::string & text) {
    return text.find("\"spacemit_ep_intra_thread_num\"") != std::string::npos ||
           text.find("\"spacemit_ep_inter_thread_num\"") != std::string::npos ||
           text.find("\"spacemit_ep_intra_thread_affinity\"") != std::string::npos;
}

static void warn_legacy_spacemit_ep_config_if_needed(const std::string & text, const char * section_name) {
    if (!contains_legacy_spacemit_ep_config(text)) {
        return;
    }

    std::cerr << "[SMT][vision] warning: detected deprecated legacy Spacemit EP config keys";
    if (section_name != nullptr && section_name[0] != '\0') {
        std::cerr << " in " << section_name;
    }
    std::cerr << "; this style will be removed in a future release. "
              << "Please migrate to the `ep_config` format.\n";
}

static void apply_legacy_spacemit_ep_config(const std::string & text, smt_vision_config & config) {
    const int32_t intra_thread_num =
        (int32_t) extract_int64_value(text, "spacemit_ep_intra_thread_num",
                                      config.ep_config.count("SPACEMIT_EP_INTRA_THREAD_NUM") ?
                                          std::stoll(config.ep_config.at("SPACEMIT_EP_INTRA_THREAD_NUM")) :
                                          4);
    const int32_t inter_thread_num =
        (int32_t) extract_int64_value(text, "spacemit_ep_inter_thread_num",
                                      config.ep_config.count("SPACEMIT_EP_INTER_THREAD_NUM") ?
                                          std::stoll(config.ep_config.at("SPACEMIT_EP_INTER_THREAD_NUM")) :
                                          1);

    if (config.ep_config.find("SPACEMIT_EP_INTRA_THREAD_NUM") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(intra_thread_num);
    }
    if (config.ep_config.find("SPACEMIT_EP_INTER_THREAD_NUM") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(inter_thread_num);
    }

    const std::string affinity = extract_string_value(text, "spacemit_ep_intra_thread_affinity");
    if (!affinity.empty() && config.ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY") == config.ep_config.end()) {
        config.ep_config["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = affinity;
    }
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

    warn_legacy_spacemit_ep_config_if_needed(vision_block, "vision_model");
    warn_legacy_spacemit_ep_config_if_needed(content, "top-level config");

    config.vision_model_path = normalize_path(config_dir, extract_string_value(vision_block, "model_path"));
    config.hidden_size       = extract_int64_value(text_block, "hidden_size", 0);
    config.architectures     = extract_string_array(content, "architectures");
    const int64_t input_size = extract_int64_value(vision_block, "input_size", 0);
    config.input_width       = (int32_t) extract_int64_value(vision_block, "input_width", input_size);
    config.input_height      = (int32_t) extract_int64_value(vision_block, "input_height", input_size);
    config.ep_config         = extract_string_map(vision_block, "ep_config");
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
    smt_vision_config                                              config;
    std::unique_ptr<onnxruntime::spacemit::SpineVisionModelEngine> vision_engine;
    std::string                                                    arch_name;
};

namespace {

static size_t get_static_input_tensor_elements(Ort::Session & session) {
    auto type_info   = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("SMT vision warmup expects float32 input tensor");
    }

    const std::vector<int64_t> input_shape = tensor_info.GetShape();
    size_t                     input_size  = 1;
    for (const int64_t dim : input_shape) {
        if (dim <= 0) {
            throw std::runtime_error("SMT vision warmup requires a static positive input shape");
        }
        if (input_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
            throw std::runtime_error("SMT vision warmup input tensor is too large");
        }
        input_size *= static_cast<size_t>(dim);
    }

    return input_size;
}

static std::string write_zero_tensor_file(size_t n_floats) {
    const std::vector<float> zeros(n_floats, 0.0f);

#if defined(_WIN32)
    char temp_path[MAX_PATH] = { 0 };
    char temp_file[MAX_PATH] = { 0 };

    if (GetTempPathA(MAX_PATH, temp_path) == 0) {
        throw std::runtime_error("failed to get temp path for SMT vision warmup");
    }
    if (GetTempFileNameA(temp_path, "lsw", 0, temp_file) == 0) {
        throw std::runtime_error("failed to create temp file for SMT vision warmup");
    }

    std::ofstream file(temp_file, std::ios::binary);
    if (!file.is_open()) {
        std::remove(temp_file);
        throw std::runtime_error("failed to open temp file for SMT vision warmup");
    }
    file.write(reinterpret_cast<const char *>(zeros.data()),
               static_cast<std::streamsize>(zeros.size() * sizeof(float)));
    if (!file) {
        file.close();
        std::remove(temp_file);
        throw std::runtime_error("failed to write temp file for SMT vision warmup");
    }
    file.close();
    return std::string(temp_file);
#else
    char      tmpl[] = "/tmp/llama-smt-vision-warmup-XXXXXX";
    const int fd     = mkstemp(tmpl);
    if (fd < 0) {
        throw std::runtime_error("failed to create temp file for SMT vision warmup");
    }

    const char * ptr        = reinterpret_cast<const char *>(zeros.data());
    size_t       bytes_left = zeros.size() * sizeof(float);
    while (bytes_left > 0) {
        const ssize_t written = write(fd, ptr, bytes_left);
        if (written <= 0) {
            close(fd);
            std::remove(tmpl);
            throw std::runtime_error("failed to write temp file for SMT vision warmup");
        }
        ptr += written;
        bytes_left -= static_cast<size_t>(written);
    }

    close(fd);
    return std::string(tmpl);
#endif
}

static void warmup_vision_engine(onnxruntime::spacemit::SpineVisionModelEngine & vision_engine,
                                 Ort::Session &                                  session,
                                 const std::string &                             arch_name) {
    const size_t      input_elements = get_static_input_tensor_elements(session);
    const std::string temp_path      = write_zero_tensor_file(input_elements);

    try {
        std::cerr << "[SMT][vision] warmup ONNX session";
        if (!arch_name.empty()) {
            std::cerr << " for " << arch_name;
        }
        std::cerr << "\n";

        std::string  path_copy    = temp_path;
        Ort::Value & input_tensor = vision_engine.SetInputTensor(path_copy);
        (void) vision_engine.RunSession(input_tensor);
    } catch (...) {
        std::remove(temp_path.c_str());
        throw;
    }

    std::remove(temp_path.c_str());
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
    d.vision_engine = create_spine_vision_model_engine(d.config.vision_model_path, d.arch_name, d.config.ep_config);
    Ort::Session & vision_session = d.vision_engine->CreateVisionModelSession();
    if (warmup) {
        warmup_vision_engine(*d.vision_engine, vision_session, d.arch_name);
    }

    std::cerr << "[SMT][vision] Spacemit EP enabled (";
    for (const auto & pair : d.config.ep_config) {
        std::cerr << ", " << pair.first << "=" << pair.second;
    }
    std::cerr << ")\n";

    return ctx;
}

std::vector<float> smt_vision_context::encode_image(const std::string & binary_path) {
    auto & d = *pimpl_;

    ggml_trace_log_begin("encode_image", "Vision", NULL);

    std::string path_copy = binary_path;

    ggml_trace_log_begin("set_input_tensor", "Vision", NULL);
    Ort::Value & input_tensor = d.vision_engine->SetInputTensor(path_copy);
    ggml_trace_log_end("set_input_tensor", "Vision", NULL);

    ggml_trace_log_begin("vision_session_run", "Vision", NULL);
    std::vector<float> result = d.vision_engine->RunSession(input_tensor);
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
