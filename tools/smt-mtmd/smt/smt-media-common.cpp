#include "smt-media-common.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>

#if !defined(_WIN32)
#    include <dlfcn.h>
#endif

namespace onnxruntime {
const OrtApi * g_ort = NULL;
}

namespace smt_media {

int get_ep_thread_num(const std::unordered_map<std::string, std::string> & ep_config,
                      const std::string &                                  key,
                      int                                                  default_value) {
    auto it = ep_config.find(key);
    if (it == ep_config.end() || it->second.empty()) {
        return default_value;
    }
    return std::stoi(it->second);
}

bool has_spacemit_ep_affinity(const std::unordered_map<std::string, std::string> & ep_config) {
    auto it = ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY");
    return it != ep_config.end() && !trim_ascii(it->second).empty();
}

std::unordered_map<std::string, std::string> make_provider_options(
    const std::unordered_map<std::string, std::string> & ep_config,
    int                                                  default_intra_thread_num,
    int                                                  default_inter_thread_num) {
    std::unordered_map<std::string, std::string> provider_options = ep_config;
    if (provider_options.find("SPACEMIT_EP_INTRA_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(default_intra_thread_num);
    }
    if (provider_options.find("SPACEMIT_EP_INTER_THREAD_NUM") == provider_options.end()) {
        provider_options["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(default_inter_thread_num);
    }
    return provider_options;
}

bool init_spacemit_execution_provider(Ort::SessionOptions &                                options,
                                      const std::unordered_map<std::string, std::string> & provider_options,
                                      std::string &                                        error_message) {
#if defined(_WIN32)
    (void) options;
    (void) provider_options;
    error_message = "SpacemiT EP dynamic loading is not supported on Windows";
    return false;
#else
    std::vector<const char *> keys;
    std::vector<const char *> values;
    keys.reserve(provider_options.size());
    values.reserve(provider_options.size());
    for (const auto & entry : provider_options) {
        keys.push_back(entry.first.c_str());
        values.push_back(entry.second.c_str());
    }

    void * handle = dlopen("libspacemit_ep.so", RTLD_NOW);
    if (!handle) {
        error_message = std::string("failed to load libspacemit_ep.so: ") + dlerror();
        return false;
    }

    auto * ep_init =
        reinterpret_cast<OrtStatus * (*) (OrtSessionOptions *, const char * const *, const char * const *, size_t)>(
            dlsym(handle, "OrtSessionOptionsSpaceMITEnvInit"));
    if (!ep_init) {
        error_message = std::string("failed to find OrtSessionOptionsSpaceMITEnvInit: ") + dlerror();
        return false;
    }

    if (OrtStatus * status = ep_init(options, keys.data(), values.data(), keys.size())) {
        error_message = Ort::GetApi().GetErrorMessage(status);
        Ort::GetApi().ReleaseStatus(status);
        return false;
    }

    return true;
#endif
}

std::vector<const char *> make_name_ptrs(const std::vector<std::string> & names) {
    std::vector<const char *> ptrs;
    ptrs.reserve(names.size());
    for (const auto & name : names) {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

std::vector<std::string> get_io_names(Ort::Session & session, bool inputs) {
    std::vector<std::string>         names;
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t                     count = inputs ? session.GetInputCount() : session.GetOutputCount();
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto allocated =
            inputs ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(allocated.get());
    }
    return names;
}

Ort::Value make_tensor_f32(const std::vector<int64_t> & shape, std::vector<float> & data) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), shape.data(), shape.size());
}

Ort::Value make_tensor_bool(const std::vector<int64_t> & shape, std::vector<uint8_t> & data) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return Ort::Value::CreateTensor(memory_info, data.data(), data.size() * sizeof(uint8_t), shape.data(), shape.size(),
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}

std::string read_file_to_string(const std::string & path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {};
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

size_t find_closing_brace(const std::string & text, size_t start_pos) {
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

std::string trim_ascii(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string extract_string_value(const std::string & text, const std::string & key) {
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

int64_t extract_int64_value(const std::string & text, const std::string & key, int64_t default_value) {
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

double extract_double_value(const std::string & text, const std::string & key, double default_value) {
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
    bool seen_dot = false;
    while (value_end < text.size()) {
        const unsigned char ch = static_cast<unsigned char>(text[value_end]);
        if (std::isdigit(ch)) {
            ++value_end;
            continue;
        }
        if (ch == '.' && !seen_dot) {
            seen_dot = true;
            ++value_end;
            continue;
        }
        break;
    }

    if (value_end == value_start) {
        return default_value;
    }

    try {
        return std::stod(text.substr(value_start, value_end - value_start));
    } catch (...) {
        return default_value;
    }
}

std::unordered_map<std::string, std::string> extract_string_map(const std::string & text, const std::string & key) {
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
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos >= content.size()) {
            break;
        }

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

        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos >= content.size() || content[pos] != ':') {
            break;
        }
        ++pos;

        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }

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

        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos < content.size() && content[pos] == ',') {
            ++pos;
        }
    }

    return values;
}

std::vector<std::string> extract_string_array(const std::string & text, const std::string & key) {
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

std::vector<double> extract_number_array(const std::string & text, const std::string & key) {
    std::vector<double> values;

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
        while (pos < content.size() &&
               (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
            ++pos;
        }
        if (pos >= content.size()) {
            break;
        }
        size_t end      = pos;
        bool   seen_dot = false;
        if (content[end] == '-' || content[end] == '+') {
            ++end;
        }
        while (end < content.size()) {
            const unsigned char ch = static_cast<unsigned char>(content[end]);
            if (std::isdigit(ch)) {
                ++end;
                continue;
            }
            if (ch == '.' && !seen_dot) {
                seen_dot = true;
                ++end;
                continue;
            }
            break;
        }
        if (end > pos) {
            try {
                values.push_back(std::stod(content.substr(pos, end - pos)));
            } catch (...) {
                values.clear();
                return values;
            }
        }
        pos = end + 1;
    }

    return values;
}

std::string normalize_path(const std::string & base_dir, const std::string & path) {
    const std::string trimmed = trim_ascii(path);
    if (trimmed.empty()) {
        return {};
    }
    if (trimmed.front() == '/') {
        return trimmed;
    }
    return base_dir + "/" + trimmed;
}

bool contains_legacy_spacemit_ep_config(const std::string & text) {
    return text.find("\"spacemit_ep_intra_thread_num\"") != std::string::npos ||
           text.find("\"spacemit_ep_inter_thread_num\"") != std::string::npos ||
           text.find("\"spacemit_ep_intra_thread_affinity\"") != std::string::npos;
}

void warn_legacy_spacemit_ep_config_if_needed(const std::string & text,
                                              const char *        backend_name,
                                              const char *        section_name) {
    if (!contains_legacy_spacemit_ep_config(text)) {
        return;
    }

    std::cerr << "[SMT][" << backend_name << "] warning: detected deprecated legacy Spacemit EP config keys";
    if (section_name != nullptr && section_name[0] != '\0') {
        std::cerr << " in " << section_name;
    }
    std::cerr << "; this style will be removed in a future release. "
              << "Please migrate to the `ep_config` format.\n";
}

void apply_legacy_spacemit_ep_config(const std::string &                            text,
                                     std::unordered_map<std::string, std::string> & ep_config,
                                     int32_t &                                      intra_thread_num,
                                     int32_t &                                      inter_thread_num) {
    intra_thread_num = (int32_t) extract_int64_value(text, "spacemit_ep_intra_thread_num", intra_thread_num);
    inter_thread_num = (int32_t) extract_int64_value(text, "spacemit_ep_inter_thread_num", inter_thread_num);

    const std::string affinity = extract_string_value(text, "spacemit_ep_intra_thread_affinity");
    if (!affinity.empty() && ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY") == ep_config.end()) {
        ep_config["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = affinity;
    }

    if (ep_config.find("SPACEMIT_EP_INTRA_THREAD_NUM") == ep_config.end()) {
        ep_config["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(intra_thread_num);
    }
    if (ep_config.find("SPACEMIT_EP_INTER_THREAD_NUM") == ep_config.end()) {
        ep_config["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(inter_thread_num);
    }
}

}  // namespace smt_media
