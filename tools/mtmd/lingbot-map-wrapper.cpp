#include "lingbot-map-wrapper.h"

#include "gguf.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <sys/stat.h>

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

struct ggml_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ggml_backend_deleter {
    void operator()(ggml_backend * backend) const {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct ggml_backend_buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct ggml_backend_sched_deleter {
    void operator()(ggml_backend_sched * sched) const {
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
        }
    }
};

using gguf_context_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_context_ptr = std::unique_ptr<ggml_context, ggml_deleter>;
using ggml_backend_ptr = std::unique_ptr<ggml_backend, ggml_backend_deleter>;
using ggml_backend_buffer_ptr = std::unique_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter>;
using ggml_backend_sched_ptr = std::unique_ptr<ggml_backend_sched, ggml_backend_sched_deleter>;

struct lingbot_map_loaded_gguf {
    gguf_context_ptr gguf;
    ggml_context_ptr ggml;
};

struct lingbot_map_runtime_weights {
    gguf_context_ptr gguf;
    ggml_context_ptr ggml;
    ggml_backend_buffer_ptr buffer;
};

struct lingbot_map_runtime_graph {
    ggml_tensor * input_tokens = nullptr;
    ggml_tensor * camera_head_input = nullptr;
    ggml_tensor * final_pose = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> selected_outputs;
    std::vector<ggml_tensor *> iteration_poses;
};




static int64_t lingbot_elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

static bool lingbot_graph_supported_by_backend(ggml_backend_t backend,
                                               ggml_backend_buffer_type_t buft,
                                               ggml_cgraph * graph,
                                               bool log_summary) {
    if (backend == nullptr || buft == nullptr || graph == nullptr) {
        return false;
    }
    bool ok = true;
    int unsupported_nodes = 0;
    if (!ggml_backend_supports_buft(backend, buft)) {
        ok = false;
    }
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            continue;
        }
        if (!ggml_backend_supports_op(backend, node)) {
            ++unsupported_nodes;
            ok = false;
        }
    }
    if (!ok && log_summary) {
        std::cerr << "[LingBot-MAP] GGML graph support check failed on backend=" << ggml_backend_name(backend)
                  << ", buffer_type=" << ggml_backend_buft_name(buft)
                  << ", unsupported_nodes=" << unsupported_nodes << "/" << n_nodes << "\n";
    }
    return ok;
}

static std::string read_file_to_string(const std::string & path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static bool file_exists(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static size_t find_closing_brace(const std::string & text, size_t start_pos) {
    if (start_pos == std::string::npos) {
        return std::string::npos;
    }
    int depth = 0;
    for (size_t i = start_pos; i < text.size(); ++i) {
        if (text[i] == '{') {
            ++depth;
        } else if (text[i] == '}') {
            --depth;
            if (depth == 0) {
                return i;
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

static std::string normalize_path(const std::string & base_dir, const std::string & path) {
    const std::string trimmed = trim_ascii(path);
    if (trimmed.empty()) {
        return {};
    }
    if (trimmed.front() == '/') {
        return trimmed;
    }
    return base_dir + "/" + trimmed;
}

static std::string extract_object_block(const std::string & text, const std::string & key) {
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return {};
    }
    const size_t brace_start = text.find('{', key_pos + marker.size());
    const size_t brace_end = find_closing_brace(text, brace_start);
    if (brace_start == std::string::npos || brace_end == std::string::npos || brace_end <= brace_start) {
        return {};
    }
    return text.substr(brace_start, brace_end - brace_start + 1);
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

static int32_t extract_int32_value(const std::string & text, const std::string & key, int32_t default_value) {
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return default_value;
    }
    const size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return default_value;
    }
    size_t pos = colon_pos + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    size_t end = pos;
    if (end < text.size() && (text[end] == '-' || text[end] == '+')) {
        ++end;
    }
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
        ++end;
    }
    if (end == pos) {
        return default_value;
    }
    try {
        return std::stoi(text.substr(pos, end - pos));
    } catch (...) {
        return default_value;
    }
}

static void extract_float_array3(const std::string & text, const std::string & key, float values[3]) {
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return;
    }
    const size_t bracket_start = text.find('[', key_pos + marker.size());
    const size_t bracket_end = text.find(']', bracket_start == std::string::npos ? key_pos : bracket_start + 1);
    if (bracket_start == std::string::npos || bracket_end == std::string::npos || bracket_end <= bracket_start) {
        return;
    }

    size_t pos = bracket_start + 1;
    for (int i = 0; i < 3 && pos < bracket_end; ++i) {
        while (pos < bracket_end && (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == ',')) {
            ++pos;
        }
        size_t end = pos;
        while (end < bracket_end && text[end] != ',') {
            ++end;
        }
        try {
            values[i] = std::stof(text.substr(pos, end - pos));
        } catch (...) {
            return;
        }
        pos = end + 1;
    }
}

static bool extract_bool_value(const std::string & text, const std::string & key, bool default_value) {
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return default_value;
    }
    const size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return default_value;
    }
    size_t pos = colon_pos + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (text.compare(pos, 4, "true") == 0) {
        return true;
    }
    if (text.compare(pos, 5, "false") == 0) {
        return false;
    }
    return default_value;
}


static std::unordered_map<std::string, std::string> extract_string_map(const std::string & text, const std::string & key) {
    std::unordered_map<std::string, std::string> values;
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return values;
    }
    const size_t brace_start = text.find('{', key_pos + marker.size());
    const size_t brace_end = find_closing_brace(text, brace_start);
    if (brace_start == std::string::npos || brace_end == std::string::npos || brace_end <= brace_start) {
        return values;
    }
    const std::string content = text.substr(brace_start + 1, brace_end - brace_start - 1);
    size_t pos = 0;
    while (pos < content.size()) {
        while (pos < content.size() && (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
            ++pos;
        }
        if (pos >= content.size() || content[pos] != '"') {
            break;
        }
        const size_t key_start = pos + 1;
        const size_t key_end = content.find('"', key_start);
        if (key_end == std::string::npos) {
            break;
        }
        const size_t colon = content.find(':', key_end + 1);
        const size_t value_quote = content.find('"', colon == std::string::npos ? key_end + 1 : colon + 1);
        if (colon == std::string::npos || value_quote == std::string::npos) {
            break;
        }
        const size_t value_end = content.find('"', value_quote + 1);
        if (value_end == std::string::npos) {
            break;
        }
        values[content.substr(key_start, key_end - key_start)] = content.substr(value_quote + 1, value_end - value_quote - 1);
        pos = value_end + 1;
    }
    return values;
}

static void merge_missing_ep_config(std::unordered_map<std::string, std::string> & dst,
                                    const std::unordered_map<std::string, std::string> & src) {
    for (const auto & kv : src) {
        if (dst.find(kv.first) == dst.end()) {
            dst[kv.first] = kv.second;
        }
    }
}

static void apply_legacy_lingbot_ep_config(const std::string & text,
                                           std::unordered_map<std::string, std::string> & ep_config) {
    if (ep_config.find("SPACEMIT_EP_INTRA_THREAD_NUM") == ep_config.end()) {
        ep_config["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(extract_int32_value(text, "spacemit_ep_intra_thread_num", 4));
    }
    if (ep_config.find("SPACEMIT_EP_INTER_THREAD_NUM") == ep_config.end()) {
        ep_config["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(extract_int32_value(text, "spacemit_ep_inter_thread_num", 1));
    }
    const std::string affinity = extract_string_value(text, "spacemit_ep_intra_thread_affinity");
    if (!affinity.empty() && ep_config.find("SPACEMIT_EP_INTRA_THREAD_AFFINITY") == ep_config.end()) {
        ep_config["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = affinity;
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
    size_t pos = bracket_start + 1;
    while (pos < bracket_end) {
        const size_t first_quote = text.find('"', pos);
        if (first_quote == std::string::npos || first_quote >= bracket_end) {
            break;
        }
        const size_t second_quote = text.find('"', first_quote + 1);
        if (second_quote == std::string::npos || second_quote > bracket_end) {
            break;
        }
        values.push_back(text.substr(first_quote + 1, second_quote - first_quote - 1));
        pos = second_quote + 1;
    }
    return values;
}

static std::vector<int32_t> extract_int32_array(const std::string & text, const std::string & key) {
    std::vector<int32_t> values;
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

    size_t pos = bracket_start + 1;
    while (pos < bracket_end) {
        while (pos < bracket_end && (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == ',')) {
            ++pos;
        }
        if (pos >= bracket_end) {
            break;
        }
        size_t end = pos;
        if (end < bracket_end && (text[end] == '-' || text[end] == '+')) {
            ++end;
        }
        while (end < bracket_end && std::isdigit(static_cast<unsigned char>(text[end]))) {
            ++end;
        }
        if (end == pos) {
            break;
        }
        values.push_back((int32_t) std::stoi(text.substr(pos, end - pos)));
        pos = end;
    }
    return values;
}

static uint32_t require_gguf_u32(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("missing GGUF uint32 metadata: ") + key);
    }
    return gguf_get_val_u32(ctx, id);
}


static bool lingbot_tensor_type_is_supported_matrix_weight(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || ggml_is_quantized(type);
}

static void require_ggml_tensor_shape(
        ggml_context *       ctx,
        const std::string &  name,
        int64_t              ne0,
        int64_t              ne1 = 1,
        int64_t              ne2 = 1,
        int64_t              ne3 = 1,
        bool                 allow_quantized_matrix_weight = false) {
    const ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
    if (tensor == nullptr) {
        throw std::runtime_error("missing LingBot-MAP tensor: " + name);
    }
    const bool shape_ok = tensor->ne[0] == ne0 && tensor->ne[1] == ne1 && tensor->ne[2] == ne2 && tensor->ne[3] == ne3;
    const bool type_ok = allow_quantized_matrix_weight ?
        lingbot_tensor_type_is_supported_matrix_weight(tensor->type) :
        tensor->type == GGML_TYPE_F32;
    if (!shape_ok || !type_ok) {
        throw std::runtime_error(
            "unexpected LingBot-MAP tensor shape/type: " + name +
            " type=" + ggml_type_name(tensor->type));
    }
}

static void validate_lingbot_map_aggregator_block_shapes(ggml_context * ctx, const lingbot_map_config & cfg) {
    const int64_t c = cfg.hidden_size;
    const int64_t mlp = c * 4;
    const int64_t head_dim = c / 16;
    if (c <= 0 || c % 16 != 0) {
        throw std::runtime_error("LingBot-MAP hidden_size must be divisible by 16 attention heads");
    }
    for (const auto & prefix : {
            std::string("aggregator.frame_blocks.0"),
            std::string("aggregator.frame_blocks.") + std::to_string(cfg.frame_block_count - 1),
            std::string("aggregator.global_blocks.0"),
            std::string("aggregator.global_blocks.") + std::to_string(cfg.global_block_count - 1),
        }) {
        require_ggml_tensor_shape(ctx, prefix + ".norm1.weight", c);
        require_ggml_tensor_shape(ctx, prefix + ".norm1.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".attn.qkv.weight", c, c * 3, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".attn.qkv.bias", c * 3);
        require_ggml_tensor_shape(ctx, prefix + ".attn.q_norm.weight", head_dim);
        require_ggml_tensor_shape(ctx, prefix + ".attn.q_norm.bias", head_dim);
        require_ggml_tensor_shape(ctx, prefix + ".attn.k_norm.weight", head_dim);
        require_ggml_tensor_shape(ctx, prefix + ".attn.k_norm.bias", head_dim);
        require_ggml_tensor_shape(ctx, prefix + ".attn.proj.weight", c, c, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".attn.proj.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".ls1.gamma", c);
        require_ggml_tensor_shape(ctx, prefix + ".norm2.weight", c);
        require_ggml_tensor_shape(ctx, prefix + ".norm2.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc1.weight", c, mlp, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc1.bias", mlp);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc2.weight", mlp, c, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc2.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".ls2.gamma", c);
    }
}


static void validate_lingbot_map_camera_head_shapes(ggml_context * ctx, const lingbot_map_config & cfg) {
    const int64_t c = cfg.camera_hidden_size;
    const int64_t pose_dim = 9;
    const int64_t mlp = c * 4;
    if (c <= 0 || c % 16 != 0) {
        throw std::runtime_error("LingBot-MAP camera_hidden_size must be divisible by 16 attention heads");
    }
    if (cfg.camera_trunk_block_count <= 0 || cfg.camera_num_iterations <= 0) {
        throw std::runtime_error("LingBot-MAP camera_head requires positive trunk block and iteration counts");
    }

    require_ggml_tensor_shape(ctx, "camera_head.empty_pose_tokens", pose_dim, 1, 1);
    require_ggml_tensor_shape(ctx, "camera_head.token_norm.weight", c);
    require_ggml_tensor_shape(ctx, "camera_head.token_norm.bias", c);
    require_ggml_tensor_shape(ctx, "camera_head.trunk_norm.weight", c);
    require_ggml_tensor_shape(ctx, "camera_head.trunk_norm.bias", c);
    require_ggml_tensor_shape(ctx, "camera_head.embed_pose.weight", pose_dim, c);
    require_ggml_tensor_shape(ctx, "camera_head.embed_pose.bias", c);
    require_ggml_tensor_shape(ctx, "camera_head.poseLN_modulation.1.weight", c, c * 3, 1, 1, true);
    require_ggml_tensor_shape(ctx, "camera_head.poseLN_modulation.1.bias", c * 3);
    require_ggml_tensor_shape(ctx, "camera_head.pose_branch.fc1.weight", c, c / 2, 1, 1, true);
    require_ggml_tensor_shape(ctx, "camera_head.pose_branch.fc1.bias", c / 2);
    require_ggml_tensor_shape(ctx, "camera_head.pose_branch.fc2.weight", c / 2, pose_dim, 1, 1, true);
    require_ggml_tensor_shape(ctx, "camera_head.pose_branch.fc2.bias", pose_dim);

    for (int32_t i = 0; i < cfg.camera_trunk_block_count; ++i) {
        const std::string prefix = "camera_head.trunk." + std::to_string(i);
        require_ggml_tensor_shape(ctx, prefix + ".norm1.weight", c);
        require_ggml_tensor_shape(ctx, prefix + ".norm1.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".attn.qkv.weight", c, c * 3, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".attn.qkv.bias", c * 3);
        require_ggml_tensor_shape(ctx, prefix + ".attn.proj.weight", c, c, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".attn.proj.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".ls1.gamma", c);
        require_ggml_tensor_shape(ctx, prefix + ".norm2.weight", c);
        require_ggml_tensor_shape(ctx, prefix + ".norm2.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc1.weight", c, mlp, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc1.bias", mlp);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc2.weight", mlp, c, 1, 1, true);
        require_ggml_tensor_shape(ctx, prefix + ".mlp.fc2.bias", c);
        require_ggml_tensor_shape(ctx, prefix + ".ls2.gamma", c);
    }
}

static std::string require_gguf_string(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("missing GGUF string metadata: ") + key);
    }
    return gguf_get_val_str(ctx, id);
}

static lingbot_map_config load_lingbot_map_config(const std::string & config_dir) {
    const std::string config_path = config_dir + "/config.json";
    const std::string content = read_file_to_string(config_path);
    if (content.empty()) {
        throw std::runtime_error("failed to read LingBot-MAP config: " + config_path);
    }

    const std::string vision_block = extract_object_block(content, "vision_model");
    const std::string agg_block = extract_object_block(content, "aggregator_camera_model");
    const std::string depth_block = extract_object_block(content, "depth_model");
    const std::string post_block = extract_object_block(content, "postprocess");
    if (vision_block.empty() || agg_block.empty() || depth_block.empty()) {
        throw std::runtime_error("LingBot-MAP config requires vision_model, aggregator_camera_model, and depth_model blocks");
    }

    lingbot_map_config cfg;
    cfg.architectures = extract_string_array(content, "architectures");
    cfg.vision_model_path = normalize_path(config_dir, extract_string_value(vision_block, "model_path"));
    cfg.aggregator_camera_model_path = normalize_path(config_dir, extract_string_value(agg_block, "model_path"));
    cfg.depth_model_path = normalize_path(config_dir, extract_string_value(depth_block, "model_path"));
    cfg.ep_config = extract_string_map(vision_block, "ep_config");
    merge_missing_ep_config(cfg.ep_config, extract_string_map(content, "ep_config"));
    apply_legacy_lingbot_ep_config(vision_block, cfg.ep_config);
    apply_legacy_lingbot_ep_config(content, cfg.ep_config);

    cfg.image_size = extract_int32_value(vision_block, "image_size", 518);
    cfg.patch_size = extract_int32_value(vision_block, "patch_size", 14);
    extract_float_array3(vision_block, "image_mean", cfg.image_mean);
    extract_float_array3(vision_block, "image_std", cfg.image_std);
    cfg.hidden_size = extract_int32_value(agg_block, "hidden_size", 0);
    cfg.camera_hidden_size = extract_int32_value(agg_block, "camera_hidden_size", 0);
    cfg.num_special_tokens = extract_int32_value(agg_block, "num_special_tokens", 0);
    cfg.num_register_tokens = extract_int32_value(agg_block, "num_register_tokens", 0);
    cfg.frame_block_count = extract_int32_value(agg_block, "frame_block_count", 0);
    cfg.global_block_count = extract_int32_value(agg_block, "global_block_count", 0);
    cfg.camera_trunk_block_count = extract_int32_value(agg_block, "camera_trunk_block_count", 0);
    cfg.camera_num_iterations = extract_int32_value(agg_block, "camera_num_iterations", 4);
    cfg.ggml_threads = extract_int32_value(agg_block, "ggml_threads", 8);
    cfg.aggregator_selected_layers = extract_int32_array(agg_block, "selected_layers");
    if (cfg.aggregator_selected_layers.empty()) {
        cfg.aggregator_selected_layers = { 4, 11, 17, 23 };
    }

    cfg.output_pose = extract_bool_value(post_block, "output_pose", true);
    cfg.output_depth = extract_bool_value(post_block, "output_depth", true);
    cfg.output_point_cloud = extract_bool_value(post_block, "output_point_cloud", true);

    if (cfg.architectures.empty()) {
        throw std::runtime_error("LingBot-MAP config requires architectures");
    }
    if (cfg.vision_model_path.empty() || cfg.aggregator_camera_model_path.empty() || cfg.depth_model_path.empty()) {
        throw std::runtime_error("LingBot-MAP config contains empty model_path");
    }
    for (const int32_t layer_idx : cfg.aggregator_selected_layers) {
        if (layer_idx < 0 || layer_idx >= cfg.frame_block_count) {
            throw std::runtime_error("LingBot-MAP aggregator selected_layers contains an invalid layer index");
        }
    }
    for (const auto & path : { cfg.vision_model_path, cfg.aggregator_camera_model_path, cfg.depth_model_path }) {
        if (!file_exists(path)) {
            throw std::runtime_error("LingBot-MAP model file not found: " + path);
        }
    }
    return cfg;
}

static lingbot_map_loaded_gguf load_and_validate_gguf(const lingbot_map_config & cfg) {
    ggml_context * ggml_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ggml_raw,
    };
    lingbot_map_loaded_gguf loaded;
    loaded.gguf.reset(gguf_init_from_file(cfg.aggregator_camera_model_path.c_str(), params));
    loaded.ggml.reset(ggml_raw);
    if (!loaded.gguf || !loaded.ggml) {
        throw std::runtime_error("failed to open LingBot-MAP GGUF: " + cfg.aggregator_camera_model_path);
    }

    const gguf_context * gguf = loaded.gguf.get();
    const std::string arch = require_gguf_string(gguf, "general.architecture");
    if (arch != "lingbot-map") {
        throw std::runtime_error("expected LingBot-MAP GGUF architecture 'lingbot-map', got '" + arch + "'");
    }
    const std::string component = require_gguf_string(gguf, "lingbot-map.component");
    if (component != "aggregator_camera_head") {
        throw std::runtime_error("unsupported LingBot-MAP GGUF component: " + component);
    }

    const uint32_t file_type = require_gguf_u32(gguf, "general.file_type");
    (void) file_type;

    const uint32_t embed_dim = require_gguf_u32(gguf, "lingbot-map.embed_dim");
    const uint32_t camera_dim = require_gguf_u32(gguf, "lingbot-map.camera_dim");
    const uint32_t special_tokens = require_gguf_u32(gguf, "lingbot-map.num_special_tokens");
    const uint32_t frame_blocks = require_gguf_u32(gguf, "lingbot-map.aggregator_frame_block_count");
    const uint32_t global_blocks = require_gguf_u32(gguf, "lingbot-map.aggregator_global_block_count");
    const uint32_t camera_blocks = require_gguf_u32(gguf, "lingbot-map.camera_trunk_block_count");

    if ((uint32_t) cfg.hidden_size != embed_dim || (uint32_t) cfg.camera_hidden_size != camera_dim ||
        (uint32_t) cfg.num_special_tokens != special_tokens || (uint32_t) cfg.frame_block_count != frame_blocks ||
        (uint32_t) cfg.global_block_count != global_blocks || (uint32_t) cfg.camera_trunk_block_count != camera_blocks) {
        throw std::runtime_error("LingBot-MAP config.json does not match GGUF metadata");
    }

    if (gguf_find_tensor(gguf, "aggregator.camera_token") < 0 ||
        gguf_find_tensor(gguf, "camera_head.pose_branch.fc2.bias") < 0 ||
        ggml_get_tensor(loaded.ggml.get(), "aggregator.camera_token") == nullptr ||
        ggml_get_tensor(loaded.ggml.get(), "camera_head.pose_branch.fc2.bias") == nullptr) {
        throw std::runtime_error("LingBot-MAP GGUF is missing required boundary tensors");
    }
    validate_lingbot_map_aggregator_block_shapes(loaded.ggml.get(), cfg);
    validate_lingbot_map_camera_head_shapes(loaded.ggml.get(), cfg);

    return loaded;
}

static const float * lingbot_tensor_f32_data(const ggml_tensor * tensor, const std::string & name) {
    if (tensor == nullptr) {
        throw std::runtime_error("missing LingBot-MAP tensor: " + name);
    }
    if (tensor->type != GGML_TYPE_F32 || tensor->data == nullptr) {
        throw std::runtime_error("LingBot-MAP tensor must be loaded as F32: " + name);
    }
    return static_cast<const float *>(tensor->data);
}

} // namespace

struct lingbot_map_context::impl {
    lingbot_map_config config;
    gguf_context_ptr gguf;
    ggml_context_ptr ggml;
    std::string arch_name;

    ggml_backend_ptr runtime_backend;
    ggml_backend_buffer_type_t runtime_buft = nullptr;
    lingbot_map_runtime_weights runtime_weights;
    bool runtime_initialized = false;
    bool runtime_prefer_smt = true;
};

lingbot_map_context::~lingbot_map_context() = default;

std::unique_ptr<lingbot_map_context> lingbot_map_context::create(const std::string & config_dir) {
    auto ctx = std::unique_ptr<lingbot_map_context>(new lingbot_map_context());
    ctx->pimpl_ = std::make_unique<impl>();
    ctx->pimpl_->config = load_lingbot_map_config(config_dir);
    auto loaded = load_and_validate_gguf(ctx->pimpl_->config);
    ctx->pimpl_->gguf = std::move(loaded.gguf);
    ctx->pimpl_->ggml = std::move(loaded.ggml);
    ctx->pimpl_->arch_name = ctx->pimpl_->config.architectures.empty() ? std::string() : ctx->pimpl_->config.architectures[0];

    std::cerr << "[LingBot-MAP] loaded config and GGUF: " << ctx->pimpl_->config.aggregator_camera_model_path
              << ", tensors=" << gguf_get_n_tensors(ctx->pimpl_->gguf.get()) << "\n";
    return ctx;
}

const lingbot_map_config & lingbot_map_context::config() const {
    return pimpl_->config;
}

const std::string & lingbot_map_context::architecture() const {
    return pimpl_->arch_name;
}

int64_t lingbot_map_context::tensor_count() const {
    return gguf_get_n_tensors(pimpl_->gguf.get());
}

ggml_context * lingbot_map_context::ggml_ctx() const {
    return pimpl_->ggml.get();
}

const ggml_tensor * lingbot_map_context::tensor(const std::string & name) const {
    if (pimpl_->ggml == nullptr) {
        return nullptr;
    }
    return ggml_get_tensor(pimpl_->ggml.get(), name.c_str());
}


lingbot_map_aggregator_input lingbot_map_context::build_aggregator_input(
        const float * vit_tokens,
        int32_t       n_frames,
        int32_t       vit_tokens_per_frame,
        int32_t       hidden_size,
        int32_t       image_h,
        int32_t       image_w,
        int32_t       num_frame_for_scale) const {
    if (vit_tokens == nullptr) {
        throw std::invalid_argument("LingBot-MAP aggregator input requires ViT tokens");
    }
    const auto & cfg = config();
    if (n_frames <= 0 || vit_tokens_per_frame <= 0 || hidden_size != cfg.hidden_size) {
        throw std::invalid_argument("Invalid LingBot-MAP ViT token shape for aggregator");
    }
    if (image_h <= 0 || image_w <= 0 || cfg.patch_size <= 0) {
        throw std::invalid_argument("Invalid LingBot-MAP image dimensions for aggregator");
    }

    const int32_t patch_h = image_h / cfg.patch_size;
    const int32_t patch_w = image_w / cfg.patch_size;
    const int32_t patch_tokens = patch_h * patch_w;
    if (patch_tokens <= 0 || vit_tokens_per_frame < patch_tokens) {
        throw std::invalid_argument("LingBot-MAP ViT output does not contain enough patch tokens for aggregator");
    }

    const int32_t vit_prefix_tokens = vit_tokens_per_frame - patch_tokens;
    const int32_t patch_start_idx = 1 + cfg.num_register_tokens + 1;
    if (patch_start_idx != cfg.num_special_tokens) {
        throw std::runtime_error("LingBot-MAP special token metadata is inconsistent");
    }

    const ggml_tensor * camera_tensor = tensor("aggregator.camera_token");
    const ggml_tensor * register_tensor = tensor("aggregator.register_token");
    const ggml_tensor * scale_tensor = tensor("aggregator.scale_token");
    const float * camera_token = lingbot_tensor_f32_data(camera_tensor, "aggregator.camera_token");
    const float * register_token = lingbot_tensor_f32_data(register_tensor, "aggregator.register_token");
    const float * scale_token = lingbot_tensor_f32_data(scale_tensor, "aggregator.scale_token");

    if (camera_tensor->ne[0] != hidden_size || camera_tensor->ne[1] != 1 || camera_tensor->ne[2] != 2 ||
        register_tensor->ne[0] != hidden_size || register_tensor->ne[1] != cfg.num_register_tokens || register_tensor->ne[2] != 2 ||
        scale_tensor->ne[0] != hidden_size || scale_tensor->ne[1] != 1 || scale_tensor->ne[2] != 2) {
        throw std::runtime_error("LingBot-MAP special token tensor shapes do not match config");
    }

    lingbot_map_aggregator_input out;
    out.n_frames = n_frames;
    out.hidden_size = hidden_size;
    out.vit_tokens_per_frame = vit_tokens_per_frame;
    out.vit_prefix_tokens = vit_prefix_tokens;
    out.patch_tokens = patch_tokens;
    out.patch_start_idx = patch_start_idx;
    out.tokens_per_frame = patch_start_idx + patch_tokens;
    out.tokens.resize((size_t) n_frames * (size_t) out.tokens_per_frame * (size_t) hidden_size);

    const int32_t scale_frames = std::max(1, std::min(num_frame_for_scale, n_frames));
    auto copy_token_variant = [&](const float * token_base, int32_t variant, int32_t n_token, int32_t frame, int32_t dst_token) {
        const size_t src_base = ((size_t) variant * (size_t) n_token) * (size_t) hidden_size;
        const size_t dst_base = ((size_t) frame * (size_t) out.tokens_per_frame + (size_t) dst_token) * (size_t) hidden_size;
        std::copy(token_base + src_base, token_base + src_base + (size_t) n_token * (size_t) hidden_size,
                  out.tokens.data() + dst_base);
    };

    for (int32_t f = 0; f < n_frames; ++f) {
        const int32_t camera_variant = f == 0 ? 0 : 1;
        const int32_t register_variant = f == 0 ? 0 : 1;
        const int32_t scale_variant = f < scale_frames ? 0 : 1;

        copy_token_variant(camera_token, camera_variant, 1, f, 0);
        copy_token_variant(register_token, register_variant, cfg.num_register_tokens, f, 1);
        copy_token_variant(scale_token, scale_variant, 1, f, 1 + cfg.num_register_tokens);

        const float * vit_frame = vit_tokens + (size_t) f * (size_t) vit_tokens_per_frame * (size_t) hidden_size;
        const float * patch_src = vit_frame + (size_t) vit_prefix_tokens * (size_t) hidden_size;
        float * patch_dst = out.tokens.data() + ((size_t) f * (size_t) out.tokens_per_frame + (size_t) patch_start_idx) * (size_t) hidden_size;
        std::copy(patch_src, patch_src + (size_t) patch_tokens * (size_t) hidden_size, patch_dst);
    }

    return out;
}


static ggml_tensor * lingbot_require_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
    if (tensor == nullptr) {
        throw std::runtime_error("missing LingBot-MAP tensor: " + name);
    }
    return tensor;
}

static ggml_tensor * lingbot_layer_norm(
        ggml_context *      ctx,
        ggml_tensor *       input,
        ggml_tensor *       weight,
        ggml_tensor *       bias,
        float               eps) {
    ggml_tensor * cur = ggml_norm(ctx, input, eps);
    cur = ggml_mul(ctx, cur, weight);
    cur = ggml_add(ctx, cur, bias);
    return cur;
}

static ggml_tensor * lingbot_linear(
        ggml_context *      ctx,
        ggml_tensor *       input,
        ggml_tensor *       weight,
        ggml_tensor *       bias) {
    ggml_tensor * cur = ggml_mul_mat(ctx, weight, input);
    if (bias != nullptr) {
        cur = ggml_add(ctx, cur, bias);
    }
    return cur;
}

static ggml_tensor * lingbot_mlp_gelu(
        ggml_context *      ctx,
        ggml_tensor *       input,
        ggml_tensor *       fc1_w,
        ggml_tensor *       fc1_b,
        ggml_tensor *       fc2_w,
        ggml_tensor *       fc2_b) {
    ggml_tensor * cur = lingbot_linear(ctx, input, fc1_w, fc1_b);
    cur = ggml_gelu(ctx, cur);
    cur = lingbot_linear(ctx, cur, fc2_w, fc2_b);
    return cur;
}

static ggml_tensor * lingbot_qkv_view(
        ggml_context * ctx,
        ggml_tensor *  qkv,
        int64_t        hidden_size,
        int            index) {
    return ggml_view_3d(ctx, qkv,
                        hidden_size,
                        qkv->ne[1],
                        qkv->ne[2],
                        qkv->nb[1],
                        qkv->nb[2],
                        (size_t) index * (size_t) hidden_size * ggml_type_size(qkv->type));
}

static ggml_tensor * lingbot_head_view(
        ggml_context * ctx,
        ggml_tensor *  x,
        int64_t        head_dim,
        int64_t        n_heads) {
    return ggml_view_4d(ctx, x,
                        head_dim,
                        n_heads,
                        x->ne[1],
                        x->ne[2],
                        (size_t) head_dim * ggml_type_size(x->type),
                        x->nb[1],
                        x->nb[2],
                        0);
}

static ggml_tensor * lingbot_frame_self_attention(
        ggml_context * ctx,
        ggml_tensor *  qkv,
        ggml_tensor *  q_norm_w,
        ggml_tensor *  q_norm_b,
        ggml_tensor *  k_norm_w,
        ggml_tensor *  k_norm_b,
        ggml_tensor *  proj_w,
        ggml_tensor *  proj_b,
        int64_t        hidden_size) {
    const int64_t head_dim = q_norm_w->ne[0];
    if (head_dim <= 0 || hidden_size % head_dim != 0 || k_norm_w->ne[0] != head_dim ||
        q_norm_b->ne[0] != head_dim || k_norm_b->ne[0] != head_dim) {
        throw std::runtime_error("LingBot-MAP q/k norm shapes do not match hidden size");
    }
    const int64_t n_heads = hidden_size / head_dim;

    ggml_tensor * q = lingbot_qkv_view(ctx, qkv, hidden_size, 0);
    ggml_tensor * k = lingbot_qkv_view(ctx, qkv, hidden_size, 1);
    ggml_tensor * v = lingbot_qkv_view(ctx, qkv, hidden_size, 2);

    q = lingbot_head_view(ctx, q, head_dim, n_heads);
    k = lingbot_head_view(ctx, k, head_dim, n_heads);
    v = lingbot_head_view(ctx, v, head_dim, n_heads);

    q = lingbot_layer_norm(ctx, q, q_norm_w, q_norm_b, 1e-6f);
    k = lingbot_layer_norm(ctx, k, k_norm_w, k_norm_b, 1e-6f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_cont_3d(ctx, attn, hidden_size, qkv->ne[1], qkv->ne[2]);

    return lingbot_linear(ctx, attn, proj_w, proj_b);
}

static ggml_tensor * lingbot_apply_aggregator_block(
        ggml_context *      ctx,
        ggml_context *      weights_ctx,
        const lingbot_map_config & cfg,
        ggml_tensor *       x,
        const std::string & prefix,
        const std::string & graph_name,
        ggml_tensor **      qkv_out) {
    ggml_tensor * norm1_w = lingbot_require_tensor(weights_ctx, prefix + ".norm1.weight");
    ggml_tensor * norm1_b = lingbot_require_tensor(weights_ctx, prefix + ".norm1.bias");
    ggml_tensor * qkv_w   = lingbot_require_tensor(weights_ctx, prefix + ".attn.qkv.weight");
    ggml_tensor * qkv_b   = lingbot_require_tensor(weights_ctx, prefix + ".attn.qkv.bias");
    ggml_tensor * q_norm_w = lingbot_require_tensor(weights_ctx, prefix + ".attn.q_norm.weight");
    ggml_tensor * q_norm_b = lingbot_require_tensor(weights_ctx, prefix + ".attn.q_norm.bias");
    ggml_tensor * k_norm_w = lingbot_require_tensor(weights_ctx, prefix + ".attn.k_norm.weight");
    ggml_tensor * k_norm_b = lingbot_require_tensor(weights_ctx, prefix + ".attn.k_norm.bias");
    ggml_tensor * proj_w  = lingbot_require_tensor(weights_ctx, prefix + ".attn.proj.weight");
    ggml_tensor * proj_b  = lingbot_require_tensor(weights_ctx, prefix + ".attn.proj.bias");
    ggml_tensor * ls1     = lingbot_require_tensor(weights_ctx, prefix + ".ls1.gamma");
    ggml_tensor * norm2_w = lingbot_require_tensor(weights_ctx, prefix + ".norm2.weight");
    ggml_tensor * norm2_b = lingbot_require_tensor(weights_ctx, prefix + ".norm2.bias");
    ggml_tensor * fc1_w   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc1.weight");
    ggml_tensor * fc1_b   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc1.bias");
    ggml_tensor * fc2_w   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc2.weight");
    ggml_tensor * fc2_b   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc2.bias");
    ggml_tensor * ls2     = lingbot_require_tensor(weights_ctx, prefix + ".ls2.gamma");

    ggml_tensor * normed = lingbot_layer_norm(ctx, x, norm1_w, norm1_b, 1e-6f);
    ggml_tensor * qkv = lingbot_linear(ctx, normed, qkv_w, qkv_b);
    ggml_set_name(qkv, (graph_name + ".qkv").c_str());
    if (qkv_out != nullptr) {
        *qkv_out = qkv;
    }

    ggml_tensor * attn = lingbot_frame_self_attention(ctx, qkv, q_norm_w, q_norm_b, k_norm_w, k_norm_b,
                                                      proj_w, proj_b, cfg.hidden_size);
    attn = ggml_mul(ctx, attn, ls1);
    ggml_tensor * attn_out = ggml_add(ctx, x, attn);
    ggml_set_name(attn_out, (graph_name + ".attn_output").c_str());

    ggml_tensor * ffn_inp = lingbot_layer_norm(ctx, attn_out, norm2_w, norm2_b, 1e-6f);
    ggml_tensor * ffn = lingbot_mlp_gelu(ctx, ffn_inp, fc1_w, fc1_b, fc2_w, fc2_b);
    ffn = ggml_mul(ctx, ffn, ls2);
    ggml_tensor * out = ggml_add(ctx, attn_out, ffn);
    ggml_set_name(out, (graph_name + ".output").c_str());
    return out;
}


static ggml_tensor * lingbot_camera_self_attention(
        ggml_context * ctx,
        ggml_tensor *  qkv,
        ggml_tensor *  proj_w,
        ggml_tensor *  proj_b,
        int64_t        hidden_size) {
    const int64_t n_heads = 16;
    if (hidden_size <= 0 || hidden_size % n_heads != 0) {
        throw std::runtime_error("LingBot-MAP camera hidden size must be divisible by 16 attention heads");
    }
    const int64_t head_dim = hidden_size / n_heads;

    ggml_tensor * q = lingbot_qkv_view(ctx, qkv, hidden_size, 0);
    ggml_tensor * k = lingbot_qkv_view(ctx, qkv, hidden_size, 1);
    ggml_tensor * v = lingbot_qkv_view(ctx, qkv, hidden_size, 2);

    q = lingbot_head_view(ctx, q, head_dim, n_heads);
    k = lingbot_head_view(ctx, k, head_dim, n_heads);
    v = lingbot_head_view(ctx, v, head_dim, n_heads);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_cont_3d(ctx, attn, hidden_size, qkv->ne[1], qkv->ne[2]);

    return lingbot_linear(ctx, attn, proj_w, proj_b);
}

static ggml_tensor * lingbot_apply_camera_trunk_block(
        ggml_context *      ctx,
        ggml_context *      weights_ctx,
        const lingbot_map_config & cfg,
        ggml_tensor *       x,
        const std::string & prefix,
        const std::string & graph_name) {
    ggml_tensor * norm1_w = lingbot_require_tensor(weights_ctx, prefix + ".norm1.weight");
    ggml_tensor * norm1_b = lingbot_require_tensor(weights_ctx, prefix + ".norm1.bias");
    ggml_tensor * qkv_w   = lingbot_require_tensor(weights_ctx, prefix + ".attn.qkv.weight");
    ggml_tensor * qkv_b   = lingbot_require_tensor(weights_ctx, prefix + ".attn.qkv.bias");
    ggml_tensor * proj_w  = lingbot_require_tensor(weights_ctx, prefix + ".attn.proj.weight");
    ggml_tensor * proj_b  = lingbot_require_tensor(weights_ctx, prefix + ".attn.proj.bias");
    ggml_tensor * ls1     = lingbot_require_tensor(weights_ctx, prefix + ".ls1.gamma");
    ggml_tensor * norm2_w = lingbot_require_tensor(weights_ctx, prefix + ".norm2.weight");
    ggml_tensor * norm2_b = lingbot_require_tensor(weights_ctx, prefix + ".norm2.bias");
    ggml_tensor * fc1_w   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc1.weight");
    ggml_tensor * fc1_b   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc1.bias");
    ggml_tensor * fc2_w   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc2.weight");
    ggml_tensor * fc2_b   = lingbot_require_tensor(weights_ctx, prefix + ".mlp.fc2.bias");
    ggml_tensor * ls2     = lingbot_require_tensor(weights_ctx, prefix + ".ls2.gamma");

    ggml_tensor * normed = lingbot_layer_norm(ctx, x, norm1_w, norm1_b, 1e-6f);
    ggml_tensor * qkv = lingbot_linear(ctx, normed, qkv_w, qkv_b);
    ggml_set_name(qkv, (graph_name + ".qkv").c_str());

    ggml_tensor * attn = lingbot_camera_self_attention(ctx, qkv, proj_w, proj_b, cfg.camera_hidden_size);
    attn = ggml_mul(ctx, attn, ls1);
    ggml_tensor * attn_out = ggml_add(ctx, x, attn);
    ggml_set_name(attn_out, (graph_name + ".attn_output").c_str());

    ggml_tensor * ffn_inp = lingbot_layer_norm(ctx, attn_out, norm2_w, norm2_b, 1e-6f);
    ggml_tensor * ffn = lingbot_mlp_gelu(ctx, ffn_inp, fc1_w, fc1_b, fc2_w, fc2_b);
    ffn = ggml_mul(ctx, ffn, ls2);
    ggml_tensor * out = ggml_add(ctx, attn_out, ffn);
    ggml_set_name(out, (graph_name + ".output").c_str());
    return out;
}

static ggml_tensor * lingbot_pose_branch(
        ggml_context * ctx,
        ggml_context * weights_ctx,
        ggml_tensor *  x) {
    ggml_tensor * fc1_w = lingbot_require_tensor(weights_ctx, "camera_head.pose_branch.fc1.weight");
    ggml_tensor * fc1_b = lingbot_require_tensor(weights_ctx, "camera_head.pose_branch.fc1.bias");
    ggml_tensor * fc2_w = lingbot_require_tensor(weights_ctx, "camera_head.pose_branch.fc2.weight");
    ggml_tensor * fc2_b = lingbot_require_tensor(weights_ctx, "camera_head.pose_branch.fc2.bias");
    return lingbot_mlp_gelu(ctx, x, fc1_w, fc1_b, fc2_w, fc2_b);
}


static ggml_tensor * lingbot_activate_pose(
        ggml_context * ctx,
        ggml_tensor *  pred_pose) {
    if (pred_pose->ne[0] != 9) {
        throw std::runtime_error("LingBot-MAP camera_head pose activation expects 9 pose channels");
    }
    ggml_tensor * trans_quat = ggml_view_3d(ctx, pred_pose, 7, pred_pose->ne[1], pred_pose->ne[2],
                                            pred_pose->nb[1], pred_pose->nb[2], 0);
    ggml_tensor * fov = ggml_view_3d(ctx, pred_pose, 2, pred_pose->ne[1], pred_pose->ne[2],
                                     pred_pose->nb[1], pred_pose->nb[2],
                                     7 * ggml_type_size(pred_pose->type));
    fov = ggml_relu(ctx, fov);
    return ggml_concat(ctx, trans_quat, fov, 0);
}

static ggml_backend_buffer_type_t lingbot_select_cpu_buffer_type(
        ggml_backend_t backend,
        bool           prefer_smt) {
    ggml_backend_buffer_type_t default_buft = ggml_backend_get_default_buffer_type(backend);
    if (!prefer_smt) {
        return default_buft;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto * get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts == nullptr) {
        return default_buft;
    }

    ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts(dev);
    if (extra_bufts == nullptr) {
        return default_buft;
    }
    for (int i = 0; extra_bufts[i] != nullptr; ++i) {
        const char * name = ggml_backend_buft_name(extra_bufts[i]);
        if (name != nullptr && std::strstr(name, "SPACEMIT") != nullptr &&
            ggml_backend_supports_buft(backend, extra_bufts[i])) {
            return extra_bufts[i];
        }
    }
    return default_buft;
}

static lingbot_map_runtime_weights lingbot_load_runtime_weights(
        const lingbot_map_config & cfg,
        ggml_backend_buffer_type_t buft) {
    ggml_context * ggml_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ggml_raw,
    };

    lingbot_map_runtime_weights weights;
    weights.gguf.reset(gguf_init_from_file(cfg.aggregator_camera_model_path.c_str(), params));
    weights.ggml.reset(ggml_raw);
    if (!weights.gguf || !weights.ggml) {
        throw std::runtime_error("failed to open LingBot-MAP GGUF for runtime: " + cfg.aggregator_camera_model_path);
    }
    validate_lingbot_map_aggregator_block_shapes(weights.ggml.get(), cfg);
    validate_lingbot_map_camera_head_shapes(weights.ggml.get(), cfg);

    weights.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.ggml.get(), buft));
    if (!weights.buffer) {
        throw std::runtime_error(std::string("failed to allocate LingBot-MAP runtime weights on buffer type: ") +
                                 ggml_backend_buft_name(buft));
    }
    ggml_backend_buffer_set_usage(weights.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::ifstream fin(cfg.aggregator_camera_model_path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("failed to read LingBot-MAP GGUF weights: " + cfg.aggregator_camera_model_path);
    }

    std::vector<uint8_t> read_buf;
    const int64_t n_tensors = gguf_get_n_tensors(weights.gguf.get());
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(weights.gguf.get(), i);
        ggml_tensor * tensor = ggml_get_tensor(weights.ggml.get(), name);
        if (tensor == nullptr) {
            throw std::runtime_error(std::string("missing LingBot-MAP runtime tensor: ") + name);
        }
        const size_t offset = gguf_get_data_offset(weights.gguf.get()) + gguf_get_tensor_offset(weights.gguf.get(), i);
        const size_t nbytes = ggml_nbytes(tensor);
        fin.seekg((std::streamoff) offset, std::ios::beg);
        if (!fin) {
            throw std::runtime_error(std::string("failed to seek LingBot-MAP runtime tensor: ") + name);
        }
        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(tensor->data), (std::streamsize) nbytes);
        } else {
            read_buf.resize(nbytes);
            fin.read(reinterpret_cast<char *>(read_buf.data()), (std::streamsize) nbytes);
            ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
        }
        if (!fin) {
            throw std::runtime_error(std::string("failed to load LingBot-MAP runtime tensor: ") + name);
        }
    }
    return weights;
}

static lingbot_map_runtime_graph lingbot_build_aggregator_camera_runtime_graph(
        ggml_context *                       ctx,
        ggml_context *                       weights_ctx,
        const lingbot_map_config &           cfg,
        const lingbot_map_aggregator_input & input) {
    ggml_tensor * tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.hidden_size,
                                              input.tokens_per_frame, input.n_frames);
    ggml_set_name(tokens, "lingbot_map.runtime.aggregator.input");
    ggml_set_input(tokens);

    lingbot_map_runtime_graph built;
    built.input_tokens = tokens;
    built.selected_outputs.reserve(cfg.aggregator_selected_layers.size());
    built.iteration_poses.reserve(cfg.camera_num_iterations);

    ggml_tensor * frame_tokens = nullptr;
    ggml_tensor * global_tokens = nullptr;
    ggml_tensor * global_as_frame = nullptr;

    for (int32_t i = 0; i < cfg.frame_block_count; ++i) {
        frame_tokens = lingbot_apply_aggregator_block(ctx, weights_ctx, cfg, tokens,
                                                      "aggregator.frame_blocks." + std::to_string(i),
                                                      "lingbot_map.runtime.aggregator.frame." + std::to_string(i),
                                                      nullptr);
        global_tokens = ggml_reshape_3d(ctx, frame_tokens, cfg.hidden_size,
                                        (int64_t) input.tokens_per_frame * input.n_frames, 1);
        global_tokens = lingbot_apply_aggregator_block(ctx, weights_ctx, cfg, global_tokens,
                                                       "aggregator.global_blocks." + std::to_string(i),
                                                       "lingbot_map.runtime.aggregator.global." + std::to_string(i),
                                                       nullptr);
        global_as_frame = ggml_reshape_3d(ctx, global_tokens, cfg.hidden_size, input.tokens_per_frame, input.n_frames);

        if (std::find(cfg.aggregator_selected_layers.begin(), cfg.aggregator_selected_layers.end(), i) !=
            cfg.aggregator_selected_layers.end()) {
            ggml_tensor * selected = ggml_concat(ctx, frame_tokens, global_as_frame, 0);
            ggml_set_name(selected, ("lingbot_map.runtime.aggregator.selected." + std::to_string(i)).c_str());
            built.selected_outputs.push_back(selected);
            if (i == cfg.frame_block_count - 1) {
                built.camera_head_input = selected;
            }
        }

        tokens = global_as_frame;
    }

    if (built.camera_head_input == nullptr) {
        built.camera_head_input = ggml_concat(ctx, frame_tokens, global_as_frame, 0);
        ggml_set_name(built.camera_head_input, "lingbot_map.runtime.aggregator.camera_head_input");
    }
    if (built.camera_head_input->ne[0] != cfg.camera_hidden_size) {
        throw std::runtime_error("LingBot-MAP runtime camera_head input width does not match camera_hidden_size");
    }

    const int64_t pose_dim = 9;
    ggml_tensor * pose_tokens = ggml_view_3d(ctx, built.camera_head_input,
                                             cfg.camera_hidden_size, input.n_frames, 1,
                                             built.camera_head_input->nb[2],
                                             (size_t) built.camera_head_input->nb[2] * (size_t) input.n_frames,
                                             0);
    pose_tokens = lingbot_layer_norm(ctx, pose_tokens,
                                     lingbot_require_tensor(weights_ctx, "camera_head.token_norm.weight"),
                                     lingbot_require_tensor(weights_ctx, "camera_head.token_norm.bias"),
                                     1e-6f);
    ggml_set_name(pose_tokens, "lingbot_map.runtime.camera_head.pose_tokens");

    ggml_tensor * empty_pose = lingbot_require_tensor(weights_ctx, "camera_head.empty_pose_tokens");
    ggml_tensor * pred_pose = nullptr;

    for (int32_t iter = 0; iter < cfg.camera_num_iterations; ++iter) {
        ggml_tensor * module_input = nullptr;
        if (pred_pose == nullptr) {
            ggml_tensor * empty_pose_target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pose_dim, input.n_frames, 1);
            module_input = ggml_repeat(ctx, empty_pose, empty_pose_target);
        } else {
            module_input = pred_pose;
        }

        module_input = lingbot_linear(ctx, module_input,
                                      lingbot_require_tensor(weights_ctx, "camera_head.embed_pose.weight"),
                                      lingbot_require_tensor(weights_ctx, "camera_head.embed_pose.bias"));

        ggml_tensor * modulation = lingbot_linear(ctx,
                                                  ggml_silu(ctx, module_input),
                                                  lingbot_require_tensor(weights_ctx, "camera_head.poseLN_modulation.1.weight"),
                                                  lingbot_require_tensor(weights_ctx, "camera_head.poseLN_modulation.1.bias"));
        ggml_tensor * shift = ggml_view_3d(ctx, modulation, cfg.camera_hidden_size, input.n_frames, 1,
                                           modulation->nb[1], modulation->nb[2], 0);
        ggml_tensor * scale = ggml_view_3d(ctx, modulation, cfg.camera_hidden_size, input.n_frames, 1,
                                           modulation->nb[1], modulation->nb[2],
                                           (size_t) cfg.camera_hidden_size * ggml_type_size(modulation->type));
        ggml_tensor * gate = ggml_view_3d(ctx, modulation, cfg.camera_hidden_size, input.n_frames, 1,
                                          modulation->nb[1], modulation->nb[2],
                                          (size_t) cfg.camera_hidden_size * 2 * ggml_type_size(modulation->type));

        ggml_tensor * adaln = ggml_norm(ctx, pose_tokens, 1e-6f);
        ggml_tensor * scale_cont = ggml_cont(ctx, scale);
        ggml_tensor * modulated = ggml_mul(ctx, adaln, ggml_scale_bias(ctx, scale_cont, 1.0f, 1.0f));
        modulated = ggml_add(ctx, modulated, shift);
        modulated = ggml_mul(ctx, modulated, gate);
        modulated = ggml_add(ctx, modulated, pose_tokens);

        for (int32_t block = 0; block < cfg.camera_trunk_block_count; ++block) {
            modulated = lingbot_apply_camera_trunk_block(ctx, weights_ctx, cfg, modulated,
                                                         "camera_head.trunk." + std::to_string(block),
                                                         "lingbot_map.runtime.camera_head.iter." + std::to_string(iter) +
                                                         ".trunk." + std::to_string(block));
        }

        ggml_tensor * trunk_norm = lingbot_layer_norm(ctx, modulated,
                                                      lingbot_require_tensor(weights_ctx, "camera_head.trunk_norm.weight"),
                                                      lingbot_require_tensor(weights_ctx, "camera_head.trunk_norm.bias"),
                                                      1e-6f);
        ggml_tensor * delta = lingbot_pose_branch(ctx, weights_ctx, trunk_norm);
        pred_pose = pred_pose == nullptr ? delta : ggml_add(ctx, pred_pose, delta);
        ggml_set_name(pred_pose, ("lingbot_map.runtime.camera_head.pose_iter." + std::to_string(iter)).c_str());
        ggml_tensor * activated_pose = lingbot_activate_pose(ctx, pred_pose);
        ggml_set_name(activated_pose, ("lingbot_map.runtime.camera_head.activated_pose_iter." + std::to_string(iter)).c_str());
        built.iteration_poses.push_back(activated_pose);
        built.final_pose = activated_pose;
    }

    built.graph = ggml_new_graph_custom(ctx, 32768, false);
    for (size_t i = 0; i < built.selected_outputs.size(); ++i) {
        ggml_tensor * selected = ggml_cont(ctx, built.selected_outputs[i]);
        ggml_set_name(selected, ("lingbot_map.runtime.aggregator.selected_output." + std::to_string(i)).c_str());
        built.selected_outputs[i] = selected;
        ggml_build_forward_expand(built.graph, selected);
    }
    built.final_pose = ggml_cont(ctx, built.final_pose);
    ggml_set_name(built.final_pose, "lingbot_map.runtime.camera_head.final_pose_output");
    ggml_build_forward_expand(built.graph, built.final_pose);
    return built;
}

static lingbot_map_graph_probe_result lingbot_build_aggregator_block_probe(
        ggml_context *                       weights_ctx,
        const lingbot_map_config &           cfg,
        const lingbot_map_aggregator_input & input,
        const std::string &                  block_prefix,
        const std::string &                  graph_name,
        bool                                 flatten_frames) {
    if (input.n_frames <= 0 || input.tokens_per_frame <= 0 || input.hidden_size != cfg.hidden_size) {
        throw std::invalid_argument("Invalid LingBot-MAP aggregator input for block probe");
    }

    const size_t mem_size = 64ull * 1024ull * 1024ull;
    ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    if (!compute_ctx) {
        throw std::runtime_error("failed to create LingBot-MAP aggregator probe ggml context");
    }

    const int64_t n_seq_tokens = flatten_frames ? (int64_t) input.tokens_per_frame * input.n_frames : input.tokens_per_frame;
    const int64_t n_batches = flatten_frames ? 1 : input.n_frames;

    ggml_context * ctx = compute_ctx.get();
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.hidden_size, n_seq_tokens, n_batches);
    ggml_set_name(x, (graph_name + ".input").c_str());
    ggml_set_input(x);

    ggml_tensor * qkv = nullptr;
    ggml_tensor * out = lingbot_apply_aggregator_block(ctx, weights_ctx, cfg, x, block_prefix, graph_name, &qkv);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, qkv);
    ggml_build_forward_expand(graph, out);

    lingbot_map_graph_probe_result result;
    result.graph_nodes = ggml_graph_n_nodes(graph);
    result.input_tokens_per_frame = (int32_t) n_seq_tokens;
    for (int i = 0; i < 4; ++i) {
        result.output_ne[i] = (int32_t) out->ne[i];
        result.qkv_ne[i] = (int32_t) qkv->ne[i];
    }
    return result;
}

lingbot_map_graph_probe_result lingbot_map_context::build_aggregator_frame_block_probe(
        const lingbot_map_aggregator_input & input,
        int32_t                              block_index) const {
    const auto & cfg = config();
    if (block_index < 0 || block_index >= cfg.frame_block_count) {
        throw std::invalid_argument("Invalid LingBot-MAP frame block index");
    }
    return lingbot_build_aggregator_block_probe(pimpl_->ggml.get(), cfg, input,
                                                "aggregator.frame_blocks." + std::to_string(block_index),
                                                "lingbot_map.aggregator.frame_probe",
                                                /* flatten_frames */ false);
}

lingbot_map_graph_probe_result lingbot_map_context::build_aggregator_global_block_probe(
        const lingbot_map_aggregator_input & input,
        int32_t                              block_index) const {
    const auto & cfg = config();
    if (block_index < 0 || block_index >= cfg.global_block_count) {
        throw std::invalid_argument("Invalid LingBot-MAP global block index");
    }
    return lingbot_build_aggregator_block_probe(pimpl_->ggml.get(), cfg, input,
                                                "aggregator.global_blocks." + std::to_string(block_index),
                                                "lingbot_map.aggregator.global_probe",
                                                /* flatten_frames */ true);
}

lingbot_map_aggregator_probe_result lingbot_map_context::build_aggregator_block_probes(
        const lingbot_map_aggregator_input & input,
        int32_t                              block_index) const {
    lingbot_map_aggregator_probe_result result;
    result.frame = build_aggregator_frame_block_probe(input, block_index);
    result.global = build_aggregator_global_block_probe(input, block_index);
    return result;
}

lingbot_map_full_aggregator_probe_result lingbot_map_context::build_full_aggregator_probe(
        const lingbot_map_aggregator_input & input) const {
    const auto & cfg = config();
    if (input.n_frames <= 0 || input.tokens_per_frame <= 0 || input.hidden_size != cfg.hidden_size) {
        throw std::invalid_argument("Invalid LingBot-MAP aggregator input for full probe");
    }
    if (cfg.frame_block_count <= 0 || cfg.global_block_count <= 0 || cfg.frame_block_count != cfg.global_block_count) {
        throw std::runtime_error("LingBot-MAP full aggregator probe requires matching frame/global block counts");
    }

    const size_t mem_size = 256ull * 1024ull * 1024ull;
    ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    if (!compute_ctx) {
        throw std::runtime_error("failed to create LingBot-MAP full aggregator probe ggml context");
    }

    ggml_context * ctx = compute_ctx.get();
    ggml_tensor * tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.hidden_size, input.tokens_per_frame, input.n_frames);
    ggml_set_name(tokens, "lingbot_map.aggregator.full_probe.input");
    ggml_set_input(tokens);

    ggml_tensor * frame_tokens = nullptr;
    ggml_tensor * global_tokens = nullptr;
    int32_t selected_outputs = 0;
    for (int32_t i = 0; i < cfg.frame_block_count; ++i) {
        frame_tokens = lingbot_apply_aggregator_block(ctx, pimpl_->ggml.get(), cfg, tokens,
                                                      "aggregator.frame_blocks." + std::to_string(i),
                                                      "lingbot_map.aggregator.full_probe.frame." + std::to_string(i),
                                                      nullptr);
        global_tokens = ggml_reshape_3d(ctx, frame_tokens, cfg.hidden_size,
                                        (int64_t) input.tokens_per_frame * input.n_frames, 1);
        global_tokens = lingbot_apply_aggregator_block(ctx, pimpl_->ggml.get(), cfg, global_tokens,
                                                       "aggregator.global_blocks." + std::to_string(i),
                                                       "lingbot_map.aggregator.full_probe.global." + std::to_string(i),
                                                       nullptr);
        tokens = ggml_reshape_3d(ctx, global_tokens, cfg.hidden_size, input.tokens_per_frame, input.n_frames);
        if (std::find(cfg.aggregator_selected_layers.begin(), cfg.aggregator_selected_layers.end(), i) !=
            cfg.aggregator_selected_layers.end()) {
            selected_outputs += 1;
        }
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, frame_tokens);
    ggml_build_forward_expand(graph, global_tokens);

    lingbot_map_full_aggregator_probe_result result;
    result.graph_nodes = ggml_graph_n_nodes(graph);
    result.selected_output_count = selected_outputs;
    result.frame_block_count = cfg.frame_block_count;
    result.global_block_count = cfg.global_block_count;
    for (int i = 0; i < 4; ++i) {
        result.final_frame_ne[i] = (int32_t) frame_tokens->ne[i];
        result.final_global_ne[i] = (int32_t) global_tokens->ne[i];
    }
    return result;
}

lingbot_map_aggregator_graph_result lingbot_map_context::build_aggregator_graph(
        const lingbot_map_aggregator_input & input) const {
    const auto & cfg = config();
    if (input.n_frames <= 0 || input.tokens_per_frame <= 0 || input.hidden_size != cfg.hidden_size) {
        throw std::invalid_argument("Invalid LingBot-MAP aggregator input for graph build");
    }
    if (cfg.frame_block_count <= 0 || cfg.global_block_count <= 0 || cfg.frame_block_count != cfg.global_block_count) {
        throw std::runtime_error("LingBot-MAP aggregator graph requires matching frame/global block counts");
    }

    const size_t mem_size = 256ull * 1024ull * 1024ull;
    ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    if (!compute_ctx) {
        throw std::runtime_error("failed to create LingBot-MAP aggregator graph ggml context");
    }

    ggml_context * ctx = compute_ctx.get();
    ggml_tensor * tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.hidden_size, input.tokens_per_frame, input.n_frames);
    ggml_set_name(tokens, "lingbot_map.aggregator.graph.input");
    ggml_set_input(tokens);

    std::vector<ggml_tensor *> selected_outputs;
    selected_outputs.reserve(cfg.aggregator_selected_layers.size());
    ggml_tensor * frame_tokens = nullptr;
    ggml_tensor * global_tokens = nullptr;

    for (int32_t i = 0; i < cfg.frame_block_count; ++i) {
        frame_tokens = lingbot_apply_aggregator_block(ctx, pimpl_->ggml.get(), cfg, tokens,
                                                      "aggregator.frame_blocks." + std::to_string(i),
                                                      "lingbot_map.aggregator.graph.frame." + std::to_string(i),
                                                      nullptr);
        global_tokens = ggml_reshape_3d(ctx, frame_tokens, cfg.hidden_size,
                                        (int64_t) input.tokens_per_frame * input.n_frames, 1);
        global_tokens = lingbot_apply_aggregator_block(ctx, pimpl_->ggml.get(), cfg, global_tokens,
                                                       "aggregator.global_blocks." + std::to_string(i),
                                                       "lingbot_map.aggregator.graph.global." + std::to_string(i),
                                                       nullptr);
        ggml_tensor * global_as_frame = ggml_reshape_3d(ctx, global_tokens, cfg.hidden_size, input.tokens_per_frame, input.n_frames);

        if (std::find(cfg.aggregator_selected_layers.begin(), cfg.aggregator_selected_layers.end(), i) !=
            cfg.aggregator_selected_layers.end()) {
            ggml_tensor * selected = ggml_concat(ctx, frame_tokens, global_as_frame, 0);
            ggml_set_name(selected, ("lingbot_map.aggregator.graph.selected." + std::to_string(i)).c_str());
            selected_outputs.push_back(selected);
        }

        tokens = global_as_frame;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, frame_tokens);
    ggml_build_forward_expand(graph, global_tokens);
    for (ggml_tensor * selected : selected_outputs) {
        ggml_build_forward_expand(graph, selected);
    }

    lingbot_map_aggregator_graph_result result;
    result.graph_nodes = ggml_graph_n_nodes(graph);
    result.selected_output_count = (int32_t) selected_outputs.size();
    result.frame_block_count = cfg.frame_block_count;
    result.global_block_count = cfg.global_block_count;
    result.tokens_per_frame = input.tokens_per_frame;
    result.patch_start_idx = input.patch_start_idx;
    result.selected_layers = cfg.aggregator_selected_layers;
    for (int i = 0; i < 4; ++i) {
        result.final_frame_ne[i] = (int32_t) frame_tokens->ne[i];
        result.final_global_ne[i] = (int32_t) global_tokens->ne[i];
    }
    result.selected_output_shapes.reserve(selected_outputs.size());
    for (const ggml_tensor * selected : selected_outputs) {
        result.selected_output_shapes.push_back({
            (int32_t) selected->ne[0],
            (int32_t) selected->ne[1],
            (int32_t) selected->ne[2],
            (int32_t) selected->ne[3],
        });
    }
    return result;
}

lingbot_map_camera_head_graph_result lingbot_map_context::build_camera_head_graph(
        const lingbot_map_aggregator_input & input) const {
    const auto & cfg = config();
    if (input.n_frames <= 0 || input.tokens_per_frame <= 0 || input.hidden_size != cfg.hidden_size ||
        cfg.camera_hidden_size <= 0) {
        throw std::invalid_argument("Invalid LingBot-MAP camera_head graph input");
    }
    if (cfg.camera_trunk_block_count <= 0 || cfg.camera_num_iterations <= 0) {
        throw std::runtime_error("LingBot-MAP camera_head graph requires trunk blocks and iterations");
    }

    const size_t mem_size = 256ull * 1024ull * 1024ull;
    ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    if (!compute_ctx) {
        throw std::runtime_error("failed to create LingBot-MAP camera_head graph ggml context");
    }

    ggml_context * weights_ctx = pimpl_->ggml.get();
    ggml_context * ctx = compute_ctx.get();
    const int64_t pose_dim = 9;
    ggml_tensor * aggregated_tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.camera_hidden_size,
                                                         input.tokens_per_frame, input.n_frames);
    ggml_set_name(aggregated_tokens, "lingbot_map.camera_head.graph.aggregated_tokens");
    ggml_set_input(aggregated_tokens);

    ggml_tensor * pose_tokens = ggml_view_3d(ctx, aggregated_tokens,
                                             cfg.camera_hidden_size, input.n_frames, 1,
                                             aggregated_tokens->nb[2],
                                             (size_t) aggregated_tokens->nb[2] * (size_t) input.n_frames,
                                             0);
    pose_tokens = lingbot_layer_norm(ctx, pose_tokens,
                                     lingbot_require_tensor(weights_ctx, "camera_head.token_norm.weight"),
                                     lingbot_require_tensor(weights_ctx, "camera_head.token_norm.bias"),
                                     1e-6f);
    ggml_set_name(pose_tokens, "lingbot_map.camera_head.graph.pose_tokens");

    ggml_tensor * empty_pose = lingbot_require_tensor(weights_ctx, "camera_head.empty_pose_tokens");
    ggml_tensor * pred_pose = nullptr;
    std::vector<ggml_tensor *> iteration_outputs;
    iteration_outputs.reserve(cfg.camera_num_iterations);

    for (int32_t iter = 0; iter < cfg.camera_num_iterations; ++iter) {
        ggml_tensor * module_input = nullptr;
        if (pred_pose == nullptr) {
            ggml_tensor * empty_pose_target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pose_dim, input.n_frames, 1);
            module_input = ggml_repeat(ctx, empty_pose, empty_pose_target);
        } else {
            module_input = pred_pose;
        }

        module_input = lingbot_linear(ctx, module_input,
                                      lingbot_require_tensor(weights_ctx, "camera_head.embed_pose.weight"),
                                      lingbot_require_tensor(weights_ctx, "camera_head.embed_pose.bias"));

        ggml_tensor * modulation = lingbot_linear(ctx,
                                                  ggml_silu(ctx, module_input),
                                                  lingbot_require_tensor(weights_ctx, "camera_head.poseLN_modulation.1.weight"),
                                                  lingbot_require_tensor(weights_ctx, "camera_head.poseLN_modulation.1.bias"));
        ggml_tensor * shift = ggml_view_3d(ctx, modulation, cfg.camera_hidden_size, input.n_frames, 1,
                                           modulation->nb[1], modulation->nb[2], 0);
        ggml_tensor * scale = ggml_view_3d(ctx, modulation, cfg.camera_hidden_size, input.n_frames, 1,
                                           modulation->nb[1], modulation->nb[2],
                                           (size_t) cfg.camera_hidden_size * ggml_type_size(modulation->type));
        ggml_tensor * gate = ggml_view_3d(ctx, modulation, cfg.camera_hidden_size, input.n_frames, 1,
                                          modulation->nb[1], modulation->nb[2],
                                          (size_t) cfg.camera_hidden_size * 2 * ggml_type_size(modulation->type));

        ggml_tensor * adaln = ggml_norm(ctx, pose_tokens, 1e-6f);
        ggml_tensor * scale_cont = ggml_cont(ctx, scale);
        ggml_tensor * modulated = ggml_mul(ctx, adaln, ggml_scale_bias(ctx, scale_cont, 1.0f, 1.0f));
        modulated = ggml_add(ctx, modulated, shift);
        modulated = ggml_mul(ctx, modulated, gate);
        modulated = ggml_add(ctx, modulated, pose_tokens);

        for (int32_t block = 0; block < cfg.camera_trunk_block_count; ++block) {
            modulated = lingbot_apply_camera_trunk_block(ctx, weights_ctx, cfg, modulated,
                                                         "camera_head.trunk." + std::to_string(block),
                                                         "lingbot_map.camera_head.graph.iter." + std::to_string(iter) + ".trunk." + std::to_string(block));
        }

        ggml_tensor * trunk_norm = lingbot_layer_norm(ctx, modulated,
                                                      lingbot_require_tensor(weights_ctx, "camera_head.trunk_norm.weight"),
                                                      lingbot_require_tensor(weights_ctx, "camera_head.trunk_norm.bias"),
                                                      1e-6f);
        ggml_tensor * delta = lingbot_pose_branch(ctx, weights_ctx, trunk_norm);
        pred_pose = pred_pose == nullptr ? delta : ggml_add(ctx, pred_pose, delta);
        ggml_set_name(pred_pose, ("lingbot_map.camera_head.graph.pose_iter." + std::to_string(iter)).c_str());
        ggml_tensor * activated_pose = lingbot_activate_pose(ctx, pred_pose);
        ggml_set_name(activated_pose, ("lingbot_map.camera_head.graph.activated_pose_iter." + std::to_string(iter)).c_str());
        iteration_outputs.push_back(activated_pose);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    for (ggml_tensor * out : iteration_outputs) {
        ggml_build_forward_expand(graph, out);
    }

    lingbot_map_camera_head_graph_result result;
    result.graph_nodes = ggml_graph_n_nodes(graph);
    result.trunk_block_count = cfg.camera_trunk_block_count;
    result.iteration_count = cfg.camera_num_iterations;
    result.pose_dim = (int32_t) pose_dim;
    for (int i = 0; i < 4; ++i) {
        result.input_ne[i] = (int32_t) aggregated_tokens->ne[i];
        result.final_pose_ne[i] = (int32_t) iteration_outputs.back()->ne[i];
    }
    result.iteration_pose_shapes.reserve(iteration_outputs.size());
    for (const ggml_tensor * out : iteration_outputs) {
        result.iteration_pose_shapes.push_back({
            (int32_t) out->ne[0],
            (int32_t) out->ne[1],
            (int32_t) out->ne[2],
            (int32_t) out->ne[3],
        });
    }
    return result;
}


lingbot_map_runtime_result lingbot_map_context::run_aggregator_camera_head(
        const lingbot_map_aggregator_input & input,
        bool                                 prefer_smt) const {
    const auto & cfg = config();
    if (input.n_frames <= 0 || input.tokens_per_frame <= 0 || input.hidden_size != cfg.hidden_size) {
        throw std::invalid_argument("Invalid LingBot-MAP runtime input");
    }
    if (cfg.frame_block_count <= 0 || cfg.global_block_count <= 0 || cfg.frame_block_count != cfg.global_block_count) {
        throw std::runtime_error("LingBot-MAP runtime requires matching frame/global block counts");
    }
    if (cfg.camera_trunk_block_count <= 0 || cfg.camera_num_iterations <= 0) {
        throw std::runtime_error("LingBot-MAP runtime requires camera_head trunk blocks and iterations");
    }

    if (!pimpl_->runtime_initialized || pimpl_->runtime_prefer_smt != prefer_smt) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        if (!backend) {
            throw std::runtime_error("failed to initialize LingBot-MAP GGML CPU/SMT backend");
        }
        ggml_backend_cpu_set_n_threads(backend.get(), cfg.ggml_threads);
        std::cerr << "[LingBot-MAP] GGML CPU backend threads=" << cfg.ggml_threads << "\n";

        ggml_backend_buffer_type_t buft = lingbot_select_cpu_buffer_type(backend.get(), prefer_smt);
        lingbot_map_runtime_weights weights;
        try {
            weights = lingbot_load_runtime_weights(cfg, buft);
        } catch (const std::exception & e) {
            ggml_backend_buffer_type_t default_buft = ggml_backend_get_default_buffer_type(backend.get());
            if (!prefer_smt || buft == default_buft) {
                throw;
            }
            std::cerr << "[LingBot-MAP] failed to allocate/load runtime weights on " << ggml_backend_buft_name(buft)
                      << ", falling back to " << ggml_backend_buft_name(default_buft) << ": " << e.what() << "\n";
            buft = default_buft;
            weights = lingbot_load_runtime_weights(cfg, buft);
        }

        pimpl_->runtime_backend = std::move(backend);
        pimpl_->runtime_buft = buft;
        pimpl_->runtime_weights = std::move(weights);
        pimpl_->runtime_prefer_smt = prefer_smt;
        pimpl_->runtime_initialized = true;
        std::cerr << "[LingBot-MAP] initialized GGML runtime backend=" << ggml_backend_name(pimpl_->runtime_backend.get())
                  << ", buffer_type=" << ggml_backend_buft_name(pimpl_->runtime_buft) << "\n";
    }

    ggml_backend_t backend = pimpl_->runtime_backend.get();
    ggml_backend_buffer_type_t buft = pimpl_->runtime_buft;
    ggml_context * weights_ctx = pimpl_->runtime_weights.ggml.get();

    const size_t mem_size = 512ull * 1024ull * 1024ull;
    ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    if (!compute_ctx) {
        throw std::runtime_error("failed to create LingBot-MAP runtime ggml context");
    }

    auto stage_start = std::chrono::steady_clock::now();
    lingbot_map_runtime_graph runtime_graph = lingbot_build_aggregator_camera_runtime_graph(
            compute_ctx.get(), weights_ctx, cfg, input);
    std::cerr << "[LingBot-MAP][time] ggml_build_graph_ms=" << lingbot_elapsed_ms(stage_start)
              << ", nodes=" << (runtime_graph.graph ? ggml_graph_n_nodes(runtime_graph.graph) : 0) << "\n";
    if (runtime_graph.selected_outputs.empty() || runtime_graph.final_pose == nullptr || runtime_graph.graph == nullptr) {
        throw std::runtime_error("LingBot-MAP runtime graph did not produce required outputs");
    }

    ggml_backend_buffer_type_t default_buft = ggml_backend_get_default_buffer_type(backend);
    const bool primary_graph_supported = lingbot_graph_supported_by_backend(
            backend, buft, runtime_graph.graph, /* log_summary */ buft != default_buft);

    ggml_backend_ptr fallback_backend;
    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_bufts;
    backend_ptrs.push_back(backend);
    backend_bufts.push_back(buft);

    bool using_hybrid_cpu_fallback = false;
    if (!primary_graph_supported && buft != default_buft) {
        if (!ggml_backend_supports_buft(backend, default_buft)) {
            std::cerr << "[LingBot-MAP] primary backend cannot use " << ggml_backend_buft_name(default_buft)
                      << ", falling back to plain CPU scheduler\n";
            buft = default_buft;
            pimpl_->runtime_buft = buft;
            pimpl_->runtime_weights = lingbot_load_runtime_weights(cfg, buft);
            weights_ctx = pimpl_->runtime_weights.ggml.get();
            stage_start = std::chrono::steady_clock::now();
            runtime_graph = lingbot_build_aggregator_camera_runtime_graph(compute_ctx.get(), weights_ctx, cfg, input);
            std::cerr << "[LingBot-MAP][time] ggml_rebuild_graph_after_cpu_fallback_ms=" << lingbot_elapsed_ms(stage_start)
                      << ", nodes=" << (runtime_graph.graph ? ggml_graph_n_nodes(runtime_graph.graph) : 0) << "\n";
            backend_ptrs.clear();
            backend_bufts.clear();
            backend_ptrs.push_back(backend);
            backend_bufts.push_back(default_buft);
        } else {
            fallback_backend.reset(ggml_backend_cpu_init());
            if (!fallback_backend) {
                throw std::runtime_error("failed to initialize LingBot-MAP GGML CPU fallback backend");
            }
            ggml_backend_cpu_set_n_threads(fallback_backend.get(), cfg.ggml_threads);

            pimpl_->runtime_buft = default_buft;
            pimpl_->runtime_weights = lingbot_load_runtime_weights(cfg, default_buft);
            weights_ctx = pimpl_->runtime_weights.ggml.get();
            stage_start = std::chrono::steady_clock::now();
            runtime_graph = lingbot_build_aggregator_camera_runtime_graph(compute_ctx.get(), weights_ctx, cfg, input);
            std::cerr << "[LingBot-MAP][time] ggml_rebuild_graph_for_hybrid_ms=" << lingbot_elapsed_ms(stage_start)
                      << ", nodes=" << (runtime_graph.graph ? ggml_graph_n_nodes(runtime_graph.graph) : 0) << "\n";
            if (runtime_graph.selected_outputs.empty() || runtime_graph.final_pose == nullptr || runtime_graph.graph == nullptr) {
                throw std::runtime_error("LingBot-MAP runtime graph did not produce required outputs after hybrid rebuild");
            }

            backend_ptrs.push_back(fallback_backend.get());
            backend_bufts.push_back(default_buft);
            using_hybrid_cpu_fallback = true;
            std::cerr << "[LingBot-MAP] using hybrid GGML scheduler with CPU-host weights: primary=" << ggml_backend_buft_name(buft)
                      << ", fallback=" << ggml_backend_buft_name(default_buft)
                      << ", threads=" << cfg.ggml_threads << "\n";
        }
    } else if (!primary_graph_supported) {
        throw std::runtime_error("LingBot-MAP GGML runtime graph contains ops unsupported by the selected backend");
    }

    ggml_backend_sched_ptr sched(
            ggml_backend_sched_new(backend_ptrs.data(), backend_bufts.data(), (int) backend_ptrs.size(),
                                   32768, false, true));
    if (!sched) {
        throw std::runtime_error("failed to create LingBot-MAP GGML scheduler");
    }

    ggml_backend_sched_reset(sched.get());
    ggml_backend_t output_backend = fallback_backend ? fallback_backend.get() : backend;
    for (ggml_tensor * selected : runtime_graph.selected_outputs) {
        ggml_backend_sched_set_tensor_backend(sched.get(), selected, output_backend);
    }
    ggml_backend_sched_set_tensor_backend(sched.get(), runtime_graph.final_pose, output_backend);

    stage_start = std::chrono::steady_clock::now();
    if (!ggml_backend_sched_alloc_graph(sched.get(), runtime_graph.graph)) {
        throw std::runtime_error("failed to allocate LingBot-MAP GGML runtime graph");
    }
    std::cerr << "[LingBot-MAP][time] ggml_alloc_graph_ms=" << lingbot_elapsed_ms(stage_start) << "\n";

    const size_t input_nbytes = ggml_nbytes(runtime_graph.input_tokens);
    if (input_nbytes != input.tokens.size() * sizeof(float)) {
        throw std::runtime_error("LingBot-MAP runtime input byte size mismatch");
    }
    ggml_backend_tensor_set(runtime_graph.input_tokens, input.tokens.data(), 0, input_nbytes);

    stage_start = std::chrono::steady_clock::now();
    std::cerr << "[LingBot-MAP][time] ggml_compute_start backend=" << ggml_backend_name(backend)
              << ", buffer_type=" << ggml_backend_buft_name(buft)
              << ", nodes=" << ggml_graph_n_nodes(runtime_graph.graph) << "\n";
    const enum ggml_status status = ggml_backend_sched_graph_compute(sched.get(), runtime_graph.graph);
    std::cerr << "[LingBot-MAP][time] ggml_compute_ms=" << lingbot_elapsed_ms(stage_start)
              << ", status=" << ggml_status_to_string(status) << "\n";
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("LingBot-MAP GGML runtime compute failed: ") + ggml_status_to_string(status));
    }

    lingbot_map_runtime_result result;
    result.backend_name = ggml_backend_name(backend);
    result.buffer_type_name = using_hybrid_cpu_fallback ?
        std::string(ggml_backend_buft_name(buft)) + "+" + ggml_backend_buft_name(default_buft) :
        ggml_backend_buft_name(buft);
    result.graph_nodes = ggml_graph_n_nodes(runtime_graph.graph);
    result.selected_output_count = (int32_t) runtime_graph.selected_outputs.size();
    result.tokens_per_frame = input.tokens_per_frame;
    result.patch_start_idx = input.patch_start_idx;
    result.frame_block_count = cfg.frame_block_count;
    result.global_block_count = cfg.global_block_count;
    result.camera_trunk_block_count = cfg.camera_trunk_block_count;
    result.camera_iteration_count = cfg.camera_num_iterations;
    result.camera_pose_dim = 9;
    result.selected_layers = cfg.aggregator_selected_layers;

    result.selected_output_shapes.reserve(runtime_graph.selected_outputs.size());
    result.selected_outputs.reserve(runtime_graph.selected_outputs.size());
    for (const ggml_tensor * selected : runtime_graph.selected_outputs) {
        if (selected->type != GGML_TYPE_F32) {
            throw std::runtime_error("LingBot-MAP runtime selected output is not F32");
        }
        result.selected_output_shapes.push_back({
            (int32_t) selected->ne[0],
            (int32_t) selected->ne[1],
            (int32_t) selected->ne[2],
            (int32_t) selected->ne[3],
        });
        std::vector<float> output(ggml_nbytes(selected) / sizeof(float));
        ggml_backend_t selected_backend = ggml_backend_sched_get_tensor_backend(sched.get(), const_cast<ggml_tensor *>(selected));
        if (selected_backend == nullptr) {
            throw std::runtime_error("LingBot-MAP selected output has no scheduled backend");
        }
        std::cerr << "[LingBot-MAP][time] reading selected_output index=" << result.selected_outputs.size()
                  << ", bytes=" << ggml_nbytes(selected)
                  << ", backend=" << ggml_backend_name(selected_backend)
                  << "\n";
        ggml_backend_tensor_get_async(selected_backend, selected, output.data(), 0, ggml_nbytes(selected));
        ggml_backend_synchronize(selected_backend);
        result.selected_outputs.push_back(std::move(output));
    }

    result.camera_head_input_shape = {
        (int32_t) runtime_graph.camera_head_input->ne[0],
        (int32_t) runtime_graph.camera_head_input->ne[1],
        (int32_t) runtime_graph.camera_head_input->ne[2],
        (int32_t) runtime_graph.camera_head_input->ne[3],
    };
    result.camera_head_final_pose_shape = {
        (int32_t) runtime_graph.final_pose->ne[0],
        (int32_t) runtime_graph.final_pose->ne[1],
        (int32_t) runtime_graph.final_pose->ne[2],
        (int32_t) runtime_graph.final_pose->ne[3],
    };
    result.camera_head_iteration_pose_shapes.reserve(runtime_graph.iteration_poses.size());
    for (const ggml_tensor * pose : runtime_graph.iteration_poses) {
        result.camera_head_iteration_pose_shapes.push_back({
            (int32_t) pose->ne[0],
            (int32_t) pose->ne[1],
            (int32_t) pose->ne[2],
            (int32_t) pose->ne[3],
        });
    }

    if (runtime_graph.final_pose->type != GGML_TYPE_F32) {
        throw std::runtime_error("LingBot-MAP runtime final pose is not F32");
    }
    result.pose_encoding.resize(ggml_nbytes(runtime_graph.final_pose) / sizeof(float));
    ggml_backend_t pose_backend = ggml_backend_sched_get_tensor_backend(sched.get(), runtime_graph.final_pose);
    if (pose_backend == nullptr) {
        throw std::runtime_error("LingBot-MAP final pose has no scheduled backend");
    }
    std::cerr << "[LingBot-MAP][time] reading final_pose bytes=" << ggml_nbytes(runtime_graph.final_pose)
              << ", backend=" << ggml_backend_name(pose_backend)
              << "\n";
    ggml_backend_tensor_get_async(pose_backend, runtime_graph.final_pose, result.pose_encoding.data(), 0, ggml_nbytes(runtime_graph.final_pose));
    ggml_backend_synchronize(pose_backend);
    return result;
}

