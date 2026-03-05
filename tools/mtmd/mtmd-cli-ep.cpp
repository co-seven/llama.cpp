// llama-mtmd-cli-ep: Multimodal CLI using Spacemit EP ONNX vision engine
// Based on mtmd-cli.cpp, replaces CLIP/GGUF vision encoding with EP's ONNX vision engine
// LLM inference logic (loading, sampling, generation) is reused from the original mtmd-cli

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "ep-vision-wrapper.h"

#include <vector>
#include <string>
#include <algorithm>
#include <limits.h>
#include <cinttypes>
#include <cctype>
#include <cmath>
#include <utility>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Multimodal CLI with Spacemit EP vision engine\n\n"
        "Usage: %s [options] -m <model> --mmproj <ep_config_dir> --image <image.bin> -p <prompt>\n\n"
        "  -m and --mmproj are required (or set EP_CONFIG_DIR)\n"
        "  --image and -p are optional, if NOT provided, the CLI will run in chat mode\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

// ============================================================
// Prompt chunking utilities
// text/image are evaluated in order, aligned with mtmd chunk strategy
// ============================================================

static const std::string k_media_marker = "<__media__>";
static const std::string k_legacy_image_marker = "<__image__>";

static void replace_all(std::string & s, const std::string & from, const std::string & to);
static std::vector<std::string> split_keep_marker(const std::string & input, const std::string & marker);

enum class ep_chunk_type {
    text,
    image,
};

struct ep_prompt_chunk {
    ep_chunk_type type = ep_chunk_type::text;
    std::vector<llama_token> tokens_text;
    std::string image_path;
};

enum class ep_image_boundary_mode {
    native,      // follow native mtmd model-family rules
    auto_detect, // probe vocab for known token pairs
    none,        // do not inject image boundary tokens
};

enum class ep_media_anchor_mode {
    auto_mode, // apply on known architectures for multi-image prompts
    on,        // always apply for multi-image prompts
    off,       // disable prompt canonicalization
};

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static ep_image_boundary_mode ep_image_boundary_mode_from_env() {
    const char * env = std::getenv("MTMD_EP_IMAGE_BOUNDARY");
    if (env == nullptr || env[0] == '\0') {
        return ep_image_boundary_mode::native;
    }

    const std::string value = to_lower_ascii(env);
    if (value == "native") {
        return ep_image_boundary_mode::native;
    }
    if (value == "auto" || value == "detect") {
        return ep_image_boundary_mode::auto_detect;
    }
    if (value == "none" || value == "off" || value == "0") {
        return ep_image_boundary_mode::none;
    }

    LOG_WRN("[EP-v3] unknown MTMD_EP_IMAGE_BOUNDARY='%s', fallback to 'native'\n", env);
    return ep_image_boundary_mode::native;
}

static const char * ep_image_boundary_mode_name(ep_image_boundary_mode mode) {
    switch (mode) {
        case ep_image_boundary_mode::native:      return "native";
        case ep_image_boundary_mode::auto_detect: return "auto";
        case ep_image_boundary_mode::none:        return "none";
    }
    return "native";
}

static bool contains_icase(const std::string & text, const std::string & pattern) {
    return to_lower_ascii(text).find(to_lower_ascii(pattern)) != std::string::npos;
}

static bool arch_requires_mrope(const std::string & arch_name) {
    return contains_icase(arch_name, "qwen2vl") ||
           contains_icase(arch_name, "qwen2_5_vl") ||
           contains_icase(arch_name, "qwen3vl") ||
           contains_icase(arch_name, "glm4v") ||
           contains_icase(arch_name, "paddleocr");
}

static std::pair<int, int> infer_image_grid_xy(int n_tokens) {
    if (n_tokens <= 0) {
        return {0, 0};
    }

    // Choose the factor pair closest to square to approximate (nx, ny).
    int best_y = 1;
    int best_x = n_tokens;
    int root = (int) std::sqrt((double) n_tokens);
    for (int y = root; y >= 1; --y) {
        if (n_tokens % y == 0) {
            best_y = y;
            best_x = n_tokens / y;
            break;
        }
    }
    return {best_x, best_y};
}

static ep_media_anchor_mode ep_media_anchor_mode_from_env() {
    const char * env = std::getenv("MTMD_EP_MEDIA_ANCHOR");
    if (env == nullptr || env[0] == '\0') {
        return ep_media_anchor_mode::off;
    }
    const std::string value = to_lower_ascii(env);
    if (value == "auto") {
        return ep_media_anchor_mode::auto_mode;
    }
    if (value == "on" || value == "1" || value == "true") {
        return ep_media_anchor_mode::on;
    }
    if (value == "off" || value == "0" || value == "false") {
        return ep_media_anchor_mode::off;
    }
    LOG_WRN("[EP-v3] unknown MTMD_EP_MEDIA_ANCHOR='%s', fallback to 'off'\n", env);
    return ep_media_anchor_mode::off;
}

static const char * ep_media_anchor_mode_name(ep_media_anchor_mode mode) {
    switch (mode) {
        case ep_media_anchor_mode::auto_mode: return "auto";
        case ep_media_anchor_mode::on:        return "on";
        case ep_media_anchor_mode::off:       return "off";
    }
    return "auto";
}

static std::vector<llama_token> tokenize_exact_special(llama_context * lctx, const std::string & token_text) {
    auto toks = common_tokenize(lctx, token_text, /*add_special*/ false, /*parse_special*/ true);
    if (toks.size() != 1) {
        return {};
    }
    // Ensure this is a true special-token hit, not byte-level fallback.
    if (common_token_to_piece(lctx, toks[0]) != token_text) {
        return {};
    }
    return toks;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>> detect_image_boundary_tokens(llama_context * lctx) {
    // Keep order aligned with mtmd.cpp init_vision() common projector families.
    static const std::vector<std::pair<std::string, std::string>> candidates = {
        {"<|vision_start|>", "<|vision_end|>"},
        {"<|image_start|>",  "<|image_end|>"},
        {"<start_of_image>", "<end_of_image>"},
        {"<img>",            "</img>"},
        {"<|begin_of_image|>", "<|end_of_image|>"},
        {"<|IMAGE_START|>",  "<|IMAGE_END|>"},
        {"<|im_start|>",     "<|im_end|>"},
        {"<image>",          "</image>"},
    };

    for (const auto & [beg_s, end_s] : candidates) {
        auto beg = tokenize_exact_special(lctx, beg_s);
        auto end = tokenize_exact_special(lctx, end_s);
        if (!beg.empty() && !end.empty()) {
            return {std::move(beg), std::move(end)};
        }
    }
    return {};
}

static bool should_apply_media_anchor(const std::string & arch_name, size_t n_images, ep_media_anchor_mode mode) {
    if (n_images < 2) {
        return false;
    }
    switch (mode) {
        case ep_media_anchor_mode::off:
            return false;
        case ep_media_anchor_mode::on:
            return true;
        case ep_media_anchor_mode::auto_mode:
            return contains_icase(arch_name, "llavaqwen2forcausallm") || contains_icase(arch_name, "llavaqwen2");
    }
    return false;
}

static std::string canonicalize_multimage_prompt(
        const std::string & prompt,
        size_t n_images,
        const std::string & arch_name,
        ep_media_anchor_mode mode,
        bool & changed) {
    changed = false;
    if (!should_apply_media_anchor(arch_name, n_images, mode)) {
        return prompt;
    }

    std::string normalized = prompt;
    replace_all(normalized, k_legacy_image_marker, k_media_marker);

    const auto parts = split_keep_marker(normalized, k_media_marker);
    size_t marker_count = 0;
    for (const auto & p : parts) {
        if (p == k_media_marker) {
            marker_count++;
        }
    }

    if (marker_count != n_images || marker_count < 2) {
        return normalized;
    }

    std::string output;
    output.reserve(normalized.size() + marker_count * 48 + 64);
    output += "Please align images by order index. ";
    output += "Image #1 maps to the first media slot, Image #2 to the second, and so on.\n";

    size_t image_idx = 0;
    for (const auto & part : parts) {
        if (part == k_media_marker) {
            image_idx++;
            output += "[Image ";
            output += std::to_string(image_idx);
            output += " Begin]\n";
            output += k_media_marker;
            output += "\n[Image ";
            output += std::to_string(image_idx);
            output += " End]\n";
        } else {
            output += part;
        }
    }

    changed = (output != normalized);
    return output;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>>
detect_image_boundary_tokens_native(llama_context * lctx, const std::string & arch_name) {
    // Match native mtmd behavior in mtmd.cpp::init_vision().
    if (contains_icase(arch_name, "qwen2vl") || contains_icase(arch_name, "qwen2_5_vl") ||
        contains_icase(arch_name, "qwen3vl") || contains_icase(arch_name, "youtuvl")) {
        return {
            tokenize_exact_special(lctx, "<|vision_start|>"),
            tokenize_exact_special(lctx, "<|vision_end|>")
        };
    }
    if (contains_icase(arch_name, "llama4")) {
        return {
            tokenize_exact_special(lctx, "<|image_start|>"),
            tokenize_exact_special(lctx, "<|image_end|>")
        };
    }
    if (contains_icase(arch_name, "gemma3")) {
        return {
            tokenize_exact_special(lctx, "<start_of_image>"),
            tokenize_exact_special(lctx, "<end_of_image>")
        };
    }
    if (contains_icase(arch_name, "internvl")) {
        return {
            tokenize_exact_special(lctx, "<img>"),
            tokenize_exact_special(lctx, "</img>")
        };
    }
    if (contains_icase(arch_name, "glm4v")) {
        return {
            tokenize_exact_special(lctx, "<|begin_of_image|>"),
            tokenize_exact_special(lctx, "<|end_of_image|>")
        };
    }
    if (contains_icase(arch_name, "paddleocr")) {
        return {
            tokenize_exact_special(lctx, "<|IMAGE_START|>"),
            tokenize_exact_special(lctx, "<|IMAGE_END|>")
        };
    }
    if (contains_icase(arch_name, "lightonocr")) {
        return {
            tokenize_exact_special(lctx, "<|im_start|>"),
            tokenize_exact_special(lctx, "<|im_end|>")
        };
    }
    return {};
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>>
resolve_image_boundary_tokens(llama_context * lctx,
                              const std::string & arch_name,
                              ep_image_boundary_mode mode) {
    if (mode == ep_image_boundary_mode::none) {
        return {};
    }
    if (mode == ep_image_boundary_mode::auto_detect) {
        return detect_image_boundary_tokens(lctx);
    }
    return detect_image_boundary_tokens_native(lctx, arch_name);
}

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::vector<std::string> split_keep_marker(const std::string & input, const std::string & marker) {
    std::vector<std::string> out;
    if (input.empty()) {
        return out;
    }

    size_t start = 0;
    while (true) {
        size_t pos = input.find(marker, start);
        if (pos == std::string::npos) {
            out.push_back(input.substr(start));
            break;
        }
        if (pos > start) {
            out.push_back(input.substr(start, pos - start));
        }
        out.push_back(marker);
        start = pos + marker.size();
    }
    return out;
}

static void append_text_chunk(std::vector<ep_prompt_chunk> & chunks, std::vector<llama_token> && tokens) {
    if (tokens.empty()) {
        return;
    }
    if (!chunks.empty() && chunks.back().type == ep_chunk_type::text) {
        auto & dst = chunks.back().tokens_text;
        dst.insert(dst.end(), tokens.begin(), tokens.end());
        return;
    }

    ep_prompt_chunk chunk;
    chunk.type = ep_chunk_type::text;
    chunk.tokens_text = std::move(tokens);
    chunks.emplace_back(std::move(chunk));
}

static bool tokenize_to_ep_chunks(
        const llama_vocab * vocab,
        const std::string & formatted_chat,
        bool add_special,
        bool parse_special,
        const std::vector<std::string> & image_paths,
        std::vector<ep_prompt_chunk> & out_chunks) {
    out_chunks.clear();

    std::string input = formatted_chat;
    replace_all(input, k_legacy_image_marker, k_media_marker);

    size_t n_images_used = 0;
    const auto parts = split_keep_marker(input, k_media_marker);
    for (const auto & part : parts) {
        if (part == k_media_marker) {
            if (n_images_used >= image_paths.size()) {
                LOG_ERR("Number of images (%zu) does not match number of media markers in prompt\n", image_paths.size());
                return false;
            }
            ep_prompt_chunk chunk;
            chunk.type = ep_chunk_type::image;
            chunk.image_path = image_paths[n_images_used++];
            out_chunks.emplace_back(std::move(chunk));
        } else {
            append_text_chunk(out_chunks, common_tokenize(vocab, part, /*add_special*/ false, parse_special));
        }
    }

    if (n_images_used != image_paths.size()) {
        LOG_ERR("Number of images (%zu) does not match number of media markers in prompt (%zu)\n",
                image_paths.size(), n_images_used);
        return false;
    }

    if (add_special && llama_vocab_get_add_bos(vocab)) {
        const llama_token bos = llama_vocab_bos(vocab);
        if (bos != LLAMA_TOKEN_NULL) {
            if (!out_chunks.empty() && out_chunks.front().type == ep_chunk_type::text) {
                out_chunks.front().tokens_text.insert(out_chunks.front().tokens_text.begin(), bos);
            } else {
                ep_prompt_chunk bos_chunk;
                bos_chunk.type = ep_chunk_type::text;
                bos_chunk.tokens_text = { bos };
                out_chunks.insert(out_chunks.begin(), std::move(bos_chunk));
            }
        }
    }

    if (add_special && llama_vocab_get_add_eos(vocab)) {
        const llama_token eos = llama_vocab_eos(vocab);
        if (eos != LLAMA_TOKEN_NULL) {
            if (!out_chunks.empty() && out_chunks.back().type == ep_chunk_type::text) {
                out_chunks.back().tokens_text.push_back(eos);
            } else {
                ep_prompt_chunk eos_chunk;
                eos_chunk.type = ep_chunk_type::text;
                eos_chunk.tokens_text = { eos };
                out_chunks.emplace_back(std::move(eos_chunk));
            }
        }
    }

    return true;
}

// ============================================================
// Decode embedding batch into LLM
// (Adapted from mtmd-helper.cpp decode_embd_batch + mtmd_helper_decode_image_chunk)
// Supports both normal positional encoding and M-RoPE positional layout.
// ============================================================

static int decode_embd(llama_context * lctx,
                       float * embd,
                       int n_tokens,
                       int n_embd,
                       llama_pos & n_past,
                       int n_batch,
                       bool logits_last,
                       bool use_mrope_pos,
                       int nx,
                       int ny) {
    const int n_pos_per_embd = use_mrope_pos ? 4 : 1;

    // Allocate auxiliary arrays (managed manually, not via llama_batch_init).
    std::vector<llama_pos>      pos((size_t) n_tokens * n_pos_per_embd);
    std::vector<int32_t>        n_seq_id(n_tokens);
    std::vector<llama_seq_id>   seq_id_0(n_tokens);
    std::vector<llama_seq_id *> seq_ids(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 0);

    for (int i = 0; i < n_tokens; ++i) {
        seq_id_0[i] = 0;
        seq_ids[i]  = &seq_id_0[i];
    }

    if (use_mrope_pos) {
        if (nx > 0 && ny > 0 && nx * ny == n_tokens) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const int i = y * nx + x;
                    pos[(size_t) i] = n_past;
                    pos[(size_t) i + n_tokens] = n_past + y;
                    pos[(size_t) i + 2 * n_tokens] = n_past + x;
                    pos[(size_t) i + 3 * n_tokens] = 0;
                }
            }
        } else {
            // Fallback to 1D M-RoPE-style positions if grid inference is inconsistent.
            for (int i = 0; i < n_tokens; ++i) {
                pos[(size_t) i] = n_past + i;
                pos[(size_t) i + n_tokens] = n_past + i;
                pos[(size_t) i + 2 * n_tokens] = n_past + i;
                pos[(size_t) i + 3 * n_tokens] = 0;
            }
        }
    } else {
        for (int i = 0; i < n_tokens; ++i) {
            pos[(size_t) i] = n_past + i;
        }
    }

    int processed = 0;
    while (processed < n_tokens) {
        int batch_size = std::min(n_batch, n_tokens - processed);
        bool is_last_batch = (processed + batch_size >= n_tokens);

        for (int i = 0; i < batch_size; ++i) {
            if (!use_mrope_pos) {
                pos[processed + i] = n_past + processed + i;
            }
            n_seq_id[processed + i] = 1;
            logits[processed + i]   = (logits_last && is_last_batch && i == batch_size - 1);
        }

        llama_pos * pos_ptr = nullptr;
        std::vector<llama_pos> pos_view;
        if (use_mrope_pos) {
            pos_view.reserve((size_t) batch_size * n_pos_per_embd);
            for (int d = 0; d < n_pos_per_embd; ++d) {
                const size_t src_idx = (size_t) d * n_tokens + processed;
                pos_view.insert(pos_view.end(), pos.data() + src_idx, pos.data() + src_idx + batch_size);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + processed;
        }

        llama_batch batch = {
            /*n_tokens  =*/ batch_size,
            /*token     =*/ nullptr,
            /*embd      =*/ embd + (size_t)processed * n_embd,
            /*pos       =*/ pos_ptr,
            /*n_seq_id  =*/ n_seq_id.data() + processed,
            /*seq_id    =*/ seq_ids.data()  + processed,
            /*logits    =*/ logits.data()   + processed,
        };

        int ret = llama_decode(lctx, batch);
        if (ret != 0) {
            LOG_ERR("Failed to decode embedding batch at offset %d\n", processed);
            return ret;
        }

        processed += batch_size;
    }

    if (use_mrope_pos) {
        n_past += std::max(nx, ny);
    } else {
        n_past += n_tokens;
    }

    return 0;
}

static int decode_tokens(llama_context * lctx,
                         const std::vector<llama_token> & tokens,
                         llama_pos & n_past,
                         int n_batch,
                         bool logits_last) {
    if (tokens.empty()) {
        return 0;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    size_t i = 0;
    while (i < tokens.size()) {
        batch.n_tokens = 0;
        for (; i < tokens.size() && batch.n_tokens < n_batch; ++i) {
            const int32_t j = batch.n_tokens;
            batch.token[j]     = tokens[i];
            batch.pos[j]       = n_past + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = false;
            batch.n_tokens++;
        }

        if (logits_last && i == tokens.size()) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        if (llama_decode(lctx, batch) != 0) {
            llama_batch_free(batch);
            return 1;
        }
        n_past += batch.n_tokens;
    }

    llama_batch_free(batch);
    return 0;
}

// ============================================================
// Context structure
// ============================================================

struct mtmd_cli_ep_context {
    std::unique_ptr<ep_vision_context> ep_ctx;
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    int64_t hidden_size;
    bool use_mrope_pos = false;
    std::vector<llama_token> tok_img_beg;
    std::vector<llama_token> tok_img_end;
    ep_image_boundary_mode img_boundary_mode = ep_image_boundary_mode::native;
    ep_media_anchor_mode media_anchor_mode = ep_media_anchor_mode::off;

    // Pending image binary paths
    std::vector<std::string> pending_images;

    // Chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;

    // Legacy template antiprompt
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;

    mtmd_cli_ep_context(common_params & params, const std::string & ep_config_dir)
        : llama_init(common_init_from_params(params))
    {
        model = llama_init->model();
        lctx  = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl  = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;

        if (!model || !lctx) {
            LOG_ERR("Failed to initialize LLM model\n");
            exit(1);
        }

        // Chat template
        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();

        // Initialize EP vision context
        ep_ctx = ep_vision_context::create(ep_config_dir);
        hidden_size = ep_ctx->hidden_size();
        use_mrope_pos = arch_requires_mrope(ep_ctx->architecture());
        img_boundary_mode = ep_image_boundary_mode_from_env();
        media_anchor_mode = ep_media_anchor_mode_from_env();
        if (hidden_size <= 0 || hidden_size > INT_MAX) {
            LOG_ERR("FATAL: invalid EP hidden_size (%" PRId64 ")\n", hidden_size);
            exit(1);
        }

        // Validate n_embd matches
        int model_n_embd = llama_model_n_embd(model);
        LOG_INF("[EP-v3] EP vision engine initialized (hidden_size=%" PRId64 ", model_n_embd=%d, arch=%s)\n",
                hidden_size, model_n_embd, ep_ctx->architecture().c_str());
        if (model_n_embd != hidden_size) {
            LOG_ERR("FATAL: model n_embd (%d) != EP hidden_size (%" PRId64 ")\n", model_n_embd, hidden_size);
            exit(1);
        }

        // Align image boundary tokens with native mtmd behavior by default.
        auto boundaries = resolve_image_boundary_tokens(lctx, ep_ctx->architecture(), img_boundary_mode);
        tok_img_beg = std::move(boundaries.first);
        tok_img_end = std::move(boundaries.second);
        LOG_INF("[EP-v3] image boundary mode: %s\n", ep_image_boundary_mode_name(img_boundary_mode));
        LOG_INF("[EP-v3] media anchor mode: %s\n", ep_media_anchor_mode_name(media_anchor_mode));
        LOG_INF("[EP-v3] mrope decode mode: %s\n", use_mrope_pos ? "enabled" : "disabled");
        if (!tok_img_beg.empty() && !tok_img_end.empty() &&
            tok_img_beg.front() != LLAMA_TOKEN_NULL && tok_img_end.front() != LLAMA_TOKEN_NULL) {
            LOG_INF("[EP-v3] image boundary tokens enabled: beg='%s', end='%s'\n",
                common_token_to_piece(lctx, tok_img_beg.front()).c_str(),
                common_token_to_piece(lctx, tok_img_end.front()).c_str());
        } else {
            LOG_INF("[EP-v3] image boundary tokens disabled for this model (arch=%s)\n",
                ep_ctx->architecture().c_str());
            tok_img_beg.clear();
            tok_img_end.clear();
        }

        // Antiprompt for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_ep_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    void add_image(const std::string & binary_path) {
        pending_images.push_back(binary_path);
    }
};

// ============================================================
// Generate response (reused from mtmd-cli.cpp)
// ============================================================

static int generate_response(mtmd_cli_ep_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break;
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        // Eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

// ============================================================
// Chat formatting (reused from mtmd-cli.cpp)
// ============================================================

static std::string chat_add_and_format(mtmd_cli_ep_context & ctx, common_chat_msg & new_msg) {
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

// ============================================================
// Eval message - core multimodal processing
// ============================================================

static int eval_message_ep(mtmd_cli_ep_context & ctx, common_chat_msg & msg) {
    bool add_bos = ctx.chat_history.empty();

    if (msg.role == "user" && !ctx.pending_images.empty()) {
        bool changed = false;
        msg.content = canonicalize_multimage_prompt(
            msg.content,
            ctx.pending_images.size(),
            ctx.ep_ctx->architecture(),
            ctx.media_anchor_mode,
            changed);
        if (changed) {
            LOG_INF("[EP-v3] media-anchor canonicalization applied (%zu images)\n", ctx.pending_images.size());
        }
    }

    auto formatted_chat = chat_add_and_format(ctx, msg);

    if (g_is_interrupted) return 0;

    std::vector<ep_prompt_chunk> chunks;
    if (!tokenize_to_ep_chunks(ctx.vocab, formatted_chat, add_bos, true, ctx.pending_images, chunks)) {
        return 1;
    }

    for (size_t i = 0; i < chunks.size(); ++i) {
        const bool logits_last = (i == chunks.size() - 1);
        const auto & chunk = chunks[i];
        if (chunk.type == ep_chunk_type::text) {
            if (decode_tokens(ctx.lctx, chunk.tokens_text, ctx.n_past, ctx.n_batch, logits_last) != 0) {
                LOG_ERR("Failed to decode text chunk %zu\n", i);
                return 1;
            }
            continue;
        }

        // image chunk
        LOG_INF("Encoding image chunk %zu with EP vision engine: %s\n", i, chunk.image_path.c_str());
        std::vector<float> image_embd = ctx.ep_ctx->encode_image(chunk.image_path);
        if (image_embd.empty() || image_embd.size() % (size_t)ctx.hidden_size != 0) {
            LOG_ERR("Invalid image embedding shape from EP (size=%zu, hidden_size=%" PRId64 ")\n",
                    image_embd.size(), ctx.hidden_size);
            return 1;
        }

        const int n_image_tokens = (int)(image_embd.size() / (size_t)ctx.hidden_size);
        int grid_nx = n_image_tokens;
        int grid_ny = 1;
        if (ctx.use_mrope_pos) {
            auto grid_xy = infer_image_grid_xy(n_image_tokens);
            grid_nx = grid_xy.first;
            grid_ny = grid_xy.second;
            LOG_INF("[EP-v3] inferred image token grid: nx=%d, ny=%d, n_tokens=%d\n",
                    grid_nx, grid_ny, n_image_tokens);
        }

        if (!ctx.tok_img_beg.empty()) {
            if (decode_tokens(ctx.lctx, ctx.tok_img_beg, ctx.n_past, ctx.n_batch, /*logits_last*/ false) != 0) {
                LOG_ERR("Failed to decode image-begin token chunk %zu\n", i);
                return 1;
            }
        }

        const bool logits_on_embd = logits_last && ctx.tok_img_end.empty();
        if (decode_embd(ctx.lctx, image_embd.data(), n_image_tokens,
                        (int)ctx.hidden_size, ctx.n_past, ctx.n_batch, logits_on_embd,
                        ctx.use_mrope_pos, grid_nx, grid_ny) != 0) {
            LOG_ERR("Failed to decode image chunk %zu\n", i);
            return 1;
        }

        if (!ctx.tok_img_end.empty()) {
            if (decode_tokens(ctx.lctx, ctx.tok_img_end, ctx.n_past, ctx.n_batch, logits_last) != 0) {
                LOG_ERR("Failed to decode image-end token chunk %zu\n", i);
                return 1;
            }
        }
    }

    ctx.pending_images.clear();

    LOG("\n");
    return 0;
}

// ============================================================
// Main function
// ============================================================

int main(int argc, char ** argv) {
    ggml_time_init();
    LOG_INF("MTMD_CLI_EP_BUILD_TAG: fastvlm-support-20260303-3 (%s %s)\n", __DATE__, __TIME__);

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    common_init();

    // Use --mmproj as EP config directory path
    std::string ep_config_dir = params.mmproj.path;
    if (ep_config_dir.empty()) {
        // Fallback: check environment variable
        const char * env = std::getenv("EP_CONFIG_DIR");
        if (env) {
            ep_config_dir = env;
        }
    }
    if (ep_config_dir.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing EP config directory (pass via --mmproj or EP_CONFIG_DIR env)\n");
        return 1;
    }

    mtmd_cli_ep_context ctx(params, ep_config_dir);

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();
    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    // Evaluate system prompt if present
    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }
        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        return eval_message_ep(ctx, msg);
    };

    LOG_WRN("Multimodal CLI with Spacemit EP vision engine\n");

    if (eval_system_prompt_if_present()) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;

        // Insert <__media__> marker if not present (same logic as mtmd-cli.cpp)
        if (params.prompt.find("<__media__>") == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); i++) {
                params.prompt = std::string("<__media__>") + params.prompt;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;

        for (const auto & image : params.image) {
            ctx.add_image(image);
        }

        if (eval_message_ep(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }
    } else {
        // Chat mode
        LOG("\n Running in chat mode (EP vision), available commands:");
        LOG("\n   /image <path>    load a preprocessed image binary");
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;

        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) {
                    return 1;
                }
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;

            bool is_image = line == "/image" || line.find("/image ") == 0;
            if (is_image) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing image filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                ctx.add_image(media_path);
                LOG("%s image binary loaded\n", media_path.c_str());
                content += "<__media__>";
                continue;
            } else {
                content += line;
            }

            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = eval_message_ep(ctx, msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (generate_response(ctx, n_predict)) {
                return 1;
            }
            content.clear();
        }
    }

    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}
