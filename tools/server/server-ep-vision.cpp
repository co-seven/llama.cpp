#include "server-ep-vision.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(LLAMA_SERVER_SMT_VISION)
#include "../mtmd/ep-vision-wrapper.h"
#include "../mtmd/ep-vision-preprocess.h"
#endif

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

struct server_ep_vision_context {
#if defined(LLAMA_SERVER_SMT_VISION)
    std::unique_ptr<ep_vision_context> ep;
#endif
    std::mutex mu;
    int32_t hidden_size = 0;
    bool use_mrope_pos = false;
    std::vector<llama_token> tok_img_beg;
    std::vector<llama_token> tok_img_end;
    std::string architecture;
};

static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
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

static std::pair<int32_t, int32_t> infer_image_grid_xy(int32_t n_tokens) {
    if (n_tokens <= 0) {
        return {0, 0};
    }

    int32_t best_y = 1;
    int32_t best_x = n_tokens;
    const int32_t root = (int32_t) std::sqrt((double) n_tokens);
    for (int32_t y = root; y >= 1; --y) {
        if (n_tokens % y == 0) {
            best_y = y;
            best_x = n_tokens / y;
            break;
        }
    }
    return {best_x, best_y};
}

static std::vector<llama_token> tokenize_exact_special(llama_context * lctx, const std::string & token_text) {
    auto toks = common_tokenize(lctx, token_text, /* add_special */ false, /* parse_special */ true);
    if (toks.size() != 1) {
        return {};
    }
    if (common_token_to_piece(lctx, toks[0]) != token_text) {
        return {};
    }
    return toks;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>>
detect_image_boundary_tokens_native(llama_context * lctx, const std::string & arch_name) {
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
detect_image_boundary_tokens_auto(llama_context * lctx) {
    static const std::array<std::pair<const char *, const char *>, 8> candidates = {{
        {"<|vision_start|>", "<|vision_end|>"},
        {"<|image_start|>", "<|image_end|>"},
        {"<start_of_image>", "<end_of_image>"},
        {"<img>", "</img>"},
        {"<|begin_of_image|>", "<|end_of_image|>"},
        {"<|IMAGE_START|>", "<|IMAGE_END|>"},
        {"<|im_start|>", "<|im_end|>"},
        {"<image>", "</image>"},
    }};

    for (const auto & candidate : candidates) {
        auto beg = tokenize_exact_special(lctx, candidate.first);
        auto end = tokenize_exact_special(lctx, candidate.second);
        if (!beg.empty() && !end.empty()) {
            return {std::move(beg), std::move(end)};
        }
    }
    return {};
}

enum class ep_image_boundary_mode {
    native,
    auto_detect,
    none,
};

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

    LOG_WRN("[server-smt] unknown MTMD_EP_IMAGE_BOUNDARY='%s', fallback to 'native'\n", env);
    return ep_image_boundary_mode::native;
}

static std::pair<std::vector<llama_token>, std::vector<llama_token>>
resolve_image_boundary_tokens(llama_context * lctx, const std::string & arch_name) {
    const auto mode = ep_image_boundary_mode_from_env();
    if (mode == ep_image_boundary_mode::none) {
        return {};
    }
    if (mode == ep_image_boundary_mode::auto_detect) {
        return detect_image_boundary_tokens_auto(lctx);
    }
    return detect_image_boundary_tokens_native(lctx, arch_name);
}

static std::string write_temp_bin_file(const std::vector<uint8_t> & data) {
#if defined(_WIN32)
    char temp_path[MAX_PATH] = {0};
    char temp_file[MAX_PATH] = {0};

    if (GetTempPathA(MAX_PATH, temp_path) == 0) {
        throw std::runtime_error("failed to get temp path");
    }
    if (GetTempFileNameA(temp_path, "lse", 0, temp_file) == 0) {
        throw std::runtime_error("failed to create temp file");
    }

    FILE * fp = std::fopen(temp_file, "wb");
    if (!fp) {
        throw std::runtime_error("failed to open temp file");
    }
    const size_t n = std::fwrite(data.data(), 1, data.size(), fp);
    std::fclose(fp);
    if (n != data.size()) {
        std::remove(temp_file);
        throw std::runtime_error("failed to write temp file");
    }
    return std::string(temp_file);
#else
    char tmpl[] = "/tmp/llama-server-ep-XXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd < 0) {
        throw std::runtime_error("failed to create temp file");
    }

    size_t written = 0;
    while (written < data.size()) {
        const ssize_t n = write(fd, data.data() + written, data.size() - written);
        if (n <= 0) {
            close(fd);
            std::remove(tmpl);
            throw std::runtime_error("failed to write temp file");
        }
        written += (size_t) n;
    }
    close(fd);
    return std::string(tmpl);
#endif
}

static int decode_tokens(
        llama_context * lctx,
        const std::vector<llama_token> & tokens,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
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
            batch.token[j] = tokens[i];
            batch.pos[j] = n_past + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j] = false;
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

static int decode_embd(
        llama_context * lctx,
        const float * embd,
        int32_t n_tokens,
        int32_t n_embd,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last,
        bool use_mrope_pos,
        int32_t nx,
        int32_t ny) {
    const int n_pos_per_embd = use_mrope_pos ? 4 : 1;

    std::vector<llama_pos> pos((size_t) n_tokens * n_pos_per_embd);
    std::vector<int32_t> n_seq_id(n_tokens);
    std::vector<llama_seq_id> seq_id_0(n_tokens);
    std::vector<llama_seq_id *> seq_ids(n_tokens);
    std::vector<int8_t> logits(n_tokens, 0);

    for (int i = 0; i < n_tokens; ++i) {
        seq_id_0[i] = seq_id;
        seq_ids[i] = &seq_id_0[i];
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
        const int batch_size = std::min(n_batch, n_tokens - processed);
        const bool is_last_batch = processed + batch_size >= n_tokens;

        for (int i = 0; i < batch_size; ++i) {
            if (!use_mrope_pos) {
                pos[processed + i] = n_past + processed + i;
            }
            n_seq_id[processed + i] = 1;
            logits[processed + i] = (logits_last && is_last_batch && i == batch_size - 1);
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
            /* n_tokens  */ batch_size,
            /* token     */ nullptr,
            /* embd      */ const_cast<float *>(embd + (size_t) processed * n_embd),
            /* pos       */ pos_ptr,
            /* n_seq_id  */ n_seq_id.data() + processed,
            /* seq_id    */ seq_ids.data() + processed,
            /* logits    */ logits.data() + processed,
        };

        if (llama_decode(lctx, batch) != 0) {
            return 1;
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

server_ep_vision_context * server_ep_vision_init(
        llama_context * lctx,
        const std::string & config_dir) {
#if defined(LLAMA_SERVER_SMT_VISION)
    auto ctx = std::make_unique<server_ep_vision_context>();
    ctx->ep = ep_vision_context::create(config_dir);
    ctx->architecture = ctx->ep->architecture();
    ctx->hidden_size = (int32_t) ctx->ep->hidden_size();
    ctx->use_mrope_pos = arch_requires_mrope(ctx->architecture);

    auto boundaries = resolve_image_boundary_tokens(lctx, ctx->architecture);
    ctx->tok_img_beg = std::move(boundaries.first);
    ctx->tok_img_end = std::move(boundaries.second);

    LOG_INF("[server-smt] initialized (hidden_size=%d, arch=%s, mrope=%s)\n",
            ctx->hidden_size,
            ctx->architecture.c_str(),
            ctx->use_mrope_pos ? "on" : "off");
    return ctx.release();
#else
    GGML_UNUSED(lctx);
    GGML_UNUSED(config_dir);
    throw std::runtime_error("SMT vision backend is not compiled. Rebuild with LLAMA_SERVER_SMT_VISION=ON.");
#endif
}

void server_ep_vision_free(server_ep_vision_context * ctx) {
    delete ctx;
}

server_ep_image_chunk server_ep_vision_encode_image_bin(
        server_ep_vision_context * ctx,
        const std::vector<uint8_t> & data) {
    if (ctx == nullptr) {
        throw std::runtime_error("SMT context is null");
    }

#if defined(LLAMA_SERVER_SMT_VISION)
    std::lock_guard<std::mutex> lock(ctx->mu);

    std::vector<uint8_t> ep_input = data;
    auto preproc = ep_vision_preprocess_if_image(data, ctx->architecture);
    if (preproc.was_image) {
        ep_input = std::move(preproc.tensor_bytes);
        LOG_INF("[server-smt] preprocessed image for arch=%s -> [1,3,%d,%d] float32 (%s)\n",
                ctx->architecture.c_str(),
                preproc.target_h,
                preproc.target_w,
                preproc.normalize_to_01 ? "0..1" : "0..255");
    }

    const std::string tmp_file = write_temp_bin_file(ep_input);

    server_ep_image_chunk out;
    try {
        out.embd = ctx->ep->encode_image(tmp_file);
        std::remove(tmp_file.c_str());
    } catch (...) {
        std::remove(tmp_file.c_str());
        throw;
    }

    if (ctx->hidden_size <= 0 || out.embd.empty() || out.embd.size() % (size_t) ctx->hidden_size != 0) {
        throw std::runtime_error("Invalid SMT embedding shape");
    }

    const int32_t n_image_tokens = (int32_t) (out.embd.size() / (size_t) ctx->hidden_size);
    auto grid = infer_image_grid_xy(n_image_tokens);

    const int32_t n_pos_img = ctx->use_mrope_pos ? std::max(grid.first, grid.second) : n_image_tokens;
    out.n_tokens = (int32_t) ctx->tok_img_beg.size() + n_image_tokens + (int32_t) ctx->tok_img_end.size();
    out.n_pos = (int32_t) ctx->tok_img_beg.size() + n_pos_img + (int32_t) ctx->tok_img_end.size();
    out.grid_nx = grid.first;
    out.grid_ny = grid.second;
    out.id = fnv_hash(data.data(), data.size());

    return out;
#else
    GGML_UNUSED(data);
    throw std::runtime_error("SMT vision backend is not compiled. Rebuild with LLAMA_SERVER_SMT_VISION=ON.");
#endif
}

int32_t server_ep_vision_decode_chunk(
        llama_context * lctx,
        const server_ep_vision_context * ctx,
        const server_ep_image_chunk & chunk,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last) {
    if (ctx == nullptr) {
        return -1;
    }

    const int32_t n_image_tokens = (int32_t) (chunk.embd.size() / (size_t) ctx->hidden_size);
    if (n_image_tokens <= 0) {
        return -1;
    }

    if (!ctx->tok_img_beg.empty()) {
        if (decode_tokens(lctx, ctx->tok_img_beg, n_past, seq_id, n_batch, false) != 0) {
            return -1;
        }
    }

    const bool logits_on_embd = logits_last && ctx->tok_img_end.empty();
    if (decode_embd(
            lctx,
            chunk.embd.data(),
            n_image_tokens,
            ctx->hidden_size,
            n_past,
            seq_id,
            n_batch,
            logits_on_embd,
            ctx->use_mrope_pos,
            chunk.grid_nx,
            chunk.grid_ny) != 0) {
        return -1;
    }

    if (!ctx->tok_img_end.empty()) {
        if (decode_tokens(lctx, ctx->tok_img_end, n_past, seq_id, n_batch, logits_last) != 0) {
            return -1;
        }
    }

    return 0;
}
