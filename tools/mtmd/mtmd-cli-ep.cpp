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
#include <fstream>
#include <limits.h>
#include <cinttypes>

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
        "Usage: %s [options] -m <model> --ep-config <config_dir> --image <image.bin> -p <prompt>\n\n"
        "  -m and --ep-config are required\n"
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
// Token embedding loading and text encoding utilities
// (Adapted from EP's SpineLLMEngine::EncodeUserText)
// ============================================================

static bool load_token_embeddings(const std::string & path,
                                  int64_t vocab_size,
                                  int64_t hidden_size,
                                  std::vector<float> & out) {
    const size_t total = vocab_size * hidden_size;
    out.resize(total);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open token embedding file: %s\n", path.c_str());
        return false;
    }
    file.read(reinterpret_cast<char *>(out.data()), total * sizeof(float));
    if (!file) {
        LOG_ERR("Token embedding file incomplete: %s\n", path.c_str());
        return false;
    }
    return true;
}

// Split text at <image> marker, tokenize both parts, look up token embeddings
struct text_encode_result {
    std::vector<float> embeddings;
    int n_before = 0;
    int n_after  = 0;
};

static text_encode_result encode_user_text(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_bos,
        const std::vector<float> & token_embeddings,
        int64_t hidden_size) {
    text_encode_result result;

    // Split at <image>
    std::string before_image, after_image;
    size_t image_pos = text.find("<image>");
    if (image_pos != std::string::npos) {
        before_image = text.substr(0, image_pos);
        size_t after_pos = image_pos + 7;
        if (after_pos < text.length() && text[after_pos] == '\n') {
            after_pos++;
        }
        after_image = text.substr(after_pos);
    } else {
        before_image = text;
    }

    // Tokenize before part
    std::vector<llama_token> before_tokens(before_image.size() + 64);
    int n_before = llama_tokenize(vocab, before_image.c_str(),
        static_cast<int>(before_image.length()),
        before_tokens.data(), static_cast<int>(before_tokens.size()),
        add_bos, true);
    if (n_before < 0) {
        before_tokens.resize(-n_before);
        n_before = llama_tokenize(vocab, before_image.c_str(),
            static_cast<int>(before_image.length()),
            before_tokens.data(), static_cast<int>(before_tokens.size()),
            add_bos, true);
    }
    before_tokens.resize(n_before);

    // Tokenize after part
    std::vector<llama_token> after_tokens;
    int n_after = 0;
    if (!after_image.empty()) {
        after_tokens.resize(after_image.size() + 64);
        n_after = llama_tokenize(vocab, after_image.c_str(),
            static_cast<int>(after_image.length()),
            after_tokens.data(), static_cast<int>(after_tokens.size()),
            false, true);
        if (n_after < 0) {
            after_tokens.resize(-n_after);
            n_after = llama_tokenize(vocab, after_image.c_str(),
                static_cast<int>(after_image.length()),
                after_tokens.data(), static_cast<int>(after_tokens.size()),
                false, true);
        }
        after_tokens.resize(n_after);
    }

    result.n_before = n_before;
    result.n_after  = n_after;

    // Look up token embedding matrix
    int total = n_before + n_after;
    result.embeddings.resize(total * hidden_size);

    for (int i = 0; i < n_before; ++i) {
        const float * src = &token_embeddings[before_tokens[i] * hidden_size];
        float * dst = &result.embeddings[i * hidden_size];
        std::copy_n(src, hidden_size, dst);
    }
    for (int i = 0; i < n_after; ++i) {
        const float * src = &token_embeddings[after_tokens[i] * hidden_size];
        float * dst = &result.embeddings[(n_before + i) * hidden_size];
        std::copy_n(src, hidden_size, dst);
    }

    return result;
}

// ============================================================
// Multimodal embedding fusion
// [before_text_embd] + [image_embd] + [after_text_embd]
// (Adapted from EP's SpineLLMEngine::MultimodalEmbeddingFusion)
// ============================================================

static std::vector<float> fuse_multimodal_embeddings(
        const std::vector<float> & image_embd,
        const text_encode_result & text_result,
        int64_t hidden_size) {
    size_t n_image  = image_embd.size() / hidden_size;
    size_t n_before = text_result.n_before;
    size_t n_after  = text_result.n_after;
    size_t total    = n_before + n_image + n_after;

    std::vector<float> fused(total * hidden_size);
    size_t offset = 0;

    // [before_text_embd]
    std::copy_n(text_result.embeddings.data(),
                n_before * hidden_size,
                fused.data() + offset);
    offset += n_before * hidden_size;

    // [image_embd]
    std::copy(image_embd.begin(), image_embd.end(),
              fused.data() + offset);
    offset += image_embd.size();

    // [after_text_embd]
    if (n_after > 0) {
        std::copy_n(text_result.embeddings.data() + n_before * hidden_size,
                    n_after * hidden_size,
                    fused.data() + offset);
    }

    LOG_INF("Multimodal fusion: [text:%zu] + [image:%zu] + [text:%zu] = %zu tokens\n",
            n_before, n_image, n_after, total);
    return fused;
}

// ============================================================
// Decode embedding batch into LLM
// (Adapted from mtmd-helper.cpp decode_embd_batch + mtmd_helper_decode_image_chunk)
// Uses normal positional encoding (no M-RoPE, as EP currently only supports LLaVA-Qwen2)
// ============================================================

static int decode_embd(llama_context * lctx,
                       float * embd,
                       int n_tokens,
                       int n_embd,
                       llama_pos & n_past,
                       int n_batch,
                       bool logits_last) {
    // Allocate auxiliary arrays (managed manually, not via llama_batch_init)
    // This follows the same pattern as mtmd-helper.cpp's decode_embd_batch
    std::vector<llama_pos>      pos(n_tokens);
    std::vector<int32_t>        n_seq_id(n_tokens);
    std::vector<llama_seq_id>   seq_id_0(n_tokens);
    std::vector<llama_seq_id *> seq_ids(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 0);

    for (int i = 0; i < n_tokens; ++i) {
        seq_id_0[i] = 0;
        seq_ids[i]  = &seq_id_0[i];
    }

    int processed = 0;
    while (processed < n_tokens) {
        int batch_size = std::min(n_batch, n_tokens - processed);
        bool is_last_batch = (processed + batch_size >= n_tokens);

        for (int i = 0; i < batch_size; ++i) {
            pos[processed + i]      = n_past + i;
            n_seq_id[processed + i] = 1;
            logits[processed + i]   = (logits_last && is_last_batch && i == batch_size - 1);
        }

        llama_batch batch = {
            /*n_tokens  =*/ batch_size,
            /*token     =*/ nullptr,
            /*embd      =*/ embd + (size_t)processed * n_embd,
            /*pos       =*/ pos.data()      + processed,
            /*n_seq_id  =*/ n_seq_id.data() + processed,
            /*seq_id    =*/ seq_ids.data()  + processed,
            /*logits    =*/ logits.data()   + processed,
        };

        int ret = llama_decode(lctx, batch);
        if (ret != 0) {
            LOG_ERR("Failed to decode embedding batch at offset %d\n", processed);
            return ret;
        }

        n_past += batch_size;
        processed += batch_size;
    }
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

    // Token embedding matrix for embedding-level fusion
    std::vector<float> token_embeddings;
    int64_t hidden_size;

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

        // Load token embeddings
        if (!load_token_embeddings(ep_ctx->token_embedding_path(),
                                   ep_ctx->vocab_size(),
                                   hidden_size,
                                   token_embeddings)) {
            LOG_ERR("Failed to load token embeddings\n");
            exit(1);
        }

        // Validate n_embd matches
        int model_n_embd = llama_model_n_embd(model);
        LOG_INF("[EP-v2] EP vision engine initialized (hidden_size=%" PRId64 ", model_n_embd=%d, arch=%s)\n",
                hidden_size, model_n_embd, ep_ctx->architecture().c_str());
        if (model_n_embd != hidden_size) {
            LOG_ERR("FATAL: model n_embd (%d) != EP hidden_size (%" PRId64 ")\n", model_n_embd, hidden_size);
            exit(1);
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
    auto formatted_chat = chat_add_and_format(ctx, msg);
    bool has_image = !ctx.pending_images.empty();

    if (g_is_interrupted) return 0;

    if (has_image) {
        // === Multimodal path: embedding-level fusion ===

        // 1. Replace <__media__> with <image> (EP uses <image> as split marker)
        std::string ep_text = formatted_chat;
        const std::string mtmd_marker = "<__media__>";
        const std::string ep_marker = "<image>";
        size_t pos = 0;
        while ((pos = ep_text.find(mtmd_marker, pos)) != std::string::npos) {
            ep_text.replace(pos, mtmd_marker.length(), ep_marker);
            pos += ep_marker.length();
        }

        // 2. Encode image using EP ONNX vision engine
        LOG_INF("Encoding image with EP vision engine...\n");
        std::vector<float> image_embd = ctx.ep_ctx->encode_image(ctx.pending_images[0]);
        LOG_INF("Image encoded: %zu floats (%zu tokens)\n",
                image_embd.size(), image_embd.size() / ctx.hidden_size);

        // 3. Encode text (split at <image>, tokenize, look up embeddings)
        auto text_result = encode_user_text(
            ctx.vocab, ep_text, add_bos,
            ctx.token_embeddings, ctx.hidden_size);

        // 4. Fuse embeddings
        auto fused = fuse_multimodal_embeddings(image_embd, text_result, ctx.hidden_size);
        int n_fused_tokens = fused.size() / ctx.hidden_size;

        // 5. Decode fused embeddings into LLM
        int ret = decode_embd(ctx.lctx, fused.data(), n_fused_tokens,
                              ctx.hidden_size, ctx.n_past, ctx.n_batch, true);
        if (ret != 0) {
            LOG_ERR("Failed to decode fused embeddings\n");
            return 1;
        }

        ctx.pending_images.clear();
    } else {
        // === Text-only path ===
        auto tokens = common_tokenize(ctx.lctx, formatted_chat, add_bos, true);

        // Decode tokens in batches
        llama_batch text_batch = llama_batch_init(ctx.n_batch, 0, 1);
        size_t i = 0;
        while (i < tokens.size()) {
            text_batch.n_tokens = 0;
            for (; i < tokens.size() && text_batch.n_tokens < ctx.n_batch; i++) {
                int32_t j = text_batch.n_tokens;
                text_batch.token[j]       = tokens[i];
                text_batch.pos[j]         = ctx.n_past++;
                text_batch.n_seq_id[j]    = 1;
                text_batch.seq_id[j][0]   = 0;
                text_batch.logits[j]      = (i == tokens.size() - 1);
                text_batch.n_tokens++;
            }
            if (llama_decode(ctx.lctx, text_batch) != 0) {
                LOG_ERR("Failed to decode text\n");
                llama_batch_free(text_batch);
                return 1;
            }
        }
        llama_batch_free(text_batch);
    }

    LOG("\n");
    return 0;
}

// ============================================================
// Main function
// ============================================================

int main(int argc, char ** argv) {
    ggml_time_init();

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
        LOG_ERR("ERR: Missing --ep-config argument (pass via --mmproj or EP_CONFIG_DIR env)\n");
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
