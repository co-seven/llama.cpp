#pragma once

#include "llama.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

enum class server_smt_media_type : uint8_t {
    image = 0,
    audio = 1,
};

struct server_smt_image_chunk {
    std::string id;
    server_smt_media_type type = server_smt_media_type::image;
    std::vector<float> embd;

    int32_t n_tokens = 0;
    int32_t n_pos = 0;
    int32_t grid_nx = 0;
    int32_t grid_ny = 0;

    double t_encode_ms = 0.0; // encoder wall-clock time in ms
};

struct server_smt_vision_context;

#if defined(LLAMA_SERVER_SMT_VISION)
server_smt_vision_context * server_smt_vision_init(
        llama_context * lctx,
        const std::string & config_dir,
        bool warmup = true);

void server_smt_vision_free(server_smt_vision_context * ctx);

bool server_smt_vision_supports_image(const server_smt_vision_context * ctx);
bool server_smt_vision_supports_audio(const server_smt_vision_context * ctx);

server_smt_image_chunk server_smt_vision_encode_media_bin(
        server_smt_vision_context * ctx,
        const std::vector<uint8_t> & data);

server_smt_image_chunk server_smt_vision_encode_image_bin(
        server_smt_vision_context * ctx,
        const std::vector<uint8_t> & data);

int32_t server_smt_vision_decode_chunk(
        llama_context * lctx,
        const server_smt_vision_context * ctx,
        const server_smt_image_chunk & chunk,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last);
#else
inline server_smt_vision_context * server_smt_vision_init(
        llama_context * /* lctx */,
        const std::string & /* config_dir */,
        bool /* warmup */ = true) {
    throw std::runtime_error("SMT media backend is not compiled");
}

inline void server_smt_vision_free(server_smt_vision_context * /* ctx */) {
}

inline bool server_smt_vision_supports_image(const server_smt_vision_context * /* ctx */) {
    return false;
}

inline bool server_smt_vision_supports_audio(const server_smt_vision_context * /* ctx */) {
    return false;
}

inline server_smt_image_chunk server_smt_vision_encode_media_bin(
        server_smt_vision_context * /* ctx */,
        const std::vector<uint8_t> & /* data */) {
    throw std::runtime_error("SMT media backend is not compiled");
}

inline server_smt_image_chunk server_smt_vision_encode_image_bin(
        server_smt_vision_context * /* ctx */,
        const std::vector<uint8_t> & /* data */) {
    throw std::runtime_error("SMT media backend is not compiled");
}

inline int32_t server_smt_vision_decode_chunk(
        llama_context * /* lctx */,
        const server_smt_vision_context * /* ctx */,
        const server_smt_image_chunk & /* chunk */,
        llama_pos & /* n_past */,
        int32_t /* seq_id */,
        int32_t /* n_batch */,
        bool /* logits_last */) {
    return -1;
}
#endif
