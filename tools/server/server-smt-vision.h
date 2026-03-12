#pragma once

#include "llama.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct server_smt_image_chunk {
    std::string id;
    std::vector<float> embd;

    int32_t n_tokens = 0;
    int32_t n_pos = 0;
    int32_t grid_nx = 0;
    int32_t grid_ny = 0;
};

struct server_smt_vision_context;

#if defined(LLAMA_SERVER_SMT_VISION)
server_smt_vision_context * server_smt_vision_init(
        llama_context * lctx,
        const std::string & config_dir);

void server_smt_vision_free(server_smt_vision_context * ctx);

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
        const std::string & /* config_dir */) {
    throw std::runtime_error("SMT vision backend is not compiled");
}

inline void server_smt_vision_free(server_smt_vision_context * /* ctx */) {
}

inline server_smt_image_chunk server_smt_vision_encode_image_bin(
        server_smt_vision_context * /* ctx */,
        const std::vector<uint8_t> & /* data */) {
    throw std::runtime_error("SMT vision backend is not compiled");
}

inline int32_t server_smt_vision_decode_chunk(
        llama_context * /* lctx */,
        const server_smt_vision_context * /* ctx */,
        const server_smt_image_chunk & /* chunk */,
        llama_pos & /* n_past */,
        int32_t /* seq_id */,
        int32_t /* n_batch */,
        bool /* logits_last */) {
    return 1;
}
#endif
