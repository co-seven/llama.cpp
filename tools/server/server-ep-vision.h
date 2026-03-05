#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct server_ep_image_chunk {
    std::string id;
    std::vector<float> embd;

    int32_t n_tokens = 0;
    int32_t n_pos = 0;
    int32_t grid_nx = 0;
    int32_t grid_ny = 0;
};

struct server_ep_vision_context;

server_ep_vision_context * server_ep_vision_init(
        llama_context * lctx,
        const std::string & config_dir);

void server_ep_vision_free(server_ep_vision_context * ctx);

server_ep_image_chunk server_ep_vision_encode_image_bin(
        server_ep_vision_context * ctx,
        const std::vector<uint8_t> & data);

int32_t server_ep_vision_decode_chunk(
        llama_context * lctx,
        const server_ep_vision_context * ctx,
        const server_ep_image_chunk & chunk,
        llama_pos & n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last);
