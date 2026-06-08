#include "models.h"

#include <algorithm>
#include <stdexcept>
#include <string>

void llama_model_lingbot_map::load_arch_hparams(llama_model_loader & ml) {
    std::string component;
    uint32_t embed_dim = 0;
    uint32_t camera_dim = 0;
    uint32_t frame_blocks = 0;
    uint32_t global_blocks = 0;
    uint32_t camera_blocks = 0;

    ml.get_key("lingbot-map.component", component);
    ml.get_key("lingbot-map.embed_dim", embed_dim);
    ml.get_key("lingbot-map.camera_dim", camera_dim);
    ml.get_key("lingbot-map.aggregator_frame_block_count", frame_blocks);
    ml.get_key("lingbot-map.aggregator_global_block_count", global_blocks);
    ml.get_key("lingbot-map.camera_trunk_block_count", camera_blocks);

    if (component != "aggregator_camera_head") {
        throw std::runtime_error("unsupported LingBot-MAP GGUF component: " + component);
    }
    if (embed_dim == 0 || camera_dim == 0 || frame_blocks == 0 || global_blocks == 0 || camera_blocks == 0) {
        throw std::runtime_error("invalid LingBot-MAP GGUF metadata");
    }

    type = LLM_TYPE_UNKNOWN;
    hparams.n_ctx_train = 0;
    hparams.n_embd = std::max(embed_dim, camera_dim);
    hparams.n_layer = frame_blocks + global_blocks + camera_blocks;
    hparams.n_expert = 0;
    hparams.n_expert_used = 0;
    hparams.causal_attn = false;
    hparams.f_norm_eps = 1e-6f;
    hparams.f_norm_rms_eps = 0.0f;
    hparams.rope_freq_base_train = 0.0f;
    hparams.rope_freq_scale_train = 1.0f;
    hparams.rope_type = LLAMA_ROPE_TYPE_NONE;

    const uint32_t n_heads = 16;
    const uint32_t n_layers = std::min<uint32_t>(hparams.n_layer, LLAMA_MAX_LAYERS);
    std::fill(hparams.n_head_arr.begin(), hparams.n_head_arr.end(), 0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(), hparams.n_ff_arr.end(), 0);
    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.swa_layers.begin(), hparams.swa_layers.end(), 0);
    for (uint32_t il = 0; il < n_layers; ++il) {
        hparams.n_head_arr[il] = n_heads;
        hparams.n_head_kv_arr[il] = n_heads;
        hparams.n_ff_arr[il] = hparams.n_embd * 4;
    }
    hparams.n_embd_head_k_full = hparams.n_embd / n_heads;
    hparams.n_embd_head_v_full = hparams.n_embd / n_heads;
    hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
    hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
    hparams.n_rot_full = 0;
    hparams.n_rot_swa = 0;
}

void llama_model_lingbot_map::load_arch_tensors(llama_model_loader &) {
    throw std::runtime_error("LingBot-MAP GGUF tensors are loaded by the mtmd SMT wrapper, not llama_model");
}

std::unique_ptr<llm_graph_context> llama_model_lingbot_map::build_arch_graph(const llm_graph_params &) const {
    throw std::runtime_error("LingBot-MAP does not support llama_model text graph execution");
}
