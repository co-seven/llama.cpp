// LingBot-MAP multimodal wrapper.
// Loads mtmd_model/config.json and validates the ONNX/GGUF artifact set.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;

struct lingbot_map_aggregator_input {
    int32_t n_frames = 0;
    int32_t hidden_size = 0;
    int32_t vit_tokens_per_frame = 0;
    int32_t vit_prefix_tokens = 0;
    int32_t patch_tokens = 0;
    int32_t patch_start_idx = 0;
    int32_t tokens_per_frame = 0;
    std::vector<float> tokens;
};


struct lingbot_map_graph_probe_result {
    int32_t graph_nodes = 0;
    int32_t input_tokens_per_frame = 0;
    int32_t output_ne[4] = { 0, 0, 0, 0 };
    int32_t qkv_ne[4] = { 0, 0, 0, 0 };
};

struct lingbot_map_aggregator_probe_result {
    lingbot_map_graph_probe_result frame;
    lingbot_map_graph_probe_result global;
};

struct lingbot_map_full_aggregator_probe_result {
    int32_t graph_nodes = 0;
    int32_t selected_output_count = 0;
    int32_t frame_block_count = 0;
    int32_t global_block_count = 0;
    int32_t final_frame_ne[4] = { 0, 0, 0, 0 };
    int32_t final_global_ne[4] = { 0, 0, 0, 0 };
};

struct lingbot_map_aggregator_graph_result {
    int32_t graph_nodes = 0;
    int32_t selected_output_count = 0;
    int32_t frame_block_count = 0;
    int32_t global_block_count = 0;
    int32_t tokens_per_frame = 0;
    int32_t patch_start_idx = 0;
    int32_t final_frame_ne[4] = { 0, 0, 0, 0 };
    int32_t final_global_ne[4] = { 0, 0, 0, 0 };
    std::vector<int32_t> selected_layers;
    std::vector<std::vector<int32_t>> selected_output_shapes;
};

struct lingbot_map_camera_head_graph_result {
    int32_t graph_nodes = 0;
    int32_t trunk_block_count = 0;
    int32_t iteration_count = 0;
    int32_t pose_dim = 0;
    int32_t input_ne[4] = { 0, 0, 0, 0 };
    int32_t final_pose_ne[4] = { 0, 0, 0, 0 };
    std::vector<std::vector<int32_t>> iteration_pose_shapes;
};

struct lingbot_map_runtime_result {
    std::string backend_name;
    std::string buffer_type_name;
    int32_t graph_nodes = 0;
    int32_t selected_output_count = 0;
    int32_t tokens_per_frame = 0;
    int32_t patch_start_idx = 0;
    int32_t frame_block_count = 0;
    int32_t global_block_count = 0;
    int32_t camera_trunk_block_count = 0;
    int32_t camera_iteration_count = 0;
    int32_t camera_pose_dim = 0;
    std::vector<int32_t> selected_layers;
    std::vector<std::vector<int32_t>> selected_output_shapes;
    std::vector<std::vector<float>> selected_outputs;
    std::vector<int32_t> camera_head_input_shape;
    std::vector<int32_t> camera_head_final_pose_shape;
    std::vector<std::vector<int32_t>> camera_head_iteration_pose_shapes;
    std::vector<float> pose_encoding;
};

struct lingbot_map_config {
    std::vector<std::string> architectures;

    std::string vision_model_path;
    std::string aggregator_camera_model_path;
    std::string depth_model_path;
    std::unordered_map<std::string, std::string> ep_config;

    int32_t image_size = 0;
    int32_t patch_size = 0;
    float   image_mean[3] = { 0.485f, 0.456f, 0.406f };
    float   image_std[3]  = { 0.229f, 0.224f, 0.225f };
    int32_t hidden_size = 0;
    int32_t camera_hidden_size = 0;
    int32_t num_special_tokens = 0;
    int32_t num_register_tokens = 0;
    int32_t frame_block_count = 0;
    int32_t global_block_count = 0;
    int32_t camera_trunk_block_count = 0;
    int32_t camera_num_iterations = 0;
    int32_t ggml_threads = 8;
    std::vector<int32_t> aggregator_selected_layers;

    bool output_pose = true;
    bool output_depth = true;
    bool output_point_cloud = true;
};

struct lingbot_map_context {
    lingbot_map_context(const lingbot_map_context &) = delete;
    lingbot_map_context & operator=(const lingbot_map_context &) = delete;
    ~lingbot_map_context();

    static std::unique_ptr<lingbot_map_context> create(const std::string & config_dir);

    const lingbot_map_config & config() const;
    const std::string & architecture() const;
    int64_t tensor_count() const;

    ggml_context * ggml_ctx() const;
    const ggml_tensor * tensor(const std::string & name) const;

    lingbot_map_aggregator_input build_aggregator_input(
            const float * vit_tokens,
            int32_t       n_frames,
            int32_t       vit_tokens_per_frame,
            int32_t       hidden_size,
            int32_t       image_h,
            int32_t       image_w,
            int32_t       num_frame_for_scale = 1) const;

    lingbot_map_graph_probe_result build_aggregator_frame_block_probe(
            const lingbot_map_aggregator_input & input,
            int32_t                              block_index = 0) const;

    lingbot_map_graph_probe_result build_aggregator_global_block_probe(
            const lingbot_map_aggregator_input & input,
            int32_t                              block_index = 0) const;

    lingbot_map_aggregator_probe_result build_aggregator_block_probes(
            const lingbot_map_aggregator_input & input,
            int32_t                              block_index = 0) const;

    lingbot_map_full_aggregator_probe_result build_full_aggregator_probe(
            const lingbot_map_aggregator_input & input) const;

    lingbot_map_aggregator_graph_result build_aggregator_graph(
            const lingbot_map_aggregator_input & input) const;

    lingbot_map_camera_head_graph_result build_camera_head_graph(
            const lingbot_map_aggregator_input & input) const;

    lingbot_map_runtime_result run_aggregator_camera_head(
            const lingbot_map_aggregator_input & input,
            bool                                 prefer_smt = true) const;

  private:
    lingbot_map_context() = default;
    struct impl;
    std::unique_ptr<impl> pimpl_;
};
