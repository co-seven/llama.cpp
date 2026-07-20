#pragma once

#include "onnxruntime_cxx_api.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
extern const OrtApi * g_ort;
}

namespace smt_media {

int get_ep_thread_num(const std::unordered_map<std::string, std::string> & ep_config,
                      const std::string &                                  key,
                      int                                                  default_value);

bool has_spacemit_ep_affinity(const std::unordered_map<std::string, std::string> & ep_config);

std::unordered_map<std::string, std::string> make_provider_options(
    const std::unordered_map<std::string, std::string> & ep_config,
    int                                                  default_intra_thread_num,
    int                                                  default_inter_thread_num);

bool init_spacemit_execution_provider(Ort::SessionOptions &                                options,
                                      const std::unordered_map<std::string, std::string> & provider_options,
                                      std::string &                                        error_message);

std::vector<const char *> make_name_ptrs(const std::vector<std::string> & names);

std::vector<std::string> get_io_names(Ort::Session & session, bool inputs);

Ort::Value make_tensor_f32(const std::vector<int64_t> & shape, std::vector<float> & data);

Ort::Value make_tensor_bool(const std::vector<int64_t> & shape, std::vector<uint8_t> & data);

std::string read_file_to_string(const std::string & path);

size_t find_closing_brace(const std::string & text, size_t start_pos);

std::string trim_ascii(std::string value);

std::string extract_string_value(const std::string & text, const std::string & key);

int64_t extract_int64_value(const std::string & text, const std::string & key, int64_t default_value);

double extract_double_value(const std::string & text, const std::string & key, double default_value);

std::unordered_map<std::string, std::string> extract_string_map(const std::string & text, const std::string & key);

std::vector<std::string> extract_string_array(const std::string & text, const std::string & key);

std::vector<double> extract_number_array(const std::string & text, const std::string & key);

std::string normalize_path(const std::string & base_dir, const std::string & path);

bool contains_legacy_spacemit_ep_config(const std::string & text);

void warn_legacy_spacemit_ep_config_if_needed(const std::string & text,
                                              const char *        backend_name,
                                              const char *        section_name);

void apply_legacy_spacemit_ep_config(const std::string &                            text,
                                     std::unordered_map<std::string, std::string> & ep_config,
                                     int32_t &                                      intra_thread_num,
                                     int32_t &                                      inter_thread_num);

}  // namespace smt_media
