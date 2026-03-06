#include "ep-vision-preprocess.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb/stb_image.h"

namespace {

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static bool contains_icase(const std::string & text, const std::string & pattern) {
    return to_lower_ascii(text).find(to_lower_ascii(pattern)) != std::string::npos;
}

struct ep_preproc_spec {
    int32_t target_w = 0;
    int32_t target_h = 0;
    bool normalize_to_01 = false;
    bool quantize_to_u8_after_resize = false;
};

static ep_preproc_spec resolve_preproc_spec(const std::string & architecture) {
    // Qwen3VL EP ONNX keeps internal (x - 127.5) / 127.5 preprocessing.
    if (contains_icase(architecture, "qwen3vl")) {
        return {/* target_w */ 768, /* target_h */ 768, /* normalize_to_01 */ false, /* quantize */ true};
    }

    // FastVLM (LlavaQwen2ForCausalLM) expects 0..1 normalized CHW float32.
    if (contains_icase(architecture, "llavaqwen2forcausallm") ||
        contains_icase(architecture, "llavaqwen2") ||
        contains_icase(architecture, "fastvlm")) {
        return {/* target_w */ 512, /* target_h */ 512, /* normalize_to_01 */ true, /* quantize */ true};
    }

    return {};
}

struct linear_contrib {
    std::vector<int32_t> idx;
    std::vector<float> w;
};

static std::vector<linear_contrib> precompute_linear_contrib(int32_t in_size, int32_t out_size) {
    if (in_size <= 0 || out_size <= 0) {
        throw std::runtime_error("Invalid resize dimensions");
    }

    const float scale = (float) in_size / (float) out_size;
    const float scale_factor = scale > 1.0f ? scale : 1.0f;
    const float support = scale > 1.0f ? scale : 1.0f;

    std::vector<linear_contrib> table((size_t) out_size);
    for (int32_t o = 0; o < out_size; ++o) {
        const float center = ((float) o + 0.5f) * scale - 0.5f;
        const int32_t left = (int32_t) std::ceil(center - support);
        const int32_t right = (int32_t) std::floor(center + support);

        auto & c = table[(size_t) o];
        c.idx.reserve((size_t) std::max(0, right - left + 1));
        c.w.reserve((size_t) std::max(0, right - left + 1));

        float w_sum = 0.0f;
        for (int32_t i = left; i <= right; ++i) {
            const float x = ((float) i - center) / scale_factor;
            const float w = std::max(0.0f, 1.0f - std::fabs(x));
            c.idx.push_back(std::clamp(i, 0, in_size - 1));
            c.w.push_back(w);
            w_sum += w;
        }

        if (w_sum > 0.0f) {
            for (auto & w : c.w) {
                w /= w_sum;
            }
        }
    }

    return table;
}

static std::vector<uint8_t> resize_rgb_u8_antialias(
        const uint8_t * src,
        int32_t src_w,
        int32_t src_h,
        int32_t dst_w,
        int32_t dst_h,
        bool quantize_u8) {
    if (src == nullptr || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        throw std::runtime_error("Invalid image dimensions");
    }

    const auto contrib_x = precompute_linear_contrib(src_w, dst_w);
    const auto contrib_y = precompute_linear_contrib(src_h, dst_h);

    std::vector<float> tmp((size_t) src_h * (size_t) dst_w * 3, 0.0f);
    for (int32_t y = 0; y < src_h; ++y) {
        for (int32_t x = 0; x < dst_w; ++x) {
            const auto & c = contrib_x[(size_t) x];
            for (size_t k = 0; k < c.idx.size(); ++k) {
                const int32_t sx = c.idx[k];
                const float w = c.w[k];
                const size_t s_idx = ((size_t) y * (size_t) src_w + (size_t) sx) * 3;
                const size_t d_idx = ((size_t) y * (size_t) dst_w + (size_t) x) * 3;
                tmp[d_idx + 0] += (float) src[s_idx + 0] * w;
                tmp[d_idx + 1] += (float) src[s_idx + 1] * w;
                tmp[d_idx + 2] += (float) src[s_idx + 2] * w;
            }
        }
    }

    std::vector<uint8_t> out((size_t) dst_h * (size_t) dst_w * 3, 0);
    for (int32_t y = 0; y < dst_h; ++y) {
        const auto & c = contrib_y[(size_t) y];
        for (int32_t x = 0; x < dst_w; ++x) {
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (size_t k = 0; k < c.idx.size(); ++k) {
                const int32_t sy = c.idx[k];
                const float w = c.w[k];
                const size_t s_idx = ((size_t) sy * (size_t) dst_w + (size_t) x) * 3;
                acc[0] += tmp[s_idx + 0] * w;
                acc[1] += tmp[s_idx + 1] * w;
                acc[2] += tmp[s_idx + 2] * w;
            }

            const size_t d_idx = ((size_t) y * (size_t) dst_w + (size_t) x) * 3;
            for (int c_id = 0; c_id < 3; ++c_id) {
                float v = quantize_u8 ? std::round(acc[c_id]) : acc[c_id];
                v = std::clamp(v, 0.0f, 255.0f);
                out[d_idx + (size_t) c_id] = (uint8_t) v;
            }
        }
    }

    return out;
}

static std::vector<float> rgb_u8_to_chw_f32(
        const std::vector<uint8_t> & src,
        int32_t w,
        int32_t h,
        bool normalize_to_01) {
    const size_t plane = (size_t) w * (size_t) h;
    if (src.size() != plane * 3) {
        throw std::runtime_error("Invalid RGB tensor size");
    }

    std::vector<float> out(plane * 3, 0.0f);
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            const size_t src_idx = ((size_t) y * (size_t) w + (size_t) x) * 3;
            const size_t dst_idx = (size_t) y * (size_t) w + (size_t) x;
            for (int32_t c = 0; c < 3; ++c) {
                float v = (float) src[src_idx + (size_t) c];
                if (normalize_to_01) {
                    v /= 255.0f;
                }
                out[(size_t) c * plane + dst_idx] = v;
            }
        }
    }
    return out;
}

static std::vector<uint8_t> pack_f32_bytes(const std::vector<float> & values) {
    if (values.empty()) {
        return {};
    }

    if (values.size() > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error("float tensor is too large");
    }

    std::vector<uint8_t> out(values.size() * sizeof(float), 0);
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

} // namespace

ep_vision_preprocess_result ep_vision_preprocess_if_image(
        const std::vector<uint8_t> & input,
        const std::string & architecture) {
    ep_vision_preprocess_result out;
    if (input.empty()) {
        return out;
    }

    if (input.size() > (size_t) std::numeric_limits<int>::max()) {
        return out;
    }

    int src_w = 0, src_h = 0, src_c = 0;
    uint8_t * pixels = stbi_load_from_memory(
            input.data(),
            (int) input.size(),
            &src_w, &src_h, &src_c,
            /* desired_channels */ 3);
    if (pixels == nullptr) {
        return out;
    }

    try {
        const ep_preproc_spec spec = resolve_preproc_spec(architecture);
        if (spec.target_w <= 0 || spec.target_h <= 0) {
            stbi_image_free(pixels);
            throw std::runtime_error(
                    "EP image preprocessing for architecture '" + architecture +
                    "' is not configured yet; please provide preprocessed .bin");
        }

        const auto resized_u8 = resize_rgb_u8_antialias(
                pixels,
                src_w, src_h,
                spec.target_w, spec.target_h,
                spec.quantize_to_u8_after_resize);
        const auto f32 = rgb_u8_to_chw_f32(
                resized_u8,
                spec.target_w, spec.target_h,
                spec.normalize_to_01);
        stbi_image_free(pixels);

        out.was_image = true;
        out.target_w = spec.target_w;
        out.target_h = spec.target_h;
        out.normalize_to_01 = spec.normalize_to_01;
        out.tensor_bytes = pack_f32_bytes(f32);
        return out;

    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
}
