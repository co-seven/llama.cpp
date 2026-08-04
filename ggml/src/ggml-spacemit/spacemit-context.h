#pragma once

#include "ggml.h"

#include <spert.hpp>

#include <cstddef>
#include <cstdint>

namespace ggml::spacemit {

inline constexpr size_t cache_line_size_f32 = 64 / sizeof(float);

struct context {
    spert::Context &        runtime;
    uint32_t                ith;
    uint32_t                nth;
    void *                  workspace;
    size_t                  workspace_size;
    spert::SharedBufferView shared;

    void sync() { runtime.sync(); }
};

class tensor_traits_base {
  public:
    virtual ~tensor_traits_base() = default;

    virtual bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const = 0;
    virtual bool compute_forward(context & ctx, ggml_tensor * op) const                 = 0;
    virtual int  repack(ggml_tensor * tensor, const void * data, size_t size) const     = 0;
};

}  // namespace ggml::spacemit
