#ifndef SPACEMIT_KERNELS_H
#define SPACEMIT_KERNELS_H

#include "ggml.h"
#include "ggml-common.h"

#include <cstddef>
#include <cstdint>

// Forward declarations for kernel dispatch.
// Actual kernel implementations live in ime1_kernels.cpp, ime2_kernels.cpp,
// rvv_kernels.cpp, and repack.cpp. The backend dispatches to them based on
// tensor type and hardware capabilities (IME1, IME2, or RVV fallback).

namespace ggml::spacemit {

// Select the optimal repack type for a weight tensor.
// Returns a pointer to a tensor_traits describing the repack layout,
// or nullptr if no repack is needed.
const void * get_optimal_repack_type(const ggml_tensor * t);

// Dispatch a single fused or unfused opnode to the appropriate kernel.
// Returns GGML_STATUS_SUCCESS on success.
enum ggml_status dispatch_op(const struct ggml_tensor * node,
                             const std::vector<struct ggml_tensor *> & fused,
                             int num_cores);

}  // namespace ggml::spacemit

#endif // SPACEMIT_KERNELS_H
