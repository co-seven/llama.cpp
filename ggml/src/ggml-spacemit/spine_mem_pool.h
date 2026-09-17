#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

enum class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

void * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
void   spine_mem_pool_free(void * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
