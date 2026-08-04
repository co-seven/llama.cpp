#pragma once

#include "spine_barrier.h"
#include "spine_mem_pool.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

inline constexpr size_t spine_init_barrier_count = 16;

enum class spine_core_arch_id : uint16_t {
    core_arch_none = 0,
    core_arch_x60  = 0x503C,
    core_arch_x100 = 0x5064,
    core_arch_x200 = 0x50C8,
    core_arch_a60  = 0xA03C,
    core_arch_a100 = 0xA064,
    core_arch_a200 = 0xA0C8,
};

struct spine_env_info {
    int                    num_cores{ 0 };
    spine_core_arch_id     perfer_core_arch_id{ spine_core_arch_id::core_arch_none };
    bool                   use_ime2{ false };
    bool                   use_ime1{ false };
    spine_mem_pool_backend mem_backend{ spine_mem_pool_backend::transparent_hugepage };
    spine_barrier_t *      init_barrier{ nullptr };

    spine_env_info();
    ~spine_env_info();
};

extern spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
