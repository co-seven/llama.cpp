// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

// Launch the whole ggml compute graph as a single spert kernel.
// n_tiles: number of AI cores to use (= num_prefer_cores, e.g. 8 on K3).
// graph_kernel_fn: callback into ggml-cpu.c that traverses all cgraph nodes.
// Returns 0 on success, non-zero on failure.
int spacemit_spert_launch_graph_kernel(
    void (*graph_kernel_fn)(int ith, int nth, void * user_data),
    void * user_data,
    int n_tiles);

// Sync primitives — callable from inside a tile (after launch_graph_kernel).
// No-op when called outside a tile.
void  spacemit_spert_grid_sync(void);           // whole-grid barrier (replaces ggml_barrier)
void *spacemit_spert_shared_buffer(long *size); // per-tile TCM scratch

#ifdef __cplusplus
}
#endif
