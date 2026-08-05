#ifndef SPACEMIT_SESSION_H
#define SPACEMIT_SESSION_H

#include "spacemit-opnode.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct spacemit_session {
    uint32_t num_cores = 0;
    int64_t  arch_id   = 0;
    int64_t  vlen      = 0;
    int64_t  tcm_size  = 0;
    bool     use_ime1  = false;
    bool     use_ime2  = false;

    std::string name = "SPACEMIT0";

    // Persistent spert stream (created on first graph_compute, reused after).
    void * stream_ptr = nullptr;  // spert::Stream*

    struct {
        uint64_t uid = 0;
        std::vector<spacemit_opnode> nodes;
    } cached_graph;

    spacemit_session() = default;

    ~spacemit_session();

    const char * c_name() const { return name.c_str(); }
};

#endif // SPACEMIT_SESSION_H
