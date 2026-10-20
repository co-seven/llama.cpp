#include "spacemit-fused.h"

#include "ggml-impl.h"
#include "ime.h"

// Fused-op dispatch. Each fused pattern is handled by the tensor_traits of the
// primary node: traits->compute_forward_fused switches on node.opcode and runs
// the matching fused kernel. This keeps the single-op and fused paths symmetric —
// both go through get_tensor_traits + a virtual compute_forward* entry point.

bool ggml_spacemit_compute_fused(ggml::spacemit::context & ctx, const spacemit_opnode & node) {
    const auto * traits = ggml_spacemit_get_tensor_traits(node.node);
    if (!traits) {
        GGML_LOG_ERROR("ggml-spacemit: no traits for fused op %s\n", node.op_name().c_str());
        return false;
    }
    return traits->compute_forward_fused(ctx, node);
}
