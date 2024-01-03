#include <torch/all.h>
#include "vertex_partition/vertex_partition.h"
#include "vertex_partition/params.h"

at::Tensor ldg_partition(at::Tensor edges,
                         at::optional<at::Tensor> vertex_weights,
                         at::optional<at::Tensor> initial_partition,
                         int64_t num_parts,
                         int64_t num_workers) {
    AT_ASSERT(edges.dim() == 2);
    auto edges_n_cols = edges.size(1);
    AT_ASSERT(edges_n_cols >= 2 && edges_n_cols <= 3);
    // Note: other checks are performed in the implementation of vp::ldg_partition function series.

    auto n = edges.slice(1, 0, 2).max().item<int64_t>() + 1;
    auto params = vp::LDGParams{.N = n, .K = num_parts, .openmp_n_threads = static_cast<int>(num_workers)};

    auto edges_clone = edges.clone();
    if (vertex_weights.has_value()) {
        auto vertex_weights_clone = vertex_weights->clone();
        if (initial_partition.has_value()) {
            auto initial_partition_clone = initial_partition->clone();
            vp::ldg_partition_v_init(edges_clone, vertex_weights_clone, initial_partition_clone, params);
            return initial_partition_clone;
        } else {
            return vp::ldg_partition_v(edges_clone, vertex_weights_clone, params);
        }
    } else {
        if (initial_partition.has_value()) {
            auto initial_partition_clone = initial_partition->clone();
            vp::ldg_partition_init(edges_clone, initial_partition_clone, params);
            return initial_partition_clone;
        } else {
            return vp::ldg_partition(edges_clone, params);
        }
    }
}
