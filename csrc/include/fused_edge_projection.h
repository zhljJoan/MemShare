#pragma once

#include <torch/extension.h>

torch::Tensor fused_edge_temporal_linear_forward(
    torch::Tensor state_feat,
    torch::Tensor edge_feat,
    torch::Tensor time_feat,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor edge_src,
    torch::Tensor edge_to_eid);
