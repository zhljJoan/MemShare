#include "fused_edge_projection.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace {

template <typename scalar_t>
__global__ void fused_edge_temporal_linear_kernel(
    const scalar_t* __restrict__ state_feat,
    const scalar_t* __restrict__ edge_feat,
    const scalar_t* __restrict__ time_feat,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int64_t* __restrict__ edge_src,
    const int64_t* __restrict__ edge_to_eid,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t state_dim,
    int64_t edge_dim,
    int64_t time_dim,
    int64_t out_dim) {
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * out_dim;
  if (linear >= total) {
    return;
  }

  int64_t row = linear / out_dim;
  int64_t col = linear - row * out_dim;
  int64_t src = edge_src[row];
  int64_t eid = edge_to_eid[row];

  scalar_t acc = bias[col];
  const scalar_t* w = weight + col * (state_dim + edge_dim + time_dim);
  for (int64_t i = 0; i < state_dim; ++i) {
    acc += state_feat[src * state_dim + i] * w[i];
  }
  for (int64_t i = 0; i < edge_dim; ++i) {
    acc += edge_feat[eid * edge_dim + i] * w[state_dim + i];
  }
  for (int64_t i = 0; i < time_dim; ++i) {
    acc += time_feat[row * time_dim + i] * w[state_dim + edge_dim + i];
  }
  out[linear] = acc;
}

}  // namespace

torch::Tensor fused_edge_temporal_linear_forward(
    torch::Tensor state_feat,
    torch::Tensor edge_feat,
    torch::Tensor time_feat,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor edge_src,
    torch::Tensor edge_to_eid) {
  TORCH_CHECK(state_feat.is_cuda(), "state_feat must be CUDA");
  TORCH_CHECK(edge_feat.is_cuda(), "edge_feat must be CUDA");
  TORCH_CHECK(time_feat.is_cuda(), "time_feat must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
  TORCH_CHECK(edge_src.is_cuda(), "edge_src must be CUDA");
  TORCH_CHECK(edge_to_eid.is_cuda(), "edge_to_eid must be CUDA");
  TORCH_CHECK(state_feat.is_contiguous(), "state_feat must be contiguous");
  TORCH_CHECK(edge_feat.is_contiguous(), "edge_feat must be contiguous");
  TORCH_CHECK(time_feat.is_contiguous(), "time_feat must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(edge_src.is_contiguous(), "edge_src must be contiguous");
  TORCH_CHECK(edge_to_eid.is_contiguous(), "edge_to_eid must be contiguous");
  TORCH_CHECK(edge_src.scalar_type() == torch::kInt64, "edge_src must be int64");
  TORCH_CHECK(edge_to_eid.scalar_type() == torch::kInt64, "edge_to_eid must be int64");
  TORCH_CHECK(state_feat.dim() == 2, "state_feat must be 2D");
  TORCH_CHECK(edge_feat.dim() == 2, "edge_feat must be 2D");
  TORCH_CHECK(time_feat.dim() == 2, "time_feat must be 2D");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

  int64_t rows = time_feat.size(0);
  int64_t state_dim = state_feat.size(1);
  int64_t edge_dim = edge_feat.size(1);
  int64_t time_dim = time_feat.size(1);
  int64_t out_dim = weight.size(0);
  TORCH_CHECK(edge_src.numel() == rows, "edge_src length mismatch");
  TORCH_CHECK(edge_to_eid.numel() == rows, "edge_to_eid length mismatch");
  TORCH_CHECK(weight.size(1) == state_dim + edge_dim + time_dim, "weight shape mismatch");
  TORCH_CHECK(bias.numel() == out_dim, "bias shape mismatch");
  TORCH_CHECK(state_feat.scalar_type() == edge_feat.scalar_type(), "dtype mismatch");
  TORCH_CHECK(state_feat.scalar_type() == time_feat.scalar_type(), "dtype mismatch");
  TORCH_CHECK(state_feat.scalar_type() == weight.scalar_type(), "dtype mismatch");
  TORCH_CHECK(state_feat.scalar_type() == bias.scalar_type(), "dtype mismatch");

  auto out = torch::empty({rows, out_dim}, time_feat.options());
  int threads = 256;
  int64_t total = rows * out_dim;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(state_feat.scalar_type(), "fused_edge_temporal_linear_forward", [&] {
    fused_edge_temporal_linear_kernel<scalar_t><<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        state_feat.data_ptr<scalar_t>(),
        edge_feat.data_ptr<scalar_t>(),
        time_feat.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        edge_src.data_ptr<int64_t>(),
        edge_to_eid.data_ptr<int64_t>(),
        out.data_ptr<scalar_t>(),
        rows,
        state_dim,
        edge_dim,
        time_dim,
        out_dim);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
