#include <torch/extension.h>
#include <omp.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "a function implemented using pybind11");
}