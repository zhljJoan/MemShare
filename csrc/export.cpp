#include "extension.h"
#include "uvm.h"

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "a function implemented using pybind11");
    m.def("uvm_storage_new", &uvm_storage_new, "return storage of unified virtual memory");
    m.def("uvm_storage_to_cuda", &uvm_storage_to_cuda, "share uvm storage with another cuda device");
    m.def("uvm_storage_to_cpu", &uvm_storage_to_cpu, "share uvm storage with cpu");
    m.def("uvm_storage_advise", &uvm_storage_advise, "apply cudaMemAdvise() to uvm storage");
    m.def("uvm_storage_prefetch", &uvm_storage_prefetch, "apply cudaMemPrefetchAsync() to uvm storage");

    py::enum_<cudaMemoryAdvise>(m, "cudaMemoryAdvise")
        .value("cudaMemAdviseSetAccessedBy", cudaMemoryAdvise::cudaMemAdviseSetAccessedBy)
        .value("cudaMemAdviseUnsetAccessedBy", cudaMemoryAdvise::cudaMemAdviseUnsetAccessedBy)
        .value("cudaMemAdviseSetPreferredLocation", cudaMemoryAdvise::cudaMemAdviseSetPreferredLocation)
        .value("cudaMemAdviseUnsetPreferredLocation", cudaMemoryAdvise::cudaMemAdviseUnsetPreferredLocation)
        .value("cudaMemAdviseSetReadMostly", cudaMemoryAdvise::cudaMemAdviseSetReadMostly)
        .value("cudaMemAdviseUnsetReadMostly", cudaMemoryAdvise::cudaMemAdviseUnsetReadMostly);
}
