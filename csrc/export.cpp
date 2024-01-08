#include "extension.h"
#include "uvm.h"
#include "partition.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    #ifdef WITH_CUDA
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
    #endif

    #ifdef WITH_METIS
    m.def("metis_partition", &metis_partition, "metis graph partition");
    m.def("metis_cache_friendly_reordering", &metis_cache_friendly_reordering, "metis cache-friendly reordering");
    #endif

    #ifdef WITH_MTMETIS
    m.def("mt_metis_partition", &mt_metis_partition, "multi-threaded metis graph partition");
    #endif

    #ifdef WITH_LGD
    // Note: the switch WITH_MULTITHREADING=ON shall be triggered during compilation
    // to enable multi-threading functionality.
    m.def("ldg_partition", &ldg_partition, "(multi-threaded optionally) LDG graph partition");
    #endif
}
