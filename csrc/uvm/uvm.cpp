#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

struct CUDAManagedContext {
    void *_ptr;
    at::DeviceIndex _dev;

    CUDAManagedContext(void* ptr, const at::DeviceIndex cuda_device): _ptr(ptr), _dev(cuda_device) {}
    ~CUDAManagedContext() {
        at::cuda::CUDAGuard device_guard(this->_dev);
        AT_CUDA_CHECK(cudaFree(this->_ptr));
    }

    static void release(void* ptr) {
        delete static_cast<CUDAManagedContext*>(ptr);
    }
};

struct CUDAManagedContextViewer {
    at::Storage _storage;

    CUDAManagedContextViewer(at::Storage storage): _storage(std::move(storage)) {}
    
    static void release(void* ptr) {
        delete static_cast<CUDAManagedContextViewer*>(ptr);
    }
};

at::Storage uvm_storage_new(const std::size_t size_bytes, const at::DeviceIndex cuda_device) {
    at::cuda::CUDAGuard device_guard(cuda_device);
    
    void *ptr;
    AT_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes));

    auto storage = at::Storage(
        at::Storage::use_byte_size_t(),
        size_bytes,
        at::DataPtr(
            ptr, new CUDAManagedContext(ptr, cuda_device),
            &CUDAManagedContext::release,
            {at::DeviceType::CUDA, cuda_device}
        ),
        /* allocator= */ nullptr,
        /* resizable= */ false
    );

    return at::Storage(
        at::Storage::use_byte_size_t(),
        size_bytes,
        at::DataPtr(
            ptr, new CUDAManagedContextViewer(storage),
            &CUDAManagedContextViewer::release,
            {at::DeviceType::CUDA, cuda_device}
        ),
        /* allocator= */ nullptr,
        /* resizable= */ false
    );
}

bool uvm_storage_check(const at::Storage& storage) {
    return storage.data_ptr().get_deleter() == &CUDAManagedContextViewer::release;
}

at::Storage uvm_storage_to_cuda(const at::Storage& storage, const at::DeviceIndex cuda_device) {
    AT_ASSERT(uvm_storage_check(storage), "must be uvm storage");
    
    auto& data_ptr = storage.data_ptr();
    auto ctx_view = static_cast<CUDAManagedContextViewer*>(data_ptr.get_context());

    at::cuda::CUDAGuard device_guard(cuda_device);
    return at::Storage(
        at::Storage::use_byte_size_t(),
        storage.nbytes(),
        at::DataPtr(
            data_ptr.get(),
            new CUDAManagedContextViewer(ctx_view->_storage),
            &CUDAManagedContextViewer::release,
            {at::DeviceType::CUDA, cuda_device}
        ),
        /* allocator= */ nullptr,
        /* resizable= */ false
    );
}

at::Storage uvm_storage_to_cpu(const at::Storage& storage) {
    AT_ASSERT(uvm_storage_check(storage), "must be uvm storage");

    auto& data_ptr = storage.data_ptr();
    auto ctx_view = static_cast<CUDAManagedContextViewer*>(data_ptr.get_context());

    return at::Storage(
        at::Storage::use_byte_size_t(),
        storage.nbytes(),
        at::DataPtr(
            data_ptr.get(),
            new CUDAManagedContextViewer(ctx_view->_storage),
            &CUDAManagedContextViewer::release,
            {at::DeviceType::CPU}
        ),
        /* allocator= */ nullptr,
        /* resizable= */ false
    );
}

void uvm_storage_advise(const at::Storage& storage, const cudaMemoryAdvise advise) {
    AT_ASSERT(uvm_storage_check(storage), "must be uvm storage");

    at::cuda::OptionalCUDAGuard device_guard;

    at::DeviceIndex hint_device = cudaCpuDeviceId;
    if (storage.device_type() == at::DeviceType::CUDA) {
        hint_device = storage.device().index();
        device_guard.set_index(hint_device);
    }

    AT_CUDA_CHECK(cudaMemAdvise(
        storage.data(),
        storage.nbytes(),
        // static_cast<enum cudaMemoryAdvise>(advise),
        advise,
        hint_device
    ));
}

void uvm_storage_prefetch(const at::Storage& storage) {
    AT_ASSERT(uvm_storage_check(storage), "must be uvm storage");

    at::cuda::OptionalCUDAGuard device_guard;

    cudaStream_t stream = 0;
    at::DeviceIndex hint_device = cudaCpuDeviceId;
    if (storage.device_type() == at::DeviceType::CUDA) {
        hint_device = storage.device().index();
        device_guard.set_index(hint_device);
        stream = at::cuda::getCurrentCUDAStream().stream();
    }

    AT_CUDA_CHECK(cudaMemPrefetchAsync(
        storage.data(),
        storage.nbytes(),
        hint_device,
        stream
    ));
}

