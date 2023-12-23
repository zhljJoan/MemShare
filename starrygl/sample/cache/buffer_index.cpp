#include <omp.h>
#include <torch/extension.h>
#include <time.h>
#include <random>
#include <phmap.h>
#include <boost/thread/mutex.hpp>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#define MTX boost::mutex

using namespace std;
namespace py = pybind11;
namespace th = torch;

typedef int64_t NodeIDType;
typedef int64_t EdgeIDType;
typedef float WeightType;
typedef float TimeStampType;


#define EXTRAARGS , phmap::priv::hash_default_hash<K>, \
                            phmap::priv::hash_default_eq<K>, \
                            std::allocator<K>, 4, MTX


template <class K, class V>
using HashM  = phmap::parallel_flat_hash_map<K, V EXTRAARGS>;

void init_cache_buffer(long long size){

}
at::Storage set_cache_buffer(at::Stroge ind){

}
//按策略更新替换cache上的数据，返回对应的索引
at::Storage update_buffer_index(string policy, at::Storge ind, u){

}
at::Storage get_buffer_index(at::Storage ind){


}


PYBIND11_MODULE(sample_cores, m)
{
    m
    .def("get_neighbors", 
        &get_neighbors, 
        py::return_value_policy::reference)    
    .
}