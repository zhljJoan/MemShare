#pragma once
#include <head.h>

void random_sample_with_collision_check(int size, th::Tensor batch_src_node_id, th::Tensor batch_dst_node_id){
    auto src_ptr = batch_src_node_id.data_ptr<NodeIDType>();
    auto dst_ptr = batch_dst_node_id.data_ptr<NodeIDType>();
    
#pragma omp for num_threads(10)
    for(int i = 0; i<size; i++){

    }
}