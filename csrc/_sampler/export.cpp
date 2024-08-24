#include<head.h>
#include <sampler.h>
#include <tppr.h>
#include <output.h>
#include <neighbors.h>
#include <temporal_utils.h>


/*------------Python Bind--------------------------------------------------------------*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m
    .def("get_neighbors", 
        &get_neighbors, 
        py::return_value_policy::reference)    
    .def("heads_unique", 
        &heads_unique, 
        py::return_value_policy::reference)
    .def("divide_nodes_to_part", 
        &divide_nodes_to_part, 
        py::return_value_policy::reference)
    .def("sparse_get_index", 
        &sparse_get_index, 
        py::return_value_policy::reference)
    .def("get_norm_temporal",
        &get_norm_temporal, 
        py::return_value_policy::reference
    );

    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(py::init<vector<NodeIDType> &, vector<NodeIDType> &,
                      vector<NodeIDType> &>())
        .def("row", [](const TemporalGraphBlock &tgb) { return vecToTensor<NodeIDType>(tgb.row); })
        .def("col", [](const TemporalGraphBlock &tgb) { return vecToTensor<NodeIDType>(tgb.col); })
        .def("eid", [](const TemporalGraphBlock &tgb) { return vecToTensor<EdgeIDType>(tgb.eid); })
        .def("delta_ts", [](const TemporalGraphBlock &tgb) { return vecToTensor<TimeStampType>(tgb.delta_ts); })
        .def("src_index", [](const TemporalGraphBlock &tgb) { return vecToTensor<EdgeIDType>(tgb.src_index); })
        .def("sample_nodes", [](const TemporalGraphBlock &tgb) { return vecToTensor<NodeIDType>(tgb.sample_nodes); })
        .def("sample_nodes_ts", [](const TemporalGraphBlock &tgb) { return vecToTensor<TimeStampType>(tgb.sample_nodes_ts); })
        .def_readonly("sample_time", &TemporalGraphBlock::sample_time, py::return_value_policy::reference)
        .def_readonly("tot_time", &TemporalGraphBlock::tot_time, py::return_value_policy::reference)
        .def_readonly("sample_edge_num", &TemporalGraphBlock::sample_edge_num, py::return_value_policy::reference);

    py::class_<T_TemporalGraphBlock>(m, "T_TemporalGraphBlock")
        .def(py::init<th::Tensor &, th::Tensor &,
                      th::Tensor &>())
        .def_readonly("row", &T_TemporalGraphBlock::row, py::return_value_policy::reference)
        .def_readonly("col", &T_TemporalGraphBlock::col, py::return_value_policy::reference)
        .def_readonly("eid", &T_TemporalGraphBlock::eid, py::return_value_policy::reference)
        .def_readonly("delta_ts", &T_TemporalGraphBlock::delta_ts, py::return_value_policy::reference)
        .def_readonly("src_index", &T_TemporalGraphBlock::src_index, py::return_value_policy::reference)
        .def_readonly("sample_nodes", &T_TemporalGraphBlock::sample_nodes, py::return_value_policy::reference)
        .def_readonly("sample_nodes_ts", &T_TemporalGraphBlock::sample_nodes_ts, py::return_value_policy::reference)
        .def_readonly("sample_time", &T_TemporalGraphBlock::sample_time, py::return_value_policy::reference)
        .def_readonly("tot_time", &T_TemporalGraphBlock::tot_time, py::return_value_policy::reference)
        .def_readonly("sample_edge_num", &T_TemporalGraphBlock::sample_edge_num, py::return_value_policy::reference);
    py::class_<TemporalNeighborList>(m,"TemporalNeighborList")
        .def(py::init<th::Tensor &, th::Tensor &, th::Tensor &,
            vector<th::Tensor> &, vector<th::Tensor> &,vector<th::Tensor> &>())
        .def_readonly("nids",&TemporalNeighborList::nids,py::return_value_policy::reference)
        .def_readonly("ts",&TemporalNeighborList::ts,py::return_value_policy::reference)
        .def_readonly("eids",&TemporalNeighborList::eids,py::return_value_policy::reference)
        .def_readonly("uid",&TemporalNeighborList::uid,py::return_value_policy::reference)
        .def_readonly("src_index",&TemporalNeighborList::src_index,py::return_value_policy::reference)
        .def_readonly("ueid",&TemporalNeighborList::ueid,py::return_value_policy::reference);
    py::class_<TemporalNeighborBlock>(m, "TemporalNeighborBlock")
        .def(py::init<vector<vector<NodeIDType>>&, 
                      vector<int64_t> &>())
        .def(py::pickle(
            [](const TemporalNeighborBlock& tnb) { return tnb.serialize(); },
            [](const std::string& s) { return TemporalNeighborBlock::deserialize(s); }
        ))
        .def("update_neighbors_with_time", 
            &TemporalNeighborBlock::update_neighbors_with_time)
        .def("update_edge_weight", 
            &TemporalNeighborBlock::update_edge_weight)
        .def("update_node_weight", 
            &TemporalNeighborBlock::update_node_weight)
        .def("update_all_node_weight", 
            &TemporalNeighborBlock::update_all_node_weight)            
        // .def("get_node_neighbor",&TemporalNeighborBlock::get_node_neighbor)
        // .def("get_node_deg", &TemporalNeighborBlock::get_node_deg)
        .def_readonly("neighbors", &TemporalNeighborBlock::neighbors, py::return_value_policy::reference)
        .def_readonly("timestamp", &TemporalNeighborBlock::timestamp, py::return_value_policy::reference)
        .def_readonly("edge_weight", &TemporalNeighborBlock::edge_weight, py::return_value_policy::reference)
        .def_readonly("eid", &TemporalNeighborBlock::eid, py::return_value_policy::reference)
        .def_readonly("deg", &TemporalNeighborBlock::deg, py::return_value_policy::reference)
        .def_readonly("with_eid", &TemporalNeighborBlock::with_eid, py::return_value_policy::reference)
        .def_readonly("with_timestamp", &TemporalNeighborBlock::with_timestamp, py::return_value_policy::reference)
        .def_readonly("weighted", &TemporalNeighborBlock::weighted, py::return_value_policy::reference);

    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<TemporalNeighborBlock &, NodeIDType, EdgeIDType, int,
                      vector<int>&, int, string, int, th::Tensor &>())
        .def_readonly("ret", &ParallelSampler::ret, py::return_value_policy::reference)
        .def("neighbor_sample_from_nodes", &ParallelSampler::neighbor_sample_from_nodes)
        .def("reset", &ParallelSampler::reset)
        .def("get_ret", [](const ParallelSampler &ps) { return ps.ret; })
        .def_readonly("block",&ParallelSampler::block, py::return_value_policy::reference);

    py::class_<ParallelTppRComputer>(m, "ParallelTppRComputer")
        .def(py::init<TemporalNeighborBlock &, NodeIDType, EdgeIDType, int,
                      int, int, int, vector<float>&, vector<float>& >())
        .def_readonly("ret", &ParallelTppRComputer::ret, py::return_value_policy::reference)
        .def("reset_ret", &ParallelTppRComputer::reset_ret)
        .def("reset_tppr", &ParallelTppRComputer::reset_tppr)
        .def("reset_val_tppr", &ParallelTppRComputer::reset_val_tppr)
        .def("backup_tppr", &ParallelTppRComputer::backup_tppr)
        .def("restore_tppr", &ParallelTppRComputer::restore_tppr)
        .def("restore_val_tppr", &ParallelTppRComputer::restore_val_tppr)
        .def("get_pruned_topk", &ParallelTppRComputer::get_pruned_topk)
        .def("extract_streaming_tppr", &ParallelTppRComputer::extract_streaming_tppr)
        .def("streaming_topk", &ParallelTppRComputer::streaming_topk)
        .def("single_streaming_topk", &ParallelTppRComputer::single_streaming_topk)
        .def("streaming_topk_no_fake", &ParallelTppRComputer::streaming_topk_no_fake)
        .def("compute_val_tppr", &ParallelTppRComputer::compute_val_tppr)
        .def("get_ret", [](const ParallelTppRComputer &ps) { return ps.ret; });

}