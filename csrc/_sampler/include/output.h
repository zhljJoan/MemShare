#pragma once
#include <head.h>

class TemporalGraphBlock
{
    public:
        vector<NodeIDType> row;
        vector<NodeIDType> col;
        vector<EdgeIDType> eid;
        vector<TimeStampType> delta_ts;
        vector<int64_t> src_index;
        vector<NodeIDType> sample_nodes;
        vector<TimeStampType> sample_nodes_ts;
        vector<WeightType> e_weights;
        double sample_time = 0;
        double tot_time = 0;
        int64_t sample_edge_num = 0;

        TemporalGraphBlock(){}
        // TemporalGraphBlock(const TemporalGraphBlock &tgb);
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes):
                           row(_row), col(_col), sample_nodes(_sample_nodes){}
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes,
                           vector<TimeStampType> &_sample_nodes_ts):
                           row(_row), col(_col), sample_nodes(_sample_nodes),
                           sample_nodes_ts(_sample_nodes_ts){}
};

class T_TemporalGraphBlock
{
    public:
        th::Tensor row;
        th::Tensor col;
        th::Tensor eid;
        th::Tensor delta_ts;
        th::Tensor src_index;
        th::Tensor sample_nodes;
        th::Tensor sample_nodes_ts;
        double sample_time = 0;
        double tot_time = 0;
        int64_t sample_edge_num = 0;

        T_TemporalGraphBlock(){}
        T_TemporalGraphBlock(th::Tensor &_row, th::Tensor &_col,
                           th::Tensor &_sample_nodes):
                           row(_row), col(_col), sample_nodes(_sample_nodes){}
};
class TemporalNeighborList{
    public:
        th::Tensor nids;
        th::Tensor ts;
        th::Tensor eids;
        vector<th::Tensor> uid;
        vector<th::Tensor> src_index;
        vector<th::Tensor> ueid;
        TemporalNeighborList(){}
        TemporalNeighborList(th::Tensor &_nids, th::Tensor &_ts, th::Tensor &_eids,
                            vector<th::Tensor> &_uid, vector<th::Tensor> &_src_index,
                            vector<th::Tensor> &_ueid):
                            nids(_nids),ts(_ts),eids(_eids),uid(_uid),src_index(_src_index),ueid(_ueid){
                                
                            }
                            
};