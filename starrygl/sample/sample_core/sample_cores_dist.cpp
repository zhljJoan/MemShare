#include <iostream>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <time.h>
#include <random>
#include <phmap.h>
#include <boost/thread/mutex.hpp>
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
template <class K>
using HashT  = phmap::parallel_flat_hash_set<K EXTRAARGS>;

template <class K, class V>
using HashM  = phmap::parallel_flat_hash_map<K, V EXTRAARGS>;


class TemporalNeighborBlock;
class TemporalGraphBlock;
TemporalNeighborBlock& get_neighbors(string graph_name, th::Tensor row, th::Tensor col, int64_t num_nodes, int is_distinct, optional<th::Tensor> eid, optional<th::Tensor> edge_weight, optional<th::Tensor> time);
th::Tensor heads_unique(th::Tensor array, th::Tensor heads, int threads);
int nodeIdToInOut(NodeIDType nid, int pid, const vector<NodeIDType>& part_ptr);
int nodeIdToPartId(NodeIDType nid, const vector<NodeIDType>& part_ptr);
vector<th::Tensor> divide_nodes_to_part(th::Tensor nodes, const vector<NodeIDType>& part_ptr, int threads);
// vector<int64_t> sample_multinomial(vector<WeightType> weights, int num_samples, bool replacement, default_random_engine e);
NodeIDType sample_multinomial(const vector<WeightType>& weights, default_random_engine& e);


template<typename T>
inline py::array vec2npy(const std::vector<T> &vec)
{
    // need to let python garbage collector handle C++ vector memory 
    // see https://github.com/pybind/pybind11/issues/1042
    // non-copy value transfer
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void *v)
                               { delete reinterpret_cast<std::vector<T> *>(v); });
    return py::array(v->size(), v->data(), capsule);
    // return py::array(vec.size(), vec.data());
}

/* 
 * NeighborSampler Utils
 */
class TemporalNeighborBlock
{
    public:
        vector<vector<NodeIDType>> neighbors;
        vector<vector<TimeStampType>> timestamp;
        vector<vector<EdgeIDType>> eid;
        vector<vector<WeightType>> edge_weight;
        vector<phmap::parallel_flat_hash_map<NodeIDType, int64_t>> inverted_index;
        vector<int64_t> deg;
        vector<phmap::parallel_flat_hash_set<NodeIDType>> neighbors_set;

        bool with_eid = false;
        bool weighted = false;
        bool with_timestamp = false;

        TemporalNeighborBlock(){}
        // TemporalNeighborBlock(const TemporalNeighborBlock &tnb);
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<int64_t> &deg):
                              neighbors(neighbors), deg(deg){}
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<vector<WeightType>>& edge_weight, 
                              vector<vector<EdgeIDType>>& eid,
                              vector<int64_t> &deg):
                              neighbors(neighbors), edge_weight(edge_weight),eid(eid), deg(deg)
                              { this->with_eid=true;
                                this->weighted=true; }
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<vector<WeightType>>& edge_weight,
                              vector<vector<TimeStampType>>& timestamp,
                              vector<vector<EdgeIDType>>& eid, 
                              vector<int64_t> &deg):
                              neighbors(neighbors), edge_weight(edge_weight), timestamp(timestamp),eid(eid), deg(deg)
                              { this->with_eid=true;
                                this->weighted=true; 
                                this->with_timestamp=true;}
        py::array get_node_neighbor(NodeIDType node_id){
            return vec2npy(neighbors[node_id]);
        }
        py::array get_node_neighbor_timestamp(NodeIDType node_id){
            return vec2npy(timestamp[node_id]);
        }
        int64_t get_node_deg(NodeIDType node_id){
            return deg[node_id];
        }
        bool empty(){
            return this->deg.empty();
        }
        void update_edge_weight(th::Tensor row_or_eid, th::Tensor col, th::Tensor edge_weight);
        void update_node_weight(th::Tensor nid, th::Tensor node_weight);
        void update_all_node_weight(th::Tensor node_weight);
        int64_t update_neighbors_with_time(th::Tensor row, th::Tensor col, th::Tensor time, th::Tensor eid, int is_distinct, std::optional<th::Tensor> edge_weight);

        
        std::string serialize() const {
            std::ostringstream oss;
            // 序列化基本类型成员
            oss << with_eid << " " << weighted << " " << with_timestamp << " ";

            // 序列化 vector<vector<T>> 类型成员
            auto serializeVecVec = [&oss](const auto& vecVec) {
                for (const auto& vec : vecVec) {
                    oss << vec.size() << " ";
                    for (const auto& elem : vec) {
                        oss << elem << " ";
                    }
                }
                oss << "|";  // 添加一个分隔符以区分不同的 vector
            };

            serializeVecVec(neighbors);
            serializeVecVec(timestamp);
            serializeVecVec(eid);
            serializeVecVec(edge_weight);

            // 序列化 vector<int64_t> 类型成员
            oss << deg.size() << " ";
            for (const auto& d : deg) {
                oss << d << " ";
            }
            oss << "|";

            // 序列化 inverted_index
            for (const auto& map : inverted_index) {
                oss << map.size() << " ";
                for (const auto& [key, value] : map) {
                    oss << key << " " << value << " ";
                }
            }
            oss << "|";

            // 序列化 neighbors_set
            for (const auto& set : neighbors_set) {
                oss << set.size() << " ";
                for (const auto& elem : set) {
                    oss << elem << " ";
                }
            }
            oss << "|";

            return oss.str();
        }

        static TemporalNeighborBlock deserialize(const std::string& s) {
            std::istringstream iss(s);
            TemporalNeighborBlock tnb;

            // 反序列化基本类型成员
            iss >> tnb.with_eid >> tnb.weighted >> tnb.with_timestamp;

            // 反序列化 vector<vector<T>> 类型成员
            auto deserializeVecLong = [&iss](vector<vector<int64_t>>& vecVec) {
                std::string segment;
                std::getline(iss, segment, '|');
                std::istringstream vec_iss(segment);
                while (!vec_iss.eof()) {
                    size_t vec_size;
                    vec_iss >> vec_size;
                    if (vec_iss.eof()) break;  // 防止多余的空白
                    vector<int64_t> vec(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        vec_iss >> vec[i];
                    }
                    vecVec.push_back(vec);
                }
            };

            
            auto deserializeVecFloat = [&iss](vector<vector<float>>& vecVec) {
                std::string segment;
                std::getline(iss, segment, '|');
                std::istringstream vec_iss(segment);
                while (!vec_iss.eof()) {
                    size_t vec_size;
                    vec_iss >> vec_size;
                    if (vec_iss.eof()) break;  // 防止多余的空白
                    vector<float> vec(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        vec_iss >> vec[i];
                    }
                    vecVec.push_back(vec);
                }
            };

            deserializeVecLong(tnb.neighbors);
            deserializeVecFloat(tnb.timestamp);
            deserializeVecLong(tnb.eid);
            deserializeVecFloat(tnb.edge_weight);

            std::string segment;
            // 反序列化 vector<int64_t> 类型成员
            segment="";
            std::getline(iss, segment, '|');
            std::istringstream vec_iss(segment);
            size_t vec_size;
            vec_iss >> vec_size;
            tnb.deg.resize(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                vec_iss >> tnb.deg[i];
            }

            // 反序列化 inverted_index
            segment="";
            std::getline(iss, segment, '|');
            std::istringstream map_iss(segment);
            while (!map_iss.eof()) {
                size_t map_size;
                map_iss >> map_size;
                if (map_iss.eof()) break;
                phmap::parallel_flat_hash_map<NodeIDType, int64_t> map;
                for (size_t i = 0; i < map_size; ++i) {
                    NodeIDType key;
                    int64_t value;
                    map_iss >> key >> value;
                    map[key] = value;
                }
                tnb.inverted_index.push_back(map);
            }

            // 反序列化 neighbors_set
            std::getline(iss, segment, '|');
            std::istringstream set_iss(segment);
            while (!set_iss.eof()) {
                size_t set_size;
                set_iss >> set_size;
                if (set_iss.eof()) break;
                phmap::parallel_flat_hash_set<NodeIDType> set;
                for (size_t i = 0; i < set_size; ++i) {
                    NodeIDType elem;
                    set_iss >> elem;
                    set.insert(elem);
                }
                tnb.neighbors_set.push_back(set);
            }

            return tnb;
        }
};

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

// 辅助函数
template <typename T>
T* get_data_ptr(const th::Tensor& tensor) {
    AT_ASSERTM(tensor.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(tensor.dim() == 1, "Offset tensor must be one-dimensional");
    return tensor.data_ptr<T>();
}
template <typename T>
torch::Tensor vecToTensor(const std::vector<T>& vec) {
    // 确定数据类型
    torch::ScalarType dtype;
    if (std::is_same<T, int64_t>::value) {
        dtype = torch::kInt64;
    } else if (std::is_same<T, float>::value) {
        dtype = torch::kFloat32;
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    // 创建Tensor
    torch::Tensor tensor = torch::from_blob(
        const_cast<T*>(vec.data()), /* 数据指针 */
        {static_cast<long>(vec.size())}, /* 尺寸 */
        dtype /* 数据类型 */
    );

    return tensor;//.clone(); // 克隆Tensor以拷贝数据
}

TemporalNeighborBlock& get_neighbors(
        string graph_name, th::Tensor row, th::Tensor col, int64_t num_nodes, int is_distinct, optional<th::Tensor> eid, optional<th::Tensor> edge_weight, optional<th::Tensor> time)
{   //row、col、time按time升序排列，由时间早的到时间晚的
    auto src = get_data_ptr<NodeIDType>(row);
    auto dst = get_data_ptr<NodeIDType>(col);
    EdgeIDType* eid_ptr = eid ? get_data_ptr<EdgeIDType>(eid.value()) : nullptr;
    WeightType* ew = edge_weight ? get_data_ptr<WeightType>(edge_weight.value()) : nullptr;
    TimeStampType* t = time ? get_data_ptr<TimeStampType>(time.value()) : nullptr;
    int64_t edge_num = row.size(0);
    static phmap::parallel_flat_hash_map<string, TemporalNeighborBlock> tnb_map;
    if(tnb_map.count(graph_name)==1)
        return tnb_map[graph_name];
    tnb_map[graph_name] = TemporalNeighborBlock();
    TemporalNeighborBlock& tnb = tnb_map[graph_name];

    double start_time = omp_get_wtime();
    //初始化
    tnb.neighbors.resize(num_nodes);
    tnb.deg.resize(num_nodes, 0);
        //初始化optional相关
    tnb.with_eid = eid.has_value();
    tnb.weighted = edge_weight.has_value();
    tnb.with_timestamp = time.has_value();
    if (tnb.with_eid) tnb.eid.resize(num_nodes);
    if (tnb.weighted) {
        tnb.edge_weight.resize(num_nodes);
        tnb.inverted_index.resize(num_nodes);
    }
    if (tnb.with_timestamp) tnb.timestamp.resize(num_nodes);
        
    //计算, 条件判断移出循环优化执行效率
    for(int64_t i=0; i<edge_num; i++){
        //计算节点邻居
        tnb.neighbors[dst[i]].emplace_back(src[i]);
    }
    //如果有eid，插入
    if(tnb.with_eid)
        for(int64_t i=0; i<edge_num; i++){
            tnb.eid[dst[i]].emplace_back(eid_ptr[i]);
        }
    //如果有权重信息，插入节点与邻居边的权重和反向索引
    if(tnb.weighted)
        for(int64_t i=0; i<edge_num; i++){
            tnb.edge_weight[dst[i]].emplace_back(ew[i]);
            if(tnb.with_eid) tnb.inverted_index[dst[i]][eid_ptr[i]]=tnb.neighbors[dst[i]].size()-1;
            else tnb.inverted_index[dst[i]][src[i]]=tnb.neighbors[dst[i]].size()-1;
        }
    //如果有时序信息，插入节点与邻居边的时间
    if(tnb.with_timestamp)
        for(int64_t i=0; i<edge_num; i++){
            tnb.timestamp[dst[i]].emplace_back(t[i]);
        }
        
    if(is_distinct){
        for(int64_t i=0; i<num_nodes; i++){
            //收集单边去重节点度
            phmap::parallel_flat_hash_set<NodeIDType> temp_s;
            temp_s.insert(tnb.neighbors[i].begin(), tnb.neighbors[i].end());
            tnb.neighbors_set.emplace_back(temp_s);
            tnb.deg[i] = tnb.neighbors_set[i].size();
        }
    }
    else{
        for(int64_t i=0; i<num_nodes; i++){
            //收集单边节点度
            tnb.deg[i] = tnb.neighbors[i].size();
        }
    }
    double end_time = omp_get_wtime();
    cout<<"get_neighbors consume: "<<end_time-start_time<<"s"<<endl;
    return tnb;
}

void TemporalNeighborBlock::update_edge_weight(
        th::Tensor row_or_eid, th::Tensor col, th::Tensor edge_weight){
    AT_ASSERTM(this->weighted, "This Graph has no edge weight infomation");
    auto dst = get_data_ptr<NodeIDType>(col);
    WeightType* ew = get_data_ptr<WeightType>(edge_weight);
    NodeIDType* src;
    EdgeIDType* eid_ptr;
    if(this->with_eid) eid_ptr = get_data_ptr<EdgeIDType>(row_or_eid);
    else src = get_data_ptr<NodeIDType>(row_or_eid);
    
    int64_t edge_num = col.size(0);

    for(int64_t i=0; i<edge_num; i++){
        //修改节点与邻居边的权重
        AT_ASSERTM(this->inverted_index[dst[i]].count(src[i])==1, "Unexist Edge Index: "+to_string(src[i])+", "+to_string(dst[i]));
		int index;
        if(this->with_eid) index = this->inverted_index[dst[i]][eid_ptr[i]];
        else index = this->inverted_index[dst[i]][src[i]];
        this->edge_weight[dst[i]][index] = ew[i];
    }
}

void TemporalNeighborBlock:: update_node_weight(th::Tensor nid, th::Tensor node_weight){
    AT_ASSERTM(this->weighted, "This Graph has no edge weight infomation");
    auto dst = get_data_ptr<NodeIDType>(nid);
    WeightType* nw = get_data_ptr<WeightType>(node_weight);
    
    int64_t node_num = nid.size(0);

    for(int64_t i=0; i<node_num; i++){
        //修改节点与邻居边的权重
        AT_ASSERTM(dst[i]<this->deg.size(), "Unexist Node Index: "+to_string(dst[i]));
        if(this->inverted_index[dst[i]].empty())
            return;
        for(auto index : this->inverted_index[dst[i]]){
            this->edge_weight[dst[i]][index.second] = nw[i];
        }
    }
}

void TemporalNeighborBlock:: update_all_node_weight(th::Tensor node_weight){
    AT_ASSERTM(this->weighted, "This Graph has no edge weight infomation");
    WeightType* nw = get_data_ptr<WeightType>(node_weight);
    int64_t node_num = node_weight.size(0);
    AT_ASSERTM(node_num==this->neighbors.size(), "The tensor node_weight size is not suitable node number.");

    for(int64_t i=0; i<node_num; i++){
        //修改节点与邻居边的权重
        for(int j=0; j<this->neighbors[i].size();j++){
            this->edge_weight[i][j] = nw[this->neighbors[i][j]];
        }
    }
}

int64_t TemporalNeighborBlock::update_neighbors_with_time(
        th::Tensor row, th::Tensor col, th::Tensor time,th::Tensor eid, int is_distinct, std::optional<th::Tensor> edge_weight){
        //row、col、time按time升序排列，由时间早的到时间晚的
    AT_ASSERTM(this->empty(), "Empty TemporalNeighborBlock, please use get_neighbors_with_time");
    AT_ASSERTM(this->with_timestamp == true, "This Graph has no time infomation!");
    auto src = get_data_ptr<NodeIDType>(row);
    auto dst = get_data_ptr<NodeIDType>(col);
    auto eid_ptr = get_data_ptr<EdgeIDType>(eid);
    auto t = get_data_ptr<TimeStampType>(time);
    WeightType* ew = edge_weight ? get_data_ptr<WeightType>(edge_weight.value()) : nullptr;
    int64_t edge_num = row.size(0);
    int64_t num_nodes = this->neighbors.size();

    //处理optional的值
    if(edge_weight.has_value()){
        AT_ASSERTM(this->weighted == true, "This Graph has no edge weight");
    }
    if(this->weighted){
        AT_ASSERTM(edge_weight.has_value(), "This Graph need edge weight");
    }

    // double start_time = omp_get_wtime();
    if(is_distinct){
        for(int64_t i=0; i<edge_num; i++){
            //如果有新节点
            if(dst[i]>=num_nodes){
                num_nodes = dst[i]+1;
                this->neighbors.resize(num_nodes);
                this->deg.resize(num_nodes, 0);
                this->eid.resize(num_nodes);
                this->timestamp.resize(num_nodes);
                    //初始化optional相关
                if (this->weighted) {
                    this->edge_weight.resize(num_nodes);
                    this->inverted_index.resize(num_nodes);
                }
            }
            //更新节点邻居
            this->neighbors[dst[i]].emplace_back(src[i]);
            //插入eid
            this->eid[dst[i]].emplace_back(eid_ptr[i]);
            //插入节点与邻居边的时间
            this->timestamp[dst[i]].emplace_back(t[i]);
            //如果有权重信息，插入节点与邻居边的权重和反向索引
            if(this->weighted){
                this->edge_weight[dst[i]].emplace_back(ew[i]);
                if(this->with_eid) this->inverted_index[dst[i]][eid_ptr[i]]=this->neighbors[dst[i]].size()-1;
                else this->inverted_index[dst[i]][src[i]]=this->neighbors[dst[i]].size()-1;
            }
            
            this->neighbors_set[dst[i]].insert(src[i]);
            this->deg[dst[i]]=this->neighbors_set[dst[i]].size();
        }
    }
    else{
        for(int64_t i=0; i<edge_num; i++){
            //更新节点邻居
            this->neighbors[dst[i]].emplace_back(src[i]);
            //插入eid
            this->eid[dst[i]].emplace_back(eid_ptr[i]);
            //插入节点与邻居边的时间
            this->timestamp[dst[i]].emplace_back(t[i]);
            //如果有权重信息，插入节点与邻居边的权重和反向索引
            if(this->weighted){
                this->edge_weight[dst[i]].emplace_back(ew[i]);
                this->inverted_index[dst[i]][src[i]]=this->neighbors[dst[i]].size()-1;
            }

            this->deg[dst[i]]=this->neighbors[dst[i]].size();
        }
    }
    // double end_time = omp_get_wtime();
    // cout<<"update_neighbors consume: "<<end_time-start_time<<"s"<<endl;
    return num_nodes;
}

class ParallelSampler
{
    public:
        TemporalNeighborBlock& tnb;
        NodeIDType num_nodes;
        EdgeIDType num_edges;
        int threads;
        vector<int> fanouts;
        // vector<NodeIDType> part_ptr;
        // int pid;
        int num_layers;
        string policy;
        std::vector<TemporalGraphBlock> ret;

        ParallelSampler(TemporalNeighborBlock& _tnb, NodeIDType _num_nodes, EdgeIDType _num_edges, int _threads, 
                        vector<int>& _fanouts, int _num_layers, string _policy) :
                        tnb(_tnb), num_nodes(_num_nodes), num_edges(_num_edges), threads(_threads), 
                        fanouts(_fanouts), num_layers(_num_layers), policy(_policy)
        {
            omp_set_num_threads(_threads);
            ret.clear();
            ret.resize(_num_layers);
        }

        void reset()
        {
            ret.clear();
            ret.resize(num_layers);
        }

        void neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts);
        void neighbor_sample_from_nodes_static(th::Tensor nodes);
        void neighbor_sample_from_nodes_static_layer(th::Tensor nodes, int cur_layer);
        void neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_nodes_with_before_layer(th::Tensor nodes, th::Tensor root_ts, int cur_layer);
};



void ParallelSampler :: neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts)
{
    omp_set_num_threads(threads);
    if(policy == "weighted")
        AT_ASSERTM(tnb.weighted, "Tnb has no weight infomation!");
    else if(policy == "recent")
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
    else if(policy == "uniform")
        ;
    else{
        throw runtime_error("The policy \"" + policy + "\" is not exit!");
    }
    if(tnb.with_timestamp){
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
        AT_ASSERTM(root_ts.has_value(), "Parameter mismatch!");
        neighbor_sample_from_nodes_with_before(nodes, root_ts.value());
    }
    else{
        neighbor_sample_from_nodes_static(nodes);
    }
}

void ParallelSampler :: neighbor_sample_from_nodes_static_layer(th::Tensor nodes, int cur_layer){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();

    TemporalGraphBlock tgb = TemporalGraphBlock();
    int fanout = fanouts[cur_layer];
    ret[cur_layer] = TemporalGraphBlock();
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);    
    vector<phmap::parallel_flat_hash_set<NodeIDType>> node_s_threads(threads);
    phmap::parallel_flat_hash_set<NodeIDType> node_s;
    vector<vector<NodeIDType>> eid_threads(threads);//row_threads(threads),col_threads(threads);
    AT_ASSERTM(tnb.with_eid, "Tnb has no eid infomation! We need eid!");
    
    // double start_time = omp_get_wtime();
    int reserve_capacity = int(ceil(nodes.size(0) / threads)) * fanout;
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = tid;
    // row_threads[tid].reserve(reserve_capacity);
    // col_threads[tid].reserve(reserve_capacity);
    eid_threads[tid].reserve(reserve_capacity);
#pragma omp for schedule(static, int(ceil(static_cast<float>((nodes.size(0)) / threads))))
    for(int64_t i=0; i<nodes.size(0); i++){
        // int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        vector<NodeIDType> nei(tnb.neighbors[node]);
        vector<EdgeIDType> edge;
        edge = tnb.eid[node];

        double s_start_time = omp_get_wtime();
        if(tnb.deg[node]>fanout){
            phmap::flat_hash_set<NodeIDType> temp_s;
            default_random_engine e(8);//(time(0));
            uniform_int_distribution<> u(0, tnb.deg[node]-1);
            while(temp_s.size()!=fanout){
                //循环选择fanout个邻居
                NodeIDType indice;
                if(policy == "weighted"){//考虑边权重信息
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    indice = sample_multinomial(ew, e);
                }
                else if(policy == "uniform"){//均匀采样
                    indice = u(e);
                }
                auto chosen_n_iter = nei.begin() + indice;
                auto rst = temp_s.insert(*chosen_n_iter);
                if(rst.second){ //不重复
                    auto chosen_e_iter = edge.begin() + indice;
                    eid_threads[tid].emplace_back(*chosen_e_iter);
                    node_s_threads[tid].insert(*chosen_n_iter);
                }
            }
            // row_threads[tid].insert(row_threads[tid].end(),temp_s.begin(),temp_s.end());
            // col_threads[tid].insert(col_threads[tid].end(), fanout, node);
        }
        else{
            node_s_threads[tid].insert(nei.begin(), nei.end());
            // row_threads[tid].insert(row_threads[tid].end(),nei.begin(),nei.end());
            // col_threads[tid].insert(col_threads[tid].end(), tnb.deg[node], node);
            eid_threads[tid].insert(eid_threads[tid].end(),edge.begin(), edge.end());
        }
        if(tid==0)
            ret[0].sample_time += omp_get_wtime() - s_start_time;
    }
}
    // double end_time = omp_get_wtime();
    
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
    
    int size = 0;
    vector<int> each_begin(threads);
    for(int i = 0; i<threads; i++){
        int s = eid_threads[i].size();
        each_begin[i]=size;
        size += s;
    }
    // ret[cur_layer].row.resize(size);
    // ret[cur_layer].col.resize(size);
    ret[cur_layer].eid.resize(size);
#pragma omp parallel for schedule(static, 1)
    for(int i = 0; i<threads; i++){
        // copy(row_threads[i].begin(), row_threads[i].end(), ret[cur_layer].row.begin()+each_begin[i]);
        // copy(col_threads[i].begin(), col_threads[i].end(), ret[cur_layer].col.begin()+each_begin[i]);
        copy(eid_threads[i].begin(), eid_threads[i].end(), ret[cur_layer].eid.begin()+each_begin[i]);
    }
    for(int i = 0; i<threads; i++)
        node_s.insert(node_s_threads[i].begin(), node_s_threads[i].end());
    ret[cur_layer].sample_nodes.assign(node_s.begin(), node_s.end());

    ret[0].tot_time += omp_get_wtime() - tot_start_time;
    ret[0].sample_edge_num += ret[cur_layer].eid.size();
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_static(th::Tensor nodes){
    for(int i=0;i<num_layers;i++){
        if(i==0) neighbor_sample_from_nodes_static_layer(nodes, i);
        else neighbor_sample_from_nodes_static_layer(vecToTensor<NodeIDType>(ret[i-1].sample_nodes), i);
    }
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before_layer(
        th::Tensor nodes, th::Tensor root_ts, int cur_layer){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();
    ret[cur_layer] = TemporalGraphBlock();
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    int fanout = fanouts[cur_layer];
    // HashT<pair<NodeIDType,TimeStampType> > node_s;
    vector<TemporalGraphBlock> tgb_i(threads);
    
    default_random_engine e(8);//(time(0));
    // double start_time = omp_get_wtime();
    int reserve_capacity = int(ceil(nodes.size(0) / threads)) * fanout;
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = tid;
    tgb_i[tid].sample_nodes.reserve(reserve_capacity);
    tgb_i[tid].sample_nodes_ts.reserve(reserve_capacity);
    tgb_i[tid].delta_ts.reserve(reserve_capacity);
    tgb_i[tid].eid.reserve(reserve_capacity);
    tgb_i[tid].src_index.reserve(reserve_capacity);
#pragma omp for schedule(static, int(ceil(static_cast<float>((nodes.size(0)) / threads))))
    for(int64_t i=0; i<nodes.size(0); i++){
        // int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        TimeStampType rtts = ts_data[i];
        int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();
        // cout<<node<<" "<<end_index<<" "<<tnb.deg[node]<<endl;

        double s_start_time = omp_get_wtime();
        if ((policy == "recent") || (end_index <= fanout)){
            int start_index = max(0, end_index-fanout);
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), end_index-start_index, i);
            tgb_i[tid].sample_nodes.insert(tgb_i[tid].sample_nodes.end(), tnb.neighbors[node].begin()+start_index, tnb.neighbors[node].begin()+end_index);
            tgb_i[tid].sample_nodes_ts.insert(tgb_i[tid].sample_nodes_ts.end(), tnb.timestamp[node].begin()+start_index, tnb.timestamp[node].begin()+end_index);
            tgb_i[tid].eid.insert(tgb_i[tid].eid.end(), tnb.eid[node].begin()+start_index, tnb.eid[node].begin()+end_index);
            for(int cid = start_index; cid < end_index;cid++){
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
            }
        }
        else{
            //可选邻居边大于扇出的话需要随机选择fanout个邻居
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), fanout, i);
            uniform_int_distribution<> u(0, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            for(int i=0; i<fanout;i++){
                int cid;
                if(policy == "uniform")
                    cid = u(e);
                    // cid = rand_r(&loc_seed) % (end_index);
                else if(policy == "weighted"){
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    cid = sample_multinomial(ew, e);
                }
                tgb_i[tid].sample_nodes.emplace_back(tnb.neighbors[node][cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(tnb.timestamp[node][cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
                tgb_i[tid].eid.emplace_back(tnb.eid[node][cid]);
            }
        }
        if(tid==0)
            ret[0].sample_time += omp_get_wtime() - s_start_time;
    }
}
    // double end_time = omp_get_wtime();
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
    
    // start_time = omp_get_wtime();

    int size = 0;
    vector<int> each_begin(threads);
    for(int i = 0; i<threads; i++){
        int s = tgb_i[i].eid.size();
        each_begin[i]=size;
        size += s;
    }
    ret[cur_layer].eid.resize(size);
    ret[cur_layer].src_index.resize(size);
    ret[cur_layer].delta_ts.resize(size);
    ret[cur_layer].sample_nodes.resize(size);
    ret[cur_layer].sample_nodes_ts.resize(size);
#pragma omp parallel for schedule(static, 1)
    for(int i = 0; i<threads; i++){
        copy(tgb_i[i].eid.begin(), tgb_i[i].eid.end(), ret[cur_layer].eid.begin()+each_begin[i]);
        copy(tgb_i[i].src_index.begin(), tgb_i[i].src_index.end(), ret[cur_layer].src_index.begin()+each_begin[i]);
        copy(tgb_i[i].delta_ts.begin(), tgb_i[i].delta_ts.end(), ret[cur_layer].delta_ts.begin()+each_begin[i]);
        copy(tgb_i[i].sample_nodes.begin(), tgb_i[i].sample_nodes.end(), ret[cur_layer].sample_nodes.begin()+each_begin[i]);
        copy(tgb_i[i].sample_nodes_ts.begin(), tgb_i[i].sample_nodes_ts.end(), ret[cur_layer].sample_nodes_ts.begin()+each_begin[i]);
    }

    // end_time = omp_get_wtime();
    // cout<<"end union consume: "<<end_time-start_time<<"s"<<endl;

    ret[0].tot_time += omp_get_wtime() - tot_start_time;
    ret[0].sample_edge_num += ret[cur_layer].eid.size();
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts){
    for(int i=0;i<num_layers;i++){
        if(i==0) neighbor_sample_from_nodes_with_before_layer(nodes, root_ts, i);
        else neighbor_sample_from_nodes_with_before_layer(vecToTensor<NodeIDType>(ret[i-1].sample_nodes), 
                                                          vecToTensor<TimeStampType>(ret[i-1].sample_nodes_ts), i);
    }
}

/*-------------------------------------------------------------------------------------**
**------------Utils--------------------------------------------------------------------**
**-------------------------------------------------------------------------------------*/

th::Tensor heads_unique(th::Tensor array, th::Tensor heads, int threads){
    auto array_ptr = array.data_ptr<NodeIDType>();
    phmap::parallel_flat_hash_set<NodeIDType> s(array_ptr, array_ptr+array.numel());
    if(heads.numel()==0) return th::tensor(vector<NodeIDType>(s.begin(), s.end()));
    AT_ASSERTM(heads.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(heads.dim() == 1, "0ffset tensor must be one-dimensional");
    auto heads_ptr = heads.data_ptr<NodeIDType>();
#pragma omp parallel for num_threads(threads)
    for(int64_t i=0; i<heads.size(0); i++){
        if(s.count(heads_ptr[i])==1){
        #pragma omp critical(erase)
            s.erase(heads_ptr[i]);
        }
    }
    vector<NodeIDType> ret;
    ret.reserve(s.size()+heads.numel());
    ret.assign(heads_ptr, heads_ptr+heads.numel());
    ret.insert(ret.end(), s.begin(), s.end());
    // cout<<"s: "<<s.size()<<" array: "<<array.size()<<endl;
    return th::tensor(ret);
}

int nodeIdToPartId(NodeIDType nid, const vector<NodeIDType>& part_ptr){
    int partitionId = -1;
    for(int i=0;i<part_ptr.size()-1;i++){
            if(nid>=part_ptr[i]&&nid<part_ptr[i+1]){
                partitionId = i;
                break;
            }
    }
    if(partitionId<0) throw "nid 不存在对应的分区";
    return partitionId;
}
//0:inner; 1:outer
int nodeIdToInOut(NodeIDType nid, int pid, const vector<NodeIDType>& part_ptr){
    if(nid>=part_ptr[pid]&&nid<part_ptr[pid+1]){
        return 0;
    }
    return 1;
}

vector<th::Tensor> divide_nodes_to_part(
        th::Tensor nodes, const vector<NodeIDType>& part_ptr, int threads){
    double start_time = omp_get_wtime();
    AT_ASSERTM(nodes.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(nodes.dim() == 1, "0ffset tensor must be one-dimensional");
    auto nodes_id = nodes.data_ptr<NodeIDType>();
    vector<vector<vector<NodeIDType>>> node_part_threads;
    vector<th::Tensor> result(part_ptr.size()-1);
    //初始化点的分区，每个分区按线程划分避免冲突
    for(int i = 0; i<threads; i++){
        vector<vector<NodeIDType>> node_parts;
        for(int j=0;j<part_ptr.size()-1;j++){
            node_parts.push_back(vector<NodeIDType>());
        }
        node_part_threads.push_back(node_parts);
    }
#pragma omp parallel for num_threads(threads) default(shared)
    for(int64_t i=0; i<nodes.size(0); i++){
        int tid = omp_get_thread_num();
        int pid = nodeIdToPartId(nodes_id[i], part_ptr);
        node_part_threads[tid][pid].emplace_back(nodes_id[i]);
    }
#pragma omp parallel for num_threads(part_ptr.size()-1) default(shared)
    for(int i = 0; i<part_ptr.size()-1; i++){
        vector<NodeIDType> temp;
        for(int j=0;j<threads;j++){
            temp.insert(temp.end(), node_part_threads[j][i].begin(), node_part_threads[j][i].end());
        }
        result[i]=th::tensor(temp);
    }
    double end_time = omp_get_wtime();
    // cout<<"end divide consume: "<<end_time-start_time<<"s"<<endl;
    return result;
}

NodeIDType sample_multinomial(const vector<WeightType>& weights, default_random_engine& e){
    NodeIDType sample_indice;
    vector<WeightType> cumulative_weights;
    partial_sum(weights.begin(), weights.end(), back_inserter(cumulative_weights));
    AT_ASSERTM(cumulative_weights.back() > 0, "Edge weight sum should be greater than 0.");
    
    uniform_real_distribution<WeightType> distribution(0.0, cumulative_weights.back());
    WeightType random_value = distribution(e);
    auto it = lower_bound(cumulative_weights.begin(), cumulative_weights.end(), random_value);
    sample_indice = distance(cumulative_weights.begin(), it);
    return sample_indice;
}

/*------------Python Bind--------------------------------------------------------------*/
PYBIND11_MODULE(sample_cores, m)
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
        py::return_value_policy::reference);

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
                      vector<int>&, int, string>())
        .def_readonly("ret", &ParallelSampler::ret, py::return_value_policy::reference)
        .def("neighbor_sample_from_nodes", &ParallelSampler::neighbor_sample_from_nodes)
        .def("reset", &ParallelSampler::reset)
        .def("get_ret", [](const ParallelSampler &ps) { return ps.ret; });
}