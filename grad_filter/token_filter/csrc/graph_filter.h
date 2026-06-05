#pragma once

#include "common.h"

namespace token_filter {

class GraphFilter {
public:
    GraphFilter(Tensor* mask_);
    ~GraphFilter();
    
    // Variable processing
    SavedVariable process_variable(Variable &var, bool is_output, bool is_lse_padding = false);
    Variable lse_padding(Variable &var, int dim);
    
    // Size processing
    FilterDims get_filter_dims_mode(std::vector<c10::SymInt> &sizes);
    void process_sizes(std::vector<c10::SymInt> &sizes, int size_index);
    void process_sizes(std::vector<c10::SymInt> &sizes);
    void process_sizes(c10::SymInt &size);

    int process_counter;
    
    // Node filter dimensions management
    void add_node_filter_dims(std::string name) {
        node_filter_dims.push_back(NodeFilterDims(name)); 
        nfd_queue_size++; 
    }
    void cache_node_filter_dims(std::string file_path);
    void load_node_filter_dims(std::string file_path);
    std::string node_filter_dims_to_string();
    void string_to_node_filter_dims(std::string input_str);
    
public:
    // Size parameters
    c10::SymInt bsz, seq_before_filter, seq_after_filter;
    
    // Node filter dimensions
    std::vector<NodeFilterDims> node_filter_dims;
    int nfd_queue_size = 0, nfd_queue_index = 0;
    bool is_loaded_from_file = false;
    
    // Mask related
    std::map<const void*, Tensor> map_;
    Tensor mask_index_flatten_, mask_index_seq_;
    Tensor mask_index_, attn_mask_softmax;
    
    // Constants
    int kAlignLSE = 32;
};

} // namespace token_filter
