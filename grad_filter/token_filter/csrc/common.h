#pragma once

#include <torch/extension.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/python_function.h>
#include <ATen/ATen.h>
#include <map>
#include <vector>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <limits>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/wait.h>

using namespace torch::autograd;
using namespace torch::autograd::generated;

namespace token_filter {

// Constants definition
#define BMM_EDGE_DEPTH 4

// Conditional compilation flags
// #define RESTORE_Q true
// #define EAGER_BMM_PROCESS true  
// #define VERBOSE true
// #define MASKED_SELECT true

// Forward declarations
class GraphFilter;

// Enum definitions
enum class FilterDimsMode {
    BTH_SEQ_LEN, 
    BTH_SEQ_LEN_MINUS_1, 
    SEQ_LEN, 
    SEQ_LEN_MINUS_1, 
    NA
};

// Data structure definitions
struct FilterDims {
    FilterDims(int index, int bsz_index, FilterDimsMode mode)
        : index(index), bsz_index(bsz_index), mode(mode) {}
    ~FilterDims() = default;
    
    int index, bsz_index;
    FilterDimsMode mode;
    
    friend std::ostream& operator<<(std::ostream& os, const FilterDims& fd) {
        os << "FilterDims(index=" << fd.index << ", bsz_index=" << 
           fd.bsz_index << ", mode=" << static_cast<int>(fd.mode) << ")";
        return os;
    }
};

struct NodeFilterDims {
    NodeFilterDims(std::string name) : name(name) {}
    ~NodeFilterDims() = default;
    
    std::string name;
    std::queue<FilterDims> changed_dims;
    
    void add_changed_dim(FilterDims dim) { changed_dims.push(dim); }
    
    FilterDims get_changed_dim() {
        if(changed_dims.empty()) std::cout << "changed_dims is empty!" << std::endl;
        FilterDims d = changed_dims.front(); 
        changed_dims.pop(); 
        return d;
    }
};

// Utility function declarations
void transfer_inputmeta_and_edge(Node* source, Node* target, 
                                bool transfer_edges=true, 
                                bool transfer_input_meta=true);

void update_grad_input_shape(Node* fn, c10::SymInt new_shape);

// Edge processing function declarations
void process_bmm_backward(Edge& edge, Tensor* mask);
void process_mm_backward(Edge& edge, Tensor* mask);
void process_scaled_dot_product_flash_attn(Edge& edge, Tensor* mask);
void process_scaled_dot_product_flash_attn_restore_q(Edge& edge, Tensor* mask, Tensor& attn_bias_restoreq);
void process_scaled_dot_product_flash_attn_restore_q(Edge* edge, Tensor* mask, Tensor& attn_bias_restoreq);
void process_scaled_dot_product_efficient_attn(Edge& edge, Tensor* mask, Tensor& attn_bias_restoreq);
void process_scaled_dot_product_cudnn_attn(Edge& edge, Tensor* mask);
void process_scaled_dot_product_efficient_attn(Edge* edge, Tensor* mask, Tensor& attn_bias_restoreq);
void process_scaled_dot_product_cudnn_attn(Edge* edge, Tensor* mask);

// Node processing function declarations
void node_processing_input_metadata(Node* fn, GraphFilter* graph_filter);
void node_processing(Node* fn, GraphFilter* graph_filter);

// Main API function declarations
at::Tensor backward_filter(at::Tensor loss, at::Tensor mask);

// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_efficient_attention_backward_without_kv_cuda(
//     const at::Tensor& grad_out_,
//     const at::Tensor& query,
//     const at::Tensor& key,
//     const at::Tensor& value,
//     const at::Tensor& attn_bias,
//     const at::Tensor& out,
//     const at::Tensor& logsumexp,
//     const at::Tensor& philox_seed,
//     const at::Tensor& philox_offset,
//     double dropout_p,
//     std::array<bool, 4> grad_input_mask,
//     bool causal,
//     std::optional<double> scale);

} // namespace token_filter
