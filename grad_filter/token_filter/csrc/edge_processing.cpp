#include "common.h"
#include "graph_filter.h"
#include "backward_filters.h"

namespace token_filter {

// ========== BMM Backward Processing ==========

void process_bmm_backward(Edge& edge, Tensor* mask) {
    auto bmm_backward_filter = std::shared_ptr<BmmBackwardFilter>(new BmmBackwardFilter(), deleteNode);
    auto edge_func = edge.function.get();
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(edge_func, static_cast<Node*>(bmm_backward_filter.get()));
    
    // Trace the next edges on self to determine process_self
    bool proc_self = false;
    for(int i=0; i<BMM_EDGE_DEPTH; i++) {
        edge_list& next_edges = edge_func->next_edges();
        if(next_edges.empty()) break;
        edge_func = next_edges[0].function.get();
        if(edge_func->name() == "SoftmaxBackward0") {proc_self = true; break;}
    }
    
    bmm_backward_filter->process_self = proc_self;
    bmm_backward_filter->transfer_saved_data(dynamic_cast<BmmBackward0*>(edge.function.get()), mask);
    
    // Update function
    edge.function = std::move(std::shared_ptr<Node>(bmm_backward_filter));
}

// ========== MM Backward Processing ==========

void process_mm_backward(Edge& edge, Tensor* mask) {
    auto mm_backward_filter = std::shared_ptr<MmBackwardFilter0>(new MmBackwardFilter0(), deleteNode);
    auto edge_func = static_cast<MmBackward0*>(edge.function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(mm_backward_filter.get()));
    
    // Transfer saved data
    mm_backward_filter->transfer_saved_data(edge_func, mask);
    
    // Copy the layout and sym_sizes
    mm_backward_filter->mat2_layout = edge_func->mat2_layout;
    mm_backward_filter->self_layout = edge_func->self_layout;
    for(auto sym_size : edge_func->mat2_sym_sizes) mm_backward_filter->mat2_sym_sizes.push_back(sym_size);
    for(auto sym_size : edge_func->mat2_sym_strides) mm_backward_filter->mat2_sym_strides.push_back(sym_size);
    for(auto sym_size : edge_func->self_sym_sizes) mm_backward_filter->self_sym_sizes.push_back(sym_size);
    for(auto sym_size : edge_func->self_sym_strides) mm_backward_filter->self_sym_strides.push_back(sym_size);
    
    // Update self_sym_sizes[0]
    mm_backward_filter->self_sym_sizes[0] = mask->sum().item<int>();
    
    // Update function
    edge.function = std::move(std::shared_ptr<Node>(mm_backward_filter));
}

// ========== Flash Attention Processing ==========

void process_scaled_dot_product_flash_attn(Edge& edge, Tensor* mask) {
    auto flash_attn_filter = std::shared_ptr<ScaledDotProductFlashAttentionFilter>(new ScaledDotProductFlashAttentionFilter(), deleteNode);
    auto edge_func = dynamic_cast<ScaledDotProductFlashAttentionBackward0*>(edge.function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(flash_attn_filter.get()));
    
    // Transfer saved data
    flash_attn_filter->transfer_saved_data(edge_func, mask);
    
    // Update function
    edge.function = std::move(std::shared_ptr<Node>(flash_attn_filter));
}

void process_scaled_dot_product_flash_attn_restore_q(Edge& edge, Tensor* mask, Tensor& attn_bias_restoreq) {
    auto flash_attn_filter = std::shared_ptr<ScaledDotProductFlashAttentionFilterRestoreQ>(new ScaledDotProductFlashAttentionFilterRestoreQ(), deleteNode);
    auto edge_func = dynamic_cast<ScaledDotProductFlashAttentionBackward0*>(edge.function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(flash_attn_filter.get()));
    
    // Transfer saved data
    flash_attn_filter->transfer_saved_data(edge_func, mask);
    flash_attn_filter->set_attn_bias_restoreq(attn_bias_restoreq);
    flash_attn_filter->split_kv();
    
    // Update function
    edge.function = std::move(std::shared_ptr<Node>(flash_attn_filter));
}

void process_scaled_dot_product_flash_attn_restore_q(Edge* edge, Tensor* mask, Tensor& attn_bias_restoreq) {
    auto flash_attn_filter = std::shared_ptr<ScaledDotProductFlashAttentionFilterRestoreQ>(new ScaledDotProductFlashAttentionFilterRestoreQ(), deleteNode);
    auto edge_func = dynamic_cast<ScaledDotProductFlashAttentionBackward0*>(edge->function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(flash_attn_filter.get()));
    
    // Transfer saved data
    flash_attn_filter->transfer_saved_data(edge_func, mask);
    flash_attn_filter->set_attn_bias_restoreq(attn_bias_restoreq);
    flash_attn_filter->split_kv();
    
    // Update function
    edge->function = std::move(std::shared_ptr<Node>(flash_attn_filter));
}

// ========== Efficient Attention Processing ==========

void process_scaled_dot_product_efficient_attn(Edge& edge, Tensor* mask, Tensor& attn_bias_restoreq) {
    auto efficient_attn_filter = std::shared_ptr<ScaledDotProductEfficientAttentionFilter>(new ScaledDotProductEfficientAttentionFilter(), deleteNode);
    auto edge_func = dynamic_cast<ScaledDotProductEfficientAttentionBackward0*>(edge.function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(efficient_attn_filter.get()));
    
    // Transfer saved data
    efficient_attn_filter->transfer_saved_data(edge_func, mask);
    efficient_attn_filter->set_attn_bias_restoreq(attn_bias_restoreq);
    
    // Update function
    edge.function = std::move(std::shared_ptr<Node>(efficient_attn_filter));
}

void process_scaled_dot_product_efficient_attn(Edge* edge, Tensor* mask, Tensor& attn_bias_restoreq) {
    auto efficient_attn_filter = std::shared_ptr<ScaledDotProductEfficientAttentionFilter>(new ScaledDotProductEfficientAttentionFilter(), deleteNode);
    auto edge_func = dynamic_cast<ScaledDotProductEfficientAttentionBackward0*>(edge->function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(efficient_attn_filter.get()));
    
    // Transfer saved data
    efficient_attn_filter->transfer_saved_data(edge_func, mask);
    efficient_attn_filter->set_attn_bias_restoreq(attn_bias_restoreq);
    
    // Update function
    edge->function = std::move(std::shared_ptr<Node>(efficient_attn_filter));
}

// ========== CUDNN Attention Processing ==========

void process_scaled_dot_product_cudnn_attn(Edge& edge, Tensor* mask) {
    auto cudnn_attn_filter = std::shared_ptr<ScaledDotProductCudnnAttentionFilter>(new ScaledDotProductCudnnAttentionFilter(), deleteNode);
    auto edge_func = dynamic_cast<ScaledDotProductCudnnAttentionBackward0*>(edge.function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(cudnn_attn_filter.get()));
    
    // Transfer saved data
    cudnn_attn_filter->transfer_saved_data(edge_func, mask);
    
    // Update function
    edge.function = std::move(std::shared_ptr<Node>(cudnn_attn_filter));
}

void process_scaled_dot_product_cudnn_attn(Edge* edge, Tensor* mask) {
    auto cudnn_attn_filter = std::shared_ptr<ScaledDotProductCudnnAttentionFilter>(new ScaledDotProductCudnnAttentionFilter(), deleteNode);
    auto edge_func = dynamic_cast<ScaledDotProductCudnnAttentionBackward0*>(edge->function.get());
    
    // Transfer input metadata and edge
    transfer_inputmeta_and_edge(static_cast<Node*>(edge_func), static_cast<Node*>(cudnn_attn_filter.get()));
    
    // Transfer saved data
    cudnn_attn_filter->transfer_saved_data(edge_func, mask);
    
    // Update function
    edge->function = std::move(std::shared_ptr<Node>(cudnn_attn_filter));
}

} // namespace token_filter
