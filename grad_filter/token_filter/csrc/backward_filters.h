#pragma once

#include "common.h"

namespace token_filter {

// Utility functions for tensor I/O
void save_tensor_to_file(const torch::Tensor& tensor, const std::string& filename);
torch::Tensor load_tensor_from_file(const std::string& filename);

// BMM Backward Filter
struct BmmBackwardFilter : public BmmBackward0 {
    std::string name() const override { return "BmmBackwardFilter"; }
    variable_list apply(variable_list&& grads) override;
    void transfer_saved_data(BmmBackward0* pre_node, at::Tensor* mask);
    
    bool process_self;
    at::Tensor mask_;
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        mat2_.reset_data();
        self_.reset_data();
        mask_.reset();
    }
};

// MM Backward Filter
struct MmBackwardFilter0 : public MmBackward0 {
    std::string name() const override { return "MmBackwardFilter0"; }
    variable_list apply(variable_list&& grads) override;
    void transfer_saved_data(MmBackward0* pre_node, at::Tensor* mask);
    
    at::Tensor mask_;
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        mat2_.reset_data();
        self_.reset_data();
        mask_.reset();
    }
};

// Flash Attention Filter (Naive version, accuracy verified but slow)
struct ScaledDotProductFlashAttentionFilter : public ScaledDotProductFlashAttentionBackward0 {
    std::string name() const override { return "ScaledDotProductFlashAttentionFilter"; }
    variable_list apply(variable_list&& grads) override;
    void transfer_saved_data(ScaledDotProductFlashAttentionBackward0* pre_node, at::Tensor* mask);
    
    at::Tensor mask_;

    /*
    SavedVariable rng_state_;
    SavedVariable unused_;
    */
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        key_.reset_data();
        query_.reset_data();
        value_.reset_data();
        cum_seq_k_.reset_data();
        cum_seq_q_.reset_data();
        logsumexp_.reset_data();
        output_.reset_data();
        #ifdef TORCH25
        philox_offset_.reset_data();
        philox_seed_.reset_data();
        #else
        rng_state_.reset_data();
        unused_.reset_data();
        #endif
        mask_.reset();
    }
};

// Flash Attention Filter with Q Restore (WARNNING: this impl cannot be used in training)
// (With Q restored but is not logically correct because flashattn does not support attn_bias)
struct ScaledDotProductFlashAttentionFilterRestoreQ : public ScaledDotProductFlashAttentionBackward0 {
    std::string name() const override { return "ScaledDotProductFlashAttentionFilterRestoreQ"; }
    variable_list apply(variable_list&& grads) override;
    void transfer_saved_data(ScaledDotProductFlashAttentionBackward0* pre_node, at::Tensor* mask);
    
    at::Tensor mask_;
    SavedVariable attn_bias_restoreq_;
    SavedVariable k_remove_;
    SavedVariable v_remove_;

    // std::shared_ptr<ScaledDotProductEfficientAttentionBackward0> efficient_attn_backward = std::shared_ptr<ScaledDotProductEfficientAttentionBackward0>(
    //     new ScaledDotProductEfficientAttentionBackward0(), deleteNode);
    
    void set_attn_bias_restoreq(Tensor& attn_bias_restoreq);
    void split_kv();
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        key_.reset_data();
        query_.reset_data();
        value_.reset_data();
        cum_seq_k_.reset_data();
        cum_seq_q_.reset_data();
        logsumexp_.reset_data();
        output_.reset_data();

        #ifdef TORCH25
        philox_offset_.reset_data();
        philox_seed_.reset_data();
        #else
        rng_state_.reset_data();
        unused_.reset_data();
        #endif
        
        mask_.reset();
        k_remove_.reset_data();
        v_remove_.reset_data();
    }
};

// Efficient Attention Filter (with Q restored using attn_bias but also slow)
struct ScaledDotProductEfficientAttentionFilter : public ScaledDotProductEfficientAttentionBackward0 {
    std::string name() const override { return "ScaledDotProductEfficientAttentionFilter"; }
    variable_list apply(variable_list&& grads) override;
    void transfer_saved_data(ScaledDotProductEfficientAttentionBackward0* pre_node, at::Tensor* mask);
    
    at::Tensor mask_;
    SavedVariable attn_bias_restoreq_;
    
    void set_attn_bias_restoreq(Tensor& attn_bias_restoreq);
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        key_.reset_data();
        query_.reset_data();
        value_.reset_data();
        attn_bias_.reset_data();
        output_.reset_data();
        log_sumexp_.reset_data();
        philox_offset_.reset_data();
        philox_seed_.reset_data();
        mask_.reset();
    }
};

// CUDNN Attention Filter 
struct ScaledDotProductCudnnAttentionFilter : public ScaledDotProductCudnnAttentionBackward0 {
    std::string name() const override { return "ScaledDotProductCudnnAttentionFilter"; }
    variable_list apply(variable_list&& grads) override;
    void transfer_saved_data(ScaledDotProductCudnnAttentionBackward0* pre_node, at::Tensor* mask);
    
    at::Tensor mask_;
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        attn_bias_.reset_data();
        key_.reset_data();
        query_.reset_data();
        value_.reset_data();
        cum_seq_k_.reset_data();
        cum_seq_q_.reset_data();
        logsumexp_.reset_data();
        output_.reset_data();
        philox_offset_.reset_data();
        philox_seed_.reset_data();
        mask_.reset();
    }
};



// Sequence Backward Nodes
#ifdef _WIN32
struct UnsequeezeBszSeqLenBackward0 : public TraceableFunction {
    TORCH_API UnsequeezeBszSeqLenBackward0() = default;
#else
struct TORCH_API UnsequeezeBszSeqLenBackward0 : public TraceableFunction {
#endif
    using TraceableFunction::TraceableFunction;
    variable_list apply(variable_list&& grads) override;
    std::string name() const override { return "UnsequeezeBszSeqLenBackward0"; }
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        mask_index_.reset_data();
    }
    
    variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
    SavedVariable mask_index_;
};

#ifdef _WIN32
struct SequeezeBszSeqLenBackward0 : public TraceableFunction {
    TORCH_API SequeezeBszSeqLenBackward0() = default;
#else
struct TORCH_API SequeezeBszSeqLenBackward0 : public TraceableFunction {
#endif
    using TraceableFunction::TraceableFunction;
    variable_list apply(variable_list&& grads) override;
    std::string name() const override { return "SequeezeBszSeqLenBackward0"; }
    
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        mask_.reset_data();
    }
    
    variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
    SavedVariable mask_;
    at::TensorOptions self_options;
    std::vector<c10::SymInt> self_sym_sizes;
};

} // namespace token_filter
