#include "backward_filters.h"
#include "common.h"
#include "graph_filter.h"
#include <fstream>
#include <iostream>

namespace token_filter {

// Utility function to save torch::Tensor to local file in binary .pt format
void save_tensor_to_file(const torch::Tensor& tensor, const std::string& filename) {
    // Check if tensor is valid and has no NaN values
    if (!tensor.defined()) {
        std::cerr << "Warning: Attempting to save undefined tensor to " << filename << std::endl;
        return;
    }
    
    // Convert tensor to CPU if it's on GPU
    torch::Tensor cpu_tensor = tensor.cpu();
    
    // Check for NaN or infinite values
    bool has_nan = torch::any(torch::isnan(cpu_tensor)).item<bool>();
    bool has_inf = torch::any(torch::isinf(cpu_tensor)).item<bool>();
    
    if (has_nan) {
        std::cerr << "Warning: Tensor contains NaN values before saving to " << filename << std::endl;
    }
    if (has_inf) {
        std::cerr << "Warning: Tensor contains infinite values before saving to " << filename << std::endl;
    }
    
    // Create filename with .pt extension if not provided
    std::string full_filename = filename;
    if (full_filename.find(".pt") == std::string::npos) {
        full_filename += ".pt";
    }
    
    try {
        // Ensure tensor is contiguous before saving
        cpu_tensor = cpu_tensor;

        auto pickled = torch::pickle_save(cpu_tensor);
        std::ofstream fout(full_filename, std::ios::out | std::ios::binary);
        fout.write(pickled.data(), pickled.size());
        fout.close();
        
        // Save the tensor using PyTorch's save function in binary format
        // torch::save(cpu_tensor, full_filename);
        std::cout << "Tensor saved to: " << full_filename << std::endl;
        std::cout << "Tensor shape: " << cpu_tensor.sizes() << std::endl;
        std::cout << "Tensor dtype: " << cpu_tensor.dtype() << std::endl;
        std::cout << "Tensor is contiguous: " << cpu_tensor.is_contiguous() << std::endl;
        
        // Additional validation: print min/max values
        if (cpu_tensor.numel() > 0) {
            std::cout << "Tensor min: " << torch::min(cpu_tensor).item<float>() << std::endl;
            std::cout << "Tensor max: " << torch::max(cpu_tensor).item<float>() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving tensor to file " << full_filename << ": " << e.what() << std::endl;
    }
}

// Utility function to load torch::Tensor from binary .pt file
torch::Tensor load_tensor_from_file(const std::string& filename) {
    // Create filename with .pt extension if not provided
    std::string full_filename = filename;
    if (full_filename.find(".pt") == std::string::npos) {
        full_filename += ".pt";
    }
    
    try {
        torch::Tensor tensor;
        torch::load(tensor, full_filename);
        std::cout << "Tensor loaded from: " << full_filename << std::endl;
        std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
        std::cout << "Tensor dtype: " << tensor.dtype() << std::endl;
        return tensor;
    } catch (const std::exception& e) {
        std::cerr << "Error loading tensor from file " << full_filename << ": " << e.what() << std::endl;
        throw;
    }
}

// ========== BmmBackwardFilter Implementation ==========

variable_list BmmBackwardFilter::apply(variable_list&& grads) {
    // update grads with mask
    auto mask_expand = mask_.unsqueeze(1).repeat_interleave(grads[0].sizes()[0]/mask_.sizes()[0], 1).flatten();
    
    // upscale the grads, shape of the grads[0] = [batch_size*num_head, sequence_len, hidden_dims]
    Variable new_grad = grads[0].new_zeros_symint({grads[0].sizes()[0] * mask_.sizes()[1], grads[0].sizes()[2]}, grads[0].options());
    
    auto indices = torch::where(mask_expand == 1)[0];
    auto reshaped_grads = grads[0].reshape({-1, grads[0].sizes()[2]});
    new_grad.index_put_({indices}, reshaped_grads);
    
    auto grad_size_upscale = {grads[0].sizes()[0], mask_.sizes()[1], grads[0].sizes()[2]};
    grads[0].reset();
    grads[0] = new_grad.view(grad_size_upscale);

    auto result = BmmBackward0::apply(std::move(grads));

    result[0] = result[0].index_select(1, torch::where(mask_.select(0, 0) == 1)[0]);
    result[1] = result[1].index_select(process_self ? 1 : 2, torch::where(mask_.select(0, 0) == 1)[0]);
    
    return result;
}

void BmmBackwardFilter::transfer_saved_data(BmmBackward0* pre_node, at::Tensor* mask) {
    Variable self = pre_node->self_.unpack();
    Variable mat2 = pre_node->mat2_.unpack();
    mask_ = mask->clone();

    self_ = SavedVariable(self, false);
    mat2_ = SavedVariable(mat2, false);
}

// ========== MmBackwardFilter0 Implementation ==========

variable_list MmBackwardFilter0::apply(variable_list&& grads) {
    Variable g = grads[0];
    grads[0] = g.index({mask_ == 1}); g.reset();
    auto front_grad = MmBackward0::apply(std::move(grads));
    
    Variable padded_grad = torch::zeros({mask_.sizes()[0], front_grad[0].sizes()[1]}, front_grad[0].options());
    padded_grad.index_copy_(0, mask_.nonzero().flatten(), front_grad[0]);
    front_grad[0].reset(); 
    front_grad[0] = padded_grad;
    return front_grad;
}

void MmBackwardFilter0::transfer_saved_data(MmBackward0* pre_node, at::Tensor* mask) {
    Variable self = pre_node->self_.unpack();
    Variable mat2 = pre_node->mat2_.unpack();
    mask_ = mask->flatten();
    
    self_ = SavedVariable(self.index({mask_ == 1}), false);
    mat2_ = SavedVariable(mat2, false);
}

// ========== ScaledDotProductFlashAttentionFilter Implementation ==========

variable_list ScaledDotProductFlashAttentionFilter::apply(variable_list&& grads) {
    #ifdef VERBOSE
    std::cout << "input grads[0]" << grads[0].select(1, 0) << std::endl;
    #endif

    auto mask_expand = mask_.unsqueeze(1).repeat_interleave(grads[0].sizes()[1], 1);
    
    #ifdef VERBOSE
    std::cout << "mask_expand " << mask_expand << std::endl;
    #endif

    auto grad_sizes = grads[0].sizes();
    auto bsz = grad_sizes[0];
    auto num_head = grad_sizes[1];
    auto seq_len_reduced = grad_sizes[2];
    auto seq_len_before_filter = mask_.sizes()[1];
    auto hidden_dims = grad_sizes[3];
    
    #ifdef VERBOSE
    std::cout << "bsz = " << bsz << std::endl;
    std::cout << "num_head = " << num_head << std::endl;
    std::cout << "seq_len_reduced = " << seq_len_reduced << std::endl;
    std::cout << "seq_len_before_filter = " << seq_len_before_filter << std::endl;
    std::cout << "hidden_dims = " << hidden_dims << std::endl;
    #endif
    
    Variable new_grad = grads[0].new_zeros_symint({bsz * num_head * seq_len_before_filter, hidden_dims}, grads[0].options());
    
    auto indices = torch::where(mask_expand.flatten() == 1)[0];
    auto reshaped_grads = grads[0].reshape({-1, hidden_dims});
    new_grad.index_put_({indices}, reshaped_grads);

    grads[0].reset();
    grads[0] = new_grad.view({bsz, num_head, seq_len_before_filter, hidden_dims});

    #ifdef VERBOSE
    std::cout << "new_grad size = " << new_grad.sizes() << std::endl;
    std::cout << "new grad[0] size = " << grads[0].sizes() << std::endl;
    #endif

    auto result = ScaledDotProductFlashAttentionBackward0::apply(std::move(grads));

    #ifdef VERBOSE
    std::cout << "result[0] size = " << result[0].sizes() << std::endl;
    std::cout << "result[1] size = " << result[1].sizes() << std::endl;
    std::cout << "result[2] size = " << result[2].sizes() << std::endl;
    #endif
    
    auto apply_per_batch_index_select = [&](const at::Tensor& t) -> at::Tensor {
        if (!t.defined()) return t;
        TORCH_CHECK(t.dim() == 4, "Expected 4D tensor [bsz, num_head, seq_len, hidden_dim], got ", t.sizes());
        const int64_t bsz = t.size(0);
        const int64_t num_head = t.size(1);
        const int64_t hidden_dim = t.size(3);

        const int64_t kept_len = seq_len_reduced;

        at::Tensor out = t.new_empty({bsz, num_head, kept_len, hidden_dim});
        for (int64_t b = 0; b < bsz; ++b) {
            auto idx_b = torch::where(mask_.select(0, b) == 1)[0];
            TORCH_CHECK(
                idx_b.size(0) == kept_len,
                "Per-batch mask must keep the same number of tokens across the batch. Batch ", b,
                " has ", idx_b.size(0), ", expected ", kept_len
            );
            out.select(0, b).copy_(t.select(0, b).index_select(1, idx_b));
        }
        return out;
    };
    
    if (result[0].defined()) {
        result[0] = apply_per_batch_index_select(result[0]);
    }
    if (result[1].defined()) {
        result[1] = apply_per_batch_index_select(result[1]);
    }
    if (result[2].defined()) {
        result[2] = apply_per_batch_index_select(result[2]);
    }

    #ifdef VERBOSE
    // std::cout << "result[0] after size = " << result[0].sizes() << std::endl;
    // std::cout << "result[1] after size = " << result[1].sizes() << std::endl;
    // std::cout << "result[2] after size = " << result[2].sizes() << std::endl;
    // std::cout << "result[0] (Q grad)" << result[0].select(1, 0) << std::endl;
    // std::cout << "result[1] (K grad)" << result[1].select(1, 0) << std::endl;
    // std::cout << "result[2] (V grad)" << result[2].select(1, 0) << std::endl;
    // exit(-1);
    #endif

    return result;
}

void ScaledDotProductFlashAttentionFilter::transfer_saved_data(ScaledDotProductFlashAttentionBackward0* pre_node, at::Tensor* mask) {
    Variable query = pre_node->query_.unpack();
    Variable key = pre_node->key_.unpack();
    Variable value = pre_node->value_.unpack();
    Variable output = pre_node->output_.unpack(pre_node->getptr());
    Variable logsumexp = pre_node->logsumexp_.unpack(pre_node->getptr());
    Variable cum_seq_q = pre_node->cum_seq_q_.unpack(pre_node->getptr());
    Variable cum_seq_k = pre_node->cum_seq_k_.unpack(pre_node->getptr());
    #ifdef TORCH25
    Variable philox_seed = pre_node->philox_seed_.unpack(pre_node->getptr());
    Variable philox_offset = pre_node->philox_offset_.unpack(pre_node->getptr());
    #else
    Variable rng_state = pre_node->rng_state_.unpack(pre_node->getptr());
    Variable unused = pre_node->unused_.unpack(pre_node->getptr());
    #endif
    
    mask_ = mask->clone();

    #ifdef VERBOSE
    // std::cout << "ScaledDotProductFlashAttentionFilter::mask_ " << mask_ << std::endl;
    #endif
    
    query_ = SavedVariable(query, false);
    key_ = SavedVariable(key, false);  
    value_ = SavedVariable(value, false);
    output_ = SavedVariable(output, true);
    logsumexp_ = SavedVariable(logsumexp, true);
    cum_seq_q_ = SavedVariable(cum_seq_q, true);
    cum_seq_k_ = SavedVariable(cum_seq_k, true);
    #ifdef TORCH25
    philox_seed_ = SavedVariable(philox_seed, true);
    philox_offset_ = SavedVariable(philox_offset, true);
    #else
    rng_state_ = SavedVariable(rng_state, true);
    unused_ = SavedVariable(unused, true);
    #endif
    
    dropout_p = pre_node->dropout_p;
    is_causal = pre_node->is_causal;
    scale = pre_node->scale;
    max_k = pre_node->max_k;
    max_q = pre_node->max_q;
}

// ========== ScaledDotProductFlashAttentionFilterRestoreQ Implementation ==========

void ScaledDotProductFlashAttentionFilterRestoreQ::set_attn_bias_restoreq(Tensor& attn_bias_restoreq){
    TORCH_CHECK(mask_.defined(), "mask_ is not defined");
    if(attn_bias_restoreq.defined()){
        attn_bias_restoreq_ = SavedVariable(attn_bias_restoreq, false);
        #ifdef VERBOSE
        std::cout << "attn_bias_restoreq is defined = " << std::endl;
        #endif
    }
    else{
        auto query = query_.unpack();
        auto bsz = query.size(0);
        auto num_head = query.size(1);
        auto seq_len_after_filter = query.size(2);
        auto seq_len_before_filter = mask_.size(1);
        auto seq_len_restore = seq_len_before_filter-seq_len_after_filter;
        #ifdef USE_CUDNN_RESTORE
        attn_bias_restoreq = query.new_zeros_symint(
            {bsz, 1, seq_len_after_filter, seq_len_restore}, query.options());
        #else
        attn_bias_restoreq = query.new_zeros_symint(
            {bsz, num_head, seq_len_after_filter, seq_len_restore}, query.options());
        #endif
        auto causal_mask = torch::full_symint(
            {seq_len_before_filter, seq_len_before_filter}, -std::numeric_limits<float>::infinity(), 
            query.options());
        causal_mask.triu_(1);
        for(int i=0; i<bsz; i++){
            auto idx_i = torch::where(mask_.select(0, i) == 1)[0];
            auto idx_j = torch::where(mask_.select(0, i) == 0)[0];
            attn_bias_restoreq.select(0, i).copy_(
                causal_mask.index_select(0, idx_i).index_select(1, idx_j).expand_as(
                    attn_bias_restoreq.select(0, i)));
        }
        // auto multiple_of = 4;
        // if(seq_len_restore % multiple_of != 0){
        //     auto padding_len = multiple_of - (seq_len_restore % multiple_of);
        //     auto padding_tensor = torch::zeros_symint(
        //         {bsz, num_head, seq_len_after_filter, padding_len}, query.options());
        //     padding_tensor.fill_(-std::numeric_limits<float>::infinity());
        //     attn_bias_restoreq = torch::concat({attn_bias_restoreq, padding_tensor}, 3);
        // }
        attn_bias_restoreq_ = SavedVariable(attn_bias_restoreq, false);
        #ifdef VERBOSE
        std::cout << "attn_bias_restoreq is not defined = " << std::endl;
        #endif
    }
}

void ScaledDotProductFlashAttentionFilterRestoreQ::split_kv(){
    Variable key = key_.unpack();
    Variable value = value_.unpack();
    auto bsz = key.sizes()[0];
    auto num_head = key.sizes()[1];
    auto seq_len_before_filter = key.sizes()[2];
    auto seq_len_after_filter = mask_.select(0, 0).sum().item<int>();
    auto hidden_dim = key.sizes()[3];
    auto seq_pending_restore = seq_len_before_filter - seq_len_after_filter;

    auto apply_per_batch_index_select = [&](const at::Tensor& t, const int select_flag) -> at::Tensor {
        if (!t.defined()) return t;
        TORCH_CHECK(t.dim() == 4, "Expected 4D tensor [bsz, num_head, seq_len, hidden_dim], got ", t.sizes());
        at::Tensor out = t.new_empty(
            {bsz, num_head, select_flag==1?seq_len_after_filter:seq_pending_restore, hidden_dim});
        for (int64_t b = 0; b < bsz; ++b) {
            auto idx_b = torch::where(mask_.select(0, b) == select_flag)[0];
            out.select(0, b).copy_(t.select(0, b).index_select(1, idx_b));
        }
        return out;
    };
    
    Variable key_keep;
    Variable key_remove;
    Variable value_keep;
    Variable value_remove;

    if(bsz == 1) {
        // new_var = var.index_select(filter_dims.index, mask_index_seq_);
        auto mask_keep = torch::where(mask_.select(0, 0) == 1)[0];
        auto mask_remove = torch::where(mask_.select(0, 0) == 0)[0];
        key_keep = key.index_select(2, mask_keep);
        key_remove = key.index_select(2, mask_remove);
        value_keep = value.index_select(2, mask_keep);
        value_remove = value.index_select(2, mask_remove);
    } else {
        key_keep = apply_per_batch_index_select(key, 1);
        value_keep = apply_per_batch_index_select(value, 1);
        key_remove = apply_per_batch_index_select(key, 0);
        value_remove = apply_per_batch_index_select(value, 0);
    }

    // Update the saved variable
    key_ = SavedVariable(key_keep, false);
    value_ = SavedVariable(value_keep, false);
    k_remove_ = SavedVariable(key_remove, false);
    v_remove_ = SavedVariable(value_remove, false);

    key.reset();
    value.reset();
}

// variable_list ScaledDotProductFlashAttentionFilterRestoreQ::apply(variable_list&& grads) {
//     // size of key and value = [bsz, num_head, seq_len, hidden_dim]
//     Variable query = query_.unpack();
//     Variable key = key_.unpack();
//     Variable value = value_.unpack();
//     Variable logsumexp = logsumexp_.unpack(this->getptr());
//     Variable output = output_.unpack(this->getptr());
//     Variable philox_seed = philox_seed_.unpack(this->getptr());
//     Variable philox_offset = philox_offset_.unpack(this->getptr());

//     Variable key_remove = k_remove_.unpack();
//     Variable value_remove = v_remove_.unpack();

//     auto result = ScaledDotProductFlashAttentionBackward0::apply(std::move(grads));

//     /*
//     _scaled_dot_product_efficient_attention_backward(const at::Tensor & grad_out_, const at::Tensor & query, 
//     const at::Tensor & key, const at::Tensor & value, const at::Tensor & attn_bias,
//      const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & philox_seed, 
//      const at::Tensor & philox_offset, double dropout_p, ::std::array<bool,4> grad_input_mask, 
//      bool is_causal=false, ::std::optional<double> scale=::std::nullopt)
//     */
//     Variable attn_bias_restoreq = attn_bias_restoreq_.unpack();
    
//     std::cout << "attn_bias_restoreq" << std::endl;
//     std::cout << attn_bias_restoreq << std::endl;
    
//     // auto result_restore = _scaled_dot_product_efficient_attention_backward(
//     //     std::move(grads)[0], query, key_remove, value_remove, attn_bias_restoreq, output, 
//     //     logsumexp, philox_seed, philox_offset, dropout_p, {true, true, true, false},
//     //     false, scale);
    
//     // result[0].add_(std::get<0>(result_restore));

//     return result;
// }

variable_list ScaledDotProductFlashAttentionFilterRestoreQ::apply(variable_list&& grads) {
    // size of key and value = [bsz, num_head, seq_len, hidden_dim]
    Variable query = query_.unpack();
    // Variable key = key_.unpack();
    // Variable value = value_.unpack();
    Variable logsumexp = logsumexp_.unpack(this->getptr());
    Variable output = output_.unpack(this->getptr());
    Variable cum_seq_q = cum_seq_q_.unpack(this->getptr());
    Variable cum_seq_k = cum_seq_k_.unpack(this->getptr());
    Variable key_remove = k_remove_.unpack();
    Variable value_remove = v_remove_.unpack();

    #ifdef TORCH25
    Variable philox_seed = philox_seed_.unpack(this->getptr()).to(query.device());
    Variable philox_offset = philox_offset_.unpack(this->getptr()).to(query.device()); 
    #else
    // Initialize zero tensors for philox_seed and philox_offset, each with one element
    Variable philox_seed = at::zeros({1}, query.options().dtype(at::kLong)).to(query.device());;
    Variable philox_offset = at::zeros({1}, query.options().dtype(at::kLong)).to(query.device());;
    #endif
    
    #ifdef VERBOSE
    std::cout << "Debug 0" << std::endl;
    #endif

    auto result = ScaledDotProductFlashAttentionBackward0::apply(std::move(grads));
    // OR
    // auto result_keep = _scaled_dot_product_flash_attention_backward_symint(
    //     grads[0], query, key, value, output, logsumexp, 
    //     cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset, scale);
    // variable_list result = {std::get<0>(result_keep), std::get<1>(result_keep), std::get<2>(result_keep)};
    
    #ifdef VERBOSE
    std::cout << "Debug 1" << std::endl;
    std::cout << "size of query " << query.sizes() << std::endl;
    std::cout << "size of key_remove " << key_remove.sizes() << std::endl;
    std::cout << "size of value_remove " << value_remove.sizes() << std::endl;
    auto key = key_.unpack();
    auto value = value_.unpack();
    std::cout << "size of key " << key.sizes() << std::endl;
    std::cout << "size of value " << value.sizes() << std::endl;
    std::cout << "size of grad[0]" << grads[0].sizes() << std::endl;
    std::cout << "size of logsumexp" << logsumexp.sizes() << std::endl;
    #endif

    /*
    _scaled_dot_product_efficient_attention_backward(const at::Tensor & grad_out_, const at::Tensor & query, 
    const at::Tensor & key, const at::Tensor & value, const at::Tensor & attn_bias,
     const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & philox_seed, 
     const at::Tensor & philox_offset, double dropout_p, ::std::array<bool,4> grad_input_mask, 
     bool is_causal=false, ::std::optional<double> scale=::std::nullopt)
    */
    #ifdef USE_CUDNN_RESTORE
    Variable attn_bias_restoreq = attn_bias_restoreq_.unpack();
    auto result_restore_cudnn = _scaled_dot_product_cudnn_attention_backward_symint(
        grads[0], query, key_remove, value_remove, output, logsumexp,
        philox_seed, philox_offset, attn_bias_restoreq, cum_seq_q, cum_seq_k, 
        max_q, max_k, dropout_p, false, scale
    );
    result[0].add_(std::get<0>(result_restore_cudnn));
    #else
    Variable attn_bias_restoreq = attn_bias_restoreq_.unpack();
    auto result_restore = _scaled_dot_product_efficient_attention_backward(
        grads[0], query, key_remove, value_remove, attn_bias_restoreq, output, 
        logsumexp, philox_seed, philox_offset, dropout_p, {true, true, true, false},
        false, scale);
    result[0].add_(std::get<0>(result_restore));
    #endif

    #ifdef VERBOSE
    std::cout << "Debug 2" << std::endl;
    #endif


    #ifdef VERBOSE
    // std::cout << "result[0] after size = " << result[0].sizes() << std::endl;
    // std::cout << "result[1] after size = " << result[1].sizes() << std::endl;
    // std::cout << "result[2] after size = " << result[2].sizes() << std::endl;
    // std::cout << "result[0] (Q grad)" << result[0].select(1, 0) << std::endl;
    // std::cout << "result[1] (K grad)" << result[1].select(1, 0) << std::endl;
    // std::cout << "result[2] (V grad)" << result[2].select(1, 0) << std::endl;
    // exit(-1);
    #endif
    
    return result;
}

void ScaledDotProductFlashAttentionFilterRestoreQ::transfer_saved_data(ScaledDotProductFlashAttentionBackward0* pre_node, at::Tensor* mask) {
    Variable query = pre_node->query_.unpack();
    Variable key = pre_node->key_.unpack();
    Variable value = pre_node->value_.unpack();
    Variable output = pre_node->output_.unpack(pre_node->getptr());
    Variable logsumexp = pre_node->logsumexp_.unpack(pre_node->getptr());
    Variable cum_seq_q = pre_node->cum_seq_q_.unpack(pre_node->getptr());
    Variable cum_seq_k = pre_node->cum_seq_k_.unpack(pre_node->getptr());
;
    
    mask_ = mask->clone();
    query_ = SavedVariable(query, false);
    key_ = SavedVariable(key, false);  
    value_ = SavedVariable(value, false);
    output_ = SavedVariable(output, true);
    logsumexp_ = SavedVariable(logsumexp, true);
    cum_seq_q_ = SavedVariable(cum_seq_q, true);
    cum_seq_k_ = SavedVariable(cum_seq_k, true);

    #ifdef TORCH25
    Variable philox_seed = pre_node->philox_seed_.unpack(pre_node->getptr());
    Variable philox_offset = pre_node->philox_offset_.unpack(pre_node->getptr());
    philox_seed_ = SavedVariable(philox_seed, true);
    philox_offset_ = SavedVariable(philox_offset, true);
    #else
    Variable rng_state = pre_node->rng_state_.unpack(pre_node->getptr());
    Variable unused = pre_node->unused_.unpack(pre_node->getptr());
    rng_state_ = SavedVariable(rng_state, true);
    unused_ = SavedVariable(unused, true);
    #endif
    
    dropout_p = pre_node->dropout_p;
    is_causal = pre_node->is_causal;
    scale = pre_node->scale;
    max_k = pre_node->max_k;
    max_q = pre_node->max_q;
}

// ========== ScaledDotProductEfficientAttentionFilter Implementation ==========

variable_list ScaledDotProductEfficientAttentionFilter::apply(variable_list&& grads) {

    // Generate the attn_bias before backward
    auto grad_sizes = grads[0].sizes();
    auto bsz = grad_sizes[0];
    auto num_head = grad_sizes[1];
    auto seq_len_reduced = grad_sizes[2];
    auto seq_len_before_filter = mask_.sizes()[1];
    auto hidden_dims = grad_sizes[3];

    #ifdef VERBOSE
    // std::cout << "query size = " << query_.unpack().sizes() << std::endl;
    // std::cout << "key size = " << key_.unpack().sizes() << std::endl;
    // std::cout << "value size = " << value_.unpack().sizes() << std::endl;
    // std::cout << "logsumexp size = " << log_sumexp_.unpack().sizes() << std::endl;
    // // std::cout << "output size = " << output_.unpack().sizes() << std::endl;
    // std::cout << "bsz = " << bsz << std::endl;
    // std::cout << "num_head = " << num_head << std::endl;
    // std::cout << "seq_len_before_filter = " << seq_len_before_filter << std::endl;
    // std::cout << "seq_len_reduced = " << seq_len_reduced << std::endl;
    // std::cout << "hidden_dims = " << hidden_dims << std::endl;
    #endif
    
    #ifdef VERBOSE
    // Variable attn_bias_restoreq = attn_bias_restoreq_.unpack();
    // std::cout << "attn_bias_restoreq size = " << attn_bias_restoreq.sizes() << std::endl;
    // std::cout << "attn_bias_restoreq " << attn_bias_restoreq.select(0, 0).select(0, 1) << std::endl;
    #endif
    
    is_causal = false;
    attn_bias_ = SavedVariable(attn_bias_restoreq_.unpack(), true);

    auto result = ScaledDotProductEfficientAttentionBackward0::apply(std::move(grads));

    auto apply_mask = [&](const at::Tensor& t) -> at::Tensor {
        if (!t.defined()) return t;
        TORCH_CHECK(t.dim() == 4, "Expected 4D tensor [bsz, num_head, seq_len, hidden_dim], got ", t.sizes());
        const int64_t bsz = t.size(0);
        const int64_t num_head = t.size(1);
        const int64_t seq_len = t.size(2);
        const int64_t hidden_dim = t.size(3);

        auto mask_sum = mask_.sum(1);
        int64_t kept_len = mask_sum[0].item<int64_t>();
        for (int64_t b = 0; b < bsz; ++b) {
            TORCH_CHECK(mask_sum[b].item<int64_t>() == kept_len, "All batch must keep the same number of tokens");
        }

        at::Tensor out = t.new_empty({bsz, num_head, kept_len, hidden_dim});
        for (int64_t b = 0; b < bsz; ++b) {
            auto idx_b = torch::where(mask_.select(0, b) == 1)[0];
            TORCH_CHECK(idx_b.size(0) == kept_len, "Batch ", b, " mask kept ", idx_b.size(0), " tokens, expected ", kept_len);
            out.select(0, b).copy_(t.select(0, b).index_select(1, idx_b));
        }
        return out;
    };

    #ifdef VERBOSE
    // std::cout << "result[0] (Q grad)" << result[0].select(1, 0) << std::endl;
    // std::cout << "result[1] (K grad)" << result[1].select(1, 0) << std::endl;
    // std::cout << "result[2] (V grad)" << result[2].select(1, 0) << std::endl;
    // exit(-1);
    #endif

    if (result[1].defined()) {
        result[1] = apply_mask(result[1]);
    }
    if (result[2].defined()) {
        result[2] = apply_mask(result[2]);
    }

    #ifdef VERBOSE
    // std::cout << "result[0] after size = " << result[0].sizes() << std::endl;
    // std::cout << "result[1] after size = " << result[1].sizes() << std::endl;
    // std::cout << "result[2] after size = " << result[2].sizes() << std::endl;
    #endif

    return result;
}

void ScaledDotProductEfficientAttentionFilter::set_attn_bias_restoreq(Tensor& attn_bias_restoreq){
    TORCH_CHECK(mask_.defined(), "mask_ is not defined");
    if(attn_bias_restoreq.defined()){
        attn_bias_restoreq_ = SavedVariable(attn_bias_restoreq, false);
        #ifdef VERBOSE
        std::cout << "attn_bias_restoreq is defined = " << std::endl;
        #endif
    }
    else{
        auto query = query_.unpack();
        auto bsz = query.size(0);
        auto num_head = query.size(1);
        auto seq_len_reduced = query.size(2);
        auto seq_len_before_filter = mask_.size(1);
        attn_bias_restoreq = query.new_zeros_symint(
            {bsz, num_head, seq_len_reduced, seq_len_before_filter}, query.options());
        auto causal_mask = torch::full_symint(
            {seq_len_before_filter, seq_len_before_filter}, -std::numeric_limits<float>::infinity(), 
            query.options());
        causal_mask.triu_(1);
        for(int i=0; i<bsz; i++){
            auto idx_i = torch::where(mask_.select(0, i) == 1)[0];
            attn_bias_restoreq.select(0, i).copy_(
                causal_mask.index_select(0, idx_i).expand_as(attn_bias_restoreq.select(0, i)));
        }
        attn_bias_restoreq_ = SavedVariable(attn_bias_restoreq, false);
        #ifdef VERBOSE
        std::cout << "attn_bias_restoreq is not defined = " << std::endl;
        #endif
    }
}

void ScaledDotProductEfficientAttentionFilter::transfer_saved_data(ScaledDotProductEfficientAttentionBackward0* pre_node, at::Tensor* mask) {
    Variable query = pre_node->query_.unpack();
    Variable key = pre_node->key_.unpack();
    Variable value = pre_node->value_.unpack();
    Variable output = pre_node->output_.unpack(pre_node->getptr());
    Variable log_sumexp = pre_node->log_sumexp_.unpack(pre_node->getptr());
    Variable philox_seed = pre_node->philox_seed_.unpack(pre_node->getptr());
    Variable philox_offset = pre_node->philox_offset_.unpack(pre_node->getptr());

    #ifdef VERBOSE
    // std::cout << "log_sumexp = " << log_sumexp.select(0, 0).select(0, 0) << std::endl;
    #endif
    
    mask_ = mask->clone();

    query_ = SavedVariable(query, false);
    key_ = SavedVariable(key, false);  
    value_ = SavedVariable(value, false);
    output_ = SavedVariable(output, true);
    log_sumexp_ = SavedVariable(log_sumexp, true);
    philox_seed_ = SavedVariable(philox_seed, true);
    philox_offset_ = SavedVariable(philox_offset, true);
    
    dropout_p = pre_node->dropout_p;
    is_causal = false;
    scale = pre_node->scale;
}

// ========== Sequence Backward Implementation ==========

variable_list UnsequeezeBszSeqLenBackward0::apply(variable_list&& grads) {
    std::lock_guard<std::mutex> lock(mutex_);
    variable_list grad_inputs(1);
    auto mask_index = mask_index_.unpack();
    grad_inputs[0] = grads[0].index({mask_index});
    return grad_inputs;
}

variable_list UnsequeezeBszSeqLenBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_index_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_index_);
    return result;
}

variable_list SequeezeBszSeqLenBackward0::apply(variable_list&& grads) {
    std::lock_guard<std::mutex> lock(mutex_);
    variable_list grad_inputs(1);
    auto mask = mask_.unpack();
    Variable padded_grad = grads[0].new_zeros_symint({self_sym_sizes[0]*self_sym_sizes[1], self_sym_sizes[2]}, self_options);
    padded_grad.index_copy_(0, mask.nonzero().flatten(), grads[0]);
    grad_inputs[0] = padded_grad.view_symint(self_sym_sizes);
    return grad_inputs;
}

variable_list SequeezeBszSeqLenBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    return result;
}

// ========== ScaledDotProductCudnnAttentionFilter Implementation ==========

variable_list ScaledDotProductCudnnAttentionFilter::apply(variable_list&& grads) {

    // Generate the attn_bias before backward
    auto grad_sizes = grads[0].sizes();
    auto bsz = grad_sizes[0];
    auto num_head = grad_sizes[1];
    auto seq_len_reduced = grad_sizes[2];
    auto seq_len_before_filter = mask_.sizes()[1];
    auto hidden_dims = grad_sizes[3];

    auto query = query_.unpack();
    auto causal_mask = torch::full_symint(
      {seq_len_before_filter, seq_len_before_filter}, -std::numeric_limits<float>::infinity(), 
      query.options());
    causal_mask.triu_(1);

    // #ifdef VERBOSE
    // std::cout << "causal_mask " << causal_mask << std::endl;
    // #endif
    
    Variable attn_bias = query.new_zeros_symint(
      {bsz, num_head, seq_len_reduced, seq_len_before_filter}, query.options());
    
    for(int i=0; i<bsz; i++){
      auto idx_i = torch::where(mask_.select(0, i) == 1)[0];
      attn_bias.select(0, i).copy_(causal_mask.index_select(0, idx_i).expand_as(attn_bias.select(0, i)));
      // std::cout << "attn_bias.select(0, i) " << attn_bias.select(0, i).select(0, 1) << std::endl;
    }
    
    is_causal = false;
    attn_bias_ = SavedVariable(attn_bias, true);

    auto result = ScaledDotProductCudnnAttentionBackward0::apply(std::move(grads));

    auto apply_mask = [&](const at::Tensor& t) -> at::Tensor {
        if (!t.defined()) return t;
        TORCH_CHECK(t.dim() == 4, "Expected 4D tensor [bsz, num_head, seq_len, hidden_dim], got ", t.sizes());
        const int64_t bsz = t.size(0);
        const int64_t num_head = t.size(1);
        const int64_t seq_len = t.size(2);
        const int64_t hidden_dim = t.size(3);

        auto mask_sum = mask_.sum(1);
        int64_t kept_len = mask_sum[0].item<int64_t>();
        for (int64_t b = 0; b < bsz; ++b) {
            TORCH_CHECK(mask_sum[b].item<int64_t>() == kept_len, "All batch must keep the same number of tokens");
        }

        at::Tensor out = t.new_empty({bsz, num_head, kept_len, hidden_dim});
        for (int64_t b = 0; b < bsz; ++b) {
            auto idx_b = torch::where(mask_.select(0, b) == 1)[0];
            TORCH_CHECK(idx_b.size(0) == kept_len, "Batch ", b, " mask kept ", idx_b.size(0), " tokens, expected ", kept_len);
            out.select(0, b).copy_(t.select(0, b).index_select(1, idx_b));
        }
        return out;
    };

    #ifdef VERBOSE
    // std::cout << "result[0] (Q grad)" << result[0].select(1, 0) << std::endl;
    // std::cout << "result[1] (K grad)" << result[1].select(1, 0) << std::endl;
    // std::cout << "result[2] (V grad)" << result[2].select(1, 0) << std::endl;
    // exit(-1);
    #endif

    if (result[1].defined()) {
        result[1] = apply_mask(result[1]);
    }
    if (result[2].defined()) {
        result[2] = apply_mask(result[2]);
    }

    #ifdef VERBOSE
    // std::cout << "result[0] after size = " << result[0].sizes() << std::endl;
    // std::cout << "result[1] after size = " << result[1].sizes() << std::endl;
    // std::cout << "result[2] after size = " << result[2].sizes() << std::endl;
    #endif

    return result;
}

void ScaledDotProductCudnnAttentionFilter::transfer_saved_data(ScaledDotProductCudnnAttentionBackward0* pre_node, at::Tensor* mask) {
    Variable query = pre_node->query_.unpack();
    Variable key = pre_node->key_.unpack();
    Variable value = pre_node->value_.unpack();
    Variable output = pre_node->output_.unpack(pre_node->getptr());
    Variable logsumexp = pre_node->logsumexp_.unpack(pre_node->getptr());
    // Variable attn_bias = pre_node->attn_bias_.unpack(pre_node->getptr());
    Variable cum_seq_q = pre_node->cum_seq_q_.unpack(pre_node->getptr());
    Variable cum_seq_k = pre_node->cum_seq_k_.unpack(pre_node->getptr());
    Variable philox_seed = pre_node->philox_seed_.unpack(pre_node->getptr());
    Variable philox_offset = pre_node->philox_offset_.unpack(pre_node->getptr());
    
    mask_ = mask->clone();

    query_ = SavedVariable(query, false);
    key_ = SavedVariable(key, false);  
    value_ = SavedVariable(value, false);
    output_ = SavedVariable(output, true);
    logsumexp_ = SavedVariable(logsumexp, true);
    cum_seq_q_ = SavedVariable(cum_seq_q, true);
    cum_seq_k_ = SavedVariable(cum_seq_k, true);
    philox_seed_ = SavedVariable(philox_seed, true);
    philox_offset_ = SavedVariable(philox_offset, true);
    
    dropout_p = pre_node->dropout_p;
    // is_causal = pre_node->is_causal;
    is_causal = false;
    scale = pre_node->scale;
    max_k = pre_node->max_k;
    max_q = pre_node->max_q;
    
    // Handle attn_bias_ if it exists
    // if (pre_node->attn_bias_.defined()) {
    //     Variable attn_bias = pre_node->attn_bias_.unpack(pre_node->getptr());
    //     attn_bias_ = SavedVariable(attn_bias, true);
    // }
}

} // namespace token_filter
