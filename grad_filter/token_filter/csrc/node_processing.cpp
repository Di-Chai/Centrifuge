#include "common.h"
#include "graph_filter.h"

namespace token_filter {

// ========== Utility Functions Implementation ==========

void transfer_inputmeta_and_edge(Node* source, Node* target, bool transfer_edges, bool transfer_input_meta) {
    // transfer input metadata
    if(transfer_input_meta)
        for(int i=0; i<source->num_inputs(); i++) {
            const InputMetadata& meta = source->input_metadata(i);
            target->add_input_metadata(meta.options(), meta.shape_as_dim_vector(), meta.is_tensor_subclass(), meta.is_nested_tensor());
        }
    // transfer edges
    if(transfer_edges)
        for(Edge& edge : source->next_edges()) target->add_next_edge(edge);
}

void update_grad_input_shape(Node* fn, c10::SymInt new_shape) {
    fn->mutable_input_metadata(0).mutable_shape_as_dim_vector()[0] = new_shape;
}

// ========== Node Processing Functions Implementation ==========

void node_processing_input_metadata(Node* fn, GraphFilter* graph_filter) {
    for(int i=0; i<fn->num_inputs(); i++) {
        auto input_metadata_shape = fn->input_metadata(i).shape_as_dim_vector();
        std::vector<c10::SymInt> input_metadata_shape_vec(input_metadata_shape.begin(), input_metadata_shape.end());
        FilterDims filter_dims = graph_filter->get_filter_dims_mode(input_metadata_shape_vec);
        
        #ifdef VERBOSE
        std::cout << "process_sizes filter_dims.index=" << filter_dims.index << " filter_dims.mode=" << static_cast<int>(filter_dims.mode) << std::endl;
        std::cout << "Input shape (Before) = " << input_metadata_shape << std::endl;
        #endif
        
        if(filter_dims.index == -1) continue;
        
        switch (filter_dims.mode) {
            case FilterDimsMode::SEQ_LEN:
                if(filter_dims.index < 10) {
                    fn->mutable_input_metadata(i).mutable_shape_as_dim_vector()[filter_dims.index] = graph_filter->seq_after_filter;
                } else {
                    fn->mutable_input_metadata(i).mutable_shape_as_dim_vector()[filter_dims.index % 10] = graph_filter->seq_after_filter;
                    fn->mutable_input_metadata(i).mutable_shape_as_dim_vector()[filter_dims.index / 10] = graph_filter->seq_after_filter;
                }
                break;
            case FilterDimsMode::BTH_SEQ_LEN:
                fn->mutable_input_metadata(i).mutable_shape_as_dim_vector()[filter_dims.index] = graph_filter->seq_after_filter * graph_filter->bsz;
                break;
            case FilterDimsMode::SEQ_LEN_MINUS_1:
                fn->mutable_input_metadata(i).mutable_shape_as_dim_vector()[filter_dims.index] = graph_filter->seq_after_filter - 1;
                break;
            case FilterDimsMode::BTH_SEQ_LEN_MINUS_1:
                fn->mutable_input_metadata(i).mutable_shape_as_dim_vector()[filter_dims.index] = (graph_filter->seq_after_filter - 1) * graph_filter->bsz;
                break;
        }
        
        #ifdef VERBOSE
        std::cout << "Input shape (After) = " << fn->input_metadata(i).shape_as_dim_vector() << std::endl;
        #endif
    }
}

void node_processing(Node* fn, GraphFilter* graph_filter) {
    #ifdef VERBOSE
    std::cout << "#############" << std::endl;
    std::cout << "Node name: " << fn->name() << std::endl;
    std::cout << "# of Input = " << fn->num_inputs() << std::endl;
    #endif

    graph_filter->process_counter += 1;

    node_processing_input_metadata(fn, graph_filter);

  /* $$ start of code generation $$ */

  if(fn->name() == "AddmmBackward0") {
    AddmmBackward0* op_fn = static_cast<AddmmBackward0*>(fn); 
    auto unpacked_mat2 = op_fn->mat2_.unpack();
    if(unpacked_mat2.defined()) op_fn->mat2_ = graph_filter->process_variable(unpacked_mat2, false);
    auto unpacked_mat1 = op_fn->mat1_.unpack();
    if(unpacked_mat1.defined()) op_fn->mat1_ = graph_filter->process_variable(unpacked_mat1, false); 
    graph_filter->process_sizes(op_fn->mat1_sym_sizes);
    graph_filter->process_sizes(op_fn->mat2_sym_sizes);
  }

  if(fn->name() == "BaddbmmBackward0") {
    BaddbmmBackward0* op_fn = static_cast<BaddbmmBackward0*>(fn); 
    auto unpacked_batch2 = op_fn->batch2_.unpack();
    if(unpacked_batch2.defined()) op_fn->batch2_ = graph_filter->process_variable(unpacked_batch2, false);
    auto unpacked_batch1 = op_fn->batch1_.unpack();
    if(unpacked_batch1.defined()) op_fn->batch1_ = graph_filter->process_variable(unpacked_batch1, false); 
    
  }

  if(fn->name() == "BmmBackward0") {
    BmmBackward0* op_fn = static_cast<BmmBackward0*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false);
    auto unpacked_mat2 = op_fn->mat2_.unpack();
    if(unpacked_mat2.defined()) op_fn->mat2_ = graph_filter->process_variable(unpacked_mat2, false); 
    
  }

  if(fn->name() == "ClampBackward1") {
    ClampBackward1* op_fn = static_cast<ClampBackward1*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false); 
    
  }

  if(fn->name() == "DivBackward0") {
    DivBackward0* op_fn = static_cast<DivBackward0*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false);
    auto unpacked_other = op_fn->other_.unpack();
    if(unpacked_other.defined()) op_fn->other_ = graph_filter->process_variable(unpacked_other, false); 
    
  }

  if(fn->name() == "NativeDropoutBackward0") {
    NativeDropoutBackward0* op_fn = static_cast<NativeDropoutBackward0*>(fn); 
    auto unpacked_result1 = op_fn->result1_.unpack(op_fn->getptr());
    if(unpacked_result1.defined()) op_fn->result1_ = graph_filter->process_variable(unpacked_result1, true); 
    
  }

  if(fn->name() == "ExpandBackward0") {
    ExpandBackward0* op_fn = static_cast<ExpandBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "MeanBackward0") {
    MeanBackward0* op_fn = static_cast<MeanBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
    graph_filter->process_sizes(op_fn->self_sym_numel);
  }

  if(fn->name() == "MeanBackward1") {
    MeanBackward1* op_fn = static_cast<MeanBackward1*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
    graph_filter->process_sizes(op_fn->self_sym_numel);
  }

  if(fn->name() == "MmBackward0") {
    MmBackward0* op_fn = static_cast<MmBackward0*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false);
    auto unpacked_mat2 = op_fn->mat2_.unpack();
    if(unpacked_mat2.defined()) op_fn->mat2_ = graph_filter->process_variable(unpacked_mat2, false); 
    graph_filter->process_sizes(op_fn->mat2_sym_sizes);
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "MulBackward0") {
    MulBackward0* op_fn = static_cast<MulBackward0*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false);
    auto unpacked_other = op_fn->other_.unpack();
    if(unpacked_other.defined()) op_fn->other_ = graph_filter->process_variable(unpacked_other, false); 
    
  }

  if(fn->name() == "NativeLayerNormBackward0") {
    NativeLayerNormBackward0* op_fn = static_cast<NativeLayerNormBackward0*>(fn); 
    auto unpacked_input = op_fn->input_.unpack();
    if(unpacked_input.defined()) op_fn->input_ = graph_filter->process_variable(unpacked_input, false);
    auto unpacked_weight = op_fn->weight_.unpack();
    if(unpacked_weight.defined()) op_fn->weight_ = graph_filter->process_variable(unpacked_weight, false);
    auto unpacked_bias = op_fn->bias_.unpack();
    if(unpacked_bias.defined()) op_fn->bias_ = graph_filter->process_variable(unpacked_bias, false);
    auto unpacked_result1 = op_fn->result1_.unpack(op_fn->getptr());
    if(unpacked_result1.defined()) op_fn->result1_ = graph_filter->process_variable(unpacked_result1, true);
    auto unpacked_result2 = op_fn->result2_.unpack(op_fn->getptr());
    if(unpacked_result2.defined()) op_fn->result2_ = graph_filter->process_variable(unpacked_result2, true); 
    
  }

  if(fn->name() == "PowBackward0") {
    PowBackward0* op_fn = static_cast<PowBackward0*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false); 
    
  }

  if(fn->name() == "ReshapeAliasBackward0") {
    ReshapeAliasBackward0* op_fn = static_cast<ReshapeAliasBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "RsqrtBackward0") {
    RsqrtBackward0* op_fn = static_cast<RsqrtBackward0*>(fn); 
    auto unpacked_result = op_fn->result_.unpack(op_fn->getptr());
    if(unpacked_result.defined()) op_fn->result_ = graph_filter->process_variable(unpacked_result, true); 
    
  }

  if(fn->name() == "SelectBackward0") {
    SelectBackward0* op_fn = static_cast<SelectBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "SliceBackward0") {
    SliceBackward0* op_fn = static_cast<SliceBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "SplitBackward0") {
    SplitBackward0* op_fn = static_cast<SplitBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "SumBackward0") {
    SumBackward0* op_fn = static_cast<SumBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "SumBackward1") {
    SumBackward1* op_fn = static_cast<SumBackward1*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "TanhBackward0") {
    TanhBackward0* op_fn = static_cast<TanhBackward0*>(fn); 
    auto unpacked_result = op_fn->result_.unpack(op_fn->getptr());
    if(unpacked_result.defined()) op_fn->result_ = graph_filter->process_variable(unpacked_result, true); 
    
  }

  if(fn->name() == "UnsafeViewBackward0") {
    UnsafeViewBackward0* op_fn = static_cast<UnsafeViewBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "ViewBackward0") {
    ViewBackward0* op_fn = static_cast<ViewBackward0*>(fn); 
     
    graph_filter->process_sizes(op_fn->self_sym_sizes);
  }

  if(fn->name() == "EmbeddingBackward0") {
    EmbeddingBackward0* op_fn = static_cast<EmbeddingBackward0*>(fn); 
    auto unpacked_indices = op_fn->indices_.unpack();
    if(unpacked_indices.defined()) op_fn->indices_ = graph_filter->process_variable(unpacked_indices, false); 
    
  }

  if(fn->name() == "NllLossBackward0") {
    NllLossBackward0* op_fn = static_cast<NllLossBackward0*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false);
    auto unpacked_target = op_fn->target_.unpack();
    if(unpacked_target.defined()) op_fn->target_ = graph_filter->process_variable(unpacked_target, false);
    auto unpacked_weight = op_fn->weight_.unpack();
    if(unpacked_weight.defined()) op_fn->weight_ = graph_filter->process_variable(unpacked_weight, false);
    auto unpacked_total_weight = op_fn->total_weight_.unpack(op_fn->getptr());
    if(unpacked_total_weight.defined()) op_fn->total_weight_ = graph_filter->process_variable(unpacked_total_weight, true); 
    
  }

  if(fn->name() == "SiluBackward0") {
    SiluBackward0* op_fn = static_cast<SiluBackward0*>(fn); 
    auto unpacked_self = op_fn->self_.unpack();
    if(unpacked_self.defined()) op_fn->self_ = graph_filter->process_variable(unpacked_self, false); 
    
  }

  if(fn->name() == "LogSoftmaxBackward0") {
    LogSoftmaxBackward0* op_fn = static_cast<LogSoftmaxBackward0*>(fn); 
    auto unpacked_result = op_fn->result_.unpack(op_fn->getptr());
    if(unpacked_result.defined()) op_fn->result_ = graph_filter->process_variable(unpacked_result, true); 
    
  }

  if(fn->name() == "SoftmaxBackward0") {
    SoftmaxBackward0* op_fn = static_cast<SoftmaxBackward0*>(fn); 
    auto unpacked_result = op_fn->result_.unpack(op_fn->getptr());
    if(unpacked_result.defined()) op_fn->result_ = graph_filter->process_variable(unpacked_result, true); 
    
  }
  /* $$ end of code generation $$ */

  if(fn->name() == "ScaledDotProductEfficientAttentionBackward0") {
    ScaledDotProductEfficientAttentionBackward0* op_fn = static_cast<ScaledDotProductEfficientAttentionBackward0*>(fn); 
    auto unpacked_query = op_fn->query_.unpack();
    if(unpacked_query.defined()) op_fn->query_ = graph_filter->process_variable(unpacked_query, false);
    // auto unpacked_key = op_fn->key_.unpack();
    // if(unpacked_key.defined()) op_fn->key_ = graph_filter->process_variable(unpacked_key, false);
    // auto unpacked_value = op_fn->value_.unpack();
    // if(unpacked_value.defined()) op_fn->value_ = graph_filter->process_variable(unpacked_value, false);
    auto unpacked_attn_bias = op_fn->attn_bias_.unpack();
    if(unpacked_attn_bias.defined()) op_fn->attn_bias_ = graph_filter->process_variable(unpacked_attn_bias, false);
    auto unpacked_output = op_fn->output_.unpack(op_fn->getptr());
    if(unpacked_output.defined()) op_fn->output_ = graph_filter->process_variable(unpacked_output, true);
    auto unpacked_log_sumexp = op_fn->log_sumexp_.unpack(op_fn->getptr());
    if(unpacked_log_sumexp.defined()) op_fn->log_sumexp_ = graph_filter->process_variable(unpacked_log_sumexp, true);
    auto unpacked_philox_seed = op_fn->philox_seed_.unpack(op_fn->getptr());
    if(unpacked_philox_seed.defined()) op_fn->philox_seed_ = graph_filter->process_variable(unpacked_philox_seed, true);
    auto unpacked_philox_offset = op_fn->philox_offset_.unpack(op_fn->getptr());
    if(unpacked_philox_offset.defined()) op_fn->philox_offset_ = graph_filter->process_variable(unpacked_philox_offset, true); 
    
  }

  if(fn->name() == "ScaledDotProductFlashAttentionBackward0") {
    ScaledDotProductFlashAttentionBackward0* op_fn = static_cast<ScaledDotProductFlashAttentionBackward0*>(fn); 
    auto unpacked_query = op_fn->query_.unpack();
    if(unpacked_query.defined()) op_fn->query_ = graph_filter->process_variable(unpacked_query, false);
    
    #ifdef FILTER_KV
    auto unpacked_key = op_fn->key_.unpack();
    if(unpacked_key.defined()) op_fn->key_ = graph_filter->process_variable(unpacked_key, false);
    auto unpacked_value = op_fn->value_.unpack();
    if(unpacked_value.defined()) op_fn->value_ = graph_filter->process_variable(unpacked_value, false);
    #endif
    
    auto unpacked_output = op_fn->output_.unpack(op_fn->getptr());
    if(unpacked_output.defined()) op_fn->output_ = graph_filter->process_variable(unpacked_output, true);
    auto unpacked_logsumexp = op_fn->logsumexp_.unpack(op_fn->getptr());
    if(unpacked_logsumexp.defined()) op_fn->logsumexp_ = graph_filter->process_variable(unpacked_logsumexp, true);
    auto unpacked_cum_seq_q = op_fn->cum_seq_q_.unpack(op_fn->getptr());
    if(unpacked_cum_seq_q.defined()) op_fn->cum_seq_q_ = graph_filter->process_variable(unpacked_cum_seq_q, true);
    auto unpacked_cum_seq_k = op_fn->cum_seq_k_.unpack(op_fn->getptr());
    if(unpacked_cum_seq_k.defined()) op_fn->cum_seq_k_ = graph_filter->process_variable(unpacked_cum_seq_k, true);
    // auto unpacked_philox_seed = op_fn->philox_seed_.unpack(op_fn->getptr());
    // if(unpacked_philox_seed.defined()) op_fn->philox_seed_ = graph_filter->process_variable(unpacked_philox_seed, true);
    // auto unpacked_philox_offset = op_fn->philox_offset_.unpack(op_fn->getptr());
    // if(unpacked_philox_offset.defined()) op_fn->philox_offset_ = graph_filter->process_variable(unpacked_philox_offset, true); 
    graph_filter->process_sizes(op_fn->max_q);
    graph_filter->process_sizes(op_fn->max_k);
  }

  if(fn->name() == "ScaledDotProductCudnnAttentionBackward0") {
    ScaledDotProductCudnnAttentionBackward0* op_fn = static_cast<ScaledDotProductCudnnAttentionBackward0*>(fn); 
    auto unpacked_query = op_fn->query_.unpack();
    if(unpacked_query.defined()) op_fn->query_ = graph_filter->process_variable(unpacked_query, false);
    // auto unpacked_key = op_fn->key_.unpack();
    // if(unpacked_key.defined()) op_fn->key_ = graph_filter->process_variable(unpacked_key, false);
    // auto unpacked_value = op_fn->value_.unpack();
    // if(unpacked_value.defined()) op_fn->value_ = graph_filter->process_variable(unpacked_value, false);
    auto unpacked_attn_bias = op_fn->attn_bias_.unpack();
    if(unpacked_attn_bias.defined()) op_fn->attn_bias_ = graph_filter->process_variable(unpacked_attn_bias, false);
    auto unpacked_output = op_fn->output_.unpack(op_fn->getptr());
    if(unpacked_output.defined()) op_fn->output_ = graph_filter->process_variable(unpacked_output, true);
    auto unpacked_logsumexp = op_fn->logsumexp_.unpack(op_fn->getptr());
    if(unpacked_logsumexp.defined()) op_fn->logsumexp_ = graph_filter->process_variable(unpacked_logsumexp, true);
    auto unpacked_cum_seq_q = op_fn->cum_seq_q_.unpack(op_fn->getptr());
    if(unpacked_cum_seq_q.defined()) op_fn->cum_seq_q_ = graph_filter->process_variable(unpacked_cum_seq_q, true);
    auto unpacked_cum_seq_k = op_fn->cum_seq_k_.unpack(op_fn->getptr());
    if(unpacked_cum_seq_k.defined()) op_fn->cum_seq_k_ = graph_filter->process_variable(unpacked_cum_seq_k, true);
    auto unpacked_philox_seed = op_fn->philox_seed_.unpack(op_fn->getptr());
    if(unpacked_philox_seed.defined()) op_fn->philox_seed_ = graph_filter->process_variable(unpacked_philox_seed, true);
    auto unpacked_philox_offset = op_fn->philox_offset_.unpack(op_fn->getptr());
    if(unpacked_philox_offset.defined()) op_fn->philox_offset_ = graph_filter->process_variable(unpacked_philox_offset, true); 
    graph_filter->process_sizes(op_fn->max_q);
    graph_filter->process_sizes(op_fn->max_k);
  }

  // PyNode from Megatron-LM
  if(
    fn->name() == "LinearWithGradAccumulationAndAsyncCommunicationBackward" ||
    fn->name() == "_ReduceFromModelParallelRegionBackward" ||
    fn->name() == "_GatherFromModelParallelRegionBackward" ||
    fn->name() == "_AllToAllBackward" ||
    fn->name() == "_LinearBackward" ||
    fn->name() == "_LayerNormLinearBackward" ||
    fn->name() == "_SplitAlongDimBackward" ||
    fn->name() == "FusedRoPEFuncBackward" ||
    fn->name() == "_ReduceScatterToSequenceParallelRegionBackward" ||
    fn->name() == "_VocabParallelCrossEntropyBackward" ||
    fn->name() == "MakeViewlessTensorBackward" ||
    fn->name() == "_LayerNormBackward" ||
    fn->name() == "_LayerNormLinearBackward" ||
    fn->name() == "CompiledFunctionBackward" ||
    fn->name() == "SwiGLUFunctionBackward" ||
    fn->name() == "FastLayerNormFNBackward" ||
    fn->name() == "ScaledUpperTriangMaskedSoftmaxBackward" // ||
    // fn->name() == "torch::autograd::CopySlices"
  ){
    PyNode* op_fn = static_cast<PyNode*>(fn);
    THPFunction* py_fn = (THPFunction*)op_fn->obj;
    // Process saved variables
    for(size_t var_index=0; var_index<py_fn->saved_variables.size(); var_index++){
      // #ifdef VERBOSE
      // if(fn->name() == "torch::autograd::CopySlices"){
      //   std::cout << "Debug 1 " << std::endl;
      //   std::cout << "saved_variables size " << py_fn->saved_variables.size() << std::endl;
      // }
      // #endif
      auto variable_unpacked = py_fn->saved_variables[var_index].unpack();
      // #ifdef VERBOSE
      // if(fn->name() == "torch::autograd::CopySlices"){
      //   std::cout << "Debug 2" << std::endl;
      // }
      // #endif
      if(variable_unpacked.defined())
      py_fn->saved_variables[var_index] = graph_filter->process_variable(variable_unpacked, false);
    }
    // Process Input and Output Size
    for(size_t var_index=0; var_index < py_fn->input_info.size(); var_index++){
      graph_filter->process_sizes(py_fn->input_info[var_index].size);
      // if(fn->name() == "torch::autograd::CopySlices")
      // std::cout << "py_fn->input_info size=" << py_fn->input_info[var_index].size << std::endl;
    }
    for(size_t var_index=0; var_index < py_fn->output_info.size(); var_index++){
      graph_filter->process_sizes(py_fn->output_info[var_index].size);
      // if(fn->name() == "torch::autograd::CopySlices")
      // std::cout << "py_fn->output_info size=" << py_fn->output_info[var_index].size << std::endl;
    }
    // if(fn->name() == "_LayerNormBackward"){
    //   std::cout << "LayerNormBackward" << std::endl;
    //   PyObject* ctx_obj = (PyObject*)py_fn;
    //   PyObject* existing_shape = PyObject_GetAttrString(ctx_obj, "inp_shape");
    //   std::cout << existing_shape << std::endl;
    // }
  }
}

} // namespace token_filter
