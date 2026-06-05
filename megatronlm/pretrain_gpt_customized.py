# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT with Custom Attention."""

import datetime
import os
import torch
import torch.nn.functional as F

import sys
sys.path.append("/data/Megatron-LM")
sys.path.append("/project")

from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module, ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    TENorm,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.utils import StragglerDetector, make_viewless_tensor
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.datasets.sft_dataset import SFTDataset

import megatron.legacy.model  # isort: skip

import torchviz
from grad_filter import token_filter

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


# ============================================================================
# Custom Attention Implementation using torch.nn.functional.scaled_dot_product_attention
# ============================================================================

class CustomScaledDotProductAttention(SelfAttention):
    """Custom Self Attention that uses PyTorch's scaled_dot_product_attention.
    
    This implementation supports tensor parallelism and maintains compatibility
    with Megatron-LM's distributed training framework.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        **kwargs  # Accept any additional arguments from Megatron-Core
    ):
        # Pass all arguments to parent class, including any extra kwargs
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs  # Forward additional arguments like model_comm_pgs
        )
        
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        # hidden_states: [sq, b, h]
        
        # =====================================================
        # Query, Key, and Value (with tensor parallelism)
        # =====================================================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.linear_qkv(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = tensor_parallel.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )
        
        # =====================================================
        # Apply rotary position embeddings if available
        # =====================================================
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2
                
            query_layer = self.apply_rotary_pos_emb(query_layer, rotary_pos_emb[0])
            key_layer = self.apply_rotary_pos_emb(key_layer, rotary_pos_emb[1])
        
        # =====================================================
        # Reshape for torch's scaled_dot_product_attention
        # [sq, b, np, hn] -> [b, np, sq, hn]
        # =====================================================
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)
        
        # =====================================================
        # Prepare attention mask
        # =====================================================
        if attention_mask is not None:
            # attention_mask shape: [b, 1, sq, sk] or similar
            # Need to ensure it's broadcastable for [b, np, sq, sk]
            if attention_mask.dim() == 4:
                # Typically [b, 1, sq, sk], which broadcasts correctly
                attn_mask = attention_mask
            elif attention_mask.dim() == 3:
                # [b, sq, sk] -> [b, 1, sq, sk]
                attn_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 2:
                # [sq, sk] -> [1, 1, sq, sk]
                attn_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_mask = attention_mask
                
            # Convert to proper format for scaled_dot_product_attention
            # scaled_dot_product_attention expects float mask where
            # values to mask out should be -inf or a large negative value
            if attn_mask.dtype == torch.bool:
                # Boolean mask: True = keep, False = mask out
                # Need to invert for scaled_dot_product_attention
                new_attn_mask = torch.zeros_like(attn_mask, dtype=query_layer.dtype)
                new_attn_mask.masked_fill_(~attn_mask, float('-inf'))
                attn_mask = new_attn_mask
            elif attn_mask.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                # Already a float mask, ensure proper dtype
                if attn_mask.dtype != query_layer.dtype:
                    attn_mask = attn_mask.to(query_layer.dtype)
            else:
                # Other types, convert to float
                attn_mask = attn_mask.to(query_layer.dtype)
        else:
            attn_mask = None
        
        # =====================================================
        # Core attention using PyTorch's scaled_dot_product_attention
        # =====================================================
        dropout_p = self.config.attention_dropout if self.training else 0.0
        
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # We handle causality through attention_mask
        )
        
        # =====================================================
        # Reshape back: [b, np, sq, hn] -> [sq, b, np, hn] -> [sq, b, h]
        # =====================================================
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # =====================================================
        # Output projection (with tensor parallelism)
        # =====================================================
        output, output_bias = self.linear_proj(context_layer)
        
        return output, output_bias


def get_bias_dropout_add(training, config):
    """Helper function for bias dropout add."""
    from megatron.core.transformer.transformer_layer import bias_dropout_add_fused_train, bias_dropout_add_fused_inference
    
    def bias_dropout_add_func(x_with_bias, residual, prob):
        x, bias = x_with_bias
        if bias is not None:
            x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out
        return out
    
    if training and config.bias_dropout_fusion:
        return bias_dropout_add_fused_train
    else:
        return bias_dropout_add_fused_inference if config.bias_dropout_fusion else bias_dropout_add_func


def get_custom_gpt_layer_spec() -> ModuleSpec:
    """Get custom GPT layer spec with scaled_dot_product_attention.
    
    This function creates a ModuleSpec that uses our custom attention implementation
    instead of the default Megatron-Core attention.
    
    Returns:
        ModuleSpec: Custom layer specification
    """
    # Define submodules for the attention block
    custom_attention_submodules = SelfAttentionSubmodules(
        linear_qkv=TEColumnParallelLinear,
        linear_proj=TERowParallelLinear,
    )
    
    # Define submodules for the MLP block
    mlp_submodules = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )
    
    # Define submodules for the entire transformer layer
    transformer_layer_submodules = TransformerLayerSubmodules(
        input_layernorm=TENorm,
        self_attention=ModuleSpec(
            module=CustomScaledDotProductAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=custom_attention_submodules,
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=TENorm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=mlp_submodules,
        ),
        mlp_bda=get_bias_dropout_add,
    )
    
    # Return the complete layer spec
    return ModuleSpec(
        module=TransformerLayer,
        submodules=transformer_layer_submodules,
    )


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.
    
    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration
        
    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Use custom layer spec with scaled_dot_product_attention
                # This supports both tensor parallel and pipeline parallel
                print_rank_0('Using custom GPT layer with torch.nn.functional.scaled_dot_product_attention')
                transformer_layer_spec = get_custom_gpt_layer_spec()
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
            )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)) and (
        not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )


# ============================================================================
# Example run commands
# ============================================================================
#
# This script uses a custom scaled_dot_product_attention with TP and PP.
#
# Single-GPU example:
# python pretrain_gpt_customized.py \
#     --tensor-model-parallel-size 1 \
#     --pipeline-model-parallel-size 1 \
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --micro-batch-size 4 \
#     --global-batch-size 8 \
#     --seq-length 1024 \
#     --max-position-embeddings 1024 \
#     --train-iters 500000 \
#     --lr-decay-iters 320000 \
#     --save /path/to/checkpoints \
#     --load /path/to/checkpoints \
#     --data-path /path/to/data \
#     --vocab-file /path/to/vocab.json \
#     --merge-file /path/to/merges.txt \
#     --split 949,50,1 \
#     --distributed-backend nccl \
#     --lr 0.00015 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --clip-grad 1.0 \
#     --lr-warmup-fraction .01 \
#     --log-interval 100 \
#     --save-interval 10000 \
#     --eval-interval 1000 \
#     --eval-iters 10 \
#     --transformer-impl local \
#     --use-checkpoint-opt_param-scheduler \
#     --no-async-tensor-model-parallel-allreduce
#
# Tensor Parallel (TP=2) example:
# torchrun --nproc_per_node=2 pretrain_gpt_customized.py \
#     --tensor-model-parallel-size 2 \
#     --pipeline-model-parallel-size 1 \
#     --num-layers 24 \
#     --hidden-size 1024 \
#     --num-attention-heads 16 \
#     --micro-batch-size 2 \
#     --global-batch-size 8 \
#     --seq-length 2048 \
#     --max-position-embeddings 2048 \
#     --train-iters 500000 \
#     --save /path/to/checkpoints \
#     --data-path /path/to/data \
#     --vocab-file /path/to/vocab.json \
#     --merge-file /path/to/merges.txt \
#     --split 949,50,1 \
#     --distributed-backend nccl \
#     --lr 0.00015 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --clip-grad 1.0 \
#     --lr-warmup-fraction .01 \
#     --log-interval 100 \
#     --save-interval 10000 \
#     --eval-interval 1000 \
#     --eval-iters 10 \
#     --transformer-impl local
#
# Pipeline Parallel (PP=2) example:
# torchrun --nproc_per_node=2 pretrain_gpt_customized.py \
#     --tensor-model-parallel-size 1 \
#     --pipeline-model-parallel-size 2 \
#     --num-layers 24 \
#     --hidden-size 1024 \
#     --num-attention-heads 16 \
#     --micro-batch-size 2 \
#     --global-batch-size 8 \
#     --seq-length 2048 \
#     --max-position-embeddings 2048 \
#     --train-iters 500000 \
#     --save /path/to/checkpoints \
#     --data-path /path/to/data \
#     --vocab-file /path/to/vocab.json \
#     --merge-file /path/to/merges.txt \
#     --split 949,50,1 \
#     --distributed-backend nccl \
#     --lr 0.00015 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --clip-grad 1.0 \
#     --lr-warmup-fraction .01 \
#     --log-interval 100 \
#     --save-interval 10000 \
#     --eval-interval 1000 \
#     --eval-iters 10 \
#     --transformer-impl local
#
# Hybrid TP=2, PP=2 (4 GPUs) example:
# torchrun --nproc_per_node=4 pretrain_gpt_customized.py \
#     --tensor-model-parallel-size 2 \
#     --pipeline-model-parallel-size 2 \
#     --num-layers 32 \
#     --hidden-size 2048 \
#     --num-attention-heads 16 \
#     --micro-batch-size 1 \
#     --global-batch-size 8 \
#     --seq-length 2048 \
#     --max-position-embeddings 2048 \
#     --train-iters 500000 \
#     --save /path/to/checkpoints \
#     --data-path /path/to/data \
#     --vocab-file /path/to/vocab.json \
#     --merge-file /path/to/merges.txt \
#     --split 949,50,1 \
#     --distributed-backend nccl \
#     --lr 0.00015 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --clip-grad 1.0 \
#     --lr-warmup-fraction .01 \
#     --log-interval 100 \
#     --save-interval 10000 \
#     --eval-interval 1000 \
#     --eval-iters 10 \
#     --transformer-impl local
#
# Multi-node example (8 nodes x 8 GPUs, TP=4, PP=4, DP=4):
# torchrun \
#     --nnodes=8 \
#     --nproc_per_node=8 \
#     --rdzv_id=123 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     pretrain_gpt_customized.py \
#     --tensor-model-parallel-size 4 \
#     --pipeline-model-parallel-size 4 \
#     --num-layers 48 \
#     --hidden-size 4096 \
#     --num-attention-heads 32 \
#     --micro-batch-size 1 \
#     --global-batch-size 64 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
#     --train-iters 500000 \
#     --save /path/to/checkpoints \
#     --data-path /path/to/data \
#     --vocab-file /path/to/vocab.json \
#     --merge-file /path/to/merges.txt \
#     --split 949,50,1 \
#     --distributed-backend nccl \
#     --lr 0.00015 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --clip-grad 1.0 \
#     --lr-warmup-fraction .01 \
#     --log-interval 100 \
#     --save-interval 10000 \
#     --eval-interval 1000 \
#     --eval-iters 10 \
#     --fp16 \
#     --transformer-impl local
#
# Notes:
# 1. --transformer-impl local: use the local transformer stack (not Transformer Engine)
# 2. Custom attention supports tensor parallel via TEColumnParallelLinear / TERowParallelLinear
# 3. Pipeline parallel splits the model across stages automatically
# 4. --data-path must point to preprocessed training data
# 5. --vocab-file and --merge-file must point to GPT-2 tokenizer files
# 6. Enable mixed precision with --fp16 or --bf16
# 7. global-batch-size = micro-batch-size × data-parallel-size × gradient-accumulation-steps
# 8. data-parallel-size = nproc_per_node × nnodes / (tensor-model-parallel-size × pipeline-model-parallel-size)