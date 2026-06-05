#!/usr/bin/env python3
"""
Custom GPT implementation with PyTorch kernels + Megatron-LM parallel infrastructure
- Attention: torch.nn.functional.scaled_dot_product_attention
- TP/PP/1F1B: Megatron-LM framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass

import sys
sys.path.append("/data/Megatron-LM")

# Megatron-LM imports for parallel infrastructure
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.gpt import GPTModel as MegatronGPTModel
from megatron.training import get_args, print_rank_0, get_timers
from megatron.training import pretrain
from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_batch_on_this_tp_rank, get_batch_on_this_cp_rank
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    TENorm,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from functools import partial


# =============================================================================
# Custom Attention using PyTorch SDPA
# =============================================================================

class CustomScaledDotProductAttention(SelfAttention):
    """
    Custom Self Attention using torch.nn.functional.scaled_dot_product_attention
    Inherits from Megatron's SelfAttention to leverage TP infrastructure
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        **kwargs
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs
        )
        
        # Ensure hidden_size_per_partition is set (for TP compatibility)
        if not hasattr(self, 'hidden_size_per_partition'):
            self.hidden_size_per_partition = (
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
            )
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        **kwargs  # Accept any additional arguments from Megatron
    ):
        # hidden_states: [sq, b, h]
        
        # QKV projection (with TP)
        mixed_x_layer, _ = self.linear_qkv(hidden_states)
        
        # Reshape to [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        
        # Split Q, K, V
        query_layer, key_layer, value_layer = tensor_parallel.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )
        
        # Apply RoPE if available (Megatron passes rotary_pos_emb or rotary_pos_cos/sin in kwargs)
        if rotary_pos_emb is not None:
            # Apply RoPE to query and key
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb, self.config)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb, self.config)
        
        # Transform to PyTorch SDPA format: [b, np, sq, hn]
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)
        
        # Prepare attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attn_mask = attention_mask
            elif attention_mask.dim() == 3:
                attn_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 2:
                attn_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_mask = attention_mask
            
            # Convert mask format
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=query_layer.dtype)
                new_attn_mask.masked_fill_(~attn_mask, float('-inf'))
                attn_mask = new_attn_mask
            elif attn_mask.dtype != query_layer.dtype:
                attn_mask = attn_mask.to(query_layer.dtype)
        else:
            attn_mask = None
        
        # Core attention computation using PyTorch SDPA
        dropout_p = self.config.attention_dropout if self.training else 0.0
        
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        
        # Transform back: [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection (with TP all-reduce)
        output, output_bias = self.linear_proj(context_layer)
        
        return output, output_bias


# =============================================================================
# Layer Spec with Custom Attention
# =============================================================================

def get_bias_dropout_add(training, config):
    """Helper function for bias dropout add"""
    from megatron.core.transformer.transformer_layer import (
        bias_dropout_add_fused_train,
        bias_dropout_add_fused_inference
    )
    
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
    """
    Get custom GPT layer spec with PyTorch SDPA
    Uses Megatron's TP layers but custom attention kernel
    """
    # Attention submodules (using Megatron's TP layers)
    # Note: core_attention is provided for compatibility but not used
    # since we override forward() to use PyTorch SDPA
    custom_attention_submodules = SelfAttentionSubmodules(
        linear_qkv=TEColumnParallelLinear,
        core_attention=DotProductAttention,  # Placeholder, not used
        linear_proj=TERowParallelLinear,
    )
    
    # MLP submodules
    mlp_submodules = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )
    
    # Complete transformer layer submodules
    transformer_layer_submodules = TransformerLayerSubmodules(
        input_layernorm=TENorm,
        self_attention=ModuleSpec(
            module=CustomScaledDotProductAttention,  # Our custom attention
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
    
    return ModuleSpec(
        module=TransformerLayer,
        submodules=transformer_layer_submodules,
    )


# =============================================================================
# Model Provider (for Megatron pretrain framework)
# =============================================================================

def model_provider(pre_process=True, post_process=True, vp_stage: Optional[int] = None):
    """
    Build GPT model with custom attention
    This function is called by Megatron's pretrain framework
    """
    args = get_args()
    print_rank_0('Building GPT model with custom PyTorch SDPA attention...')
    
    # Get config from args
    from megatron.training.arguments import core_transformer_config_from_args
    config = core_transformer_config_from_args(args)
    
    # Use custom layer spec
    transformer_layer_spec = get_custom_gpt_layer_spec()
    
    # Create Megatron GPT model (handles PP automatically)
    model = MegatronGPTModel(
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
    )
    
    return model


# =============================================================================
# Data and Training Functions
# =============================================================================

def get_batch(data_iterator):
    """Generate a batch - required by Megatron pretrain"""
    # For mock data, we generate on-the-fly
    args = get_args()
    
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)) and (
        not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ):
        return None, None, None, None, None
    
    # Generate mock data
    tokens = torch.randint(
        0, args.padded_vocab_size,
        (args.seq_length, args.micro_batch_size),
        device=torch.cuda.current_device()
    )
    labels = torch.randint(
        0, args.padded_vocab_size,
        (args.seq_length, args.micro_batch_size),
        device=torch.cuda.current_device()
    )
    loss_mask = torch.ones_like(tokens, dtype=torch.float32)
    attention_mask = None
    position_ids = torch.arange(
        args.seq_length, dtype=torch.long, device=torch.cuda.current_device()
    ).unsqueeze(1).expand(-1, args.micro_batch_size)
    
    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function - required by Megatron pretrain"""
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    
    num_tokens = loss_mask.sum()
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
    
    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model):
    """Forward step - required by Megatron pretrain"""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    
    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
    )
    
    return output_tensor, partial(loss_func, loss_mask)


# =============================================================================
# Mock Dataset Provider
# =============================================================================

class MockDataset:
    """Mock dataset for testing"""
    def __init__(self, num_samples):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return dummy data - actual data is generated in get_batch
        return {'text': torch.zeros(1)}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """
    Create mock datasets for train/valid/test
    Actual data generation happens in get_batch()
    """
    train_ds = MockDataset(train_val_test_num_samples[0]) if train_val_test_num_samples else MockDataset(100)
    valid_ds = MockDataset(train_val_test_num_samples[1]) if train_val_test_num_samples and len(train_val_test_num_samples) > 1 else MockDataset(10)
    test_ds = MockDataset(train_val_test_num_samples[2]) if train_val_test_num_samples and len(train_val_test_num_samples) > 2 else MockDataset(10)
    
    return train_ds, valid_ds, test_ds

# Mark as distributed so it runs on all ranks
train_valid_test_datasets_provider.is_distributed = True


# =============================================================================
# Main Training Function
# =============================================================================

if __name__ == "__main__":
    # Use Megatron's pretrain framework
    # This handles all TP/PP/1F1B scheduling automatically
    
    # Default args for mock training
    import argparse
    
    def add_custom_args(parser):
        """Add any custom arguments"""
        return parser
    
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',  # Use NullTokenizer for mock data
            'no_load_optim': True,
            'no_load_rng': True,
            'no_save_optim': True,
            'no_save_rng': True,
            'mock_data': True,
        },
        extra_args_provider=add_custom_args,
    )


# End of file - all parallel logic handled by Megatron-LM framework
