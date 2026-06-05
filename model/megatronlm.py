import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.core.model_parallel_config import ModelParallelConfig


def create_model_parallel_config(tp_size):
    megatron_config = ModelParallelConfig()
    
    # Core parallel sizes
    megatron_config.tensor_model_parallel_size = tp_size
    megatron_config.pipeline_model_parallel_size = 1
    megatron_config.sequence_parallel = False
    megatron_config.expert_model_parallel_size = 1
    megatron_config.context_parallel_size = 1
    megatron_config.virtual_pipeline_model_parallel_size = None
    
    # Dtype and weight init
    megatron_config.use_cpu_initialization = False
    megatron_config.perform_initialization = True
    megatron_config.params_dtype = torch.bfloat16
    
    # Optimizer / comm fusion flags
    megatron_config.gradient_accumulation_fusion = False
    megatron_config.async_tensor_model_parallel_allreduce = False
    megatron_config.defer_embedding_wgrad_compute = False
    
    # Attention settings
    megatron_config.use_flash_attn = True
    megatron_config.attention_type = "multihead_attention"
    megatron_config.apply_query_key_layer_scaling = False
    megatron_config.attention_softmax_in_fp32 = False
    
    # Misc Megatron flags
    megatron_config.fp16 = False
    megatron_config.bf16 = True
    megatron_config.bias_gelu_fusion = False
    megatron_config.hidden_dropout = 0.0
    megatron_config.attention_dropout = 0.0
    
    return megatron_config


class MegatronLlamaAttention(nn.Module):
    """MegatronLM TP-aware TinyLlama Attention"""
    def __init__(self, llama_config, megatron_config):
        super().__init__()
        self.llama_config = llama_config
        self.megatron_config = megatron_config
        
        self.hidden_size = llama_config.hidden_size
        self.num_attention_heads = llama_config.num_attention_heads
        self.num_key_value_heads = getattr(llama_config, 'num_key_value_heads', self.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Per-TP partition sizes
        tp_size = megatron_config.tensor_model_parallel_size
        self.num_attention_heads_per_partition = self.num_attention_heads // tp_size
        self.num_key_value_heads_per_partition = self.num_key_value_heads // tp_size
        self.hidden_size_per_partition = self.hidden_size // tp_size
        
        self.q_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            gather_output=False
        )
        
        self.k_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            gather_output=False
        )
        
        self.v_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            gather_output=False
        )
        
        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False
        )
         
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Unwrap Megatron tuple outputs (bias, etc.)
        if isinstance(q, tuple):
            q = q[0]
        if isinstance(k, tuple):
            k = k[0]
        if isinstance(v, tuple):
            v = v[0]
        
        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_attention_heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)
        
        # GQA: repeat KV heads when KV < Q
        if self.num_key_value_heads_per_partition < self.num_attention_heads_per_partition:
            repeat_factor = self.num_attention_heads_per_partition // self.num_key_value_heads_per_partition
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Flash Attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size_per_partition)
        
        output = self.o_proj(attn_output)
        if isinstance(output, tuple):
            output = output[0]
            
        return output


class MegatronLlamaMLP(nn.Module):
    """MegatronLM TP-aware TinyLlama MLP (SwiGLU)"""
    def __init__(self, llama_config, megatron_config):
        super().__init__()
        self.llama_config = llama_config
        self.megatron_config = megatron_config
        
        self.hidden_size = llama_config.hidden_size
        self.intermediate_size = llama_config.intermediate_size
        
        self.gate_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            gather_output=False
        )
        
        self.up_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            gather_output=False
        )
        
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False
        )
        
    
    def forward(self, hidden_states):
        # SwiGLU: SiLU(gate) * up
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        
        if isinstance(gate, tuple):
            gate = gate[0]
        if isinstance(up, tuple):
            up = up[0]
        
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(gate)
        intermediate = gate * up
        
        # Down projection
        output = self.down_proj(intermediate)
        if isinstance(output, tuple):
            output = output[0]
            
        return output

class MegatronLlamaLayer(nn.Module):
    """MegatronLM TP-aware TinyLlama Transformer Layer"""
    def __init__(self, llama_config, megatron_config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.input_layernorm = nn.RMSNorm(llama_config.hidden_size, eps=llama_config.rms_norm_eps, dtype=torch.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(llama_config.hidden_size, eps=llama_config.rms_norm_eps, dtype=torch.bfloat16)
        
        self.self_attn = MegatronLlamaAttention(llama_config, megatron_config)
        self.mlp = MegatronLlamaMLP(llama_config, megatron_config)
            
    def forward(self, hidden_states):
        # Self-attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class MegatronLlamaModel(nn.Module):
    """MegatronLM TP-aware TinyLlama Model"""
    def __init__(self, llama_config, megatron_config, num_layers=None):
        super().__init__()
        self.llama_config = llama_config
        self.megatron_config = megatron_config

        self.num_layers = llama_config.num_hidden_layers
        
        self.embed_tokens = nn.Embedding(llama_config.vocab_size, llama_config.hidden_size, dtype=torch.bfloat16)
        # self.embed_tokens = VocabParallelEmbedding(
        #     num_embeddings=llama_config.vocab_size,
        #     embedding_dim=llama_config.hidden_size,
        #     config=megatron_config,
        #     init_method=lambda x: nn.init.normal_(x, mean=0.0, std=0.02)
        # )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            MegatronLlamaLayer(llama_config, megatron_config, i) 
            for i in range(self.num_layers)
        ])
        
        # Final RMSNorm
        self.norm = nn.RMSNorm(llama_config.hidden_size, eps=llama_config.rms_norm_eps, dtype=torch.bfloat16)
        
        # LM head
        self.lm_head = ColumnParallelLinear(
            input_size=llama_config.hidden_size,
            output_size=llama_config.vocab_size,
            config=megatron_config,
            init_method=lambda x: nn.init.xavier_uniform_(x),
            bias=False,
            gather_output=True
        )
    
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # Run all transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final RMSNorm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        return logits