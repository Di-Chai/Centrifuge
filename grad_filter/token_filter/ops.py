import torch
from torch import Tensor
from typing import List, Dict, Any
from . import _C

__all__ = ["backward_filter", "backward_filter_with_dims"]


def backward_filter(loss: Tensor, mask: Tensor) -> Tensor:
    return torch.ops.token_filter.backward_filter.default(loss, mask)


def backward_filter_with_dims(loss: Tensor, mask: Tensor, input_node_filter_dims: str = "") -> str:
    """
    Backward filter that accepts and returns node filter dimensions as string.
    
    Args:
        loss: Input loss tensor
        mask: Input mask tensor  
        input_node_filter_dims: String representation of node filter dimensions to use as input.
                               If empty, will attempt to load from cache.
                               Format follows the same as cache_node_filter_dims/load_node_filter_dims.
    
    Returns:
        String representation of node filter dimensions collected during the backward pass.
        Format:
        function_name1
        index1 bsz_index1 mode1
        index2 bsz_index2 mode2
        ---
        function_name2
        index3 bsz_index3 mode3
        ---
    """
    return torch.ops.token_filter.backward_filter_with_dims.default(loss, mask, input_node_filter_dims)


def collect_node_names(loss: Tensor) -> List[str]:
    return torch.ops.token_filter.collect_node_names.default(loss)

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("token_filter::backward_filter")
def _(loss: Tensor, mask: Tensor) -> Tensor:
    return torch.empty_like(loss)

@torch.library.register_fake("token_filter::backward_filter_with_dims")
def _(loss: Tensor, mask: Tensor, input_node_filter_dims: str) -> str:
    return ""

@torch.library.register_fake("token_filter::collect_node_names")
def _(loss: Tensor) -> List[str]:
    return []