import os
import time

from transformers import AutoTokenizer, LlamaTokenizerFast, AutoConfig, AutoModelForCausalLM
from torch.autograd import Function
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.attention import SDPBackend

from model.utils import *
# from model.dropping_model import *
from data.build_data import build_pretrain_data

from grad_filter import token_filter
from datetime import datetime
from arguments import parse_args

# from torchtune.models.llama3_2 import llama3_2_1b, lora_llama3_2_1b
# from torchtune.modules.peft import get_adapter_params, set_trainable_params


torch.manual_seed(1234)

sdpa_backend = SDPBackend.FLASH_ATTENTION

def gen_node_tracing_with_model(model, bsz, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len)).to('cuda') 
    attn_mask = torch.ones(bsz, seq_len).to('cuda') # ref mask will be generated based on attn mask
    with torch.nn.attention.sdpa_kernel(sdpa_backend):
        output = model(input_ids, attention_mask=attn_mask)
        loss, ref_mask = token_filter_loss(
            inputs=input_ids, logits=output.logits, attention_mask=attn_mask,
            ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(), 
            dropping_strategy='fixed', drop_rate=0.4, # the drop_rate does not impact generating the node tracing files
            is_left_padding=True, return_mask=True
        )
    token_filter.ops.backward_filter(loss, ref_mask)
    try:
        # Run backward to test the node tracing file
        loss.backward()
    except RuntimeError as e:
        print("Generated tracing file failed in running backwark with error:", e)

def gen_node_tracing_with_model_with_dims_lora(model, bsz, seq_len, vocab_size, node_filter_dims='', device='cuda'):
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len)).to(device) 
    attn_mask = torch.ones(bsz, seq_len).to(device) # ref mask will be generated based on attn mask
    with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = model(input_ids)
    loss, ref_mask = token_filter_loss(
        inputs=input_ids,
        logits=output, 
        attention_mask=attn_mask,
        ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(), 
        dropping_strategy='fixed', drop_rate=0.5, # the drop_rate does not impact generating the node tracing files
        is_left_padding=True, return_mask=True
    )
    node_filter_dims = token_filter.ops.backward_filter_with_dims(loss, ref_mask, node_filter_dims)
    try:
        # Run backward to test the node tracing file
        loss.backward()
        return node_filter_dims
    except RuntimeError as e:
        print("Generated tracing file failed in running backwark with error:", e)
        return ""

def gen_node_tracing_with_model_with_dims(model, bsz, seq_len, vocab_size, node_filter_dims='', device='cuda'):
    if hasattr(model, "device"):
        device = model.device
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len)).to(device) 
    attn_mask = torch.ones(bsz, seq_len).to(device) # ref mask will be generated based on attn mask
    with torch.nn.attention.sdpa_kernel(sdpa_backend):
        output = model(input_ids, attention_mask=attn_mask)
    loss, ref_mask = token_filter_loss(
        inputs=input_ids, 
        logits=output.logits, 
        attention_mask=attn_mask,
        ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(), 
        dropping_strategy='fixed', drop_rate=0.5, # the drop_rate does not impact generating the node tracing files
        is_left_padding=True, return_mask=True
    )
    node_filter_dims = token_filter.ops.backward_filter_with_dims(loss, ref_mask, node_filter_dims)
    try:
        # Run backward to test the node tracing file
        loss.backward()
        return node_filter_dims
    except RuntimeError as e:
        print("Generated tracing file failed in running backwark with error:", e)
        return ""

def gen_node_tracing_with_model_with_dims_megatronlm(model, bsz, seq_len, vocab_size, node_filter_dims=''):
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len)).to('cuda') 
    attn_mask = torch.ones(bsz, seq_len).to('cuda') # ref mask will be generated based on attn mask
    output = model(input_ids, attention_mask=attn_mask)
    loss, ref_mask = token_filter_loss(
        inputs=input_ids, logits=output, attention_mask=attn_mask,
        ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(), 
        dropping_strategy='fixed', drop_rate=0.5, # the drop_rate does not impact generating the node tracing files
        is_left_padding=True, return_mask=True
    )
    node_filter_dims = token_filter.ops.backward_filter_with_dims(loss, ref_mask, node_filter_dims)
    try:
        # Run backward to test the node tracing file
        loss.backward()
        return node_filter_dims
    except RuntimeError as e:
        print("Generated tracing file failed in running backwark with error:", e)
        return ""
    return node_filter_dims

def is_node_tracing_generated():
    return os.path.exists("/tmp/node_tracing.txt")

def gen_graph_with_model(loss, model, filename=None):
    from torchviz import make_dot
    filename = "tmp/tmp.svg" if filename is None else f"tmp/{filename}"
    dot = make_dot(
        loss, params=dict(model.named_parameters()),
        show_saved=True, show_attrs=True
    )
    dot.graph_attr['fontname'] = 'Times New Roman'
    dot.node_attr['fontname'] = 'Times New Roman'
    dot.edge_attr['fontname'] = 'Times New Roman'
    dot.graph_attr['fontsize'] = '16'
    dot.node_attr['fontsize'] = '16'
    dot.edge_attr['fontsize'] = '16'
    dot.graph_attr.update(ranksep='0.2', nodesep='0.1', pack='true', packmode='clust', concentrate='true', ratio='compress')
    dot.attr(size='8,200')
    # dot.attr(dpi='500')
    return dot.render(filename, format="svg")
    

def gen_node_tracing(args=None, gen_computational_graph=False, filter_graph=False):
    if args is None:
        args = parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    if gen_computational_graph:
        config.num_hidden_layers = 2
        # config._attn_implementation = "sdpa"
        # config._attn_implementation_autoset = False
    
    if args.attn_impl == "eager":
        config._attn_implementation = args.attn_impl

    if args.use_lora:
        # from peft import LoraConfig, get_peft_model
        # lora_config = LoraConfig(
        #     r=8, 
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        #     lora_dropout=0.1, lora_alpha=16
        # )
        # model = get_peft_model(model, lora_config)
        model = lora_llama3_2_1b(
            lora_attn_modules=["q_proj"], # "k_proj", "v_proj", "k_proj", "output_proj"
            apply_lora_to_mlp=True).to(torch.bfloat16).to('cuda')
        lora_params = get_adapter_params(model)
        set_trainable_params(model, lora_params)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, use_safetensors=False).to(torch.bfloat16).to('cuda')
    
    print(model)
    
    # Node tracing filename (defined in backward_filter.cpp)
    node_tracing_filename = "tmp/node_tracing.txt"

    # Synthetic data: select a sequence length that does not match the model's parameters (e.g., 77)
    bsz = args.batch_size
    seq_len = args.context_length
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len)).to('cuda') 
    attn_mask = torch.ones(bsz, seq_len).to('cuda') # ref mask will be generated based on attn mask

    # with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION): # CUDNN_ATTENTION
    # with torch.nn.attention.sdpa_kernel(sdpa_backend):
    output = model(
        input_ids, 
        # attention_mask=attn_mask
    )
    loss, ref_mask = token_filter_loss(
        inputs=input_ids, 
        logits=output if not hasattr(output, 'logits') else output.logits, 
        attention_mask=attn_mask,
        ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(), 
        dropping_strategy='fixed', 
        drop_rate=0.5, # the drop_rate does not impact generating the node tracing files
        is_left_padding=True, return_mask=True
    )

    print(token_filter.ops.collect_node_names(loss))
    
    # Remove the existing node tracing file
    if not gen_computational_graph or filter_graph:
        if os.path.exists(node_tracing_filename):
            os.remove(node_tracing_filename)
        
        start = time.time()
        # token_filter.ops.backward_filter(loss, ref_mask)
        filter_dims = token_filter.ops.backward_filter_with_dims(loss, ref_mask, "")
        with open(node_tracing_filename, "w") as f:
            f.write(filter_dims)
        
        end = time.time()
        print(f"Time taken by backward_filter (python get str): {end - start} seconds")

        # start = time.time()
        # gen_node_tracing_with_model_with_dims(model, bsz, seq_len, config.vocab_size, filter_dims)
        # end = time.time()
        print(f"Time taken by backward_filter (python execute): {end - start} seconds")
        
        # Check if the node tracing file is generated
        # assert os.path.exists(node_tracing_filename)
    
    # Run backward to test the node tracing file, or generating the computational graph
    try:
        if gen_computational_graph:
            # WARNING: Successfully generating the computational graph does NOT guarantee
            #  the correctness of the node tracing file, unless running backward.
            from torchviz import make_dot
            dot = make_dot(
                loss, params=dict(model.named_parameters()),
                show_saved=True, show_attrs=True
            )
            dot.graph_attr['fontname'] = 'Times New Roman'
            dot.node_attr['fontname'] = 'Times New Roman'
            dot.edge_attr['fontname'] = 'Times New Roman'
            dot.graph_attr['fontsize'] = '16'
            dot.node_attr['fontsize'] = '16'
            dot.edge_attr['fontsize'] = '16'
            dot.graph_attr.update(ranksep='0.2', nodesep='0.1', pack='true', packmode='clust', concentrate='true', ratio='compress')
            dot.attr(size='8,200')
            # dot.attr(dpi='500')
            dot.render(f"tmp/computational_graph_gen_tracing_eager_{args.eager_attn}", format="svg")
        else:
            # Run backward to test the node tracing file
            loss.backward()
        if not gen_computational_graph:
            print(f"Node tracing file is successfully generated at {node_tracing_filename}")
        else:
            print("Computational graph is successfully generated")
    except RuntimeError as e:
        print("Generated tracing file failed in running backwark with error:", e)


if __name__ == "__main__":
    args = parse_args(return_parsed=False)
    args.add_argument("--gen_computational_graph", action="store_true", default=False)
    args.add_argument("--filter_graph", action="store_true", default=False)
    args.add_argument("--eager_attn", action="store_true", default=False)
    args = args.parse_args()
    gen_node_tracing(
        args, 
        gen_computational_graph=args.gen_computational_graph, 
        filter_graph=args.filter_graph
    )


# tiny_llama_config = AutoConfig.from_pretrained("TinyLlama/TinyLlama_v1.1")
# # tiny_llama_config.num_hidden_layers = 2
# # tiny_llama_config._attn_implementation = "sdpa" # sdpa flash_attention_2
# # tiny_llama_config.torch_dtype = torch.bfloat16

# print(tiny_llama_config, tiny_llama_config._attn_implementation)

# filtering_llama = FilteringLlamaForCausalLM(tiny_llama_config).to(torch.bfloat16).to('cuda')
# filtering_llama.from_pretrained("TinyLlama/TinyLlama_v1.1")

# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
# tokenizer.pad_token = tokenizer.eos_token

# print(filtering_llama)

# datasets = build_pretrain_data(['open-web-math/open-web-math'], [0.9, 0.1], seed=1234, data_select_ratio=0.1, data_select_strategy="random")['train']
# tokenized_datasets = datasets.select(range(1)).map(
#     lambda x: tokenizer(x['text'], padding=True, truncation=True, max_length=2048, return_tensors='pt'), batched=True)

# filtering_llama.train()

# def pack_hook(x):
#     print("Packing", x.shape)
#     return x

# def unpack_hook(x):
#     print("Unpacking", x.shape)
#     return x


# class filter_mul(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, filter_mask):
#         ctx.save_for_backward(filter_mask)
#         return input * filter_mask

#     @staticmethod
#     def backward(ctx, grad_output):
#         filter_mask = ctx.saved_tensors[0]
#         return grad_output[:, :3], None


# def grad_hook(grad):
#     print("Gradient", grad.shape)
#     return grad[:, :3]

# # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):

# compute_ref_mask = False
# apply_grad_filter = False

# start = time.time()

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:

#     # torch.cuda.cudart().cudaProfilerStart()
#     # torch.cuda.nvtx.range_push("iteration{}".format(0))

#     input_ids = torch.asarray(tokenized_datasets['input_ids']).to('cuda')
#     attn_mask = torch.asarray(tokenized_datasets['attention_mask']).to('cuda')

#     # torch.cuda.nvtx.range_push("forward")
#     output = filtering_llama(input_ids, attention_mask=attn_mask)

#     if compute_ref_mask:
#         loss, ref_mask = token_filter_loss(
#             inputs=input_ids, logits=output.logits, attention_mask=attn_mask,
#             ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(), dropping_strategy='fixed', drop_rate=0.5, 
#             is_left_padding=True, return_mask=True
#         )
#     else:
#         loss = token_filter_loss(
#             inputs=input_ids, logits=output.logits, attention_mask=attn_mask,
#             ref_loss=None, is_left_padding=True, return_mask=False
#         )

#     print("#" * 20)
#     if apply_grad_filter:
#         token_filter.ops.backward_filter(loss, ref_mask)
#     print("#" * 20)
    
#     # torch.cuda.nvtx.range_pop()

#     # torch.cuda.nvtx.range_push("backward")

#     loss.backward()

#     # torch.cuda.nvtx.range_pop()
    
#     # make_dot(
#     #     loss, params=dict(filtering_llama.named_parameters()),
#     #     show_saved=True, show_attrs=True
#     # ).render(f"tmp_gradient_graph_{apply_grad_filter}", format="svg")

# # torch.cuda.cudart().cudaProfilerStop()

# print("Time", time.time() - start)

# current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# prof.export_chrome_trace(f"tmp_profile/tmp_profiler_trace_{apply_grad_filter}_{current_time}.json")

# result = prof.key_averages(group_by_input_shape=True).table(
#     sort_by="cuda_time_total", row_limit=1000, max_src_column_width=200, max_name_column_width=200, max_shapes_column_width=200)
 
# with open(f"tmp_profile/tmp_profiler_result_{apply_grad_filter}_{current_time}.txt", "w") as f:
#     f.write(result)

# print("Done")