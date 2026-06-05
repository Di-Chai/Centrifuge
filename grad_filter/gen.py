import os
import re
import yaml
import torchgen

from torchgen.packaged.autograd.load_derivatives import load_derivatives
from torchgen.packaged.autograd.gen_autograd_functions import get_infos_with_derivatives_list, FUNCTION_DECLARATION, process_function
from torchgen.api.types import BaseCType, tensorT, OptionalCType, MutRefCType, scalarT, symIntArrayRefT, SymIntT


if __name__ == "__main__":

    if os.path.exists("target_node_names.yaml"):
        with open("target_node_names.yaml", "r") as f:
            target_node_nams = yaml.load(f, Loader=yaml.FullLoader)["target_node_names"]
    else:
        target_node_nams = [
            "UnsqueezeBackward0", "AddmmBackward0", "BmmBackward0", "MeanBackward0", "ToCopyBackward0", 
            "CatBackward0", "MulBackward0", "MeanBackward1", "NativeDropoutBackward0", 
            "_ReduceFromModelParallelRegionBackward", "_GatherFromModelParallelRegionBackward", 
            "SiluBackward0", "LinearWithGradAccumulationAndAsyncCommunicationBackward", 
            "SumBackward1", "ViewBackward0", "TanhBackward0", "NativeLayerNormBackward0", 
            "torch::autograd::AccumulateGrad", "TransposeBackward0", "SoftmaxBackward0", 
            "CloneBackward0", "NegBackward0", "TBackward0", "SliceBackward0", "AddBackward0", 
            "ExpandBackward0", "AddBackward1", "ReshapeAliasBackward0", "LogSoftmaxBackward0", 
            "RsqrtBackward0", "EmbeddingBackward0", "PowBackward0", "DivBackward0", 
            "UnsafeViewBackward0", "MmBackward0", "NllLossBackward0"
        ]

    exluded_node_names = [
        "ScaledDotProductFlashAttentionBackward0"
    ]
    
    code_template = \
"""
  if(fn->name() == "{op_name}") {{
    {op_name}* op_fn = static_cast<{op_name}*>(fn); 
    {process_tensors} 
    {process_sizes}
  }}
"""
    process_var_template = "auto {new_var_name} = op_fn->{var_name}.unpack({ptr_name});\n    if({new_var_name}.defined()) op_fn->{var_name} = graph_filter->process_variable({new_var_name}, {is_output});"
    process_sizes_template = "graph_filter->process_sizes(op_fn->{var_name});"

    src_file = "token_filter/csrc/node_processing.cpp"
    insert_start = "/* $$ start of code generation $$ */"
    insert_end = "/* $$ end of code generation $$ */"
    
    torchgen_path = os.path.dirname(os.path.abspath(torchgen.__file__))

    native_functions_path = os.path.join(torchgen_path, "packaged", "ATen", "native", "native_functions.yaml")
    tags_path = os.path.join(torchgen_path, "packaged", "ATen", "native", "tags.yaml")
    autograd_dir = os.path.join(torchgen_path, "packaged", "autograd")

    differentiability_infos, used_dispatch_keys = load_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), native_functions_path, tags_path
    )

    infos = get_infos_with_derivatives_list(differentiability_infos)

    gen_codes = []
    processed_node_names = []
    for info in infos:
        r = process_function(info, FUNCTION_DECLARATION)
        op_name = re.match(".*\nstruct (.+?) : public.*", r).group(1)
        
        if op_name not in target_node_nams:
            if "CopySlices" in op_name:
                print(op_name)
            continue

        if "Attention" in op_name:
            print("Attention op will be sequerated processed in token filtering", op_name)
            continue
        
        if op_name in exluded_node_names:
            continue

        processed_node_names.append(op_name)
        
        print("##" * 20)
        print(op_name)

        process_vars = []
        process_sizes = []
        var_lists = [[e, False] for e in info.all_saved_inputs] + [[e, True] for e in info.all_saved_outputs]
        for var, is_output in var_lists:
            v_name, v_type = var.nctype.name, var.nctype.type
            print(is_output, v_name, v_type)
            if(
                v_type == BaseCType(tensorT)
                or v_type == OptionalCType(BaseCType(tensorT))
                or v_type == MutRefCType(OptionalCType(BaseCType(tensorT)))
                or (v_type == BaseCType(scalarT) and is_output)
            ):
                process_vars.append(process_var_template.format(
                    new_var_name="unpacked_" + v_name, 
                    var_name=v_name + "_", 
                    ptr_name="" if not is_output else "op_fn->getptr()",
                    is_output="true" if is_output else "false"))

            elif v_type == BaseCType(symIntArrayRefT) and v_name.endswith("sizes"):
                process_sizes.append(process_sizes_template.format(var_name=v_name))

            elif v_type == BaseCType(SymIntT) and (v_name.endswith("numel") or v_name in ["max_k", "max_q"]):
                process_sizes.append(process_sizes_template.format(var_name=v_name))

        if len(process_vars) > 0 or len(process_sizes) > 0:
            gen_code = code_template.format(
                op_name=op_name,
                process_tensors="\n    ".join(process_vars),
                process_sizes="\n    ".join(process_sizes)
            )
            gen_codes.append(gen_code)
    
    print("".join(gen_codes))
    with open(src_file, "r") as f:
        src_code = f.readlines()
    updated_src_code = []
    i = 0
    while i < len(src_code):
        updated_src_code.append(src_code[i])
        if insert_start in src_code[i]:
            updated_src_code.append("".join(gen_codes))
            while i < len(src_code) and insert_end not in src_code[i]:
                i += 1
            i -= 1
        i += 1
    with open(src_file, "w") as f:
        f.write("".join(updated_src_code))
    
    print("-" * 20)
    print("Following nodes are not included in torch-native autograd functions.")
    for node_name in target_node_nams:
        if node_name not in processed_node_names:
            print(node_name)

    