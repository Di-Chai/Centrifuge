# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "token_filter"


def parse_define_args():
    """Parse --define arguments from command line"""
    define_flags = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--define':
            if i + 1 < len(sys.argv):
                define_flags.append(f"-D{sys.argv[i + 1]}")
                # Remove the --define and its value from sys.argv
                sys.argv.pop(i)
                sys.argv.pop(i)
            else:
                print("Error: --define requires a value")
                sys.exit(1)
        else:
            i += 1
    return define_flags


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    # Parse --define arguments
    define_flags = parse_define_args()
    if define_flags:
        print(f"Compiling with define flags: {define_flags}")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ] + define_flags,  # Add define flags to C++ compiler
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ] + define_flags,  # Add define flags to NVCC compiler
    }
    
    # Add PyTorch library linking to resolve symbols like SymInt::sym_eq_slow_path
    torch_path = torch.__path__[0]
    torch_lib_path = os.path.join(torch_path, "lib")
    
    extra_link_args.extend([
        f"-L{torch_lib_path}",        # Library search path
        "-lc10",                       # c10 core library (contains SymInt symbols)
        "-ltorch_cpu",                 # torch CPU library
        "-ltorch",                     # torch main library
        f"-Wl,-rpath,{torch_lib_path}" # Runtime library search path
    ])
    
    # Add cuBLAS library linking for CUDA builds
    if use_cuda:
        extra_link_args.extend(["-lcublas", "-ltorch_cuda"])
    
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    print("#" * 10)
    print(extensions_dir)
    print(cuda_sources)
    print("#" * 10)

    if use_cuda:
        sources += cuda_sources

    torch_path = torch.__path__[0]
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    include_dirs = [
        "include",
        os.path.join(torch_path, "include"),
        os.path.join(torch_path, "include", "torch", "csrc", "api", "include"),
        os.path.join(cuda_home, "include"),
        "/usr/include",
        "/opt/pytorch/pytorch/third_party/fbgemm/third_party/cutlass/include",
        os.path.abspath('./token_filter')
    ]

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
            # libraries=['cuda', 'cudart', 'cublas', 'curand'],
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="1.0.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Token filtering extention for torch",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    # url="https://github.com/pytorch/extension-cpp",
    cmdclass={"build_ext": BuildExtension},
)
