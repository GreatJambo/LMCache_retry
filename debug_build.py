#!/usr/bin/env python3

import os
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch ABI: {torch.compiled_with_cxx11_abi()}")
print(f"PyTorch C ABI flag: {torch._C._GLIBCXX_USE_CXX11_ABI}")

# Check environment
print(f"_GLIBCXX_USE_CXX11_ABI env: {os.environ.get('_GLIBCXX_USE_CXX11_ABI', 'Not set')}")

# Create a test extension to see what flags are used
test_ext = CUDAExtension(
    name="test",
    sources=["dummy.cpp"],  # We won't actually build this
    extra_compile_args={
        "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=0", "-v"],
        "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=0", "-v"],
    },
)

print("Extension created successfully")

# Check if BuildExtension modifies flags
builder = BuildExtension()
print("BuildExtension created successfully")