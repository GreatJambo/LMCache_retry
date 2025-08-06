#!/usr/bin/env python3

# 创建一个最小的测试扩展来验证编译设置

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

print(f"PyTorch ABI setting: {torch.compiled_with_cxx11_abi()}")
print(f"PyTorch internal ABI flag: {torch._C._GLIBCXX_USE_CXX11_ABI}")

# 创建最小测试源文件
test_cpp = """
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Test extension";
}
"""

test_cu = """
// Empty CUDA kernel file
"""

# 写入测试文件
with open("/tmp/test_ext.cpp", "w") as f:
    f.write(test_cpp)

with open("/tmp/test_ext.cu", "w") as f:
    f.write(test_cu)

setup(
    name="test_ext",
    ext_modules=[
        CUDAExtension(
            name="test_ext",
            sources=["/tmp/test_ext.cpp", "/tmp/test_ext.cu"],
            extra_compile_args={
                "cxx": ["-D_GLIBCXX_USE_CXX11_ABI=0", "-v", "-DDEBUG_ABI"],
                "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=0", "-v", "-DDEBUG_ABI"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)