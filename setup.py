from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
# import torch # torch import is no longer needed here as torch_lib_path is removed

# Get the directory of the current setup.py file
setup_dir = os.path.dirname(os.path.realpath(__file__))
kernel_file_cu = os.path.join(setup_dir, "flash_attention_kernels.cu")
kernel_file_cpp = os.path.join(setup_dir, "flash_attn_cuda.cpp") # New C++ source file

setup(
    name='flash_attn_cuda_kernels', # Name of the Python package
    ext_modules=[
        CUDAExtension(
            name='flash_attn_cuda_kernels', # Must match PYBIND11_MODULE name and import name
            sources=[kernel_file_cu, kernel_file_cpp], # Added .cpp file
            extra_compile_args={ # Added compile args
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr"]
            }
            # extra_link_args=['-Wl,-rpath,' + torch_lib_path] # Removed RPATH line
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# To build this extension:
# 1. Make sure you have PyTorch installed with CUDA support.
# 2. Make sure you have the CUDA toolkit (nvcc) installed and in your PATH.
# 3. Run for development: python setup.py build_ext --inplace
# Or to install: python setup.py install
# Recommended clean build:
# python setup.py clean --all
# python setup.py build_ext --inplace
