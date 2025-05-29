from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory of the current setup.py file
setup_dir = os.path.dirname(os.path.realpath(__file__))
kernel_file = os.path.join(setup_dir, "flash_attention_kernels.cu")

setup(
    name='flash_attn_cuda_kernels', # Name of the Python package
    ext_modules=[
        CUDAExtension(
            name='flash_attn_cuda_kernels', # Must match the name passed to import
            sources=[kernel_file],
            # You might need to specify include_dirs if your .cu file includes
            # headers from other locations, or pass specific compiler args.
            # For example:
            # include_dirs=[os.path.join(setup_dir, 'include')],
            # extra_compile_args={'cxx': ['-g'],
            #                     'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# To build this extension:
# 1. Make sure you have PyTorch installed with CUDA support.
# 2. Make sure you have the CUDA toolkit (nvcc) installed and in your PATH.
# 3. Run: python setup.py install
# Or for development: python setup.py build_ext --inplace
