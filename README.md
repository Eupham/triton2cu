# Custom CUDA Flash Attention Implementation

This project provides a framework for implementing Flash Attention using custom CUDA kernels in PyTorch. It includes placeholder CUDA kernels, a C++ extension build system, a Python wrapper, and a benchmark script to compare against PyTorch's built-in MultiheadAttention.

## Project Structure

*   `flash_attention_kernels.cu`: Contains the C++ wrapper functions and placeholder CUDA kernels for the forward and backward passes of Flash Attention. **Currently, the CUDA kernels are placeholders and need to be fully implemented.**
*   `setup.py`: Python script to build the CUDA code in `flash_attention_kernels.cu` into a PyTorch C++ extension. The compiled module will be named `flash_attn_cuda_kernels`.
*   `flash_attn_cuda.py`: Python script that defines a `torch.autograd.Function` to wrap the compiled CUDA kernels, making them callable from PyTorch and integrating with its autograd system. It provides the `flash_attention_cuda` function.
*   `benchmark.py`: Python script to benchmark the custom CUDA Flash Attention implementation against `torch.nn.MultiheadAttention` for speed and numerical accuracy.

## Prerequisites

*   PyTorch installed with CUDA support.
*   NVIDIA CUDA Toolkit (including `nvcc` compiler) installed and configured in your system's PATH.
*   A C++ compiler compatible with your CUDA version (e.g., g++).

## Build Instructions

To build the custom CUDA extension, navigate to the root directory of this project in your terminal and run:

```bash
python setup.py install
```

Alternatively, for development, you can build the extension in-place:

```bash
python setup.py build_ext --inplace
```

This will compile `flash_attention_kernels.cu` and create a Python module (e.g., in a `build` directory and/or installing it into your Python environment) that can be imported as `flash_attn_cuda_kernels`.

If the build is successful, you should be able to import the custom module in Python, e.g., `import flash_attn_cuda_kernels`.

## Running Benchmarks

Once the CUDA extension is successfully built, you can run the benchmark script:

```bash
python benchmark.py
```

The script will:
1.  Test several configurations of batch size, number of heads, sequence length, and head dimension.
2.  For each configuration, it will run both the custom CUDA Flash Attention and PyTorch's `nn.MultiheadAttention`.
3.  Print the execution times for forward and backward passes for both implementations.
4.  Print accuracy comparison results.

**Note:** Since the CUDA kernels in `flash_attention_kernels.cu` are currently placeholders, the "custom CUDA" path in the benchmark will report execution of these placeholders (which return zero tensors). The accuracy comparison will likely show mismatches until the kernels are fully implemented. The primary purpose of running the benchmark with placeholder kernels is to verify that the build system, Python wrapper, and benchmark script are correctly integrated.
