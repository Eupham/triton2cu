#include <torch/extension.h>
#include <vector> // Required for std::vector

// Declare the C++ functions that are implemented in flash_attention_kernels.cu
// Ensure these declarations match exactly the function signatures in flash_attention_kernels.cu

// Forward pass function (returns output tensor and log-sum-exp tensor)
std::vector<torch::Tensor> flash_attn_forward_cuda(
    torch::Tensor q,         // [B, H, N_q, D_head]
    torch::Tensor k,         // [B, H, N_kv, D_head]
    torch::Tensor v,         // [B, H, N_kv, D_head]
    float sm_scale,
    bool causal
);

// Backward pass function (returns gradients for q, k, v)
std::vector<torch::Tensor> flash_attn_backward_cuda(
    torch::Tensor dout,        // [B, H, N_q, D_head] (gradient of o)
    torch::Tensor q,           // [B, H, N_q, D_head] (saved from forward)
    torch::Tensor k,           // [B, H, N_kv, D_head] (saved from forward)
    torch::Tensor v,           // [B, H, N_kv, D_head] (saved from forward)
    torch::Tensor o,           // [B, H, N_q, D_head] (saved from forward)
    torch::Tensor softmax_lse, // [B, H, N_q] (saved from forward)
    torch::Tensor delta,       // New
    float sm_scale,
    bool causal
);

// Declaration for the backward preprocess function (implemented in .cu file)
torch::Tensor flash_bwd_preprocess_cuda(
    torch::Tensor o,
    torch::Tensor dout
);

// PYBIND11_MODULE macro defines the module structure
// The first argument to PYBIND11_MODULE (flash_attn_cuda_kernels) MUST match
// the 'name' argument given to CUDAExtension in setup.py.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("name_in_python", &function_name_in_cpp, "docstring", py::arg("arg_name_in_python") = default_value);
    m.def(
        "forward", // Name of the function as it will be called from Python
        &flash_attn_forward_cuda, // Pointer to the C++ function
        "Flash Attention Forward Pass (CUDA)" // Docstring
        // pybind11 will automatically handle arguments by inspecting the C++ function signature.
        // For keyword arguments in Python, you can add py::arg("q"), py::arg("k"), etc.
        // For now, positional arguments are fine.
    );
    m.def(
        "backward", // Name of the function as it will be called from Python
        &flash_attn_backward_cuda, // Pointer to the C++ function
        "Flash Attention Backward Pass (CUDA)" // Docstring
    );
    // New binding for the preprocess backward function:
    m.def(
        "preprocess_backward",
        &flash_bwd_preprocess_cuda,
        "Flash Attention Backward Preprocess (Delta Computation CUDA)"
    );
}
