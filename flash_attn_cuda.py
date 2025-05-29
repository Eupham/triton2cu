import sys
import os

# --- Diagnostic Info ---
print("--- flash_attn_cuda.py: Diagnostic Info ---")
print(f"Current Working Directory: {os.getcwd()}")
print("Files in CWD:")
try:
    for item in os.listdir(os.getcwd()):
        print(f"  - {item}")
except Exception as e:
    print(f"  Error listing CWD contents: {e}")
print("Python sys.path:")
for pth in sys.path:
    print(f"  - {pth}")
print("Attempting to import flash_attn_cuda_kernels...")
# --- End Diagnostic Info ---

_FLASH_CUDA_KERNELS_AVAILABLE = False
try:
    print("flash_attn_cuda.py: Attempting to import flash_attn_cuda_kernels...")
    import flash_attn_cuda_kernels
    print("flash_attn_cuda.py: Successfully imported flash_attn_cuda_kernels.")
    _FLASH_CUDA_KERNELS_AVAILABLE = True
except ImportError as e:
    print(f"!!! flash_attn_cuda.py: FAILED to import flash_attn_cuda_kernels. !!!")
    print(f"!!! Detailed ImportError below: !!!")
    print(f"{e}") # Print the detailed error message
    # Re-raise to ensure benchmark.py's try-except (for importing from this file) catches it.
    raise
except Exception as e_other:
    print(f"!!! flash_attn_cuda.py: An UNEXPECTED error occurred during import of flash_attn_cuda_kernels. !!!")
    print(f"!!! Detailed Exception: {e_other} !!!")
    raise

import torch
# The original import line for flash_attn_cuda_kernels might be here or in benchmark.py.
# For now, this script is where the direct import happens for the autograd.Function.
# The actual 'import flash_attn_cuda_kernels' is now inside the try-except block above.

class FlashAttentionCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        # q, k, v: [B, H, N, D_head]
        # sm_scale: float
        # causal: bool

        # Ensure inputs are contiguous and on CUDA device, and correct dtype (e.g., float16)
        # The C++ wrapper (flash_attn_forward_cuda) already performs some checks,
        # but good practice to have checks here too or ensure consistency.
        for tensor_name, tensor_val in [('q', q), ('k', k), ('v', v)]:
            if not tensor_val.is_cuda:
                raise ValueError(f"{tensor_name} must be a CUDA tensor")
            if not tensor_val.is_contiguous():
                # Consider making it contiguous: tensor_val = tensor_val.contiguous()
                raise ValueError(f"{tensor_name} must be contiguous")
            if tensor_val.dtype != torch.float16:
                # Consider casting: tensor_val = tensor_val.to(torch.float16)
                raise ValueError(f"{tensor_name} must be of dtype torch.float16")

        # Call the C++ / CUDA forward function
        # flash_attn_forward_cuda returns [o, softmax_lse]
        # These are placeholder calls for now, actual kernel is not implemented
        o, softmax_lse = flash_attn_cuda_kernels.forward( # Corrected function name
            q, k, v, sm_scale, causal
        )

        # Save tensors for backward pass
        # Detach tensors that are not needed for gradient computation related to themselves
        # but are needed for other gradient computations.
        # q, k, v, o are needed for gradient computation.
        # softmax_lse is an intermediate result for backward.
        # causal is a boolean, sm_scale is a float. We need to pass them to backward.
        # Store sm_scale and causal in ctx directly or as dummy tensors if preferred.
        ctx.save_for_backward(q, k, v, o, softmax_lse)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        
        return o

    @staticmethod
    def backward(ctx, dout):
        # dout: [B, H, N, D_head] (gradient of o)

        if not dout.is_cuda:
            raise ValueError("dout must be a CUDA tensor")
        if not dout.is_contiguous(): # Important for CUDA
            dout = dout.contiguous() 
        if dout.dtype != torch.float16:
            # Consider casting: dout = dout.to(torch.float16)
            raise ValueError("dout must be of dtype torch.float16")

        q, k, v, o, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal

        # Call the C++ / CUDA backward function
        # flash_attn_backward_cuda returns [dq, dk, dv]
        # These are placeholder calls for now, actual kernel is not implemented
        dq, dk, dv = flash_attn_cuda_kernels.backward( # Corrected function name
            dout, q, k, v, o, softmax_lse, sm_scale, causal
        )

        # Gradients for q, k, v. No gradients for sm_scale, causal.
        return dq, dk, dv, None, None

# Convenience wrapper function
def flash_attention_cuda(q, k, v, sm_scale, causal=False):
    """
    Computes Flash Attention using custom CUDA kernels.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, N_q, D_head], dtype float16, CUDA.
        k (torch.Tensor): Key tensor of shape [B, H, N_kv, D_head], dtype float16, CUDA.
        v (torch.Tensor): Value tensor of shape [B, H, N_kv, D_head], dtype float16, CUDA.
        sm_scale (float): Scale factor for QK^T. Typically 1/sqrt(D_head).
        causal (bool): Whether to apply causal masking. Default is False.

    Returns:
        torch.Tensor: Output tensor of shape [B, H, N_q, D_head].
    """
    # Basic shape and type checks can be added here for user-friendliness
    # For example:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("Inputs q, k, v must be 4-dimensional tensors [B, H, N, D_head].")
    # Add more checks as needed (e.g. matching dimensions, head_dim > 0 etc.)

    return FlashAttentionCUDAFunction.apply(q, k, v, sm_scale, causal)
