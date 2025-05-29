# This script benchmarks the custom CUDA Flash Attention implementation
# against PyTorch's nn.MultiheadAttention.
# It measures forward and backward pass execution times and compares
# the numerical accuracy of the outputs and gradients.

import torch
import torch.nn as nn
import time

try:
    from flash_attn_cuda import flash_attention_cuda
    FLASH_CUDA_AVAILABLE = True
except ImportError:
    print("CUDA Flash Attention module not found or not built. Please build it using 'python setup.py install'.")
    print("Falling back to Triton version if available, or skipping custom tests.")
    FLASH_CUDA_AVAILABLE = False
    # Try to import Triton version as a fallback or for comparison
    try:
        from flash_attention import attention as triton_flash_attention
        TRITON_AVAILABLE = True
        print("Using Triton implementation as fallback/comparison.")
    except ImportError:
        TRITON_AVAILABLE = False
        print("Triton implementation also not found.")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_benchmark(BATCH, H, N_CTX, HEAD_DIM, causal, device=DEVICE):
    if not FLASH_CUDA_AVAILABLE:
        print(f"\nSkipping benchmark for B={BATCH}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM}, Causal={causal} as Custom CUDA module is not available.")
        # Optionally, run Triton if available
        if TRITON_AVAILABLE:
             print("Attempting to run with Triton implementation instead...")
             # Call a modified run_benchmark or a specific Triton benchmark function here
             # For simplicity, we'll just note it.
        else:
            return

    print(f"\n--- Running Benchmark: B={BATCH}, H={H}, N_CTX={N_CTX}, D={HEAD_DIM}, Causal={causal}, Device={device} ---")

    # Initialize tensors
    dtype = torch.float16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn_like(q)

    sm_scale = HEAD_DIM ** -0.5

    # --- Custom CUDA Flash Attention ---
    print("\n--- Custom CUDA Flash Attention ---")
    # Reset grads for custom op
    if q.grad is not None: q.grad.zero_()
    if k.grad is not None: k.grad.zero_()
    if v.grad is not None: v.grad.zero_()
    
    # Warmup
    custom_out_warmup = flash_attention_cuda(q, k, v, sm_scale, causal)
    torch.cuda.synchronize()

    start_time_fwd_custom = time.time()
    custom_out = flash_attention_cuda(q, k, v, sm_scale, causal)
    torch.cuda.synchronize() # Wait for GPU to finish
    end_time_fwd_custom = time.time()
    fwd_time_custom = end_time_fwd_custom - start_time_fwd_custom
    print(f"Forward pass time: {fwd_time_custom:.6f} seconds")

    # Backward pass
    # Warmup for backward
    custom_out.backward(dout, retain_graph=True) # Use custom_out from actual run for backward warmup
    if q.grad is not None: q.grad.zero_() # Reset grads
    if k.grad is not None: k.grad.zero_()
    if v.grad is not None: v.grad.zero_()
    torch.cuda.synchronize()
    
    start_time_bwd_custom = time.time()
    custom_out.backward(dout, retain_graph=True)
    torch.cuda.synchronize() # Wait for GPU to finish
    end_time_bwd_custom = time.time()
    bwd_time_custom = end_time_bwd_custom - start_time_bwd_custom
    print(f"Backward pass time: {bwd_time_custom:.6f} seconds")

    custom_dq, q.grad = q.grad.clone(), None
    custom_dk, k.grad = k.grad.clone(), None
    custom_dv, v.grad = v.grad.clone(), None

    
    # --- PyTorch nn.MultiheadAttention Implementation ---
    print("\n--- PyTorch nn.MultiheadAttention ---")
    # Reset grads for PyTorch MHA section
    # Detach and re-require grad for q, k, v to ensure clean state for MHA
    q_mha_input = q.detach().clone().requires_grad_(True)
    k_mha_input = k.detach().clone().requires_grad_(True)
    v_mha_input = v.detach().clone().requires_grad_(True)

    embed_dim = HEAD_DIM * H
    mha_module = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=H, batch_first=True, dtype=dtype, device=device)

    # Reshape q, k, v for MHA: (BATCH, H, N_CTX, HEAD_DIM) -> (BATCH, N_CTX, H, HEAD_DIM) -> (BATCH, N_CTX, embed_dim)
    q_mha = q_mha_input.permute(0, 2, 1, 3).reshape(BATCH, N_CTX, embed_dim)
    k_mha = k_mha_input.permute(0, 2, 1, 3).reshape(BATCH, N_CTX, embed_dim)
    v_mha = v_mha_input.permute(0, 2, 1, 3).reshape(BATCH, N_CTX, embed_dim)
    dout_mha_input = dout.permute(0, 2, 1, 3).reshape(BATCH, N_CTX, embed_dim)

    attn_mask = None
    if causal:
        # MHA expects mask where True means "masked out"
        attn_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=device, dtype=torch.bool), diagonal=1)

    # Warmup
    _, _ = mha_module(q_mha, k_mha, v_mha, attn_mask=attn_mask, average_attn_weights=False)
    torch.cuda.synchronize()

    start_time_fwd_pytorch = time.time()
    mha_out_tuple = mha_module(q_mha, k_mha, v_mha, attn_mask=attn_mask, average_attn_weights=False)
    mha_out = mha_out_tuple[0]
    torch.cuda.synchronize()
    end_time_fwd_pytorch = time.time()
    fwd_time_pytorch = end_time_fwd_pytorch - start_time_fwd_pytorch
    print(f"Forward pass time: {fwd_time_pytorch:.6f} seconds")

    # Backward pass
    # Warmup for backward
    mha_out.backward(dout_mha_input, retain_graph=True)
    if q_mha_input.grad is not None: q_mha_input.grad.zero_() # Reset MHA grads
    if k_mha_input.grad is not None: k_mha_input.grad.zero_()
    if v_mha_input.grad is not None: v_mha_input.grad.zero_()
    torch.cuda.synchronize()

    start_time_bwd_pytorch = time.time()
    mha_out.backward(dout_mha_input, retain_graph=True)
    torch.cuda.synchronize()
    end_time_bwd_pytorch = time.time()
    bwd_time_pytorch = end_time_bwd_pytorch - start_time_bwd_pytorch
    print(f"Backward pass time: {bwd_time_pytorch:.6f} seconds")

    # Reshape MHA grads back to (BATCH, H, N_CTX, HEAD_DIM)
    mha_dq = q_mha_input.grad.reshape(BATCH, N_CTX, H, HEAD_DIM).permute(0, 2, 1, 3)
    mha_dk = k_mha_input.grad.reshape(BATCH, N_CTX, H, HEAD_DIM).permute(0, 2, 1, 3)
    mha_dv = v_mha_input.grad.reshape(BATCH, N_CTX, H, HEAD_DIM).permute(0, 2, 1, 3)

    # Reshape MHA output back to (BATCH, H, N_CTX, HEAD_DIM)
    mha_out_reshaped = mha_out.reshape(BATCH, N_CTX, H, HEAD_DIM).permute(0, 2, 1, 3)

    # --- Timing Summary ---
    print("\n--- Timing Summary ---")
    print(f"  Custom CUDA Flash Attention Forward Time: {fwd_time_custom:.6f} s")
    print(f"  PyTorch MHA Forward Time:               {fwd_time_pytorch:.6f} s")
    print(f"  Custom CUDA Flash Attention Backward Time: {bwd_time_custom:.6f} s")
    print(f"  PyTorch MHA Backward Time:              {bwd_time_pytorch:.6f} s")

    # --- Accuracy Comparison ---
    print("\n--- Accuracy Comparison (Custom CUDA vs PyTorch MHA) ---")
    # We use a relatively high tolerance due to potential differences in implementation details (e.g. exact softmax computation)
    # and fp16 arithmetic. The custom CUDA kernels are placeholders and will output zeros.
    atol = 1e-2 
    rtol = 1e-2 

    outputs_match = torch.allclose(custom_out, mha_out_reshaped, atol=atol, rtol=rtol)
    print(f"Forward outputs match: {outputs_match}")
    if not outputs_match:
        print("Max diff in forward outputs:", (custom_out - mha_out_reshaped).abs().max().item())
        # print("Custom out sample:", custom_out[0,0,0,:5])
        # print("MHA out sample:", mha_out_reshaped[0,0,0,:5])


    dq_match = torch.allclose(custom_dq, mha_dq, atol=atol, rtol=rtol)
    print(f"dQ gradients match: {dq_match}")
    if not dq_match:
        print("Max diff in dQ gradients:", (custom_dq - mha_dq).abs().max().item())
        # print("Custom dQ sample:", custom_dq[0,0,0,:5])
        # print("MHA dQ sample:", mha_dq[0,0,0,:5])


    dk_match = torch.allclose(custom_dk, mha_dk, atol=atol, rtol=rtol)
    print(f"dK gradients match: {dk_match}")
    if not dk_match:
        print("Max diff in dK gradients:", (custom_dk - mha_dk).abs().max().item())
        # print("Custom dK sample:", custom_dk[0,0,0,:5])
        # print("MHA dK sample:", mha_dk[0,0,0,:5])
        
    dv_match = torch.allclose(custom_dv, mha_dv, atol=atol, rtol=rtol)
    print(f"dV gradients match: {dv_match}")
    if not dv_match:
        print("Max diff in dV gradients:", (custom_dv - mha_dv).abs().max().item())
        # print("Custom dV sample:", custom_dv[0,0,0,:5])
        # print("MHA dV sample:", mha_dv[0,0,0,:5])


if __name__ == "__main__":
    # --- Example Usage ---
    # To run the benchmark, execute this script directly from your terminal:
    # python benchmark.py
    #
    # You can add more configurations to test by calling `run_benchmark`
    # with different parameters for BATCH, H (num_heads), N_CTX (sequence_length),
    # HEAD_DIM, and causal (True or False).
    #
    # Each call to run_benchmark will print:
    # - The configuration parameters being tested.
    # - Forward pass execution time for both Custom CUDA Flash Attention and PyTorch MHA.
    #   A lower time indicates better performance.
    # - Backward pass execution time for both implementations.
    #   A lower time indicates better performance.
    # - Accuracy comparison results for forward outputs and gradients (dQ, dK, dV).
    #   'True' signifies that the outputs/gradients are numerically close within the defined tolerances.
    #   'False' along with the maximum difference will be printed if they are not close.
    #   (Note: With placeholder CUDA kernels, outputs/grads will be zeros, so comparisons will likely fail
    #    against a functional PyTorch MHA unless PyTorch MHA also produces zeros for some reason,
    #    or if inputs lead to zero outputs naturally).

    # Example configurations:
    # Small config for quick test
    print("Running small test configurations...")
    if DEVICE == 'cpu':
        print("CUDA not available. Skipping benchmarks that require CUDA.")
    else:
        run_benchmark(BATCH=2, H=4, N_CTX=256, HEAD_DIM=32, causal=True, device=DEVICE)
        run_benchmark(BATCH=2, H=4, N_CTX=256, HEAD_DIM=32, causal=False, device=DEVICE)

    # Larger configurations (uncomment the lines below to run more extensive tests if CUDA is available)
    # if DEVICE == 'cuda':
        # print("\nRunning larger test configurations (these may take a while)...")
        # run_benchmark(BATCH=4, H=8, N_CTX=1024, HEAD_DIM=64, causal=True, device=DEVICE)
        # run_benchmark(BATCH=4, H=8, N_CTX=1024, HEAD_DIM=64, causal=False, device=DEVICE)
        # run_benchmark(BATCH=2, H=12, N_CTX=2048, HEAD_DIM=64, causal=True, device=DEVICE) # Potentially OOM on some GPUs

        # Test with different head dimensions
        # print("\nRunning configurations with different head dimensions...")
        # run_benchmark(BATCH=2, H=4, N_CTX=512, HEAD_DIM=128, causal=True, device=DEVICE)
    
    if not FLASH_CUDA_AVAILABLE and not TRITON_AVAILABLE and DEVICE == 'cuda':
        print("\nNote: Neither Custom CUDA nor Triton Flash Attention modules were found.")
        print("The benchmark compared PyTorch MHA against (non-functional) placeholders for the custom CUDA version.")
    elif not FLASH_CUDA_AVAILABLE and TRITON_AVAILABLE and DEVICE == 'cuda':
        print("\nNote: Custom CUDA Flash Attention module was not found.")
        print("If Triton benchmarks were run, they would be against PyTorch MHA.")
        print("Currently, the script is set to skip if custom CUDA is not found, or this message implies it tried to run something else.")


    print("\nBenchmark script finished.")
