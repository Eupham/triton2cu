// flash_attention_kernels.cu

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> // Added header
#include <cuda_runtime.h>
#include <cuda_fp16.h> // For __half
#include <vector>
#include <iostream> // For placeholder messages
#include <cstdio> // For fprintf

// Define fixed tile sizes for the kernel compilation
#define KERNEL_BLOCK_M 64
#define KERNEL_BLOCK_N 64

__global__ void flash_fwd_kernel(
    const __half* __restrict__ q_ptr,    // Query tensor [B, H, N_q, D_head]
    const __half* __restrict__ k_ptr,    // Key tensor [B, H, N_kv, D_head]
    const __half* __restrict__ v_ptr,    // Value tensor [B, H, N_kv, D_head]
    __half* __restrict__ o_ptr,          // Output tensor [B, H, N_q, D_head]
    float* __restrict__ softmax_lse_ptr, // Log-sum-exp [B, H, N_q]
    const int B,
    const int H,
    const int N_q,
    const int N_kv,
    const int D_head,
    const float sm_scale,
    const bool is_causal
    // BLOCK_M_KERNEL and BLOCK_N_KERNEL removed from parameters
) {
    // --- Thread and Block Indexing ---
    // Each block processes one KERNEL_BLOCK_M segment of Q for a specific batch and head.
    // blockIdx.x maps to the Q sequence block.
    // blockIdx.y maps to (batch * num_heads).
    const int q_block_idx_m = blockIdx.x; // Index of the current Q block along N_q
    const int bh_idx = blockIdx.y;        // Combined batch and head index

    const int current_batch = bh_idx / H;
    const int current_head = bh_idx % H;

    // Starting row index of Q for this block
    const int q_start_row_idx = q_block_idx_m * KERNEL_BLOCK_M; // Use define

    // Thread indexing (example, will be refined for actual computation)
    const int tidx = threadIdx.x; // Assuming 1D thread block for now for loading
    // const int warp_id = tidx / 32;
    // const int lane_id = tidx % 32;

    // Shared memory for tiles (using defines)
    __shared__ __half q_tile[KERNEL_BLOCK_M][D_head];
    __shared__ __half k_tile[KERNEL_BLOCK_N][D_head];
    __shared__ __half v_tile[KERNEL_BLOCK_N][D_head];

    // Online Softmax Accumulators (per thread, for its assigned Q row m)
    // Assumes blockDim.x == KERNEL_BLOCK_M, so threadIdx.x maps directly to a row in q_tile.
    const int m = threadIdx.x; // Q row index within the tile for this thread

    float m_i = -INFINITY; // Current max for this Q row
    float l_i = 1.0f;      // Current sum_exp for this Q row
    float acc[D_head];     // Accumulator for O for this Q row
    for (int d_acc = 0; d_acc < D_head; ++d_acc) {
        acc[d_acc] = 0.0f;
    }

    // --- Load Q tile for this block ---
    // Base pointer for Q for the current batch and head
    const __half* q_batch_head_ptr = q_ptr + current_batch * H * N_q * D_head + current_head * N_q * D_head;

    // Load Q tile into shared memory (q_tile[KERNEL_BLOCK_M][D_head])
    for (int i = tidx; i < KERNEL_BLOCK_M * D_head; i += blockDim.x) { // Use define
        int row = i / D_head;
        int col = i % D_head;
        if (q_start_row_idx + row < N_q) { // Boundary check for Q sequence
            ((__half*)q_tile)[row * D_head + col] = q_batch_head_ptr[(q_start_row_idx + row) * D_head + col];
        } else {
            ((__half*)q_tile)[row * D_head + col] = static_cast<__half>(0.0f); // Padding
        }
    }
    __syncthreads(); // Ensure Q tile is loaded before use

    // --- Determine K/V Loop Bounds based on Causality ---
    // This translates Triton's STAGE logic and _attn_fwd_inner's lo/hi.
    // There will be two main scenarios:
    // 1. Processing K/V blocks *before* the diagonal (only if causal).
    // 2. Processing the K/V block *at* the diagonal (causal or non-causal, masking differs).
    // 3. Processing K/V blocks *after* the diagonal (only if causal and if full context not covered by 1 & 2,
    //    but FlashAttention usually avoids this by design for the second loop pass in Triton's causal path).
    //    Alternatively, for non-causal, it's one loop over all K/V.

    // Simplified: One main loop over K/V blocks (start_n_kv).
    // Masking and exact range will refine this.
    // The loop iterates over K/V sequence in steps of KERNEL_BLOCK_N.
    
    // Loop 1: Pre-diagonal blocks if causal
    if (is_causal) {
        for (int start_n_kv = 0; start_n_kv < q_start_row_idx; start_n_kv += KERNEL_BLOCK_N) { // Use define
            // Base pointer for K for the current batch and head
            const __half* k_batch_head_ptr = k_ptr + current_batch * H * N_kv * D_head + current_head * N_kv * D_head;
            // Base pointer for V for the current batch and head
            const __half* v_batch_head_ptr = v_ptr + current_batch * H * N_kv * D_head + current_head * N_kv * D_head;

            // Load K tile into shared memory (k_tile[KERNEL_BLOCK_N][D_head])
            for (int i = tidx; i < KERNEL_BLOCK_N * D_head; i += blockDim.x) { // Use define
                int row = i / D_head; // Index within K tile (0 to KERNEL_BLOCK_N-1)
                int col = i % D_head; // Index within head dim (0 to D_head-1)
                if (start_n_kv + row < N_kv) { // Boundary check for K sequence
                    ((__half*)k_tile)[row * D_head + col] = k_batch_head_ptr[(start_n_kv + row) * D_head + col];
                } else {
                    ((__half*)k_tile)[row * D_head + col] = static_cast<__half>(0.0f); // Padding
                }
            }
            // Load V tile into shared memory (v_tile[KERNEL_BLOCK_N][D_head])
            for (int i = tidx; i < KERNEL_BLOCK_N * D_head; i += blockDim.x) { // Use define
                int row = i / D_head;
                int col = i % D_head;
                if (start_n_kv + row < N_kv) { // Boundary check for V sequence
                    ((__half*)v_tile)[row * D_head + col] = v_batch_head_ptr[(start_n_kv + row) * D_head + col];
                } else {
                    ((__half*)v_tile)[row * D_head + col] = static_cast<__half>(0.0f); // Padding
                }
            }
            __syncthreads(); // Ensure K and V tiles are loaded

            // QK^T computation and online softmax update
            float qk_tile_row[KERNEL_BLOCK_N];  // Use define
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
                float qk_val = 0.0f;
                for (int d = 0; d < D_head; ++d) {
                    qk_val += __half2float(q_tile[m][d]) * __half2float(k_tile[n][d]);
                }
                qk_tile_row[n] = qk_val;
            }

            const float qk_scale_factor = sm_scale * 1.44269504f; // 1/log(2)
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
                qk_tile_row[n] *= qk_scale_factor;
            }
            
            // No intra-tile causal masking for pre-diagonal blocks

            float current_row_max_qk = -INFINITY;
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
                if (qk_tile_row[n] > current_row_max_qk) {
                    current_row_max_qk = qk_tile_row[n];
                }
            }

            float new_m_i = fmaxf(m_i, current_row_max_qk);
            __half p_ij_row[KERNEL_BLOCK_N]; // Use define
            float p_sum_numerator = 0.0f;
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
                float p_val_float = exp2f(qk_tile_row[n] - new_m_i);
                p_ij_row[n] = __float2half(p_val_float);
                p_sum_numerator += p_val_float;
            }

            float alpha = exp2f(m_i - new_m_i);
            for (int d_acc = 0; d_acc < D_head; ++d_acc) {
                acc[d_acc] *= alpha;
            }

            for (int d_acc = 0; d_acc < D_head; ++d_acc) {
                float pv_sum_d = 0.0f;
                for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
                    pv_sum_d += __half2float(p_ij_row[n]) * __half2float(v_tile[n][d_acc]);
                }
                acc[d_acc] += pv_sum_d;
            }
            
            l_i = l_i * alpha + p_sum_numerator;
            m_i = new_m_i;
            __syncthreads(); // Sync before next K/V iteration if K/V tiles are overwritten
        }
    }

    // Loop 2: Diagonal block (and all blocks if not causal)
    // The starting point for this loop depends on whether we are causal or not.
    // If causal, it's the diagonal block. If not causal, it's from the beginning.
    int diagonal_and_post_loop_start_n_kv = is_causal ? q_start_row_idx : 0;

    for (int start_n_kv = diagonal_and_post_loop_start_n_kv; start_n_kv < N_kv; start_n_kv += KERNEL_BLOCK_N) { // Use define
        // Base pointer for K for the current batch and head
        const __half* k_batch_head_ptr = k_ptr + current_batch * H * N_kv * D_head + current_head * N_kv * D_head;
        // Base pointer for V for the current batch and head
        const __half* v_batch_head_ptr = v_ptr + current_batch * H * N_kv * D_head + current_head * N_kv * D_head;

        // Load K tile into shared memory (k_tile[KERNEL_BLOCK_N][D_head])
        for (int i = tidx; i < KERNEL_BLOCK_N * D_head; i += blockDim.x) { // Use define
            int row = i / D_head; // Index within K tile (0 to KERNEL_BLOCK_N-1)
            int col = i % D_head; // Index within head dim (0 to D_head-1)
            if (start_n_kv + row < N_kv) { // Boundary check for K sequence
                ((__half*)k_tile)[row * D_head + col] = k_batch_head_ptr[(start_n_kv + row) * D_head + col];
            } else {
                ((__half*)k_tile)[row * D_head + col] = static_cast<__half>(0.0f); // Padding
            }
        }
        // Load V tile into shared memory (v_tile[KERNEL_BLOCK_N][D_head])
        for (int i = tidx; i < KERNEL_BLOCK_N * D_head; i += blockDim.x) { // Use define
            int row = i / D_head;
            int col = i % D_head;
            if (start_n_kv + row < N_kv) { // Boundary check for V sequence
                ((__half*)v_tile)[row * D_head + col] = v_batch_head_ptr[(start_n_kv + row) * D_head + col];
            } else {
                ((__half*)v_tile)[row * D_head + col] = static_cast<__half>(0.0f); // Padding
            }
        }
        __syncthreads(); // Ensure K and V tiles are loaded
        
        // QK^T computation and online softmax update
        float qk_tile_row[KERNEL_BLOCK_N]; // Use define
        for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
            float qk_val = 0.0f;
            for (int d = 0; d < D_head; ++d) {
                qk_val += __half2float(q_tile[m][d]) * __half2float(k_tile[n][d]);
            }
            qk_tile_row[n] = qk_val;
        }

        const float qk_scale_factor = sm_scale * 1.44269504f; // 1/log(2)
        for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
            qk_tile_row[n] *= qk_scale_factor;
        }

        // Apply intra-tile causal mask if this is the diagonal block in a causal pass
        // Check if the current K/V block overlaps with the Q block this thread block is processing
        // and if q_global_idx < k_global_idx
        if (is_causal && start_n_kv < (q_start_row_idx + KERNEL_BLOCK_M) &&  // Use define
            (q_start_row_idx < (start_n_kv + KERNEL_BLOCK_N)) ) { // Use define
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
                if ((q_start_row_idx + m) < (start_n_kv + n)) {
                    qk_tile_row[n] = -INFINITY;
                }
            }
        }
        
        float current_row_max_qk = -INFINITY;
        for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
            if (qk_tile_row[n] > current_row_max_qk) {
                current_row_max_qk = qk_tile_row[n];
            }
        }

        float new_m_i = fmaxf(m_i, current_row_max_qk);
        __half p_ij_row[KERNEL_BLOCK_N]; // Use define
        float p_sum_numerator = 0.0f;
        for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
            float p_val_float = exp2f(qk_tile_row[n] - new_m_i);
            p_ij_row[n] = __float2half(p_val_float);
            p_sum_numerator += p_val_float;
        }

        float alpha = exp2f(m_i - new_m_i);
        for (int d_acc = 0; d_acc < D_head; ++d_acc) {
            acc[d_acc] *= alpha;
        }

        for (int d_acc = 0; d_acc < D_head; ++d_acc) {
            float pv_sum_d = 0.0f;
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) { // Use define
                pv_sum_d += __half2float(p_ij_row[n]) * __half2float(v_tile[n][d_acc]);
            }
            acc[d_acc] += pv_sum_d;
        }
        
        l_i = l_i * alpha + p_sum_numerator;
        m_i = new_m_i;
        __syncthreads(); // Sync before next K/V iteration
    }
    
    // --- Finalize output ---
    // m_i, l_i, acc are per-thread registers holding values for Q row m = threadIdx.x.
    // (This assumes blockDim.x == BLOCK_M_KERNEL for direct mapping m=threadIdx.x)

    if (q_start_row_idx + m < N_q) { // Check if the Q row is within bounds (m == threadIdx.x)

        float inv_l_i = (l_i == 0.0f || l_i != l_i) ? 0.0f : 1.0f / l_i; // Avoid division by zero or NaN
        float final_lse_val = m_i + log2f(l_i);
        if (l_i == 0.0f || l_i != l_i) { // if l_i was 0 or NaN, lse might be NaN too
            final_lse_val = -INFINITY; // or some other indicator of invalid stats
        }


        // Store softmax_lse for this Q row
        // softmax_lse_ptr is [B, H, N_q]
        // bh_idx = current_batch * H + current_head
        // Global index for this Q row in LSE tensor: bh_idx * N_q + (q_start_row_idx + m)
        softmax_lse_ptr[bh_idx * N_q + q_start_row_idx + m] = final_lse_val;

        // Calculate and store final O values for this Q row
        // o_ptr is [B, H, N_q, D_head]
        // Base pointer for the current Q output row in global memory:
        // (bh_idx * N_q + (q_start_row_idx + m)) * D_head
        __half* o_row_global_ptr = o_ptr + (bh_idx * N_q + q_start_row_idx + m) * D_head;
        for (int d = 0; d < D_head; ++d) {
            o_row_global_ptr[d] = __float2half(acc[d] * inv_l_i);
        }
    }
    // No __syncthreads() needed here at the very end of the kernel for storing,
    // as threads are writing to distinct global memory locations for their assigned Q row.
}

std::vector<torch::Tensor> flash_attn_forward_cuda(
    torch::Tensor q,         // [B, H, N_q, D_head]
    torch::Tensor k,         // [B, H, N_kv, D_head]
    torch::Tensor v,         // [B, H, N_kv, D_head]
    float sm_scale,
    bool causal
) {
    // Input validation (basic checks)
    // TODO: Add BLOCK_M_KERNEL and BLOCK_N_KERNEL to parameters and pass them to kernel - This is now handled by defines
    TORCH_CHECK(q.device().is_cuda(), "Input Q must be a CUDA tensor");
    TORCH_CHECK(k.device().is_cuda(), "Input K must be a CUDA tensor");
    TORCH_CHECK(v.device().is_cuda(), "Input V must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "Input Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input V must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "Input Q must be FP16");
    TORCH_CHECK(k.scalar_type() == torch::kFloat16, "Input K must be FP16");
    TORCH_CHECK(v.scalar_type() == torch::kFloat16, "Input V must be FP16");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N_q = q.size(2);
    const int D_head = q.size(3);
    const int N_kv = k.size(2);

    TORCH_CHECK(D_head == k.size(3) && D_head == v.size(3), "Head dimensions must match");
    TORCH_CHECK(H == k.size(1) && H == v.size(1), "Number of heads must match");
    TORCH_CHECK(B == k.size(0) && B == v.size(0), "Batch sizes must match");
    TORCH_CHECK(N_kv == v.size(2), "Seq len for K and V must match");

    // Output tensor for attention output
    auto opts = q.options();
    torch::Tensor o = torch::empty_like(q, opts);

    // Output tensor for log-sum-exp (LSE) for backward pass
    // Shape: (B, H, N_q)
    torch::Tensor softmax_lse = torch::empty({B, H, N_q}, opts.dtype(torch::kFloat32));

    // Define block and grid dimensions (these are placeholders and require tuning)
    // For FlashAttention, tiling is crucial. A common approach is to tile N_q and N_kv.
    // Example: dim3 threads(128); // 128 threads per block
    // Example: dim3 blocks(cdiv(N_q, KERNEL_BLOCK_M), H * B); // Updated grid calculation
    // These need to be carefully chosen based on head dimension, shared memory, etc.

    // Updated grid calculation using defines:
    // Each block handles a KERNEL_BLOCK_M segment of Q.
    // Grid dim for x: (N_q + KERNEL_BLOCK_M - 1) / KERNEL_BLOCK_M
    // Grid dim for y: B * H
    dim3 threads(KERNEL_BLOCK_M); // Threads per block is KERNEL_BLOCK_M
    dim3 blocks(
        (N_q + KERNEL_BLOCK_M - 1) / KERNEL_BLOCK_M, // Number of Q blocks
        B * H                                         // Batch size * Number of heads
    );

    // Get data pointers
    const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr());
    const __half* k_ptr = reinterpret_cast<const __half*>(k.data_ptr());
    const __half* v_ptr = reinterpret_cast<const __half*>(v.data_ptr());
    __half* o_ptr = reinterpret_cast<__half*>(o.data_ptr());
    float* softmax_lse_ptr = softmax_lse.data_ptr<float>();

    // Launch the CUDA kernel
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream(); // Get current PyTorch stream
    // flash_fwd_kernel<<<blocks, threads, 0, stream>>>(
    //     q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
    //     B, H, N_q, N_kv, D_head,
    //     sm_scale, causal
    // );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(); // Get current PyTorch stream
    flash_fwd_kernel<<<blocks, threads, 0, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
        B, H, N_q, N_kv, D_head,
        sm_scale, causal
        // BLOCK_M_KERNEL_val, BLOCK_N_KERNEL_val removed from launch call
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error in %s: %s\n", "flash_attn_forward_cuda", cudaGetErrorString(err));
        // TORCH_CHECK could be used here if preferred and if it handles CUDA errors appropriately
        // For now, just printing to stderr and continuing, as outputs are placeholders
    }

    // Fill output tensors with zeros as the kernel is a placeholder.
    // In a real implementation, the kernel would populate these.
    o.zero_();
    softmax_lse.zero_();

    return {o, softmax_lse};
}

// Placeholder for the actual CUDA kernel for the backward pass
// This kernel would perform the core Flash Attention backward computation.
__global__ void flash_bwd_kernel(
    const __half* __restrict__ dout_ptr,
    const __half* __restrict__ q_ptr,
    const __half* __restrict__ k_ptr,
    const __half* __restrict__ v_ptr,
    const __half* __restrict__ o_ptr,
    const float* __restrict__ softmax_lse_ptr, // From forward
    __half* __restrict__ dq_ptr,
    __half* __restrict__ dk_ptr,
    __half* __restrict__ dv_ptr,
    int B, int H, int N_q, int N_kv, int D,
    float sm_scale,
    bool is_causal
    // TODO: Add strides and other necessary parameters
) {
    // This is a placeholder. A full implementation is complex.
    // It would involve:
    // 1. Recomputing attention scores (or using saved ones if memory allows, though Flash Attention aims to avoid this).
    // 2. Calculating dP = dO V^T.
    // 3. Calculating dS = P * (dP - sum(P * dP, dim=-1)).
    // 4. Calculating dQ = dS K.
    // 5. Calculating dK = dS^T Q.
    // 6. Calculating dV = P^T dO.
    // All while handling tiling and shared memory.

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // This printf is a placeholder
        // printf("CUDA flash_bwd_kernel: Placeholder executed.\n");
    }
}

std::vector<torch::Tensor> flash_attn_backward_cuda(
    torch::Tensor dout,        // [B, H, N_q, D_head] (gradient of o)
    torch::Tensor q,           // [B, H, N_q, D_head] (saved from forward)
    torch::Tensor k,           // [B, H, N_kv, D_head] (saved from forward)
    torch::Tensor v,           // [B, H, N_kv, D_head] (saved from forward)
    torch::Tensor o,           // [B, H, N_q, D_head] (saved from forward)
    torch::Tensor softmax_lse, // [B, H, N_q] (saved from forward)
    float sm_scale,
    bool causal
) {
    // Input validation (basic checks)
    TORCH_CHECK(dout.device().is_cuda(), "Input dout must be a CUDA tensor");
    TORCH_CHECK(q.device().is_cuda(), "Input Q must be a CUDA tensor");
    TORCH_CHECK(k.device().is_cuda(), "Input K must be a CUDA tensor");
    TORCH_CHECK(v.device().is_cuda(), "Input V must be a CUDA tensor");
    TORCH_CHECK(o.device().is_cuda(), "Input O must be a CUDA tensor");
    TORCH_CHECK(softmax_lse.device().is_cuda(), "Input softmax_lse must be a CUDA tensor");

    TORCH_CHECK(dout.is_contiguous(), "Input dout must be contiguous");
    TORCH_CHECK(q.is_contiguous(), "Input Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input V must be contiguous");
    TORCH_CHECK(o.is_contiguous(), "Input O must be contiguous");
    TORCH_CHECK(softmax_lse.is_contiguous(), "Input softmax_lse must be contiguous");

    TORCH_CHECK(dout.scalar_type() == torch::kFloat16, "Input dout must be FP16");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "Input Q must be FP16");
    TORCH_CHECK(k.scalar_type() == torch::kFloat16, "Input K must be FP16");
    TORCH_CHECK(v.scalar_type() == torch::kFloat16, "Input V must be FP16");
    TORCH_CHECK(o.scalar_type() == torch::kFloat16, "Input O must be FP16");
    TORCH_CHECK(softmax_lse.scalar_type() == torch::kFloat32, "Input softmax_lse must be FP32");


    const int B = q.size(0);
    const int H = q.size(1);
    const int N_q = q.size(2);
    const int D_head = q.size(3);
    const int N_kv = k.size(2);

    // Output tensors for gradients
    torch::Tensor dq = torch::empty_like(q);
    torch::Tensor dk = torch::empty_like(k);
    torch::Tensor dv = torch::empty_like(v);

    // Define block and grid dimensions (placeholders, require tuning)
    dim3 threads(32); // Small placeholder
    dim3 blocks(1, H, B); // Minimal placeholder for now

    // Get data pointers
    const __half* dout_ptr = reinterpret_cast<const __half*>(dout.data_ptr());
    const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr());
    const __half* k_ptr = reinterpret_cast<const __half*>(k.data_ptr());
    const __half* v_ptr = reinterpret_cast<const __half*>(v.data_ptr());
    const __half* o_ptr = reinterpret_cast<const __half*>(o.data_ptr());
    const float* softmax_lse_ptr = softmax_lse.data_ptr<float>();

    __half* dq_ptr = reinterpret_cast<__half*>(dq.data_ptr());
    __half* dk_ptr = reinterpret_cast<__half*>(dk.data_ptr());
    __half* dv_ptr = reinterpret_cast<__half*>(dv.data_ptr());

    // Launch the CUDA kernel
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // flash_bwd_kernel<<<blocks, threads, 0, stream>>>(
    //     dout_ptr, q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
    //     dq_ptr, dk_ptr, dv_ptr,
    //     B, H, N_q, N_kv, D_head,
    //     sm_scale, causal
    // );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    flash_bwd_kernel<<<blocks, threads, 0, stream>>>(
        dout_ptr, q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        B, H, N_q, N_kv, D_head,
        sm_scale, causal
    );
    cudaError_t err_bwd = cudaGetLastError();
    if (err_bwd != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error in %s: %s\n", "flash_attn_backward_cuda", cudaGetErrorString(err_bwd));
        // TORCH_CHECK could be used here
    }

    // Fill gradient tensors with zeros as the kernel is a placeholder.
    dq.zero_();
    dk.zero_();
    dv.zero_();

    return {dq, dk, dv};
}
