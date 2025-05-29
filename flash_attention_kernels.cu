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
#define KERNEL_MAX_D_HEAD 128 // Max head dimension supported by shared memory
#define KERNEL_PRE_BLOCK 128  // Tile size for N_CTX in preprocess kernel
// Backward pass tile sizes from Triton _attention.backward method
#define KERNEL_BLOCK_M1_BWD 32  // For dK/dV Q tile processing height
#define KERNEL_BLOCK_N1_BWD 128 // For dK/dV K/V tile height (also output tile height for dK, dV)
#define KERNEL_BLOCK_M2_BWD 128 // For dQ Q tile processing height (also output tile height for dQ)
#define KERNEL_BLOCK_N2_BWD 32  // For dQ K/V tile processing height
#define KERNEL_BLK_SLICE_FACTOR_BWD 2 // From Triton

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
    const int q_block_idx_m = blockIdx.x; 
    const int bh_idx = blockIdx.y;       

    const int current_batch = bh_idx / H;
    const int current_head = bh_idx % H;

    const int q_start_row_idx = q_block_idx_m * KERNEL_BLOCK_M;

    const int m = threadIdx.x; // Local row index in q_tile, assuming blockDim.x == KERNEL_BLOCK_M

    // --- Shared Memory Declaration ---
    __shared__ __half q_tile[KERNEL_BLOCK_M][KERNEL_MAX_D_HEAD];
    __shared__ __half k_tile[KERNEL_BLOCK_N][KERNEL_MAX_D_HEAD];
    __shared__ __half v_tile[KERNEL_BLOCK_N][KERNEL_MAX_D_HEAD];

    // --- Load Q tile ---
    // Base pointer for Q for the current batch and head. Using bh_idx directly.
    const __half* q_bh_base_ptr = q_ptr + bh_idx * N_q * D_head;

    // Each thread m (threadIdx.x) loads one row of q_tile.
    // This assumes blockDim.x (number of threads in x-dim) == KERNEL_BLOCK_M.
    // The C++ wrapper flash_attn_forward_cuda sets threads(KERNEL_BLOCK_M).
    if (m < KERNEL_BLOCK_M) { // Check if thread is within the KERNEL_BLOCK_M rows
        for (int d_load = 0; d_load < D_head; ++d_load) { // Load up to actual D_head
            if (q_start_row_idx + m < N_q) { // Boundary check for Q sequence (rows)
                q_tile[m][d_load] = q_bh_base_ptr[(q_start_row_idx + m) * D_head + d_load];
            } else {
                q_tile[m][d_load] = __float2half(0.0f); // Padding for out-of-bounds rows
            }
        }
    }
    __syncthreads(); 

    // --- Softmax Accumulator Initialization (Per-thread for Q-row m) ---
    float m_i = -INFINITY; 
    float l_i = 1.0f;    
    float acc[KERNEL_MAX_D_HEAD]; 
    
    for (int d_acc = 0; d_acc < D_head; ++d_acc) { 
        acc[d_acc] = 0.0f;
    }

    // ... (Rest of the kernel to be added in subsequent steps) ...

    // --- Determine K/V Loop Bounds based on Causality ---
    // Loop 1: Pre-diagonal blocks if causal
    if (is_causal) {
        for (int start_n_kv = 0; start_n_kv < q_start_row_idx; start_n_kv += KERNEL_BLOCK_N) {
            // Base pointer for K for the current batch and head
            const __half* k_batch_head_ptr = k_ptr + bh_idx * N_kv * D_head; // Using bh_idx directly
            // Base pointer for V for the current batch and head
            const __half* v_batch_head_ptr = v_ptr + bh_idx * N_kv * D_head; // Using bh_idx directly

            // Load K tile into shared memory
            // Each thread m loads a portion of K tile rows. This is a common strategy but needs careful indexing.
            // For simplicity here if KERNEL_BLOCK_M (threads) loads KERNEL_BLOCK_N rows:
            // This assumes KERNEL_BLOCK_M is large enough or threads loop.
            // The original loading was: for (int i = tidx; i < KERNEL_BLOCK_N * D_head; i += blockDim.x)
            // Let's use that more general cooperative loading. tidx is m.
            for (int i = m; i < KERNEL_BLOCK_N * D_head; i += KERNEL_BLOCK_M) { // blockDim.x is KERNEL_BLOCK_M
                int row = i / D_head;
                int col = i % D_head;
                if (start_n_kv + row < N_kv) { // Boundary check for K sequence
                    ((__half*)k_tile)[row * D_head + col] = k_batch_head_ptr[(start_n_kv + row) * D_head + col];
                } else {
                    ((__half*)k_tile)[row * D_head + col] = __float2half(0.0f); // Padding
                }
            }
            // Load V tile into shared memory
            for (int i = m; i < KERNEL_BLOCK_N * D_head; i += KERNEL_BLOCK_M) { // blockDim.x is KERNEL_BLOCK_M
                int row = i / D_head;
                int col = i % D_head;
                if (start_n_kv + row < N_kv) { // Boundary check for V sequence
                    ((__half*)v_tile)[row * D_head + col] = v_batch_head_ptr[(start_n_kv + row) * D_head + col];
                } else {
                    ((__half*)v_tile)[row * D_head + col] = __float2half(0.0f); // Padding
                }
            }
            __syncthreads(); // Ensure K and V tiles are loaded

            // --- QK^T computation and online softmax update (Step 3c) ---
            float qk_tile_row_loop1[KERNEL_BLOCK_N]; // Renamed from qk_tile_row for clarity
            
            if (m < KERNEL_BLOCK_M) { // Thread active for a Q row
                // Compute QK^T for the current Q row (m) and all K rows in k_tile
                for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                    float qk_val = 0.0f;
                    for (int d = 0; d < D_head; ++d) {
                        qk_val += __half2float(q_tile[m][d]) * __half2float(k_tile[n][d]);
                    }
                    qk_tile_row_loop1[n] = qk_val;
                }

                // Scale QK^T
                const float qk_scale_factor = sm_scale * 1.44269504f;
                for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                    qk_tile_row_loop1[n] *= qk_scale_factor;
                }
                // No intra-block causal masking in Loop 1 (pre-diagonal blocks)

                // --- Online Softmax Update (Step 3d) ---
                // 1. Find current_row_max_qk from qk_tile_row_loop1
                float current_row_max_qk = -INFINITY;
                for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                    if (qk_tile_row_loop1[n] > current_row_max_qk) {
                        current_row_max_qk = qk_tile_row_loop1[n];
                    }
                }

                // 2. Calculate new_m_i and handle fully masked rows
                float new_m_i_loop1; // Renamed for clarity
                if (current_row_max_qk == -INFINITY) { 
                    new_m_i_loop1 = m_i; 
                } else {
                    new_m_i_loop1 = fmaxf(m_i, current_row_max_qk);
                }

                // 3. Calculate P_ij values and sum for l_ij (p_sum_numerator)
                float p_sum_numerator_loop1 = 0.0f;
                __half p_ij_row_loop1[KERNEL_BLOCK_N]; 

                for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                    if (current_row_max_qk == -INFINITY) {
                        p_ij_row_loop1[n] = __float2half(0.0f);
                    } else {
                        float p_val_float = exp2f(qk_tile_row_loop1[n] - new_m_i_loop1);
                        p_ij_row_loop1[n] = __float2half(p_val_float);
                    }
                    p_sum_numerator_loop1 += __half2float(p_ij_row_loop1[n]);
                }
                
                // 4. Calculate alpha
                float alpha_loop1 = exp2f(m_i - new_m_i_loop1);

                // 5. Rescale existing acc
                for (int d_acc = 0; d_acc < D_head; ++d_acc) {
                    acc[d_acc] *= alpha_loop1;
                }

                // 6. Accumulate P.V
                if (current_row_max_qk != -INFINITY) { // Only if row not fully masked
                    for (int d_acc = 0; d_acc < D_head; ++d_acc) { 
                        float pv_sum_d = 0.0f;
                        for (int n = 0; n < KERNEL_BLOCK_N; ++n) { 
                            pv_sum_d += __half2float(p_ij_row_loop1[n]) * __half2float(v_tile[n][d_acc]);
                        }
                        acc[d_acc] += pv_sum_d;
                    }
                }
                
                // 7. Update l_i and m_i
                l_i = l_i * alpha_loop1 + p_sum_numerator_loop1;
                m_i = new_m_i_loop1;
            } // end if (m < KERNEL_BLOCK_M)
            
            __syncthreads(); // Sync before next K/V iteration
        }
    }

    // Loop 2: Diagonal block (and all blocks if not causal)
    int diagonal_and_post_loop_start_n_kv = is_causal ? q_start_row_idx : 0;
    for (int start_n_kv = diagonal_and_post_loop_start_n_kv; start_n_kv < N_kv; start_n_kv += KERNEL_BLOCK_N) {
        // Base pointer for K for the current batch and head
        const __half* k_batch_head_ptr = k_ptr + bh_idx * N_kv * D_head;
        // Base pointer for V for the current batch and head
        const __half* v_batch_head_ptr = v_ptr + bh_idx * N_kv * D_head;

        // Load K tile
        for (int i = m; i < KERNEL_BLOCK_N * D_head; i += KERNEL_BLOCK_M) {
            int row = i / D_head;
            int col = i % D_head;
            if (start_n_kv + row < N_kv) {
                ((__half*)k_tile)[row * D_head + col] = k_batch_head_ptr[(start_n_kv + row) * D_head + col];
            } else {
                ((__half*)k_tile)[row * D_head + col] = __float2half(0.0f);
            }
        }
        // Load V tile
        for (int i = m; i < KERNEL_BLOCK_N * D_head; i += KERNEL_BLOCK_M) {
            int row = i / D_head;
            int col = i % D_head;
            if (start_n_kv + row < N_kv) {
                ((__half*)v_tile)[row * D_head + col] = v_batch_head_ptr[(start_n_kv + row) * D_head + col];
            } else {
                ((__half*)v_tile)[row * D_head + col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // --- QK^T computation, causal masking (Step 3c) ---
        float qk_tile_row_loop2[KERNEL_BLOCK_N]; // Renamed for clarity
        if (m < KERNEL_BLOCK_M) { // Thread active for a Q row
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                float qk_val = 0.0f;
                for (int d = 0; d < D_head; ++d) {
                    qk_val += __half2float(q_tile[m][d]) * __half2float(k_tile[n][d]);
                }
                qk_tile_row_loop2[n] = qk_val;
            }

            const float qk_scale_factor = sm_scale * 1.44269504f;
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                qk_tile_row_loop2[n] *= qk_scale_factor;
            }

            // Apply intra-block causal mask
            if (is_causal && start_n_kv == q_start_row_idx) { 
                for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                    if (m < n) { 
                        qk_tile_row_loop2[n] = -INFINITY; 
                    }
                }
            }
            
            // --- Online Softmax Update (Step 3d) ---
            // 1. Find current_row_max_qk from qk_tile_row_loop2
            float current_row_max_qk = -INFINITY;
            for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                if (qk_tile_row_loop2[n] > current_row_max_qk) {
                    current_row_max_qk = qk_tile_row_loop2[n];
                }
            }

            // 2. Calculate new_m_i and handle fully masked rows
            float new_m_i_loop2; // Renamed for clarity
            if (current_row_max_qk == -INFINITY) { 
                new_m_i_loop2 = m_i; 
            } else {
                new_m_i_loop2 = fmaxf(m_i, current_row_max_qk);
            }

            // 3. Calculate P_ij values and sum for l_ij
            float p_sum_numerator_loop2 = 0.0f;
            __half p_ij_row_loop2[KERNEL_BLOCK_N]; 

            for (int n = 0; n < KERNEL_BLOCK_N; ++n) {
                if (current_row_max_qk == -INFINITY) {
                    p_ij_row_loop2[n] = __float2half(0.0f);
                } else {
                    float p_val_float = exp2f(qk_tile_row_loop2[n] - new_m_i_loop2);
                    p_ij_row_loop2[n] = __float2half(p_val_float);
                }
                p_sum_numerator_loop2 += __half2float(p_ij_row_loop2[n]);
            }
            
            // 4. Calculate alpha
            float alpha_loop2 = exp2f(m_i - new_m_i_loop2);

            // 5. Rescale existing acc
            for (int d_acc = 0; d_acc < D_head; ++d_acc) {
                acc[d_acc] *= alpha_loop2;
            }

            // 6. Accumulate P.V
            if (current_row_max_qk != -INFINITY) { // Only if row not fully masked
                for (int d_acc = 0; d_acc < D_head; ++d_acc) { 
                    float pv_sum_d = 0.0f;
                    for (int n = 0; n < KERNEL_BLOCK_N; ++n) { 
                        pv_sum_d += __half2float(p_ij_row_loop2[n]) * __half2float(v_tile[n][d_acc]);
                    }
                    acc[d_acc] += pv_sum_d;
                }
            }
            
            // 7. Update l_i and m_i
            l_i = l_i * alpha_loop2 + p_sum_numerator_loop2;
            m_i = new_m_i_loop2;
        } // end if (m < KERNEL_BLOCK_M)
        
        __syncthreads(); // Sync before next K/V iteration
    }

    // --- Finalize output (Step 3e) ---
    // m = threadIdx.x (local Q row index)
    // m_i, l_i, acc are per-thread registers.

    if (m < KERNEL_BLOCK_M) { // Ensure thread is active for a Q row it was responsible for
        // Boundary check: only compute and write if the Q row is within actual N_q bounds
        if (q_start_row_idx + m < N_q) {

            // 1. Finalize LSE value for this Q row
            float final_lse_val;
            if (l_i <= 0.0f || l_i != l_i) { // Check for zero, negative, or NaN l_i
                final_lse_val = -INFINITY; 
            } else {
                final_lse_val = m_i + log2f(l_i);
            }
            
            // Store softmax_lse for this Q row
            softmax_lse_ptr[bh_idx * N_q + (q_start_row_idx + m)] = final_lse_val;

            // 2. Finalize Output O values for this Q row
            float inv_l_i = (l_i == 0.0f || l_i != l_i) ? 0.0f : 1.0f / l_i; 

            __half* o_row_global_ptr = o_ptr + (bh_idx * N_q + (q_start_row_idx + m)) * D_head;
            for (int d = 0; d < D_head; ++d) {
                o_row_global_ptr[d] = __float2half(acc[d] * inv_l_i);
            }
        }
    }
    // No __syncthreads() needed here at the very end of the kernel.
}

// Placeholder for the actual CUDA kernel for the backward pass
// This kernel would perform the core Flash Attention backward computation.
__global__ void flash_bwd_kernel(
    const __half* __restrict__ dout_ptr,         // [B, H, N_q, D_head]
    const __half* __restrict__ q_ptr,            // [B, H, N_q, D_head]
    const __half* __restrict__ k_prescaled_ptr,  // [B, H, N_kv, D_head], pre-scaled K
    const __half* __restrict__ v_ptr,            // [B, H, N_kv, D_head]
    const __half* __restrict__ o_ptr,            // [B, H, N_q, D_head]
    const float* __restrict__ softmax_lse_ptr,  // [B, H, N_q] (M)
    const float* __restrict__ delta_ptr,        // [B, H, N_q] (D)
    __half* __restrict__ dq_ptr,                 // Output dQ [B, H, N_q, D_head]
    __half* __restrict__ dk_ptr,                 // Output dK [B, H, N_kv, D_head]
    __half* __restrict__ dv_ptr,                 // Output dV [B, H, N_kv, D_head]
    const int B,
    const int H,
    const int N_q,  // Sequence length for Q, O, dO, dQ, LSE, Delta
    const int N_kv, // Sequence length for K, V, dK, dV
    const int D_head,
    const float sm_scale, // Original sm_scale, not the 1/ln(2) version
    const bool is_causal  // Note: Triton _attn_bwd MASK is True/False based on section
) {
    // Empty body
}

// The above problematic duplicate flash_bwd_preprocess_kernel definition was targeted for removal.
// If this search block is found and replaced with nothing, it means the duplicate is gone.
// If not found, it implies it was already removed or the file state is different.

__global__ void flash_bwd_preprocess_kernel(
    const __half* __restrict__ o_ptr,    // Input O [B, H, N_CTX, D_head]
    const __half* __restrict__ do_ptr,   // Input dO [B, H, N_CTX, D_head]
    float* __restrict__ delta_ptr,       // Output Delta [B, H, N_CTX]
    const int B,
    const int H,
    const int N_CTX,
    const int D_head
) {
    // Empty body
}

torch::Tensor flash_bwd_preprocess_cuda(
    torch::Tensor o,    // Output tensor from forward pass [B, H, N_CTX, D_head]
    torch::Tensor dout  // Gradient dO [B, H, N_CTX, D_head]
) {
    // Minimal body for flash_bwd_preprocess_cuda
    const int B = o.size(0);
    const int H = o.size(1);
    const int N_CTX = o.size(2);
    // const int D_head = o.size(3); // Not needed for delta shape

    auto delta_options = o.options().dtype(torch::kFloat32);
    torch::Tensor delta_dummy = torch::empty({B, H, N_CTX}, delta_options);
    return delta_dummy;
}

std::vector<torch::Tensor> flash_attn_forward_cuda(
    torch::Tensor q,         // [B, H, N_q, D_head]
    torch::Tensor k,         // [B, H, N_kv, D_head]
    torch::Tensor v,         // [B, H, N_kv, D_head]
    float sm_scale,
    bool causal
) {
    // Minimal body for flash_attn_forward_cuda
    const int B = q.size(0);
    const int H = q.size(1);
    const int N_q = q.size(2);
    const int D_head = q.size(3); 

    torch::Tensor o_dummy = torch::empty({B, H, N_q, D_head}, q.options());
    torch::Tensor softmax_lse_dummy = torch::empty({B, H, N_q}, q.options().dtype(torch::kFloat32));
    return {o_dummy, softmax_lse_dummy};
}

std::vector<torch::Tensor> flash_attn_backward_cuda(
    torch::Tensor dout,        // [B, H, N_q, D_head] (gradient of o)
    torch::Tensor q,           // [B, H, N_q, D_head] (saved from forward)
    torch::Tensor k,           // [B, H, N_kv, D_head] (saved from forward)
    torch::Tensor v,           // [B, H, N_kv, D_head] (saved from forward)
    torch::Tensor o,           // [B, H, N_q, D_head] (saved from forward)
    torch::Tensor softmax_lse, // M
    torch::Tensor delta,       // D (New)
    float sm_scale,
    bool causal
) {
    // Minimal body for flash_attn_backward_cuda
    torch::Tensor dq_dummy = torch::empty_like(q);
    torch::Tensor dk_dummy = torch::empty_like(k);
    torch::Tensor dv_dummy = torch::empty_like(v);
    return {dq_dummy, dk_dummy, dv_dummy};
}

[end of flash_attention_kernels.cu]
    TORCH_CHECK(q.device().is_cuda(), "Input Q must be a CUDA tensor");
    TORCH_CHECK(k.device().is_cuda(), "Input K must be a CUDA tensor");
    TORCH_CHECK(v.device().is_cuda(), "Input V must be a CUDA tensor");
    TORCH_CHECK(o.device().is_cuda(), "Input O must be a CUDA tensor");
    TORCH_CHECK(softmax_lse.device().is_cuda(), "Input softmax_lse must be a CUDA tensor");
    TORCH_CHECK(delta.device().is_cuda(), "Input delta must be a CUDA tensor");

    TORCH_CHECK(dout.is_contiguous(), "Input dout must be contiguous");
    TORCH_CHECK(q.is_contiguous(), "Input Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input V must be contiguous");
    TORCH_CHECK(o.is_contiguous(), "Input O must be contiguous");
    TORCH_CHECK(softmax_lse.is_contiguous(), "Input softmax_lse must be contiguous");
    TORCH_CHECK(delta.is_contiguous(), "Input delta must be contiguous");

    TORCH_CHECK(dout.scalar_type() == torch::kFloat16, "Input dout must be FP16");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "Input Q must be FP16");
    TORCH_CHECK(k.scalar_type() == torch::kFloat16, "Input K must be FP16");
    TORCH_CHECK(v.scalar_type() == torch::kFloat16, "Input V must be FP16");
    TORCH_CHECK(o.scalar_type() == torch::kFloat16, "Input O must be FP16");
    TORCH_CHECK(softmax_lse.scalar_type() == torch::kFloat32, "Input softmax_lse must be FP32");
    TORCH_CHECK(delta.scalar_type() == torch::kFloat32, "Input delta must be FP32");
    // Add size checks for delta: [B, H, N_q]
    TORCH_CHECK(delta.size(0) == q.size(0) && delta.size(1) == q.size(1) && delta.size(2) == q.size(2),
                "Delta tensor dimensions must match [B, H, N_q]");


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
    // These would need to be defined based on KERNEL_BLOCK_M_BWD, KERNEL_BLOCK_N_BWD if we had them
    dim3 threads(32); // Small placeholder
    dim3 blocks(1, H, B); // Minimal placeholder for now

    // Get data pointers
    const __half* dout_ptr_in = reinterpret_cast<const __half*>(dout.data_ptr());
    const __half* q_ptr_in = reinterpret_cast<const __half*>(q.data_ptr());
    const __half* k_ptr_in = reinterpret_cast<const __half*>(k.data_ptr());
    const __half* v_ptr_in = reinterpret_cast<const __half*>(v.data_ptr());
    const __half* o_ptr_in = reinterpret_cast<const __half*>(o.data_ptr());
    const float* softmax_lse_ptr_in = softmax_lse.data_ptr<float>();
    const float* delta_ptr_in = delta.data_ptr<float>(); // New pointer

    __half* dq_ptr_out = reinterpret_cast<__half*>(dq.data_ptr());
    __half* dk_ptr_out = reinterpret_cast<__half*>(dk.data_ptr());
    __half* dv_ptr_out = reinterpret_cast<__half*>(dv.data_ptr());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Update kernel launch call to include delta_ptr
    flash_bwd_kernel<<<blocks, threads, 0, stream>>>(
        dout_ptr_in, q_ptr_in, k_ptr_in, v_ptr_in, o_ptr_in, softmax_lse_ptr_in,
        delta_ptr_in, // Pass new delta_ptr
        dq_ptr_out, dk_ptr_out, dv_ptr_out,
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
