#ifndef __FLASH_ATTENTION_KERNEL_CUH__
#define __FLASH_ATTENTION_KERNEL_CUH__

#include <cuda_runtime.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../devices/cuda/cuda_kernel_common.cuh"

template <typename Tdata, typename Tmask>
INFINIOP_CUDA_KERNEL flashAttentionForward(
    Tdata *out_,
    const Tdata *q_, const Tdata *k_, const Tdata *v_, const Tmask *mask_,
    const size_t seq_len_q, const size_t seq_len_kv, const size_t head_dim,
    const size_t B_c, const size_t B_r, const size_t T_c, const size_t T_r, 
    const Tdata softmax_scale,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n) {
        size_t bx = blockIdx.x;  // batch_size
        size_t by = blockIdx.y;  // nums_head_q
        size_t tx = threadIdx.x; // B_c

        size_t offset;

        extern __shared__ __align__(sizeof(Tdata)) char shared_mem[];
        Tdata* q_i = reinterpret_cast<Tdata*>(shared_mem);
        Tdata* k_j = reinterpret_cast<Tdata*>(q_i + B_r * head_dim);
        Tdata* v_j = reinterpret_cast<Tdata*>(k_j + B_c * head_dim);
        Tdata* s_i = reinterpret_cast<Tdata*>(v_j + B_c * head_dim);
        
        Tdata m_i = -INFINITY;
        Tdata l_i;

        for (size_t j = 0; j < T_c; j++) {
            // load K_j, V_j (暂时只考虑了 seq_len_kv == seq_len_q)
            for (int x = 0; x < head_dim; x++) {
                offset = bx * kv_stride_b + tx * kv_stride_s + by * kv_stride_n + x;
                k_j[tx * head_dim + x] = k_[offset];
                v_j[tx * head_dim + x] = v_[offset];
            }
            __syncthreads();

            for (size_t i = 0; i < T_r; i++) {
                // load Q_i
                for (int x = 0; x < head_dim; x++) {
                    offset = bx * qo_stride_b + tx * qo_stride_s + by * qo_stride_n + x;
                    q_i[tx * head_dim + x] = q_[offset];
                }
                __syncthreads();

                // S = Q @ K^T
                Tdata m_ij = -INFINITY;
                for (int y = 0; y < B_r; y++) {
                    Tdata s_ij = 0;
                    for (int x = 0; x < head_dim; x++) {
                        s_ij += q_i[tx * head_dim + x] * k_j[y * head_dim + x];
                    }
                    s_ij *= softmax_scale;
                    s_i[tx * B_r + y] = s_ij;

                    // max of s_i
                    if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, __nv_bfloat16>) {
                        m_ij = __hmax(m_ij, s_ij);
                    } else {
                        m_ij = max(m_ij, s_ij);
                    }
                }
                __syncthreads();

                // TODO: apply mask
                // Tmask mask_j = mask_[mask_offset + j * seq_len];
                // __syncthreads();

                // p = exp(s - m), l = sum(p)
                Tdata l_ij = 0;
                for (int x = 0; x < B_c; x++) {
                    s_i[tx * B_c + x] = __expf(s_i[tx * B_c + x] - m_ij);
                    l_ij += s_i[tx * B_c + x];
                }
                __syncthreads();

                // compute new m and l
                if constexpr (std::is_same_v<Tdata, half>) {
                    Tdata m_i_new = __hmax(m_i, m_ij);
                    Tdata l_i_new = hexp(m_i - m_i_new) * l_i + hexp(m_ij - m_i_new) * l_ij;
                    // compute new out
                    for (int y = 0; y < head_dim; y++) {
                        Tdata p_v = 0; // p_ij @ v_j
                        for (int x = 0; x < B_c; x++) {
                            p_v += s_i[tx * B_c + x] * v_j[tx * head_dim + x];
                        }
                        offset = bx * qo_stride_b + tx * qo_stride_s + by * qo_stride_n + y;
                        out_[offset] = __hmul(__hdiv(__float2half(1.0f), l_i_new), (l_i * hexp(m_i - m_i_new) * out_[offset] + hexp(m_ij - m_i_new) * p_v));
                    }
                    m_i = m_i_new;
                    l_i = l_i_new;
                } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                    Tdata m_i_new = __hmax(m_i, m_ij);
                    Tdata l_i_new = hexp(m_i - m_i_new) * l_i + hexp(m_ij - m_i_new) * l_ij;
                    // compute new out
                    for (int y = 0; y < head_dim; y++) {
                        Tdata p_v = 0; // p_ij @ v_j
                        for (int x = 0; x < B_c; x++) {
                            p_v += s_i[tx * B_c + x] * v_j[tx * head_dim + x];
                        }
                        offset = bx * qo_stride_b + tx * qo_stride_s + by * qo_stride_n + y;
                        out_[offset] = __hmul(__hdiv(__float2bfloat16(1.0f), l_i_new), (l_i * hexp(m_i - m_i_new) * out_[offset] + hexp(m_ij - m_i_new) * p_v));
                    }
                    m_i = m_i_new;
                    l_i = l_i_new;
                } else {
                    Tdata m_i_new = max(m_i, m_ij);
                    Tdata l_i_new = __expf(m_i - m_i_new) * l_i + __expf(m_ij - m_i_new) * l_ij;
                    // compute new out
                    for (int y = 0; y < head_dim; y++) {
                        Tdata p_v = 0; // p_ij @ v_j
                        for (int x = 0; x < B_c; x++) {
                            p_v += s_i[tx * B_c + x] * v_j[tx * head_dim + x];
                        }
                        offset = bx * qo_stride_b + tx * qo_stride_s + by * qo_stride_n + y;
                        out_[offset] = (1 / l_i_new) * (l_i * __expf(m_i - m_i_new) * out_[offset] + __expf(m_ij - m_i_new) * p_v);
                    }
                    m_i = m_i_new;
                    l_i = l_i_new;
                }
                __syncthreads();
            }
            __syncthreads();
        }
    }

#endif // __FLASH_ATTENTION_KERNEL_CUH__