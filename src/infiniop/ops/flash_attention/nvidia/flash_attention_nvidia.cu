#include "../../../devices/cuda/cuda_common.cuh"
#include "flash_attention_cuda.cuh"
#include "flash_attention_kernel.cuh"

#include <cuda_bf16.h>

namespace op::flash_attention::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t out_desc,
                                  infiniopTensorDescriptor_t q_desc,
                                  infiniopTensorDescriptor_t k_desc,
                                  infiniopTensorDescriptor_t v_desc,
                                  infiniopTensorDescriptor_t mask_desc,
                                  infiniopTensorDescriptor_t mask_type) {

    auto handle = reinterpret_cast<device::cuda::Handle *>(handle_);

    auto info = FlashAttentionInfo::create(out_desc, q_desc, k_desc, v_desc, mask_desc, mask_type);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::cuda::Handle *>(handle)->internal()},
        info.take(),
        0,
        handle->device,
        handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tmask>
infiniStatus_t calculateFlashAttention(const FlashAttentionInfo &info,
                                       Tdata *out,
                                       const Tdata *q, const Tdata *k, const Tdata *v,
                                       Tmask *mask,
                                       cudaStream_t stream) {
    
    // TODO: determine B_c, B_r automatically
    size_t B_r = 32;
    size_t B_c = 32;

    size_t batch_size = info.batch_size;
    size_t seq_len_q = info.seq_len_q;
    size_t seq_len_kv = info.seq_len_kv;
    size_t nums_head_q = info.num_heads_q;
    size_t nums_head_kv = info.num_heads_kv;
    size_t group = nums_head_q / nums_head_kv;
    size_t head_dim = info.head_dim;

    size_t T_r = ceil(float(seq_len_q) / B_r);
    size_t T_c = ceil(float(seq_len_kv) / B_c);
    Tdata softmax_scale = 1.0 / sqrt(head_dim);

    dim3 grid_dim(batch_size, nums_head_q);
    dim3 block_dim(B_c);

    flashAttentionForward<Tdata, Tmask><<<grid_dim, block_dim, 0, stream>>>(
        out, q, k, v, mask,
        seq_len_q, seq_len_kv, head_dim,
        B_c, B_r, T_c, T_r,
        softmax_scale,
        info.qo_stride_b, info.qo_stride_s, info.qo_stride_n,
        info.kv_stride_b, info.kv_stride_s, info.kv_stride_n);
    
    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_FLASH_ATTENTION(Tdata, Tmask)           \
    calculateFlashAttention<Tdata, Tmask>(                \
        _info,                                            \
        (Tdata *)out,                                     \
        (const Tdata *)q,                                 \
        (const Tdata *)k,                                 \
        (const Tdata *)v,                                 \
        (Tmask *)mask,                                    \
        (cudaStream_t)stream)

#define MASK_TYPE(Tdata)                                  \
    switch (_info.mask_dtype) {                           \
    case INFINI_DTYPE_U8:                                 \
        return CALCULATE_FLASH_ATTENTION(Tdata, uint8_t); \
    case INFINI_DTYPE_BOOL:                               \
        return CALCULATE_FLASH_ATTENTION(Tdata, bool);    \
    default:                                              \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;            \
    }

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *out,
                                     const void *q, const void *k, const void *v, 
                                     void *mask,
                                     void *stream) const {

    switch (_info.dtype) {
    {
    case INFINI_DTYPE_F16:
        MASK_TYPE(half);
    case INFINI_DTYPE_F32:
        MASK_TYPE(float);
    case INFINI_DTYPE_BF16:
        MASK_TYPE(__nv_bfloat16);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
    }

}
} // namespace op::flash_attention::cuda