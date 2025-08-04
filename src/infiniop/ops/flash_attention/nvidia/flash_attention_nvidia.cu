#include "../../../devices/nvidia/nvidia_common.cuh"
#include "flash_attention_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"

template <typename Tdata>
INFINIOP_CUDA_KERNEL flashAttentionKernel(
    Tdata *__restrict__ out_,
    const Tdata *q_,
    const Tdata *k_,
    const Tdata *v_,
    const float *mask_,
    const size_t seq_len_q, const size_t seq_len_kv, const size_t head_dim,
    const size_t B_c, const size_t B_r, const size_t T_c, const size_t T_r,
    const Tdata softmax_scale,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n
) {
    flashAttentionBlock(
        out_,
        q_, k_, v_, mask_,
        seq_len_q, seq_len_kv, head_dim,
        B_c, B_r, T_c, T_r,
        softmax_scale,
        qo_stride_b, qo_stride_s, qo_stride_n,
        kv_stride_b, kv_stride_s, kv_stride_n
    );
}


namespace op::flash_attention::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopAttentionMaskType_t mask_type) {
    printf("create: begin\n");

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info = FlashAttentionInfo::create(out_desc, q_desc, k_desc, v_desc, mask_desc, mask_type);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);
    
    printf("create: end\n");
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateFlashAttention(const FlashAttentionInfo &info,
                                       Tdata *out,
                                       const Tdata *q, const Tdata *k, const Tdata *v,
                                       float *mask,
                                       cudaStream_t stream) {
    printf("calculateFlashAttention: begin\n");

    // TODO: determine B_c, B_r automatically
    // size_t B_r = 32;
    // size_t B_c = 32;
    size_t B_r = 2;
    size_t B_c = 2;

    size_t batch_size = info.batch_size;
    size_t seq_len_q = info.seq_len_q;
    size_t seq_len_kv = info.seq_len_kv;
    size_t nums_head_q = info.num_heads_q;
    size_t nums_head_kv = info.num_heads_kv;
    size_t group = nums_head_q / nums_head_kv;
    size_t head_dim = info.head_dim;
    float *mask_input = info.mask != nullptr ? info.mask : mask;

    size_t T_r = ceil(float(seq_len_q) / B_r);
    size_t T_c = ceil(float(seq_len_kv) / B_c);
    Tdata softmax_scale = 1.0 / sqrt(head_dim);

    dim3 grid_dim(batch_size, nums_head_q);
    dim3 block_dim(B_c);

    printf("k[0]: %f\n", k[0]);

    flashAttentionKernel<Tdata><<<grid_dim, block_dim, 0, stream>>>(
        out, q, k, v, mask_input,
        seq_len_q, seq_len_kv, head_dim,
        B_c, B_r, T_c, T_r,
        softmax_scale,
        info.qo_stride_b, info.qo_stride_s, info.qo_stride_n,
        info.kv_stride_b, info.kv_stride_s, info.kv_stride_n);

    printf("calculateFlashAttention: end\n");

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_FLASH_ATTENTION(Tdata)                  \
    calculateFlashAttention<Tdata>(                       \
        _info,                                            \
        (Tdata *)out,                                     \
        (const Tdata *)q,                                 \
        (const Tdata *)k,                                 \
        (const Tdata *)v,                                 \
        (float *)mask,                                    \
        (cudaStream_t)stream)

infiniStatus_t Descriptor::calculate(
    void *workspace, 
    size_t workspace_size,
    void *out,
    const void *q, const void *k, const void *v, 
    const void *mask,
    void *stream) const {

    switch (_info.dtype) {
    // case INFINI_DTYPE_F16:
    //     CALCULATE_FLASH_ATTENTION(half);
    case INFINI_DTYPE_F32:
        CALCULATE_FLASH_ATTENTION(float);
    // case INFINI_DTYPE_BF16:
    //     CALCULATE_FLASH_ATTENTION(cuda_bfloat16);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

#undef CALCULATE_FLASH_ATTENTION

} // namespace op::flash_attention::nvidia