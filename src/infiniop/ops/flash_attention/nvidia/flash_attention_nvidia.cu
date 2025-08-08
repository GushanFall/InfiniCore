#include "../../../devices/nvidia/nvidia_common.cuh"
#include "flash_attention_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"

template <typename Tdata>
INFINIOP_CUDA_KERNEL flashAttentionKernel(
    Tdata *__restrict__ out_,
    const Tdata *__restrict__ q_,
    const Tdata *__restrict__ k_,
    const Tdata *__restrict__ v_,
    const float *mask_,
    const size_t seq_len_q, const size_t seq_len_kv, const size_t head_dim,
    const size_t B_c, const size_t B_r, const size_t T_c, const size_t T_r,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n
) {
    Tdata softmax_scale = 1.0 / sqrt(head_dim);
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

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info = FlashAttentionInfo::create(out_desc, q_desc, k_desc, v_desc, mask_desc, mask_type);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    void *out,
    const void *q, const void *k, const void *v,
    const void *mask,
    size_t batch_size,
    size_t nums_head_q, size_t nums_head_kv,
    size_t seq_len_q, size_t seq_len_kv, 
    size_t head_dim,
    size_t B_c, size_t B_r, size_t T_c, size_t T_r,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    infiniDtype_t dtype,
    cudaStream_t stream) {

    dim3 grid_dim(batch_size, nums_head_q);
    dim3 block_dim(B_c);

#define LAUNCHI_KERNEL(Tdata)                                        \
    flashAttentionKernel<Tdata><<<grid_dim, block_dim, 0, stream>>>( \
        reinterpret_cast<Tdata *>(out),                              \
        reinterpret_cast<const Tdata *>(q),                          \
        reinterpret_cast<const Tdata *>(k),                          \
        reinterpret_cast<const Tdata *>(v),                          \
        reinterpret_cast<const float *>(mask),                       \
        seq_len_q, seq_len_kv, head_dim,                             \
        B_c, B_r, T_c, T_r,                                          \
        qo_stride_b, qo_stride_s, qo_stride_n,                       \
        kv_stride_b, kv_stride_s, kv_stride_n)

    if (dtype == INFINI_DTYPE_F16) {
        LAUNCHI_KERNEL(half);
    } else if (dtype == INFINI_DTYPE_F32) {
        LAUNCHI_KERNEL(float);
    } else if (dtype == INFINI_DTYPE_BF16) {
        LAUNCHI_KERNEL(__nv_bfloat16);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, 
    size_t workspace_size,
    void *out,
    const void *q, const void *k, const void *v, 
    const void *mask,
    void *stream) const {

    // TODO: determine B_c, B_r automatically
    // size_t B_r = 32;
    // size_t B_c = 32;
    size_t B_r = 2;
    size_t B_c = 2;

    size_t batch_size = _info.batch_size;
    size_t seq_len_q = _info.seq_len_q;
    size_t seq_len_kv = _info.seq_len_kv;
    size_t nums_head_q = _info.num_heads_q;
    size_t nums_head_kv = _info.num_heads_kv;
    // size_t group = nums_head_q / nums_head_kv;
    size_t head_dim = _info.head_dim;
    const void *mask_input = nullptr;
    if (_info.is_masked) {
        mask_input = _info.mask != nullptr ? _info.mask : mask;
    }

    size_t T_r = ceil(float(seq_len_q) / B_r);
    size_t T_c = ceil(float(seq_len_kv) / B_c);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    CHECK_STATUS(launchKernel(
        out, q, k, v, mask_input,
        batch_size, nums_head_q, nums_head_kv,
        seq_len_q, seq_len_kv,
        head_dim,
        B_c, B_r, T_c, T_r,
        _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
        _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
        _info.dtype,
        cuda_stream));

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::flash_attention::nvidia