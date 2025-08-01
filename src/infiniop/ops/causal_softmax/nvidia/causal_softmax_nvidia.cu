#include "../../../devices/nvidia/nvidia_common.cuh"
#include "causal_softmax_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL causalSoftmax(
    Tdata *y, const Tdata *x,
    size_t batch, size_t height, size_t width,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_h,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_h) {
    causalSoftmaxKernel<BLOCK_SIZE, Tdata, Tcompute>(y, x, batch, height, width, y_stride_b, y_stride_h, x_stride_b, x_stride_h);
}

namespace op::causal_softmax::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto info = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t dtype,
                            size_t batch_size, size_t seq_len, size_t total_seq_len,
                            ptrdiff_t y_stride_b, ptrdiff_t y_stride_i,
                            ptrdiff_t x_stride_b, ptrdiff_t x_stride_i,
                            cudaStream_t stream) {
    dim3 grid(uint32_t(seq_len), uint32_t(batch_size), 1);
    if (dtype == INFINI_DTYPE_F16) {
        causalSoftmax<BLOCK_SIZE, half, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((half *)y, (const half *)x,
                                              batch_size, seq_len, total_seq_len,
                                              y_stride_b, y_stride_i,
                                              x_stride_b, x_stride_i);
    } else if (dtype == INFINI_DTYPE_BF16) {
        causalSoftmax<BLOCK_SIZE, __nv_bfloat16, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16 *)y, (const __nv_bfloat16 *)x,
                                              batch_size, seq_len, total_seq_len,
                                              y_stride_b, y_stride_i,
                                              x_stride_b, x_stride_i);
    } else if (dtype == INFINI_DTYPE_F32) {
        causalSoftmax<BLOCK_SIZE, float, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const float *)x,
                                              batch_size, seq_len, total_seq_len,
                                              y_stride_b, y_stride_i,
                                              x_stride_b, x_stride_i);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            y, x, _info.dtype, _info.batch_size, _info.seq_len, _info.total_seq_len,
            _info.y_stride_b, _info.y_stride_i, _info.x_stride_b, _info.x_stride_i, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            y, x, _info.dtype, _info.batch_size, _info.seq_len, _info.total_seq_len,
            _info.y_stride_b, _info.y_stride_i, _info.x_stride_b, _info.x_stride_i, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            y, x, _info.dtype, _info.batch_size, _info.seq_len, _info.total_seq_len,
            _info.y_stride_b, _info.y_stride_i, _info.x_stride_b, _info.x_stride_i, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::nvidia
