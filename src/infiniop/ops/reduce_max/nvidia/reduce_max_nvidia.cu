#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "reduce_max_nvidia.cuh"
#include "../cuda/kernel.cuh"

template <typename Tdata>
INFINIOP_CUDA_KERNEL reduceMax(
    Tdata *output,
    ptrdiff_t output_stride_outer,
    const Tdata *input,
    size_t input_shape_dim,
    ptrdiff_t input_stride_outer,
    ptrdiff_t input_stride_dim) {
    printf("reduceMax input[0]: %f\n", input[0]); // Debugging output
    reduceMaxKernel<Tdata>(
        output, output_stride_outer,
        input, input_shape_dim, 
        input_stride_outer, input_stride_dim);
}

namespace op::reduce_max::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int dim) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info = ReduceMaxInfo::create(output_desc, input_desc, dim);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    infiniDtype_t dtype,
    void *output, ptrdiff_t output_stride_outer,
    const void *input, size_t input_shape_dim,
    ptrdiff_t input_stride_outer, ptrdiff_t input_stride_dim,
    size_t outer_size, size_t inner_size,
    cudaStream_t stream) {
    printf("launchKernel input[0]: %f\n", ((const float *)input)[0]); // Debugging output

    // dim3 block = dim3(BLOCK_SIZE);
    dim3 block = dim3(1);
    dim3 grid = dim3(
        (inner_size + block.x - 1) / block.x,
        outer_size
    );

    if (dtype == INFINI_DTYPE_F16) {
        reduceMax<half>
            <<<grid, block, 0, stream>>>((half *)output,
                                         output_stride_outer,
                                         (const half *)input,
                                         input_shape_dim,
                                         input_stride_outer, input_stride_dim);
    } else if (dtype == INFINI_DTYPE_BF16) {
        reduceMax<__nv_bfloat16>
            <<<grid, block, 0, stream>>>((__nv_bfloat16 *)output,
                                         output_stride_outer,
                                         (const __nv_bfloat16 *)input,
                                         input_shape_dim,
                                         input_stride_outer, input_stride_dim);
    } else if (dtype == INFINI_DTYPE_F32) {
        reduceMax<float>
            <<<grid, block, 0, stream>>>((float *)output,
                                         output_stride_outer,
                                         (const float *)input,
                                         input_shape_dim,
                                         input_stride_outer, input_stride_dim);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    printf("calculate input[0]: %f\n", ((const float *)input)[0]); // Debugging output

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            _info.dtype, 
            output, _info.output_stride_outer,
            input, _info.input_shape_dim, 
            _info.input_stride_outer, _info.input_stride_dim,
            _info.outer_size, _info.inner_size, 
            cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            _info.dtype, 
            output, _info.output_stride_outer,
            input, _info.input_shape_dim, 
            _info.input_stride_outer, _info.input_stride_dim,
            _info.outer_size, _info.inner_size, 
            cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            _info.dtype, 
            output, _info.output_stride_outer,
            input, _info.input_shape_dim, 
            _info.input_stride_outer, _info.input_stride_dim,
            _info.outer_size, _info.inner_size,
            cuda_stream));
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::reduce_max::nvidia
