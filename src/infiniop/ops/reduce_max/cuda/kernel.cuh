#ifndef __REDUCE_MAX_CUDA_KERNEL_H__
#define __REDUCE_MAX_CUDA_KERNEL_H__

template <typename Tdata>
__device__ void reduceMaxKernel(
    Tdata *output,
    ptrdiff_t output_stride_outer,
    const Tdata *input,
    size_t input_shape_dim,
    ptrdiff_t input_stride_outer,
    ptrdiff_t input_stride_dim) {

    size_t outer_idx = blockIdx.y;
    size_t inner_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Tdata max_val = -INFINITY;

    for (size_t i = 0; i < input_shape_dim; ++i) {
        ptrdiff_t input_offset = outer_idx * input_stride_outer + i * input_stride_dim + inner_idx;
        Tdata val = input[input_offset];
        if (val > max_val) {
            max_val = val;
        }
    }
    __syncthreads();

    ptrdiff_t output_offset = outer_idx * output_stride_outer + inner_idx;
    output[output_offset] = max_val;

}

#endif // __REDUCE_MAX_CUDA_KERNEL_H__
