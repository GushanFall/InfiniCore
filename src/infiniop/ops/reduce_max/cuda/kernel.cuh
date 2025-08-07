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

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("=== Kernel Debug Info ===\n");
        printf("Input ptr: %p\n", input);
        printf("Grid: (%d, %d), Block: %d\n", 
               gridDim.x, gridDim.y, blockDim.x);
        printf("First 4 elements: ");
        for (int i = 0; i < 4 && i < input_shape_dim; ++i) {
            printf("%f ", static_cast<float>(input[i]));
        }
        printf("\n");
    }

    Tdata max_val = -INFINITY;

    for (size_t i = 0; i < input_shape_dim; ++i) {
        ptrdiff_t input_offset = outer_idx * input_stride_outer + i * input_stride_dim + inner_idx;
        // printf("Processing input offset: %ld\n", input_offset); // Debugging output
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
