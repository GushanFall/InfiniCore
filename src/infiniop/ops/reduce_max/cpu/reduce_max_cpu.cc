#include "reduce_max_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

#include <limits>

namespace op::reduce_max::cpu {
    
Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int dim) {
    auto result = ReduceMaxInfo::create(output_desc, input_desc, dim);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t reduceMax(
    const ReduceMaxInfo *info,
    Tdata *output,
    const Tdata *input) {

    const auto input_shape_dim = info->input_shape_dim;
    const ptrdiff_t input_stride_outer = info->input_stride_outer;
    const ptrdiff_t input_stride_dim = info->input_stride_dim;
    const ptrdiff_t output_stride_outer = info->output_stride_outer;

    const size_t outer_size = info->outer_size;
    const size_t inner_size = info->inner_size;

#pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        for (size_t k = 0; k < inner_size; k++) {
            Tdata max_val = std::numeric_limits<Tdata>::lowest();

            for (size_t j = 0; j < input_shape_dim; j++) {
                size_t idx = i * input_stride_outer + j * input_stride_dim + k;
                const Tdata current = input[idx];
                bool is_greater;
                if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
                    float current_val = utils::cast<float>(current);
                    float max_val_val = utils::cast<float>(max_val);
                    is_greater = current_val > max_val_val;
                } else {
                    is_greater = current > max_val;
                    // printf("idx: %zu, i: %zu, j: %zu, k: %zu, current: %f, max_val: %f\n", idx, i, j, k, current, max_val);
                }

                if (is_greater) max_val = current;
            }
            output[i * output_stride_outer + k] = max_val;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    if (_info.dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(reduceMax(&_info, (fp16_t *)output, (const fp16_t *)input));
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(reduceMax(&_info, (bf16_t *)output, (const bf16_t *)input));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(reduceMax(&_info, (float *)output, (const float *)input));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::reduce_max::cpu
