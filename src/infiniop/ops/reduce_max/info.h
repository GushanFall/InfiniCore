#ifndef __REDUCE_MAX_INFO_H__
#define __REDUCE_MAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
#include <numeric>

namespace op::reduce_max {

class ReduceMaxInfo {
    ReduceMaxInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> input_strides;
    ptrdiff_t output_stride_outer;
    ptrdiff_t input_stride_outer;
    ptrdiff_t input_stride_dim;
    size_t input_shape_dim;
    size_t outer_size;
    size_t inner_size;

    static utils::Result<ReduceMaxInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int dim) {
        
        auto dtype = output_desc->dtype();
        CHECK_OR_RETURN(dtype == input_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        size_t ndim = output_desc->ndim();
        CHECK_OR_RETURN(ndim == input_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(dim >= 0 && dim < int(ndim), INFINI_STATUS_BAD_PARAM);
        
        auto input_shape = input_desc->shape();
        size_t outer_size = std::accumulate(input_shape.begin(), input_shape.begin() + dim, 1, std::multiplies<size_t>());
        size_t inner_size = std::accumulate(input_shape.begin() + dim + 1, input_shape.end(), 1, std::multiplies<size_t>());

        auto output_strides = output_desc->strides();
        ptrdiff_t output_stride_outer = (dim == 0) ? 1 : output_strides[dim - 1];

        auto input_strides = input_desc->strides();
        ptrdiff_t input_stride_outer = (dim == 0) ? 1 : input_strides[dim - 1];
        ptrdiff_t input_stride_dim = input_strides[dim];

        return utils::Result<ReduceMaxInfo>{ReduceMaxInfo{
            dtype,
            output_desc->strides(),
            input_desc->strides(),
            output_stride_outer,
            input_stride_outer,
            input_stride_dim,
            input_shape[dim],
            outer_size,
            inner_size
        }};
    }
};

} // namespace op::reduce_max

#endif // __REDUCE_MAX_INFO_H__
