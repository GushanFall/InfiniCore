#ifndef __FLASH_ATTENTION_INFO_H__
#define __FLASH_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::flash_attention {

class FlashAttentionInfo {
private:
    FlashAttentionInfo() = default;

public:
    infiniDtype_t dtype, mask_dtype;
    size_t batch_size;
    size_t seq_len_q, seq_len_kv;
    size_t num_heads_q, num_heads_kv;
    size_t head_dim;

    ptrdiff_t qo_stride_b;
    ptrdiff_t qo_stride_s;
    ptrdiff_t qo_stride_n;
    ptrdiff_t qo_stride_d;

    ptrdiff_t kv_stride_b;
    ptrdiff_t kv_stride_s;
    ptrdiff_t kv_stride_n;
    ptrdiff_t kv_stride_d;

    // ptrdiff_t mask_stride_b;
    // ptrdiff_t mask_stride_s;

    static utils::Result<FlashAttentionInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t mask_desc,
        infiniopTensorDescriptor_t mask_type) {
        // 检查数据类型
        auto dtype = out_desc->dtype();
        if (dtype != q_desc->dtype() || dtype != k_desc->dtype() || dtype != v_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

        auto mask_dtype = mask_desc->dtype();
        CHECK_DTYPE(mask_dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_U8);

        // 检查 qkvo 张量形状
        // q 和 out 的形状必须相同
        auto q_shape = out_desc->shape();
        CHECK_SAME_SHAPE(q_shape, q_desc->shape());
        // k 和 v 的形状必须相同
        auto kv_shape = k_desc->shape();
        CHECK_SAME_SHAPE(kv_shape, v_desc->shape());
        
        // 检查输入的维度
        auto ndim = q_desc->ndim();
        if (ndim != k_desc->ndim()) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        if (ndim != 3 && ndim != 4) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        // auto mask_ndim = mask_desc->ndim();
        // CHECK_OR_RETURN(mask_ndim == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t batch_size_q = 1;
        size_t seq_len_q = q_shape[ndim - 3];
        size_t num_heads_q = q_shape[ndim - 2];
        size_t head_dim_q = q_shape[ndim - 1];

        size_t batch_size_kv = 1;
        size_t seq_len_kv = kv_shape[ndim - 3];
        size_t num_heads_kv = kv_shape[ndim - 2];
        size_t head_dim_kv = kv_shape[ndim - 1];

        ptrdiff_t qo_stride_b = 0,
                  qo_stride_s = out_desc->stride(ndim - 3),
                  qo_stride_n = out_desc->stride(ndim - 2),
                  qo_stride_d = out_desc->stride(ndim - 1);
        
        ptrdiff_t kv_stride_b = 0,
                  kv_stride_s = k_desc->stride(ndim - 3),
                  kv_stride_n = k_desc->stride(ndim - 2),
                  kv_stride_d = k_desc->stride(ndim - 1);

        if (ndim == 4) {
            qo_stride_b = out_desc->stride(0);
            kv_stride_b = k_desc->stride(0);
            batch_size_q = q_shape[0];
            batch_size_kv = kv_shape[0];
        }

        if (batch_size_q != batch_size_kv && head_dim_q != head_dim_kv) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (num_heads_q % num_heads_kv != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t batch_size = batch_size_q;
        size_t head_dim = head_dim_q;

        // ptrdiff_t mask_stride_b = mask_desc->stride(0),
        //           mask_stride_s = mask_desc->stride(1);

        return utils::Result<FlashAttentionInfo>(FlashAttentionInfo{
            dtype, mask_dtype,
            batch_size,
            seq_len_q, seq_len_kv,
            num_heads_q, num_heads_kv,
            head_dim,
            qo_stride_b, qo_stride_s, qo_stride_n, qo_stride_d,
            kv_stride_b, kv_stride_s, kv_stride_n, kv_stride_d,
        });
    }
};

} // namespace op::flash_attention

#endif // __FLASH_ATTENTION_INFO_H__