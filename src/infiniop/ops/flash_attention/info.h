#ifndef __FLASH_ATTENTION_INFO_H__
#define __FLASH_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>
#include <cmath>

namespace op::flash_attention {

class FlashAttentionInfo {
private:
    FlashAttentionInfo() = default;

public:
    infiniDtype_t dtype;
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

    ptrdiff_t mask_stride_sq;
    ptrdiff_t mask_stride_sk;

    float *mask = nullptr;

    static utils::Result<FlashAttentionInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t mask_desc,
        infiniopAttentionMaskType_t mask_type) {
        // 检查 qkv 数据类型
        auto dtype = out_desc->dtype();
        CHECK_OR_RETURN(
            dtype == q_desc->dtype()
                && dtype == k_desc->dtype()
                && dtype == v_desc->dtype(),
            INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
        // 检查 qkvo 张量形状
        // q 和 out 的形状必须相同
        auto q_shape = out_desc->shape();
        CHECK_SAME_SHAPE(q_shape, q_desc->shape());
        // k 和 v 的形状必须相同
        auto kv_shape = k_desc->shape();
        CHECK_SAME_SHAPE(kv_shape, v_desc->shape());
        // 检查输入的维度
        auto ndim = q_desc->ndim();
        CHECK_OR_RETURN(ndim == k_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(ndim == 3 || ndim == 4, INFINI_STATUS_BAD_TENSOR_SHAPE);


        size_t batch_size_q = 1;
        size_t seq_len_q = q_shape[ndim - 3];
        size_t num_heads_q = q_shape[ndim - 2];
        size_t head_dim_q = q_shape[ndim - 1];

        size_t batch_size_kv = 1;
        size_t seq_len_kv = kv_shape[ndim - 3];
        size_t num_heads_kv = kv_shape[ndim - 2];
        size_t head_dim_kv = kv_shape[ndim - 1];

        size_t batch_size = batch_size_q;
        size_t head_dim = head_dim_q;

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
        // batch_size 和 head_dim 是否一致
        CHECK_OR_RETURN(batch_size_q == batch_size_kv, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(head_dim_q == head_dim_kv, INFINI_STATUS_BAD_TENSOR_SHAPE);
        // 多头注意力是否整除
        CHECK_OR_RETURN(num_heads_q % num_heads_kv == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(out_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(q_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(k_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(v_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

        ptrdiff_t mask_stride_sq = seq_len_kv, 
                  mask_stride_sk = 1;
        float *mask = nullptr;

        if (mask_type == INFINIOP_ATTENTION_MASK_TYPE_NONE) {
            mask_stride_sq = 0;
            mask_stride_sk = 0;
        } else if (mask_type == INFINIOP_ATTENTION_MASK_TYPE_FULL) {
            auto mask_dtype = mask_desc->dtype();
            CHECK_DTYPE(mask_dtype, INFINI_DTYPE_F32);
            CHECK_OR_RETURN(mask_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
            CHECK_OR_RETURN(mask_desc->dim(0) == seq_len_q && mask_desc->dim(1) == seq_len_kv, INFINI_STATUS_BAD_TENSOR_SHAPE);
            CHECK_OR_RETURN(mask_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        } else if (mask_type == INFINIOP_ATTENTION_MASK_TYPE_CAUSAL) {
            size_t mask_size = seq_len_q * seq_len_kv;
            float *causal_mask = new float[mask_size];

            for (size_t i = 0; i < seq_len_q; ++i) {
                for (size_t j = 0; j < seq_len_kv; ++j) {
                    if (j < i) {
                        causal_mask[i * seq_len_kv + j] = -INFINITY;
                    } else {
                        causal_mask[i * seq_len_kv + j] = 0.0f;
                    }
                }
            }

            mask = causal_mask;
        }

        return utils::Result<FlashAttentionInfo>(FlashAttentionInfo{
            dtype,
            batch_size,
            seq_len_q, seq_len_kv,
            num_heads_q, num_heads_kv,
            head_dim,
            qo_stride_b, qo_stride_s, qo_stride_n, qo_stride_d,
            kv_stride_b, kv_stride_s, kv_stride_n, kv_stride_d,
            mask_stride_sq, mask_stride_sk,
            mask,
        });
    }
};

} // namespace op::flash_attention

#endif // __FLASH_ATTENTION_INFO_H__