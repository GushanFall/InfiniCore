#include "flash_attention_ascend.h"

#include "../../../devices/ascend/common_ascend.h"
#include "aclnnop/aclnn_prompt_flash_attention_v3.h"

#include <cmath>
#include <vector>

namespace op::flash_attention::ascend {

struct Descriptor::Opaque {
    aclOpExecutor *executor;
    aclnnTensorDescriptor_t out, q, k, v, mask;
    aclnnTensorDescriptor_t softmax_max, softmax_sum;
    size_t softmax_max_offset, softmax_sum_offset;
    std::vector<char> layout;
    ~Opaque() {
        delete out;
        delete q;
        delete k;
        delete v;
        if (mask != nullptr) {
            delete mask;
        }
        delete softmax_max;
        delete softmax_sum;
        aclDestroyAclOpExecutor(executor);
    }
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
    infiniopTensorDescriptor_t mask_desc) {
    size_t workspace_size = 0;
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto result = FlashAttentionInfo::create(out_desc, q_desc, k_desc, v_desc, mask_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    auto out = new aclnnTensorDescriptor(
        toAclDataType(out_desc->dtype()),
        {int64_t(info.b), int64_t(info.q_len), int64_t(info.nh), int64_t(info.d_v)});

    auto q = new aclnnTensorDescriptor(
        toAclDataType(q_desc->dtype()),
        {int64_t(info.b), int64_t(info.q_len), int64_t(info.nh), int64_t(info.d_qk)});

    auto k = new aclnnTensorDescriptor(
        toAclDataType(k_desc->dtype()),
        {int64_t(info.b), int64_t(info.kv_len), int64_t(info.nkvh), int64_t(info.d_qk)});

    auto v = new aclnnTensorDescriptor(
        toAclDataType(v_desc->dtype()),
        {int64_t(info.b), int64_t(info.kv_len), int64_t(info.nkvh), int64_t(info.d_v)});

    aclnnTensorDescriptor_t mask = info.has_mask == 1 ? new aclnnTensorDescriptor(mask_desc) : nullptr;

    auto softmax_max = new aclnnTensorDescriptor(
        aclDataType::ACL_FLOAT,
        {int64_t(1), int64_t(info.nh), int64_t(info.q_len), int64_t(8)},
        nullptr);
    auto softmax_sum = new aclnnTensorDescriptor(
        aclDataType::ACL_FLOAT,
        {int64_t(1), int64_t(info.nh), int64_t(info.q_len), int64_t(8)},
        nullptr);
    size_t softmax_out_size = softmax_max->numel() * sizeof(float);

    aclTensor *pseShift = nullptr;
    aclIntArray *actualSeqLengths = nullptr;
    aclIntArray *actualSeqLengthsKv = nullptr;
    aclTensor *deqScale1 = nullptr;
    aclTensor *quantScale1 = nullptr;
    aclTensor *deqScale2 = nullptr;
    aclTensor *quantScale2 = nullptr;
    aclTensor *quantOffset2 = nullptr;

    double scale = double(1) / std::sqrt(double(info.d_qk));
    int64_t pre_tokens = info.q_len;
    int64_t next_tokens = info.kv_len - info.q_len;
    int64_t nh = int64_t(info.nh);
    int64_t nkvh = int64_t(info.nkvh);
    int64_t inner_precise = 0;
    int64_t sparse_mode = info.has_mask == 1 ? 1 : 0;
    // int64_t sparse_mode = 0;

    std::vector<char> layout = {'B', 'S', 'N', 'D', 0};
    aclOpExecutor *executor;
    CHECK_ACL(aclnnPromptFlashAttentionV3GetWorkspaceSize(
        q->tensor, k->tensor, v->tensor, pseShift,
        info.has_mask ? mask->tensor : nullptr,
        actualSeqLengths, actualSeqLengthsKv, deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
        nh, scale, pre_tokens, next_tokens, layout.data(), nkvh,
        sparse_mode, inner_precise, out->tensor, &workspace_size, &executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));
    size_t total_workspace_size = workspace_size + softmax_out_size * 2;

    *desc_ptr = new Descriptor(
        info,
        total_workspace_size,
        new Opaque{
            executor,
            out,
            q,
            k,
            v,
            mask,
            softmax_max,
            softmax_sum,
            workspace_size,
            workspace_size + softmax_out_size,
            layout,
        },
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    void *mask,
    void *stream) const {
    if (workspace_size < _min_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    void *softmax_max = (char *)workspace + _opaque->softmax_max_offset;
    void *softmax_sum = (char *)workspace + _opaque->softmax_sum_offset;
    AclSetInputTensorAddr(_opaque->executor, 0, _opaque->q->tensor, (void *)q);
    AclSetInputTensorAddr(_opaque->executor, 1, _opaque->k->tensor, (void *)k);
    AclSetInputTensorAddr(_opaque->executor, 2, _opaque->v->tensor, (void *)v);
    if (mask != nullptr) {
        AclSetInputTensorAddr(_opaque->executor, 3, _opaque->mask->tensor, (void *)mask);
    }
    AclSetOutputTensorAddr(_opaque->executor, 0, _opaque->softmax_max->tensor, softmax_max);
    AclSetOutputTensorAddr(_opaque->executor, 1, _opaque->softmax_sum->tensor, softmax_sum);
    AclSetOutputTensorAddr(_opaque->executor, 2, _opaque->out->tensor, out);
    CHECK_ACL(aclnnPromptFlashAttentionV3(workspace, workspace_size, _opaque->executor, stream));
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::flash_attention::ascend
