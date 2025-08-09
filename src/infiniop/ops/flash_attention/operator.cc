#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/flash_attention.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/flash_attention_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopAttentionMaskType_t mask_type) {

#define CREATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        return op::flash_attention::NAMESPACE::Descriptor::create(                     \
            handle,                                                                    \
            reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                                  \
            q_desc,                                                                    \
            k_desc,                                                                    \
            v_desc,                                                                    \
            mask_desc,                                                                 \
            mask_type);

    switch (handle->device) {

#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
    
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc, 
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                           \
    case CASE:                                                                                         \
        *size = reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}


__C infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *mask,
    void *stream) {
        
#define CALCULATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                            \
        return reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc)       \
            ->calculate(workspace, workspace_size, out, q, k, v, mask, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyFlashAttentionDescriptor(infiniopFlashAttentionDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                         \
    case CASE:                                                                           \
        delete reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc);     \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
