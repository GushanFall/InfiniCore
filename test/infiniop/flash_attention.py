from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from libinfiniop import (
    open_lib,
    to_tensor,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    check_error,
    rearrange_tensor,
    create_workspace,
    get_args,
    get_test_devices,
    test_operator,
    debug,
    get_tolerance,
    profile_operation,
)


import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


class FlashAttentionDescriptor(Structure):
    _fields_ = [("device", c_int32)]
    
    
infiniopFlashAttentionDescriptor_t = POINTER(FlashAttentionDescriptor)


def flashAttention(q, k, v, mask_type, mask_dtype, mask=None):
    # mask
    if mask_type == "INFINI_MASK_FULL":
        if mask is None:
            raise ValueError("Mask must be provided for INFINI_MASK_FULL")
        if mask_dtype == torch.uint8:
            mask = 1 - mask
        elif mask_dtype == torch.bool:
            mask = ~mask
    elif mask_type == "INFINI_MASK_CAUSAL":
        mask = torch.tril(torch.ones(q.shape[-3], k.shape[-3], dtype=mask_dtype), diagonal=0).to(q.device)
    elif mask_type == "INFINI_MASK_NO":
        mask = None
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")
    
    if q.ndim == 3:
        q = q.unsqueeze(0)
    if k.ndim == 3:
        k = k.unsqueeze(0)
    if v.ndim == 3:
        v = v.unsqueeze(0)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    
    with sdpa_kernel(SDPBackend.MATH):
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True
    )
    attn_output = attn_output.permute(0, 2, 1, 3)
    
    return attn_output
    
    
def test(
    lib,
    handle,
    torch_device,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_heads_q,
    num_heads_kv,
    head_dim,
    mask_type,
    mask_dtype,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing FlashAttention on {torch_device} with batch_size:{batch_size} seq_len_q:{seq_len_q} seq_len_kv:{seq_len_kv} num_heads_q:{num_heads_q} num_heads_kv:{num_heads_kv} head_dim:{head_dim} mask_type:{mask_type} dtype:{dtype}"
    )
    
    out = torch.zeros([batch_size, seq_len_q, num_heads_q, head_dim], dtype=dtype, device=torch_device)
    q = torch.rand([batch_size, seq_len_q, num_heads_q, head_dim], dtype=dtype, device=torch_device) * 0.1
    k = torch.rand([batch_size, seq_len_kv, num_heads_kv, head_dim], dtype=dtype, device=torch_device) * 0.1
    v = torch.rand([batch_size, seq_len_kv, num_heads_kv, head_dim], dtype=dtype, device=torch_device) * 0.1
    mask = torch.zeros([seq_len_q, seq_len_kv], dtype=mask_dtype, device=torch_device)
    
    ans = flashAttention(q, k, v, mask_type, mask_dtype, mask)
    
    out_tensor = to_tensor(out, lib)
    q_tensor = to_tensor(q, lib)
    k_tensor = to_tensor(k, lib)
    v_tensor = to_tensor(v, lib)
    mask_tensor = to_tensor(mask, lib)
    
    if sync is not None:
        sync()
    
    descriptor = infiniopFlashAttentionDescriptor_t()
    check_error(
        lib.infiniopCreateFlashAttentionDescriptor(
            handle, 
            ctypes.byref(descriptor), 
            out_tensor.descriptor, 
            q_tensor.descriptor, 
            k_tensor.descriptor, 
            v_tensor.descriptor,
            mask_tensor.descriptor,
            mask_type,
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [
        out_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        mask_tensor,
    ]:
        tensor.destroyDesc(lib)
    
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetFlashAttentionWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, out.device)
    
    def lib_flash_attention():
        check_error(
            lib.infiniopFlashAttention(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                out_tensor.data,
                q_tensor.data,
                k_tensor.data,
                v_tensor.data,
                mask_tensor.data,
                None,
            )
        )
    
    lib_flash_attention()
    
    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out, ans, atol=atol, rtol=rtol)
    assert torch.allclose(out, ans, atol=atol, rtol=rtol)
    
    # Profiling workflow
    if PROFILE:
        profile_operation("PyTorch", lambda: flashAttention(q, k, v, mask_type, mask_dtype, mask), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_flash_attention(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
    check_error(lib.infiniopDestroyFlashAttentionDescriptor(descriptor))


if __name__ == "__main__":
    _TENSOR_DTYPES = [torch.float16, torch.float32]
    
    _TOLERANCE_MAP = {
        torch.float16: {"atol": 1e-4, "rtol": 1e-2},
        torch.float32: {"atol": 1e-5, "rtol": 1e-3},
    }
    
    DEBUG = False
    PROFILE = False
    NUM_PRERUN = 10
    NUM_ITERATIONS = 1000
    test_cases = [
        (
            1, # batch_size
            20, # seq_len_q
            20, # seq_len_kv
            4,  # num_heads_q
            4,  # num_heads_kv
            64, # head_dim
            "INFINI_MASK_NO", # mask_type
            torch.uint8, # mask_dtype
        )
    ]
    
    args = get_args()
    lib = open_lib()
    
    lib.infiniopCreateFlashAttentionDescriptor.restype = c_int32
    lib.infiniopCreateFlashAttentionDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopFlashAttentionDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetFlashAttentionWorkspaceSize.restype = c_int32
    lib.infiniopGetFlashAttentionWorkspaceSize.argtypes = [
        infiniopFlashAttentionDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopFlashAttention.restype = c_int32
    lib.infiniopFlashAttention.argtypes = [
        infiniopFlashAttentionDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    
    lib.infiniopDestroyFlashAttentionDescriptor.restype = c_int32
    lib.infiniopDestroyFlashAttentionDescriptor.argtypes = [
        infiniopFlashAttentionDescriptor_t,
    ]
    
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    for device in get_test_devices(args):
        test_operator(lib, device, test, test_cases, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")