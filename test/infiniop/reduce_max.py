import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

_TEST_CASES = [
    # shape_input, stride_input, shape_output, stride_output, dim
    ((4, 3), None, (4, 1), None, 1), # test
    # ((4, 2, 3), None, (1, 2, 3), None, 0), # test
    # ((13, 4), None, (1, 4), None, 0),
    # ((13, 4), None, (13, 1), None, 1),
    # ((13, 4), (10, 1), (13, 1), (10, 1), 1),
    # ((13, 4, 4), None, (1, 4, 4), None, 0),
    # ((13, 4, 4), None, (13, 4, 1), None, 2),
    # ((16, 5632), None, (16, 1), None, 1),
    # ((16, 5632), (6000, 1), (1, 5632), (6000, 1), 0),
    # ((4, 4, 5632), None, (4, 4, 1), None, 2),
    # ((16, 8, 4, 8), None, (1, 8, 4, 8), None, 0),
    # ((16, 8, 4, 8), None, (16, 8, 1, 8), None, 2),
]

# Data types used for testing
_TENSOR_DTYPES = [
    # InfiniDtype.F16, 
    InfiniDtype.F32, 
    # InfiniDtype.BF16,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def reduce_max(input, dim):
    return torch.max(input, dim=dim, keepdim=True).values
    
    
def test(
    handle,
    device,
    shape_input,
    stride_input,
    shape_output,
    stride_output,
    dim,
    dtype=InfiniDtype.F16,
    sync=None,
):
    input = TestTensor(shape_input, stride_input, dtype, device)
    output = TestTensor(shape_output, stride_output, dtype, device, mode="zeros")
    
    print(input.torch_tensor())
    print(hex(input.torch_tensor().data_ptr()))

    print(
        f"Testing ReduceMax on {InfiniDeviceNames[device]} with shape_input:{shape_input} stride_input:{stride_input} output_shape:{shape_output} stride_output:{stride_output} dim:{dim} dtype:{InfiniDtypeNames[dtype]}"
    )
    
    def torch_reduce_max():
        return reduce_max(input.torch_tensor(), dim)
    
    ans = torch_reduce_max()
    
    if sync is not None:
        sync()
        
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input.descriptor,
            dim,
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input, output]:
        tensor.destroy_desc()
        
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_reduce_max():
        check_error(
            LIBINFINIOP.infiniopReduceMax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output.data(),
                input.data(),
                None,
            )
        )
        
    lib_reduce_max()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(output.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_reduce_max(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_reduce_max(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyReduceMaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
