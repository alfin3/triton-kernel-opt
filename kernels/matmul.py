"""
matmul.py

An evolving collection of matrix multiplication kernels with various
optimizations.

The implementation supports FP8, FP16, and FP32 data types.

The implementation is designed for experimentation, such that optimizations
for specific kernels can be made without affecting the compilation of other
kernels for benchmarking and profiling purposes.

Usage example after imports:

    a = torch.randn((16384, 16384), device='cuda:0', dtype=torch.float16)
    b = torch.randn((16384, 16384), device='cuda:0', dtype=torch.float16)
    matmul_group_m = matmul(mode="group_m")
    matmul_group_m(a,
                   b,
                   block_size_m=128,
                   block_size_n=128,
                   block_size_k=32,
                   group_size_m=32,
                   dtype_in_fp8=False,
                   dtype_out=torch.float16,
                   dtype_acc=tl.float32,
                   dtype_acc_to=tl.float16,
                   num_stages=3,
                   num_warps=4,
    )
"""

from __future__ import annotations
from typing import Any, Callable, Iterable, List, Mapping, Sequence

import torch
from triton import cdiv, jit
from triton import language as tl


# Optimized kernels and kernel launch function with the L2 cache efficient
# program id mapping scheme.


@jit
def _group_m_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes are tl.constexpr for recompilation if tiling changes.
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE_IN_FP8: tl.constexpr,
    DTYPE_ACC: tl.constexpr,
    DTYPE_ACC_TO: tl.constexpr,
    # num_stages and num_warps can be included.
):
    """
    Multiplies FP8, FP16, and FP32 matrices with the L2 cache efficient program
    id mapping scheme. The accumulator is FP32.

    Given a computed (m, n) program id, m corresponds to a block row of the
    A matrix, n corresponds to a block column of the B matrix, and the (m, n)
    program corresponds to a thread block that multiplies and accumulates
    across the blocks in the K dimension. During the launch of the programs,
    m is within a group of GROUP_SIZE_M block rows until n is completed, and m
    is advanced to the next group. Therefore, m is grouped and n is not. The
    A matrix should be contiguous for optimal cache line use.

    The resulting increase in the L2 cache hit rate depends on the ordered
    scheduling of thread blocks, which is not guaranteed by NVIDIA, but
    approximately happens in practice at least on some systems.
    """
    pid = tl.program_id(axis=0)
    pid_m, pid_n = _compute_group_2d_pid(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # Compute tile pointers. Each tile is mapped to a thread block.
    # The below modulo operation ensures valid memory accesses for the values
    # that are later excluded with c_mask.
    # According to the source relating to contiguity and divisibility
    # https://github.com/triton-lang/triton/blob/main/include/triton/Analysis/AxisInfo.h
    # - the usage of 'tl.max_contiguous' without checking if the current
    # block exceeds matrix dimensions, frequently seen in other
    # implementations (see the below usage with checking), may result in
    # invalid memory accesses in vectorized load/store memory transactions, and
    # - block sizes must be powers of two for 'tl.multiple_of' to be applied
    # correctly.
    # Block size dimensions should also be >= 16, and I believe Triton requires
    # it. Given FP8 or higher precision data and a matrix allocation at the
    # highest memory alignment requirement, the minimum block size requirement
    # ensures that 128 bit vectorized accesses with 'tl.multiple_of' are
    # correctly aligned in memory.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    if (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M < M):
        offs_am = tl.max_contiguous(
            tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    if (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N < N):
        offs_bn = tl.max_contiguous(
            tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (a_ptr +
              offs_am[:, None] * stride_am +
              offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr +
              offs_k[:, None] * stride_bk +
              offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE_ACC)

    # Iterate across the K dimension within the tile.
    load_block_size_k = K % BLOCK_SIZE_K == 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining = K - k * BLOCK_SIZE_K
        a, b = _global_vectorized_load(
            a_ptrs,
            b_ptrs,
            offs_k,
            remaining,
            LOAD_BLOCK_SIZE_K=load_block_size_k,
            DTYPE_IN_FP8=DTYPE_IN_FP8,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result in unique location in global memory.
    c = accumulator.to(DTYPE_ACC_TO)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr +
              stride_cm * offs_cm[:, None] +
              stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def group_m_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    dtype_in_fp8: bool,
    dtype_out: torch.dtype,
    dtype_acc: tl.dtype,
    dtype_acc_to: tl.dtype,
    num_stages: int,
    num_warps: int,
) -> torch.Tensor:
    """
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions."
    assert a.is_contiguous(), "Matrix A must be contiguous."
    assert a.dtype == b.dtype, "Incompatible types."
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=dtype_out)
    grid = lambda META: (cdiv(M, META["BLOCK_SIZE_M"]) *
                         cdiv(N, META["BLOCK_SIZE_N"]), )
    _group_m_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        DTYPE_IN_FP8=dtype_in_fp8,
        DTYPE_ACC=dtype_acc,
        DTYPE_ACC_TO=dtype_acc_to,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return c


# Optimized kernels and kernel launch function with the SplitK method.


@jit
def _group_m_split_k_kernel_fp16_fp32(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes are tl.constexpr for recompilation if tiling changes.
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    DTYPE_ACC: tl.constexpr,
    DTYPE_ACC_TO: tl.constexpr,
    # num_stages and num_warps can be included.
):
    """
    Multiplies FP16 and FP32 matrices with the L2 cache efficient program id
    mapping scheme and the SplitK method.

    The kernel is launched with a 2D grid of consecutive program ids and SplitK
    start indices. The consecutive program ids are then mapped to (m, n) pairs
    for L2 cache efficiency. Each (m, n) pair also corresponds to SPLIT_K
    programs, each with a different start index in the K dimension. As a
    result, the SplitK method adds thread blocks for each (m, n) pair.
    """
    pid = tl.program_id(axis=0)
    pid_m, pid_n = _compute_group_2d_pid(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # Compute tile pointers according to a SplitK start index.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    if (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M < M):
        offs_am = tl.max_contiguous(
            tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    if (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N < N):
        offs_bn = tl.max_contiguous(
            tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.program_id(axis=1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (a_ptr +
              offs_am[:, None] * stride_am +
              offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr +
              offs_k[:, None] * stride_bk +
              offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE_ACC)

    # Iterate across the K dimension within the tile.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        remaining = K - k * BLOCK_SIZE_K * SPLIT_K
        a, b = _global_vectorized_load_fp16_fp32(
            a_ptrs,
            b_ptrs,
            offs_k,
            remaining,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    # Store the partial accumulations in global memory.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr +
              stride_cm * offs_cm[:, None] +
              stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # Variance in reduction results due to repeat conversions across programs
    # and variance in the order of floating point operations and rounding.
    c = accumulator.to(DTYPE_ACC_TO)
    tl.atomic_add(c_ptrs, c, mask=c_mask)


def group_m_split_k_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    dtype_in_fp8: bool,
    dtype_out: torch.dtype,
    dtype_acc: tl.dtype,
    dtype_acc_to: tl.dtype,
    num_stages: int,
    num_warps: int,
) -> torch.Tensor:
    """
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions."
    assert a.is_contiguous(), "Matrix A must be contiguous."
    assert a.dtype == b.dtype, "Incompatible types."
    if dtype_in_fp8:
        raise NotImplementedError(
            "The _group_m_split_k_kernel does not work with FP8 data."
        )
    M, K = a.shape
    _, N = b.shape
    # Initialize to zeros, not empty, due to reduction.
    c = torch.zeros((M, N), device=a.device, dtype=dtype_out)
    grid = lambda META: (
        cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )
    _group_m_split_k_kernel_fp16_fp32[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        SPLIT_K=4,
        DTYPE_ACC=dtype_acc,
        DTYPE_ACC_TO=dtype_acc_to,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return c


# Optimized kernels and kernel launch function with the SplitK method
# implemented with condition variables and mutex locks to evaluate the cost
# of thread block synchronization overhead in the reductions and to enable the
# SplitK method for the FP8 data formats.


@jit
def _group_m_split_k_ref_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    lock_ptr,
    M,
    N,
    K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes are tl.constexpr for recompilation if tiling changes.
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    DTYPE_IN_FP8: tl.constexpr,
    DTYPE_ACC: tl.constexpr,
    DTYPE_ACC_TO: tl.constexpr,
    # num_stages and num_warps can be included.
):
    """
    Multiplies FP8, FP16, and FP32 matrices with the L2 cache efficient program
    id mapping scheme and the SplitK method, implemented with condition
    variables and mutex locks.

    The kernel is launched with a 2D grid of consecutive program ids and SplitK
    start indices, as described in '_group_m_split_k_kernel_fp16_fp32'.
    """
    pid = tl.program_id(axis=0)
    pid_m, pid_n = _compute_group_2d_pid(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # Compute tile pointers according to a SplitK start index.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    if (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M < M):
        offs_am = tl.max_contiguous(
            tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    if (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N < N):
        offs_bn = tl.max_contiguous(
            tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.program_id(axis=1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (a_ptr +
              offs_am[:, None] * stride_am +
              offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr +
              offs_k[:, None] * stride_bk +
              offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE_ACC)

    # Iterate across the K dimension within the tile.
    load_block_size_k = K % (BLOCK_SIZE_K * SPLIT_K) == 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        remaining = K - k * BLOCK_SIZE_K * SPLIT_K
        a, b = _global_vectorized_load(
            a_ptrs,
            b_ptrs,
            offs_k,
            remaining,
            LOAD_BLOCK_SIZE_K=load_block_size_k,
            DTYPE_IN_FP8=DTYPE_IN_FP8,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    # Store the partial accumulations in global memory.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr +
              stride_cm * offs_cm[:, None] +
              stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # Use a condition variable and mutex lock implementation to evaluate the
    # overhead of thread block synchronization in reductions.
    lock = lock_ptr + pid
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass
    accumulator += tl.load(c_ptrs, mask=c_mask)
    # Variance in reduction results due to repeat conversions across programs
    # and variance in the order of floating point operations and rounding.
    c = accumulator.to(DTYPE_ACC_TO)
    tl.store(c_ptrs, c, mask=c_mask)
    tl.atomic_xchg(lock, 0)


def group_m_split_k_ref_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    dtype_in_fp8: bool,
    dtype_out: torch.dtype,
    dtype_acc: tl.dtype,
    dtype_acc_to: tl.dtype,
    num_stages: int,
    num_warps: int,
) -> torch.Tensor:
    """
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions."
    assert a.is_contiguous(), "Matrix A must be contiguous."
    assert a.dtype == b.dtype, "Incompatible types."
    M, K = a.shape
    _, N = b.shape
    # Initialize to zeros, not empty, due to reduction.
    c = torch.zeros((M, N), device=a.device, dtype=dtype_out)
    locks = torch.zeros(
        cdiv(M, block_size_m) * cdiv(N, block_size_n),
        device=a.device,
        dtype=torch.int32,
    )
    grid = lambda META: (
        cdiv(M, META["BLOCK_SIZE_M"]) * cdiv(N, META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )
    _group_m_split_k_ref_kernel[grid](
        a,
        b,
        c,
        locks,
        M,
        N,
        K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        SPLIT_K=(2 if dtype_in_fp8 else 4),
        DTYPE_IN_FP8=dtype_in_fp8,
        DTYPE_ACC=dtype_acc,
        DTYPE_ACC_TO=dtype_acc_to,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return c


# Auxiliary functions.


@jit
def _compute_group_2d_pid(
    pid,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Maps a consecutive 1D program id to a grouped 2D program id for L2
    cache hit rate efficiency.
    """
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# The accumulation with FP32 precision showed different sets of assembly
# instructions for FP16 vs. FP8 data:
# - FP32 matrix multiplication and addition instructions were used for
# FP16 data, and
# - integer multiplication and addition instructions (including 32-bit), as
# well as register permutation instructions were used for FP8 data.
# The following separation into '_global_vectorized_load_fp8' and
# '_global_vectorized_load_fp16_fp32' was determined to be necessary for the
# generation of 128-bit vectorized global memory load instructions across
# the FP8, FP16, and FP32 data formats on the system that was used for
# profiling to ensure uniformly high performance. This separation was
# necessary in addition to the above compiler hints.


@jit
def _global_vectorized_load(
    a_ptrs,
    b_ptrs,
    offs_k,
    remaining,
    LOAD_BLOCK_SIZE_K: tl.constexpr,
    DTYPE_IN_FP8: tl.constexpr,
    ):
    if DTYPE_IN_FP8:
        a, b = _global_vectorized_load_fp8(
            a_ptrs,
            b_ptrs,
            offs_k,
            remaining,
            LOAD_BLOCK_SIZE_K,
        )
    else:
        a, b = _global_vectorized_load_fp16_fp32(
            a_ptrs,
            b_ptrs,
            offs_k,
            remaining,
        )
    return a, b


@jit
def _global_vectorized_load_fp8(
    a_ptrs,
    b_ptrs,
    offs_k,
    remaining,
    LOAD_BLOCK_SIZE_K: tl.constexpr,
):
    if LOAD_BLOCK_SIZE_K:
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
    else:
        a = tl.load(a_ptrs,
                    mask=offs_k[None, :] < remaining,
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < remaining,
                    other=0.0)
    return a, b


@jit
def _global_vectorized_load_fp16_fp32(
    a_ptrs,
    b_ptrs,
    offs_k,
    remaining,
):
    a = tl.load(a_ptrs,
                mask=offs_k[None, :] < remaining,
                other=0.0)
    b = tl.load(b_ptrs,
                mask=offs_k[:, None] < remaining,
                other=0.0)
    return a, b


# API for accessing kernels available through __init__.py.


class _matmul(torch.autograd.Function):

    fn = {
        "group_m": group_m_matmul,
        "group_m_split_k": group_m_split_k_matmul,
        "group_m_split_k_ref": group_m_split_k_ref_matmul,
    }

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        mode: str,
        block_size_m: int,
        block_size_n: int,
        block_size_k: int,
        group_size_m: int,
        dtype_in_fp8: bool,
        dtype_out: torch.dtype,
        dtype_acc: tl.dtype,
        dtype_acc_to: tl.dtype,
        num_stages: int,
        num_warps: int,
    ):
        c = _matmul.fn[mode](
            a,
            b,
            block_size_m,
            block_size_n,
            block_size_k,
            group_size_m,
            dtype_in_fp8,
            dtype_out,
            dtype_acc,
            dtype_acc_to,
            num_stages,
            num_warps,
        )
        return c


class matmul:

    def __init__(self, mode: str = "group_m"):
        if mode not in ["group_m",
                        "group_m_split_k",
                        "group_m_split_k_ref"]:
            raise NotImplementedError(
               "Usage: mode='group_m', mode='group_m_split_k'" +
               "mode='group_m_split_k_ref'"
            )
        self.mode = mode

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        block_size_m: int = 128,
        block_size_n: int = 128,
        block_size_k: int = 32,
        group_size_m: int = 8,
        dtype_in_fp8: bool = False,
        dtype_out: torch.dtype = torch.float16,
        dtype_acc: tl.dtype = tl.float32,
        dtype_acc_to: tl.dtype = tl.float16,
        num_stages: int = 3,
        num_warps: int = 4,
    ) -> torch.Tensor:
        c = _matmul.apply(
            a,
            b,
            self.mode,
            block_size_m,
            block_size_n,
            block_size_k,
            group_size_m,
            dtype_in_fp8,
            dtype_out,
            dtype_acc,
            dtype_acc_to,
            num_stages,
            num_warps,
        )
        return c
