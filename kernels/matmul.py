"""
matmul.py

An evolving collection of matrix multiplication kernels with various
optimizations.

The implementation supports FP8, FP16, and FP32 data types.

The implementation is designed for experimentation, such that optimizations
for specific kernels can be made without affecting the compilation of other
kernels for benchmarking and profiling purposes.

Usage example: TODO
"""

from __future__ import annotations
from typing import Any, Callable, Iterable, List, Mapping, Sequence

import torch
from triton import cdiv, jit
from triton import language as tl


# Kernels and kernel launch function with the naive program id mapping scheme.


@jit
def _naive_kernel(
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
    DTYPE_ACC: tl.constexpr,
    DTYPE_ACC_TO: tl.constexpr,
    FP8: tl.constexpr,
    # num_stages and num_warps can be included.
):
    """
    Multiplies FP8, FP16, and FP32 matrices with the naive program id mapping
    scheme. The accumulator is FP32.
    """
    # Compute 2D program ids.
    pid = tl.program_id(axis=0)
    pid_m, pid_n = _compute_2d_pid(pid, N, BLOCK_SIZE_N)

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
    rem = K % BLOCK_SIZE_K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining = K - k * BLOCK_SIZE_K
        a, b = _global_vectorized_load(
            a_ptrs,
            b_ptrs,
            offs_k,
            remaining,
            REM=rem,
            FP8=FP8,
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


def naive_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    dtype_out: torch.dtype,
    dtype_acc: tl.dtype,
    dtype_acc_to: tl.dtype,
    num_stages: int,
    num_warps: int,
    fp8: bool,
) -> torch.Tensor:
    """
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions."
    assert a.is_contiguous(), "Matrix A must be contiguous."
    assert a.dtype == b.dtype, "Incompatible types."
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=dtype_out)
    grid = lambda META: (cdiv(M, META['BLOCK_SIZE_M']) *
                         cdiv(N, META['BLOCK_SIZE_N']), )
    _naive_kernel[grid](
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
        DTYPE_ACC=dtype_acc,
        DTYPE_ACC_TO=dtype_acc_to,
        FP8=fp8,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return c


# Optimized kernels and kernel launch function with the L2 cache efficient
# program id mapping scheme.


# Optimized kernels and kernel launch function with the SplitK method.


# Auxiliary functions.


@jit
def _compute_2d_pid(
    pid,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Maps a consecutive 1D program id to a 2D program id.
    """
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    return pid_m, pid_n


# The accumulation with FP32 precision showed different sets of assembly
# instructions for FP16 vs. FP8 data:
# - FP32 matrix multiplication and addition instructions were used for
# FP16 data, and
# - 32-bit integer multiplication and addition instructions as well as
# register permutation instructions were used for FP8 data.
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
    REM: tl.constexpr,
    FP8: tl.constexpr,
    ):
    if FP8:
        a, b = _global_vectorized_load_fp8(
            a_ptrs,
            b_ptrs,
            offs_k,
            remaining,
            REM,
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
    REM: tl.constexpr,
):
    if REM == 0:
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
        "naive": naive_matmul,
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
        dtype_out: torch.dtype,
        dtype_acc: tl.dtype,
        dtype_acc_to: tl.dtype,
        num_stages: int,
        num_warps: int,
        fp8: bool,
    ):
        c = _matmul.fn[mode](
            a,
            b,
            block_size_m,
            block_size_n,
            block_size_k,
            dtype_out,
            dtype_acc,
            dtype_acc_to,
            num_stages,
            num_warps,
            fp8,
        )
        return c


class matmul:

    def __init__(self, mode: str = "naive"):
        if mode not in ["naive"]:
            raise NotImplementedError(
               "Usage: mode='naive'"
            )
        self.mode = mode

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        block_size_m: int = 128,
        block_size_n: int = 128,
        block_size_k: int = 32,
        dtype_out: torch.dtype = torch.float16,
        dtype_acc: tl.dtype = tl.float32,
        dtype_acc_to: tl.dtype = tl.float16,
        num_stages: int = 3,
        num_warps: int = 4,
        fp8: bool = False,
    ) -> torch.Tensor:
        c = _matmul.apply(
            a,
            b,
            self.mode,
            block_size_m,
            block_size_n,
            block_size_k,
            dtype_out,
            dtype_acc,
            dtype_acc_to,
            num_stages,
            num_warps,
            fp8,
        )
        return c
