#!/usr/bin/env python3

"""
benchmark_matmul.py

Configurations and functions for benchmarking matrix multiplication kernels.

Usage example:

    python benchmark_matmul.py
    python benchmark_matmul.py --save_path=matmul_results
    python benchmark_matmul.py config_batch matmul_results
"""

from __future__ import annotations
from typing import Any, Callable, Iterable, List, Mapping, Sequence

import torch
from triton import testing
from triton import language as tl
import fire

from kernels import matmul


_REF_LIB_NAME = "cuBLAS"


def get_benchmark_configs(
    dtype_in_fp8: bool,
    dtype_out: torch.dtype,
    dtype_acc: tl.dtype,
    dtype_acc_to: tl.dtype,
) -> List[testing.Benchmark]:
    """
    Creates a list of benchmark configurations. A BenchmarkMatmul instance
    can then be used to vary the warmup, rep, quantiles parameter values.

    The cuBLAS reference and group_m_split_k kernel are excluded from the
    configurations with the FP8 data due to PyTorch and Triton limitations.
    """
    idx = [1, 2, 4] if dtype_in_fp8 else [0, 1, 2, 3, 4]
    line_vals = [[
        _REF_LIB_NAME.lower(),
        "group_m (group_size_m=1)",
        "group_m",
        "group_m_split_k",
        "group_m_split_k_ref",
        ][i] for i in idx]
    line_names = [[
        _REF_LIB_NAME,
        "group_m (group_size_m=1)",
        "group_m",
        "group_m_split_k",
        "group_m_split_k_ref",
        ][i] for i in idx]
    styles = [[
        ("red", "-"),
        ("green", "-"),
        ("green", "--"),
        ("blue", "-"),
        ("blue", "--"),
        ][i] for i in idx]
    configs = []
    for block_size_m in [128, 256]: # [32, 64, 128, 256]
        for block_size_n in [128, 256]: # [32, 64, 128, 256]
            for block_size_k in [32, 64]:
                for group_size_m in [8]: # [2, 4, 8]
                    for num_warps in [4, 8]: # [2, 4, 8]
                        # Exclude configs not fitting into RTX 4060 shared memory.
                        if not (block_size_m == 256 and block_size_n == 256):
                            plot_param_name = (
                                "m{}_n{}_k{}_group{}_warps{}".
                                format(
                                    block_size_m,
                                    block_size_n,
                                    block_size_k,
                                    group_size_m,
                                    num_warps,
                                )
                            )
                            configs.append(
                                testing.Benchmark(
                                    x_names=["M", "N", "K"],
                                    x_vals=[
                                        256, 512, 1024, 2048, 4096, 8192, 16384],
                                    line_arg="provider",
                                    line_vals=line_vals,
                                    line_names=line_names,
                                    styles=styles,
                                    ylabel="TFLOPS",
                                    plot_name=f"matmul_{plot_param_name}",
                                    args={
                                        "block_size_m": block_size_m,
                                        "block_size_n": block_size_n,
                                        "block_size_k": block_size_k,
                                        "group_size_m": group_size_m,
                                        "dtype_in_fp8": dtype_in_fp8,
                                        "dtype_out": dtype_out,
                                        "dtype_acc": dtype_acc,
                                        "dtype_acc_to": dtype_acc_to,
                                        "num_warps": num_warps,
                                    },
                                )
                            )
    return configs


class BenchmarkMatmul:

    def __init__(
        self,
        configs: List[testing.Benchmark],
        device: str = "cuda:0",
    ):
        self._configs = configs
        self._provider2mode = {
            "group_m (group_size_m=1)": "group_m",
            "group_m": "group_m",
            "group_m_split_k": "group_m_split_k",
            "group_m_split_k_ref": "group_m_split_k_ref",
        }
        self._device = device
        self._ref_lib = _REF_LIB_NAME.lower()
        self._group_m_naive = "group_m (group_size_m=1)"

    def benchmark_config_batch(
        self,
        warmup: int,
        rep: int,
        quantiles: List[float],
        save_path: str,
        configs: List[testing.Benchmark] = [],
    )-> None:
        """
        Runs a batch of configurations in a single benchmark.run.
        """
        if not configs:
            configs = self._configs
        @testing.perf_report(configs)
        def _benchmark(
            M,
            N,
            K,
            provider,
            block_size_m,
            block_size_n,
            block_size_k,
            group_size_m,
            dtype_in_fp8,
            dtype_out,
            dtype_acc,
            dtype_acc_to,
            num_warps,
            warmup=warmup,
            rep=rep,
        ):
            a = torch.randn(
                (M, K), device=self._device, dtype=torch.float32)
            b = torch.randn(
                (K, N), device=self._device, dtype=torch.float32)
            a = a.to(dtype_out)
            b = b.to(dtype_out)
            if provider == self._ref_lib:
                ms, min_ms, max_ms = testing.do_bench(
                    lambda: torch.matmul(a, b),
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                )
            else:
                bench = self._create_bench(
                    mode=self._provider2mode[provider],
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                )
                if provider == self._group_m_naive:
                    group_size_m = 1
                ms, min_ms, max_ms = bench(
                    a,
                    b,
                    block_size_m=block_size_m,
                    block_size_n=block_size_n,
                    block_size_k=block_size_k,
                    group_size_m=group_size_m,
                    dtype_in_fp8=dtype_in_fp8,
                    dtype_out=dtype_out,
                    dtype_acc=dtype_acc,
                    dtype_acc_to=dtype_acc_to,
                    num_warps=num_warps,
                )
            perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
            return perf(ms), perf(max_ms), perf(min_ms)
        _benchmark.run(
            print_data=True,
            show_plots=False,
            save_path=save_path,
        )

    def benchmark_single_configs(
        self,
        warmup: int,
        rep: int,
        quantiles: List[float],
        save_path: str,
    ) -> None:
        """
        Runs single configurations across a set of configurations.
        This function may help verify benchmarking results.
        """
        for config in self._configs:
            self.benchmark_config_batch(
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
                save_path=save_path,
                configs=[config],
            )

    @staticmethod
    def _create_bench(
        mode,
        warmup,
        rep,
        quantiles,
    ) -> Callable[..., Any]:
        matmul_fn = matmul(mode=mode)
        return (lambda
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
                num_warps,
                :
                testing.do_bench(
                    lambda: matmul_fn(
                        a,
                        b,
                        block_size_m=block_size_m,
                        block_size_n=block_size_n,
                        block_size_k=block_size_k,
                        group_size_m=group_size_m,
                        dtype_in_fp8=dtype_in_fp8,
                        dtype_out=dtype_out,
                        dtype_acc=dtype_acc,
                        dtype_acc_to=dtype_acc_to,
                        num_warps=num_warps,
                    ),
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                )
                )


def main(
    mode: str = "config_batch",
    prec: str = "fp16",
    save_path: str = "matmul_results",
    warmup: int = 100,
    rep: int = 100,
    quantiles: List[float] = [0.5, 0.2, 0.8],
):
    torch.manual_seed(0)
    if mode not in ["config_batch", "single_configs"]:
        raise NotImplementedError(
            "Usage: mode='config_batch', mode='single_configs'"
        )
    if prec not in ["fp8_e4m3", "fp8_e5m2", "fp16", "fp32"]:
        raise NotImplementedError(
            "Usage: prec='fp8_e4m3', prec='fp8_e5m2', prec='fp16', prec='fp32'"
        )
    if mode == "config_batch":
        if prec == "fp8_e4m3":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=True,
                    dtype_out=torch.float8_e4m3fn,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float8e4nv,
                )
            )
            benchmark.benchmark_config_batch(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
        if prec == "fp8_e5m2":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=True,
                    dtype_out=torch.float8_e5m2,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float8e5,
                )
            )
            benchmark.benchmark_config_batch(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
        if prec == "fp16":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=False,
                    dtype_out=torch.float16,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float16,
                )
            )
            benchmark.benchmark_config_batch(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
        if prec == "fp32":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=False,
                    dtype_out=torch.float32,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float32,
                )
            )
            benchmark.benchmark_config_batch(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
    if mode == "single_configs":
        if prec == "fp8_e4m3":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=True,
                    dtype_out=torch.float8_e4m3fn,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float8e4nv,
                )
            )
            benchmark.benchmark_single_configs(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
        if prec == "fp8_e5m2":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=True,
                    dtype_out=torch.float8_e5m2,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float8e5,
                )
            )
            benchmark.benchmark_single_configs(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
        if prec == "fp16":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=False,
                    dtype_out=torch.float16,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float16,
                )
            )
            benchmark.benchmark_single_configs(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )
        if prec == "fp32":
            benchmark = BenchmarkMatmul(
                configs=get_benchmark_configs(
                    dtype_in_fp8=False,
                    dtype_out=torch.float32,
                    dtype_acc=tl.float32,
                    dtype_acc_to=tl.float32,
                )
            )
            benchmark.benchmark_single_configs(
                save_path=save_path,
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )


if __name__ == "__main__":
    fire.Fire(main)
