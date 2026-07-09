#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone real URMA TGET/TPUT correctness smoke for A5 hardware."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for path in (str(REPO_ROOT), str(REPO_ROOT / "python")):
    if path not in sys.path:
        sys.path.insert(0, path)
sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != "_simpler_editable"]

import pytest
import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE_NBYTES = 4
SIGNAL_NBYTES = 16 * 4
STATUS_WORDS = 8
CASES = (16, 4096)


def parse_device_range(spec: str) -> list[int]:
    if "," in spec:
        return [int(x) for x in spec.split(",") if x]
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    extra_includes = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    kernel = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/kernel_urma_real_async.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=extra_includes,
    )
    if not platform.endswith("sim"):
        kernel = extract_text_section(kernel)

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/urma_real_async_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )

    signature = [
        ArgDirection.IN,
        ArgDirection.INOUT,
        ArgDirection.INOUT,
        ArgDirection.INOUT,
        ArgDirection.OUT,
        ArgDirection.IN,
        ArgDirection.IN,
    ]
    return ChipCallable.build(
        signature=signature,
        func_name="urma_real_async_orchestration",
        config_name="urma_real_async_orchestration_config",
        binary=orch,
        children=[(0, CoreCallable.build(signature=signature, binary=kernel))],
    )


def _send_pattern(rank: int, count: int) -> torch.Tensor:
    return torch.tensor([float(rank * 100000 + i) for i in range(count)], dtype=torch.float32).share_memory_()


def _zero_float(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.float32).share_memory_()


def _zero_i32(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.int32).share_memory_()


def run_case(platform: str, device_ids: list[int], elem_count: int, *, build: bool = False) -> bool:
    if platform != "a5":
        raise ValueError("urma_real_async_demo requires platform 'a5'; a5sim cannot validate real URMA")
    if len(device_ids) != 2:
        raise ValueError(f"urma_real_async_demo needs exactly 2 devices, got {device_ids}")

    nranks = len(device_ids)
    send_nbytes = elem_count * DTYPE_NBYTES
    tget_nbytes = elem_count * DTYPE_NBYTES
    tput_nbytes = nranks * elem_count * DTYPE_NBYTES
    window_size = max(send_nbytes + tget_nbytes + tput_nbytes + SIGNAL_NBYTES, 4 * 1024 * 1024)

    send_host = [_send_pattern(rank, elem_count) for rank in range(nranks)]
    tget_zero = [_zero_float(elem_count) for _ in range(nranks)]
    tput_zero = [_zero_float(nranks * elem_count) for _ in range(nranks)]
    signal_zero = [_zero_i32(16) for _ in range(nranks)]
    status = [_zero_i32(STATUS_WORDS) for _ in range(nranks)]

    chip_callable = build_chip_callable(platform)
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        build=build,
    )
    chip_handle = worker.register(chip_callable)
    try:
        worker.init()

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name=f"urma_real_{elem_count}",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    CommBufferSpec(name="send", dtype="float32", count=elem_count, nbytes=send_nbytes),
                    CommBufferSpec(name="tget_recv", dtype="float32", count=elem_count, nbytes=tget_nbytes),
                    CommBufferSpec(
                        name="tput_recv",
                        dtype="float32",
                        count=nranks * elem_count,
                        nbytes=tput_nbytes,
                    ),
                    CommBufferSpec(name="signal", dtype="int32", count=16, nbytes=SIGNAL_NBYTES),
                ],
            ) as handle:
                for rank in range(nranks):
                    domain = handle[rank]
                    orch.copy_to(rank, domain.buffer_ptrs["send"], send_host[rank].data_ptr(), send_nbytes)
                    orch.copy_to(rank, domain.buffer_ptrs["tget_recv"], tget_zero[rank].data_ptr(), tget_nbytes)
                    orch.copy_to(rank, domain.buffer_ptrs["tput_recv"], tput_zero[rank].data_ptr(), tput_nbytes)
                    orch.copy_to(rank, domain.buffer_ptrs["signal"], signal_zero[rank].data_ptr(), SIGNAL_NBYTES)

                for rank in range(nranks):
                    domain = handle[rank]
                    args = TaskArgs()
                    args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["send"],
                            shapes=(elem_count,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INPUT,
                    )
                    args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["tget_recv"],
                            shapes=(elem_count,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INOUT,
                    )
                    args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["tput_recv"],
                            shapes=(nranks * elem_count,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INOUT,
                    )
                    args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["signal"],
                            shapes=(16,),
                            dtype=DataType.INT32,
                            child_memory=True,
                        ),
                        TensorArgType.INOUT,
                    )
                    args.add_tensor(make_tensor_arg(status[rank]), TensorArgType.OUTPUT_EXISTING)
                    args.add_scalar(domain.device_ctx)
                    args.add_scalar(elem_count)
                    orch.submit_next_level(chip_handle, args, cfg, worker=rank)

        worker.run(orch_fn, args=None, config=CallConfig())
    finally:
        worker.close()

    ok = True
    for rank in range(nranks):
        words = [int(x) for x in status[rank].tolist()]
        print(f"[urma_real_async_demo] count={elem_count} rank={rank} status={words}")
        ok = ok and words[0] == 0
    return ok


def run(platform: str = "a5", device_ids: list[int] | None = None, *, build: bool = False) -> int:
    if device_ids is None:
        device_ids = [0, 1]
    ok = True
    for elem_count in CASES:
        ok = run_case(platform, device_ids, elem_count, build=build) and ok
    return 0 if ok else 1


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
def test_urma_real_async_demo(st_platform, st_device_ids) -> None:
    assert run(st_platform, [int(st_device_ids[0]), int(st_device_ids[1])]) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="0-1")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device), build=args.build)


if __name__ == "__main__":
    raise SystemExit(main())
