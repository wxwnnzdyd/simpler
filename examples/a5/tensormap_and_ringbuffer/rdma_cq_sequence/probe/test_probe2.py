#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step 2: the full case's arg shape and domain, with a kernel that only echoes.

If this prints ``status=[0, 0, 1, 0x5a5a5a5a]`` and ``summary[0]=1`` then the
six-tensor/four-scalar shape and the five-buffer domain are fine, and the
remaining breakage is in the RDMA post or the summary.
"""

from __future__ import annotations

import argparse
import os

import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
RDMA_DATA_OFFSET_NBYTES = 64 * 4
SLICE_BYTES = 1024
SIGNAL_BYTES = 256
SUMMARY_WORDS = 10
STATUS_WORDS = 4


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    extra_includes = list(kc.get_orchestration_include_dirs(runtime)) + [str(kc.project_root / "src" / "common")]
    pto_isa_root = ensure_pto_isa_root()

    kernel = kc.compile_incore(
        source_path=os.path.join(HERE, "probe2_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=extra_includes,
    )
    if not platform.endswith("sim"):
        kernel = extract_text_section(kernel)
    child = CoreCallable.build(
        signature=[
            ArgDirection.IN,
            ArgDirection.INOUT,
            ArgDirection.INOUT,
            ArgDirection.INOUT,
            ArgDirection.INOUT,
            ArgDirection.OUT,
        ]
        + [ArgDirection.IN] * 4,
        binary=kernel,
    )

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "probe2_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[
            ArgDirection.IN,
            ArgDirection.INOUT,
            ArgDirection.INOUT,
            ArgDirection.INOUT,
            ArgDirection.OUT,
            ArgDirection.OUT,
            ArgDirection.IN,
            ArgDirection.IN,
            ArgDirection.IN,
            ArgDirection.IN,
        ],
        func_name="probe2_orch",
        config_name="probe2_orch_config",
        binary=orch,
        children=[(0, child)],
    )


def run(platform: str, device_ids: list[int]) -> int:
    nranks = len(device_ids)
    chip_callable = build_chip_callable(platform)
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        build=False,
    )
    chip_handle = worker.register(chip_callable)

    summary = [torch.zeros(SUMMARY_WORDS, dtype=torch.int64).share_memory_() for _ in range(nranks)]
    status = [torch.full((STATUS_WORDS,), -1, dtype=torch.int32).share_memory_() for _ in range(nranks)]
    slice_elems = SLICE_BYTES // 4
    signal_elems = SIGNAL_BYTES // 4
    window_size = 4 * 1024 * 1024
    bufs = [
        [worker.create_buffer(SLICE_BYTES) for _ in range(nranks)] for _ in range(4)
    ]

    def orch_fn(orch, _args, cfg):
        with orch.allocate_domain(
            name="probe2",
            workers=list(range(nranks)),
            window_size=window_size,
            buffers=[
                CommBufferSpec(
                    name="rdma_reserved", dtype="int32", count=RDMA_DATA_OFFSET_NBYTES // 4, nbytes=RDMA_DATA_OFFSET_NBYTES
                ),
                CommBufferSpec(name="send", dtype="float32", count=slice_elems, nbytes=SLICE_BYTES),
                CommBufferSpec(name="recv", dtype="float32", count=slice_elems, nbytes=SLICE_BYTES),
                CommBufferSpec(name="tput", dtype="float32", count=slice_elems, nbytes=SLICE_BYTES),
                CommBufferSpec(name="signal", dtype="int32", count=signal_elems, nbytes=SIGNAL_BYTES),
            ],
        ) as handle:
            for rank in range(nranks):
                domain = handle[rank]
                orch.copy_to(domain.buffers["send"], bufs[0][rank])
                orch.copy_to(domain.buffers["recv"], bufs[1][rank])
                orch.copy_to(domain.buffers["tput"], bufs[2][rank])
                orch.copy_to(domain.buffers["signal"], bufs[3][rank])

            for rank in range(nranks):
                domain = handle[rank]
                args = TaskArgs()
                args.add_tensor(domain.buffers["send"].tensor((slice_elems,), DataType.FLOAT32), TensorArgType.INPUT)
                args.add_tensor(domain.buffers["recv"].tensor((slice_elems,), DataType.FLOAT32), TensorArgType.INOUT)
                args.add_tensor(domain.buffers["tput"].tensor((slice_elems,), DataType.FLOAT32), TensorArgType.INOUT)
                args.add_tensor(
                    domain.buffers["signal"].tensor((signal_elems,), DataType.INT32), TensorArgType.INOUT
                )
                args.add_tensor(make_tensor_arg(worker, summary[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(make_tensor_arg(worker, status[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_ctx)
                args.add_scalar(0)  # mode
                args.add_scalar(64)  # posts
                args.add_scalar(1)  # tasks
                orch.submit_next_level(chip_handle, args, cfg, worker=rank)

    try:
        worker.init()
        worker.run(orch_fn, args=None, config=CallConfig())
    finally:
        for row in bufs:
            for b in row:
                worker.release_buffer(b)
        worker.close()

    ok = True
    for rank in range(nranks):
        words = [int(x) for x in status[rank].tolist()]
        summary_words = [int(x) for x in summary[rank].tolist()]
        print(f"[probe2] rank={rank} status={words} summary={summary_words}")
        ok = ok and words[0] == 0 and words[1] == 0 and words[2] == 1 and words[3] == 0x5A5A5A5A
        ok = ok and summary_words[0] == 1
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="9,10")
    args = parser.parse_args()
    return run(args.platform, [int(x) for x in args.device.split(",") if x])


if __name__ == "__main__":
    raise SystemExit(main())
