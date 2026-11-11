#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step 1 of the RDMA CQ sequence case: prove one task can write a host tensor.

No RDMA, no comm domain, no chained dependencies. If this prints
``status=[0, 1234, 42, -1]`` the harness, the orchestrator and the kernel wiring
all work, and later steps can add the domain, the posts and the summary.
"""

from __future__ import annotations

import argparse
import os

import torch
from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, CoreCallable, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS_WORDS = 4


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    extra_includes = list(kc.get_orchestration_include_dirs(runtime)) + [str(kc.project_root / "src" / "common")]
    pto_isa_root = ensure_pto_isa_root()

    kernel = kc.compile_incore(
        source_path=os.path.join(HERE, "probe_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=extra_includes,
    )
    if not platform.endswith("sim"):
        kernel = extract_text_section(kernel)
    child = CoreCallable.build(signature=[ArgDirection.OUT], binary=kernel)

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "probe_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[ArgDirection.OUT],
        func_name="probe_orch",
        config_name="probe_orch_config",
        binary=orch,
        children=[(0, child)],
    )


def run(platform: str, device_ids: list[int]) -> int:
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
    status = [torch.full((STATUS_WORDS,), -1, dtype=torch.int32).share_memory_() for _ in device_ids]
    ok = True
    try:
        worker.init()

        def orch_fn(orch, _args, cfg):
            for rank in range(len(device_ids)):
                args = TaskArgs()
                args.add_tensor(make_tensor_arg(worker, status[rank]), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(chip_handle, args, cfg, worker=rank)

        worker.run(orch_fn, args=None, config=CallConfig())
    finally:
        worker.close()

    for rank, rank_status in enumerate(status):
        words = [int(x) for x in rank_status.tolist()]
        print(f"[probe] rank={rank} status={words}")
        ok = ok and words[0] == 0 and words[1] == 1234 and words[2] == 42
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="9,10")
    args = parser.parse_args()
    device_ids = [int(x) for x in args.device.split(",") if x]
    return run(args.platform, device_ids)


if __name__ == "__main__":
    raise SystemExit(main())
