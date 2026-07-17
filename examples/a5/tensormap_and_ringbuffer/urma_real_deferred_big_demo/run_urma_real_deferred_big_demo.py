#!/usr/bin/env python3
"""Experimental real A5 URMA deferred large-transfer smoke test.

This is not part of the current acceptance gate. It allocates a very large
communication window and may leave the local A5/HCCL environment unable to
initialize new communicators until the device/runtime state is recovered.
"""

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
STATUS_WORDS = 8
URMA_DATA_OFFSET_NBYTES = 64 * 4
URMA_SINGLE_WQE_FLOATS = (256 * 1024 * 1024) // DTYPE_NBYTES
BIG_COUNT = URMA_SINGLE_WQE_FLOATS + 1
SIGNAL_NBYTES = 16 * 4


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
    extra_includes = list(include_dirs) + [
        str(kc.project_root / "src" / "common"),
        str(kc.project_root / "examples" / "a5" / "tensormap_and_ringbuffer"),
    ]

    children = []
    for func_id, rel, child_signature in [
        (
            0,
            "kernels/aiv/kernel_urma_real_deferred_big_tget.cpp",
            [
                ArgDirection.OUT,
                ArgDirection.OUT,
                ArgDirection.OUT,
                ArgDirection.OUT,
                ArgDirection.OUT,
                ArgDirection.IN,
                ArgDirection.IN,
            ],
        ),
        (
            1,
            "kernels/aiv/kernel_urma_real_deferred_big_consumer.cpp",
            [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN, ArgDirection.IN],
        ),
    ]:
        kernel = kc.compile_incore(
            source_path=os.path.join(HERE, rel),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=extra_includes,
        )
        if not platform.endswith("sim"):
            kernel = extract_text_section(kernel)
        children.append((func_id, CoreCallable.build(signature=child_signature, binary=kernel)))

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/urma_real_deferred_big_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    signature = [
        ArgDirection.INOUT,
        ArgDirection.INOUT,
        ArgDirection.INOUT,
        ArgDirection.OUT,
        ArgDirection.IN,
        ArgDirection.IN,
    ]
    return ChipCallable.build(
        signature=signature,
        func_name="urma_real_deferred_big_orchestration",
        config_name="urma_real_deferred_big_orchestration_config",
        binary=orch,
        children=children,
    )


def _zero_i32(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.int32).share_memory_()


def run(platform: str = "a5", device_ids: list[int] | None = None, *, build: bool = False) -> int:
    if device_ids is None:
        device_ids = [0, 1]
    if platform != "a5":
        raise ValueError("urma_real_deferred_big_demo requires platform 'a5'")
    if len(device_ids) != 2:
        raise ValueError(f"urma_real_deferred_big_demo needs exactly 2 devices, got {device_ids}")

    nranks = len(device_ids)
    elem_count = BIG_COUNT
    send_nbytes = elem_count * DTYPE_NBYTES
    recv_nbytes = elem_count * DTYPE_NBYTES
    window_size = max(URMA_DATA_OFFSET_NBYTES + send_nbytes + recv_nbytes + SIGNAL_NBYTES, 4 * 1024 * 1024)

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

        status = [_zero_i32(STATUS_WORDS) for _ in range(nranks)]

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name="urma_real_deferred_big",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    CommBufferSpec(
                        name="urma_reserved",
                        dtype="int32",
                        count=URMA_DATA_OFFSET_NBYTES // 4,
                        nbytes=URMA_DATA_OFFSET_NBYTES,
                    ),
                    CommBufferSpec(name="send", dtype="float32", count=elem_count, nbytes=send_nbytes),
                    CommBufferSpec(name="recv", dtype="float32", count=elem_count, nbytes=recv_nbytes),
                    CommBufferSpec(name="signal", dtype="int32", count=16, nbytes=SIGNAL_NBYTES),
                ],
            ) as handle:
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
                        TensorArgType.INOUT,
                    )
                    args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["recv"],
                            shapes=(elem_count,),
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
        print(
            f"[urma_real_deferred_big_demo] count={elem_count} rank={rank} "
            f"status={words}"
        )
        ok = ok and words[0] == 0 and words[1] == elem_count
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="0-1")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device), build=args.build)


if __name__ == "__main__":
    raise SystemExit(main())
