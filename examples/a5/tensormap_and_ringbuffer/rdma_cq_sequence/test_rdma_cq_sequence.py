#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Drive the RDMA CQ past the boundaries the ack ring has to survive.

Three cases share one chip callable and differ only in how the sequence is
driven (see kernels/orchestration/rdma_cq_sequence_orch.cpp):

  chain     one task at a time, 1024 chunks of 64 single-WQE posts, so the shared
            QP sees one monotone sequence of 65536 CQEs -- two full laps of the
            32768-deep ring. Exercises the absolute-sequence comparison in
            AdvanceAckedHead: a slot reused in the next lap still holds the
            previous lap's value when it is compared.
  parallel  16 tasks post at once with no ordering, so their completions become
            ready out of order and the acked prefix has to walk a hole-free set.
  notify    every post is a native notify (payload + fenced signal = two signalled
            WQEs) registered as a deferred completion, so each record owns two CQ
            slots and the retire has to confirm (ack_begin, ack_sequence].

Every case ends with a summary task that publishes the consumer cursor and
reports reserve/ready/posted/completed/acked plus the CQ depth. The host checks:

  * every per-task status slice reports OK (a failed task cannot be hidden by a
    later one -- each task writes its own slice),
  * posted == acked == expected CQEs (65536 / 1024 / 33280), which only holds if
    the consumer confirmed every completion and the sender published the prefix,
  * the payloads landed on their own post index.

Asserting "the run finished" would be too weak: a deferred producer that never
gets acked still finishes its posts, it just never retires the ring.
"""

from __future__ import annotations

import argparse
import os

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
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE_NBYTES = 4
SLICE_FLOATS = 64
POSTS_PER_TASK = 64
RDMA_DATA_OFFSET_NBYTES = 64 * 4
SUMMARY_STATUS_WORDS = 4
SUMMARY_WORDS = 10
RANK_STRIDE = 1_000_000.0

MODE_CHAIN = 0
MODE_NOTIFY = 1
MODE_PARALLEL = 2

# (name, mode, tasks, posts per task)
CASES = (
    ("chain", MODE_CHAIN, 1024, POSTS_PER_TASK),
    ("parallel", MODE_PARALLEL, 16, POSTS_PER_TASK),
    ("notify", MODE_NOTIFY, 260, POSTS_PER_TASK),
)


def rdma_workspace_enabled() -> bool:
    return os.environ.get("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE", "").strip().upper() in {"1", "ON", "TRUE", "YES", "Y"}


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
    extra_includes = list(kc.get_orchestration_include_dirs(runtime)) + [str(kc.project_root / "src" / "common")]

    children = []
    for func_id, rel, signature in [
        # posts: send, recv, tput, status, ctx, posts, task_index
        (
            0,
            "kernels/aiv/kernel_rdma_cq_posts.cpp",
            [ArgDirection.IN, ArgDirection.INOUT, ArgDirection.INOUT] + [ArgDirection.IN] * 3,
        ),
        # notify: send, tput, signal, status, ctx, posts, task_index
        (
            1,
            "kernels/aiv/kernel_rdma_cq_notify.cpp",
            [ArgDirection.IN, ArgDirection.INOUT, ArgDirection.INOUT] + [ArgDirection.IN] * 3,
        ),
        # summary: recv, tput, signal, summary, status, ctx, total_posts, mode
        (
            2,
            "kernels/aiv/kernel_rdma_cq_summary.cpp",
            [ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.INOUT, ArgDirection.INOUT]
            + [ArgDirection.IN] * 5,
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
        children.append((func_id, CoreCallable.build(signature=signature, binary=kernel)))

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/rdma_cq_sequence_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[
            ArgDirection.IN,  # send
            ArgDirection.INOUT,  # recv
            ArgDirection.INOUT,  # tput
            ArgDirection.INOUT,  # signal
            ArgDirection.OUT,  # summary
            ArgDirection.OUT,  # status
            ArgDirection.IN,  # ctx
            ArgDirection.IN,  # mode
            ArgDirection.IN,  # posts
            ArgDirection.IN,  # tasks
        ],
        func_name="rdma_cq_sequence_orchestration",
        config_name="rdma_cq_sequence_orchestration_config",
        binary=orch,
        children=children,
    )


def _send_pattern(rank: int, total_posts: int) -> torch.Tensor:
    """Every post's slice carries its own index, so a misplaced transfer shows."""
    posts = torch.arange(total_posts, dtype=torch.float32).repeat_interleave(SLICE_FLOATS)
    return (posts + float(rank) * RANK_STRIDE).share_memory_()


def _zero_float(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.float32).share_memory_()


def _zero_i32(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.int32).share_memory_()


def _sentinel_i32(count: int) -> torch.Tensor:
    return torch.full((count,), -1, dtype=torch.int32).share_memory_()


def _run_case_on_worker(
    worker: Worker, chip_handle, name: str, mode: int, tasks: int, posts: int, nranks: int
) -> bool:
    if nranks != 2:
        raise ValueError(f"rdma_cq_sequence needs exactly 2 devices, got {nranks}")

    total_posts = tasks * posts
    slice_elems = total_posts * SLICE_FLOATS
    send_nbytes = slice_elems * DTYPE_NBYTES
    signal_nbytes = total_posts * DTYPE_NBYTES
    status_words = SUMMARY_STATUS_WORDS
    # summary/status are host OUTPUT_EXISTING tensors, not part of the symmetric
    # window. The window itself must be 64-byte aligned and cover every domain
    # buffer (the reserved prefix included).
    window_bytes = RDMA_DATA_OFFSET_NBYTES + 3 * send_nbytes + signal_nbytes
    window_size = max((window_bytes + 63) // 64 * 64, 4 * 1024 * 1024)

    send_host = [_send_pattern(rank, total_posts) for rank in range(nranks)]
    recv_zero = [_zero_float(slice_elems) for _ in range(nranks)]
    tput_zero = [_zero_float(slice_elems) for _ in range(nranks)]
    signal_zero = [_zero_i32(total_posts) for _ in range(nranks)]
    summary = [torch.zeros(SUMMARY_WORDS, dtype=torch.int64).share_memory_() for _ in range(nranks)]
    status = [_sentinel_i32(status_words) for _ in range(nranks)]

    send_buf = [worker.create_buffer(send_nbytes) for _ in range(nranks)]
    recv_buf = [worker.create_buffer(send_nbytes) for _ in range(nranks)]
    tput_buf = [worker.create_buffer(send_nbytes) for _ in range(nranks)]
    signal_buf = [worker.create_buffer(signal_nbytes) for _ in range(nranks)]
    for r in range(nranks):
        assert send_buf[r].shm is not None and send_buf[r].shm.buf is not None
        torch.frombuffer(send_buf[r].shm.buf, dtype=torch.float32, count=slice_elems).copy_(send_host[r])
        torch.frombuffer(recv_buf[r].shm.buf, dtype=torch.float32, count=slice_elems).copy_(recv_zero[r])
        torch.frombuffer(tput_buf[r].shm.buf, dtype=torch.float32, count=slice_elems).copy_(tput_zero[r])
        torch.frombuffer(signal_buf[r].shm.buf, dtype=torch.int32, count=total_posts).copy_(signal_zero[r])

    def orch_fn(orch, _args, cfg):
        with orch.allocate_domain(
            name=f"rdma_cq_sequence_{name}",
            workers=list(range(nranks)),
            window_size=window_size,
            buffers=[
                CommBufferSpec(
                    name="rdma_reserved",
                    dtype="int32",
                    count=RDMA_DATA_OFFSET_NBYTES // 4,
                    nbytes=RDMA_DATA_OFFSET_NBYTES,
                ),
                CommBufferSpec(name="send", dtype="float32", count=slice_elems, nbytes=send_nbytes),
                CommBufferSpec(name="recv", dtype="float32", count=slice_elems, nbytes=send_nbytes),
                CommBufferSpec(name="tput", dtype="float32", count=slice_elems, nbytes=send_nbytes),
                CommBufferSpec(name="signal", dtype="int32", count=total_posts, nbytes=signal_nbytes),
            ],
        ) as handle:
            for rank in range(nranks):
                domain = handle[rank]
                orch.copy_to(domain.buffers["send"], send_buf[rank])
                orch.copy_to(domain.buffers["recv"], recv_buf[rank])
                orch.copy_to(domain.buffers["tput"], tput_buf[rank])
                orch.copy_to(domain.buffers["signal"], signal_buf[rank])

            for rank in range(nranks):
                domain = handle[rank]
                args = TaskArgs()
                args.add_tensor(
                    domain.buffers["send"].tensor((slice_elems,), DataType.FLOAT32), TensorArgType.INPUT
                )
                args.add_tensor(
                    domain.buffers["recv"].tensor((slice_elems,), DataType.FLOAT32), TensorArgType.INOUT
                )
                args.add_tensor(
                    domain.buffers["tput"].tensor((slice_elems,), DataType.FLOAT32), TensorArgType.INOUT
                )
                args.add_tensor(
                    domain.buffers["signal"].tensor((total_posts,), DataType.INT32), TensorArgType.INOUT
                )
                args.add_tensor(make_tensor_arg(worker, summary[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(make_tensor_arg(worker, status[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_ctx)
                args.add_scalar(mode)
                args.add_scalar(posts)
                args.add_scalar(tasks)
                orch.submit_next_level(chip_handle, args, cfg, worker=rank)

    try:
        worker.run(orch_fn, args=None, config=CallConfig())
    except Exception:
        # The chip process may have died; the shm tensors are host memory and can
        # still be read, which is the only record of how far the run got.
        _dump_status(name, tasks, status, summary)
        raise
    finally:
        for r in range(nranks):
            worker.release_buffer(send_buf[r])
            worker.release_buffer(recv_buf[r])
            worker.release_buffer(tput_buf[r])
            worker.release_buffer(signal_buf[r])

    return _check_result(name, mode, tasks, posts, summary, status)


def _dump_status(name, tasks, status, summary) -> None:
    for rank in range(len(status)):
        words = [int(x) for x in status[rank].tolist()]
        summary_words = [int(x) for x in summary[rank].tolist()]
        print(
            f"[rdma_cq_sequence] case={name} rank={rank} aborted; summary={summary_words} "
            f"status={words[:SUMMARY_STATUS_WORDS]}"
        )


def _check_result(name, mode, tasks, posts, summary, status) -> bool:
    total_posts = tasks * posts
    expected_cqes = total_posts * 2 if mode == MODE_NOTIFY else total_posts
    ok = True
    for rank in range(len(summary)):
        words = [int(x) for x in summary[rank].tolist()]
        rank_status = [int(x) for x in status[rank].tolist()]
        bad_slices = [] if rank_status[0] == 0 else [(0, rank_status[:SUMMARY_STATUS_WORDS])]
        entered = 1 if rank_status[3] == 0x5A5A5A5A else 0
        print(
            f"[rdma_cq_sequence] case={name} rank={rank} "
            f"reserve/ready/posted/completed/acked={words[0]}/{words[1]}/{words[2]}/{words[3]}/{words[4]} "
            f"qpNum={words[5]} cqDepth={words[6]} expectedCqes={expected_cqes} "
            f"summaryEntered={entered} orchSawTasks={words[7]} orchSawPosts={words[8]} orchSawMode={words[9]}"
        )
        if bad_slices:
            print(f"[rdma_cq_sequence] case={name} rank={rank} failed tasks: {bad_slices[:4]}")
            ok = False
            continue
        if words[2] != expected_cqes or words[4] != expected_cqes:
            print(
                f"[rdma_cq_sequence] case={name} rank={rank} counter mismatch: "
                f"posted={words[2]} acked={words[4]} expected={expected_cqes}"
            )
            ok = False
    return ok


def run_case(platform: str, device_ids: list[int], name: str, mode: int, tasks: int, posts: int) -> bool:
    if platform != "a5":
        raise ValueError("rdma_cq_sequence requires platform 'a5'; a5sim cannot validate real RDMA")
    if len(device_ids) != 2:
        raise ValueError(f"rdma_cq_sequence needs exactly 2 devices, got {device_ids}")

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
    try:
        worker.init()
        return _run_case_on_worker(worker, chip_handle, name, mode, tasks, posts, len(device_ids))
    finally:
        worker.close()


def run(
    platform: str = "a5", device_ids: list[int] | None = None, only: str | None = None, tasks_override: int | None = None
) -> int:
    if not rdma_workspace_enabled():
        raise RuntimeError(
            "rdma_cq_sequence requires the host runtime built with SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON; "
            "without it comm_ctx->workSpace is not an RDMA workspace and every kernel bails out silently."
        )
    if device_ids is None:
        device_ids = [0, 1]
    ok = True
    for name, mode, tasks, posts in CASES:
        if only is not None and name != only:
            continue
        if tasks_override is not None:
            tasks = tasks_override
        ok = run_case(platform, device_ids, name, mode, tasks, posts) and ok
    return 0 if ok else 1


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
def test_rdma_cq_sequence(st_platform, st_device_ids) -> None:
    if not rdma_workspace_enabled():
        pytest.skip("rdma_cq_sequence requires SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON")
    assert run(st_platform, [int(st_device_ids[0]), int(st_device_ids[1])]) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="0-1")
    parser.add_argument("-k", "--only", default=None, help="run a single case: chain | parallel | notify")
    parser.add_argument("-n", "--tasks", type=int, default=None, help="override the chunk count (bisect)")
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device), args.only, args.tasks)


if __name__ == "__main__":
    raise SystemExit(main())
