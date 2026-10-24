#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproduce a URMA CQ slot overwrite before the deferred consumer polls it.

Each rank submits one victim URMA TGET and registers its CQE. A filler task
then posts TGETs until the CQ wraps, reusing the victim slot. The AICPU poll is
delayed by a test-only build hook so the overwrite lands before the victim
completion is observed.

The filler reads the actual CQ depth from the workspace. ``wrap=0`` posts
``depth - 1`` filler WQEs, leaving the ring one entry short of a wrap, and must
pass on both sides of the fix. ``wrap=1`` posts ``depth`` filler WQEs, so the
last post is the first wrap: without the ack gating the victim completion stays
PENDING and the consumer never runs; with the gating the filler blocks until
the victim CQE is acknowledged. Stopping at the first wrap also keeps the CQE
owner bit inverted, avoiding an ABA false positive.
"""

import ctypes
import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, Scalar, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

N = 128 * 128
NRANKS = 2
DTYPE_NBYTES = 4
URMA_DATA_OFFSET_NBYTES = 64 * DTYPE_NBYTES
SCRATCH_ELEMS = 1024
_URMA_WORKSPACE_ENV = "SIMPLER_ENABLE_PTO_URMA_WORKSPACE"
_TEST_HOOK_ENV = "SIMPLER_TEST_URMA_CQ_OVERWRITE"
_WORKSPACE_TRUTHY = {"1", "ON", "TRUE", "YES"}


def _env_enabled(name):
    return os.environ.get(name, "").upper() in _WORKSPACE_TRUTHY


def _require_env_enabled():
    if not _env_enabled(_URMA_WORKSPACE_ENV):
        raise RuntimeError(
            "urma_cq_overwrite requires host runtime built with "
            f"{_URMA_WORKSPACE_ENV}=ON; set it before rebuilding simpler."
        )
    if not _env_enabled(_TEST_HOOK_ENV):
        raise RuntimeError(
            "urma_cq_overwrite requires the test-only AICPU poll delay; build with "
            f"{_TEST_HOOK_ENV}=ON before rebuilding simpler."
        )


def urma_cq_overwrite_orch_fn(orch, callables, task_args, config):
    _require_env_enabled()
    input_nbytes = N * DTYPE_NBYTES
    scratch_nbytes = SCRATCH_ELEMS * DTYPE_NBYTES
    wrap = int(task_args.wrap.value)
    with orch.allocate_domain(
        name="urma_cq_overwrite",
        workers=list(range(NRANKS)),
        window_size=max(URMA_DATA_OFFSET_NBYTES + input_nbytes + scratch_nbytes + 4, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(
                name="urma_reserved",
                dtype="int32",
                count=URMA_DATA_OFFSET_NBYTES // DTYPE_NBYTES,
                nbytes=URMA_DATA_OFFSET_NBYTES,
            ),
            CommBufferSpec(name="input_window", dtype="float32", count=N, nbytes=input_nbytes),
            CommBufferSpec(name="scratch", dtype="float32", count=SCRATCH_ELEMS, nbytes=scratch_nbytes),
            CommBufferSpec(name="flag", dtype="int32", count=1, nbytes=4),
        ],
    ) as handle:
        for rank in range(NRANKS):
            orch.copy_to(
                handle[rank].buffers["input_window"],
                _rehosted_buffer_for(task_args, getattr(task_args, f"in_{rank}")),
            )
            orch.copy_to(
                handle[rank].buffers["flag"],
                _rehosted_buffer_for(task_args, getattr(task_args, f"flag_init_{rank}")),
            )
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(domain.buffers["flag"].tensor((1,), DataType.INT32), TensorArgType.INPUT)
            args.add_tensor(
                domain.buffers["scratch"].tensor((SCRATCH_ELEMS,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_ctx)
            args.add_scalar(wrap)
            orch.submit_next_level(callables.urma_cq_overwrite, args, config, worker=rank)


class _UrmaCqOverwriteBase(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_cq_overwrite_orch_fn,
        "callables": [
            {
                "name": "urma_cq_overwrite",
                "orchestration": {
                    "source": "kernels/orchestration/urma_cq_overwrite_orch.cpp",
                    "function_name": "urma_cq_overwrite_orchestration",
                    "config_name": "urma_cq_overwrite_orchestration_config",
                    "signature": [D.IN, D.OUT, D.OUT, D.IN, D.OUT, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_cq_victim.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_urma_cq_filler.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN, D.IN, D.IN],
                    },
                    {
                        "func_id": 2,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                ],
            }
        ],
    }
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            inp = torch.tensor([float(rank * 1000 + (i % 251)) / 10.0 for i in range(N)], dtype=torch.float32)
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(N, dtype=torch.float32)),
                    TensorArg(f"result_{rank}", torch.zeros(N, dtype=torch.float32)),
                    TensorArg(f"flag_init_{rank}", torch.zeros(1, dtype=torch.int32)),
                ]
            )
        specs.append(Scalar("wrap", ctypes.c_int64(int(params["wrap"]))))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            expected_out = getattr(args, f"in_{1 - rank}")
            getattr(args, f"out_{rank}").copy_(expected_out)
            getattr(args, f"result_{rank}").copy_(expected_out + 1.0)


@pytest.mark.skipif(
    not (_env_enabled(_URMA_WORKSPACE_ENV) and _env_enabled(_TEST_HOOK_ENV)),
    reason="URMA workspace overlay or the test-only AICPU poll delay is not enabled. "
    f"Set both {_URMA_WORKSPACE_ENV}=ON and {_TEST_HOOK_ENV}=ON before rebuilding simpler.",
)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaCqOverwriteBaseline(_UrmaCqOverwriteBase):
    CASES = [
        {
            "name": "baseline",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"wrap": 0},
        }
    ]


@pytest.mark.skipif(
    not (_env_enabled(_URMA_WORKSPACE_ENV) and _env_enabled(_TEST_HOOK_ENV)),
    reason="URMA workspace overlay or the test-only AICPU poll delay is not enabled. "
    f"Set both {_URMA_WORKSPACE_ENV}=ON and {_TEST_HOOK_ENV}=ON before rebuilding simpler.",
)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaCqOverwriteWrap(_UrmaCqOverwriteBase):
    CASES = [
        {
            "name": "overwrite",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"wrap": 1},
        }
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
