#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Show the deferred ack gap when a single TGET slice exceeds one WQE.

A 257 MB payload is split by UrmaPostSliceOnJetty into two WQEs (sequence 1 and
2), but GetUrmaCompletionRecordAt publishes only one record with sequence 2.
The AICPU therefore writes only ack[slot(sequence 2)]. The prefix scan starts at
sequence 1, whose slot was never acked, so AdvanceAcked returns 0 even though
sequence 2 was retired.

The probe task runs after the victim and reports:
  probe[0] = submittedWqeCount   (expected 2)
  probe[1] = AdvanceAcked(...)   (expected 0 while the gap exists)
  probe[2] = ack ring slot 0     (expected 0)
  probe[3] = ack ring slot 1     (expected 2)
"""

import ctypes
import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, Scalar, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

# 257 MB of float: one slice that exceeds the 256 MB single-WQE cap.
N = 67371008
NRANKS = 2
DTYPE_NBYTES = 4
URMA_DATA_OFFSET_NBYTES = 64 * DTYPE_NBYTES
_URMA_WORKSPACE_ENV = "SIMPLER_ENABLE_PTO_URMA_WORKSPACE"
_TEST_HOOK_ENV = "SIMPLER_TEST_URMA_CQ_OVERWRITE"
_WORKSPACE_TRUTHY = {"1", "ON", "TRUE", "YES"}


def _env_enabled(name):
    return os.environ.get(name, "").upper() in _WORKSPACE_TRUTHY


def _require_env_enabled():
    if not _env_enabled(_URMA_WORKSPACE_ENV):
        raise RuntimeError(
            "urma_cq_chunked requires host runtime built with "
            f"{_URMA_WORKSPACE_ENV}=ON; set it before rebuilding simpler."
        )
    if not _env_enabled(_TEST_HOOK_ENV):
        raise RuntimeError(
            "urma_cq_chunked requires the test-only AICPU poll delay; build with "
            f"{_TEST_HOOK_ENV}=ON before rebuilding simpler."
        )


def urma_cq_chunked_orch_fn(orch, callables, task_args, config):
    _require_env_enabled()
    input_nbytes = N * DTYPE_NBYTES
    with orch.allocate_domain(
        name="urma_cq_chunked",
        workers=list(range(NRANKS)),
        window_size=max(URMA_DATA_OFFSET_NBYTES + input_nbytes + 4096, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(
                name="urma_reserved",
                dtype="int32",
                count=URMA_DATA_OFFSET_NBYTES // DTYPE_NBYTES,
                nbytes=URMA_DATA_OFFSET_NBYTES,
            ),
            CommBufferSpec(name="input_window", dtype="float32", count=N, nbytes=input_nbytes),
        ],
    ) as handle:
        for rank in range(NRANKS):
            orch.copy_to(
                handle[rank].buffers["input_window"],
                _rehosted_buffer_for(task_args, getattr(task_args, f"in_{rank}")),
            )
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"probe_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"flag_{rank}")), TensorArgType.INPUT
            )
            args.add_scalar(domain.device_ctx)
            orch.submit_next_level(callables.urma_cq_chunked, args, config, worker=rank)


class _UrmaCqChunkedBase(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_cq_chunked_orch_fn,
        "callables": [
            {
                "name": "urma_cq_chunked",
                "orchestration": {
                    "source": "kernels/orchestration/urma_cq_chunked_orch.cpp",
                    "function_name": "urma_cq_chunked_orchestration",
                    "config_name": "urma_cq_chunked_orchestration_config",
                    "signature": [D.IN, D.OUT, D.OUT, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_cq_victim_chunked.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_urma_ack_probe.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
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
            # Vectorized: a Python loop over 67M elements would take minutes.
            idx = torch.arange(N, dtype=torch.float32)
            inp = ((idx % 251) + rank * 1000.0) / 10.0
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(N, dtype=torch.float32)),
                    TensorArg(f"probe_{rank}", torch.zeros(4, dtype=torch.int32)),
                    TensorArg(f"flag_{rank}", torch.zeros(1, dtype=torch.int32)),
                ]
            )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            expected_out = getattr(args, f"in_{1 - rank}")
            getattr(args, f"out_{rank}").copy_(expected_out)

            # submitted=2 (two WQEs), acked=0 (slot 0 never acked),
            # ack[0]=0, ack[1]=2 (sequence 2 was retired and acked).
            probe = getattr(args, f"probe_{rank}")
            probe[0] = 2
            probe[1] = 0
            probe[2] = 0
            probe[3] = 2


@pytest.mark.skipif(
    not (_env_enabled(_URMA_WORKSPACE_ENV) and _env_enabled(_TEST_HOOK_ENV)),
    reason="URMA workspace overlay or the test-only AICPU poll delay is not enabled. "
    f"Set both {_URMA_WORKSPACE_ENV}=ON and {_TEST_HOOK_ENV}=ON before rebuilding simpler.",
)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaCqChunkedDeferredAckGap(_UrmaCqChunkedBase):
    CASES = [
        {
            "name": "chunked_deferred_gap",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {},
        }
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
