#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the deferred ack prefix over a long mixed TGET/TPUT sequence.

Each rank chained 16 AIV tasks; a task posts 64 instructions and every
instruction registers its own deferred completion record. Tasks are ordered by
an explicit dependency, so a single sequence task owns the shared PER_PEER queue
at a time and the whole run is one monotonically increasing sequence.

Instructions alternate TGET (into out) and TPUT (out into the peer scratch),
which puts both opcodes on the same CQ. The run posts 16 * 64 = 1024 WQEs, i.e.
it crosses the ring several times for any CQ depth up to 512.

The summary task reports:

  summary[0] = submittedWqeCount        (expected 1024)
  summary[1] = AdvanceAcked(...)        (expected 1024)
  summary[2] = 1  two or more laps crossed
               2  one lap crossed
               3  the ring never wrapped
               0  submitted/acked mismatch

summary[2] == 2 or 3 means the run was too short to reuse slots; raise
kNTaskChunks in urma_cq_sequence_orch.cpp. TPUT data is not checked here: the
test asserts the completion sequence, not the write path.
"""

import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

NRANKS = 2
DTYPE_NBYTES = 4
ELEMS_PER_POST = 128 * 128
POSTS_PER_TASK = 64
NTASK_CHUNKS = 16
N = ELEMS_PER_POST * POSTS_PER_TASK * NTASK_CHUNKS
EXPECTED_WQES = POSTS_PER_TASK * NTASK_CHUNKS
URMA_DATA_OFFSET_NBYTES = 64 * DTYPE_NBYTES
_URMA_WORKSPACE_ENV = "SIMPLER_ENABLE_PTO_URMA_WORKSPACE"
_WORKSPACE_TRUTHY = {"1", "ON", "TRUE", "YES"}


def _env_enabled(name):
    return os.environ.get(name, "").upper() in _WORKSPACE_TRUTHY


def _require_env_enabled():
    if not _env_enabled(_URMA_WORKSPACE_ENV):
        raise RuntimeError(
            "urma_cq_sequence requires host runtime built with "
            f"{_URMA_WORKSPACE_ENV}=ON; set it before rebuilding simpler."
        )


def urma_cq_sequence_orch_fn(orch, callables, task_args, config):
    _require_env_enabled()
    input_nbytes = N * DTYPE_NBYTES
    with orch.allocate_domain(
        name="urma_cq_sequence",
        workers=list(range(NRANKS)),
        window_size=URMA_DATA_OFFSET_NBYTES + 2 * input_nbytes + 4096,
        buffers=[
            CommBufferSpec(
                name="urma_reserved",
                dtype="int32",
                count=URMA_DATA_OFFSET_NBYTES // DTYPE_NBYTES,
                nbytes=URMA_DATA_OFFSET_NBYTES,
            ),
            CommBufferSpec(name="input_window", dtype="float32", count=N, nbytes=input_nbytes),
            CommBufferSpec(name="scratch_window", dtype="float32", count=N, nbytes=input_nbytes),
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
            args.add_tensor(
                domain.buffers["input_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT
            )
            args.add_tensor(
                domain.buffers["scratch_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"summary_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_ctx)
            args.add_scalar(POSTS_PER_TASK)
            orch.submit_next_level(callables.urma_cq_sequence, args, config, worker=rank)


class _UrmaCqSequenceBase(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_cq_sequence_orch_fn,
        "callables": [
            {
                "name": "urma_cq_sequence",
                "orchestration": {
                    "source": "kernels/orchestration/urma_cq_sequence_orch.cpp",
                    "function_name": "urma_cq_sequence_orchestration",
                    "config_name": "urma_cq_sequence_orchestration_config",
                    "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_cq_sequence.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_urma_ack_summary.cpp",
                        "core_type": "aiv",
                        "signature": [D.OUT, D.IN, D.IN],
                    },
                ],
            }
        ],
    }
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        specs = []
        # Vectorized: a Python loop over 16M elements would take minutes.
        idx = torch.arange(N, dtype=torch.float32)
        for rank in range(NRANKS):
            inp = ((idx % 251) + rank * 1000.0) / 10.0
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(N, dtype=torch.float32)),
                    TensorArg(f"summary_{rank}", torch.zeros(3, dtype=torch.int32)),
                ]
            )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        slots = torch.arange(NTASK_CHUNKS * POSTS_PER_TASK)
        # Post i of a task is a TGET when i is even and a TPUT when it is odd.
        tget = ((slots % POSTS_PER_TASK) % 2) == 0
        for rank in range(NRANKS):
            peer_in = getattr(args, f"in_{1 - rank}").view(NTASK_CHUNKS * POSTS_PER_TASK, ELEMS_PER_POST)
            golden = torch.where(tget.unsqueeze(1), peer_in, torch.zeros_like(peer_in))
            getattr(args, f"out_{rank}").copy_(golden.reshape(-1))

            summary = getattr(args, f"summary_{rank}")
            summary[0] = EXPECTED_WQES
            summary[1] = EXPECTED_WQES
            summary[2] = 1


@pytest.mark.skipif(
    not _env_enabled(_URMA_WORKSPACE_ENV),
    reason="URMA workspace overlay is not enabled. "
    f"Set {_URMA_WORKSPACE_ENV}=ON before rebuilding simpler.",
)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaCqSequenceAckRange(_UrmaCqSequenceBase):
    CASES = [
        {
            "name": "sequence_mixed_ops",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {},
        }
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
