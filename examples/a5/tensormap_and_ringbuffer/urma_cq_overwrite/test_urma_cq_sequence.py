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

Each rank chains 64 AIV tasks; a task posts 64 instructions and every
instruction registers its own deferred completion record. Tasks are ordered by
an explicit dependency, so a single sequence task owns the shared PER_PEER queue
at a time and the whole run is one monotonically increasing sequence.

Instructions alternate TGET (into out) and TPUT (out into the peer scratch),
which puts both opcodes on the same CQ. The run posts 64 * 64 = 4096 WQEs.

The summary task reports:

  summary[0] = submittedWqeCount        (expected 4096)
  summary[1] = AdvanceAcked(...)        (expected 4096)
  summary[2] = cq->depth                (the device's CQ depth)

summary[2] is a property of the device, not of the run, so the check is a
relation: submitted and acked must both equal 4096, and 4096 must exceed twice
the reported depth, i.e. the run has to reuse CQ slots at least twice. A shorter
run fails with the measured depth in the message.

TPUT data is not checked here: the test asserts the completion sequence, not the
write path.
"""

import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _describe_mismatch, _rehosted_buffer_for, _rehosted_ref_for

NRANKS = 2
DTYPE_NBYTES = 4
ELEMS_PER_POST = 128 * 32
POSTS_PER_TASK = 64
NTASK_CHUNKS = 64
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
                        "signature": [D.OUT, D.IN],
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
            # summary[2] carries the device CQ depth, which the host cannot
            # predict; compare_outputs checks it as a relation instead.
            summary[2] = 0

    def compare_outputs(self, test_args, golden_args, output_names, params):
        for name in output_names:
            actual = getattr(test_args, name)
            expected = getattr(golden_args, name)
            if name.startswith("summary_"):
                submitted, acked, depth = (int(value) for value in actual[:3].tolist())
                assert submitted == EXPECTED_WQES, (
                    f"{name}: submittedWqeCount {submitted}, expected {EXPECTED_WQES}"
                )
                assert acked == EXPECTED_WQES, f"{name}: AdvanceAcked {acked}, expected {EXPECTED_WQES}"
                assert depth > 0, f"{name}: reported CQ depth {depth} is not positive"
                assert EXPECTED_WQES > 2 * depth, (
                    f"{name}: {EXPECTED_WQES} WQEs cover {EXPECTED_WQES / depth:.1f} laps of a "
                    f"depth-{depth} CQ; raise NTASK_CHUNKS past {2 * depth} WQEs"
                )
                continue
            assert torch.allclose(actual, expected, rtol=self.RTOL, atol=self.ATOL), (
                f"{name}: {_describe_mismatch(actual, expected, self.RTOL, self.ATOL)}"
            )


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
