/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

// Step 2 orchestrator: the full case's argument shape (6 tensors + 4 scalars),
// one task, no dependencies. The kernel only echoes scalars.

#include <stdint.h>

#include "orchestration_api.h"
#include "platform_comm/comm_context.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig probe2_orch_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 10};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return probe2_orch_config(orch_args);
}

__attribute__((visibility("default"))) void probe2_orch(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &send = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &recv = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &tput = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &signal = orch_args.tensor(3).ref();
    const simpler::tmr::Tensor &summary = orch_args.tensor(4).ref();
    const simpler::tmr::Tensor &status = orch_args.tensor(5).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(0));
    const uint64_t mode = orch_args.scalar<uint64_t>(1);
    const uint64_t posts = orch_args.scalar<uint64_t>(2);
    const uint64_t tasks = orch_args.scalar<uint64_t>(3);

    CoreTaskArgs args;
    args.add_input(send);
    args.add_inout(recv);
    args.add_inout(tput);
    args.add_inout(signal);
    args.add_inout(summary);
    args.add_output(status);
    args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    args.add_scalar(mode);
    args.add_scalar(posts);
    args.add_scalar(tasks);
    rt_submit_aiv_task(0, args);
}

}  // extern "C"
