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

#include <stdint.h>

#include "platform_comm/comm_context.h"
#include "orchestration_api.h"

namespace {

// Each sequence task posts POSTS_PER_TASK WQEs; the tasks are chained with an
// ordering dependency so exactly one sequence task posts to the shared PER_PEER
// queue at a time.
constexpr uint32_t kNTaskChunks = 16;
constexpr uint32_t kPostsPerTask = 64;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
urma_cq_sequence_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return urma_cq_sequence_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void urma_cq_sequence_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() != 4 || orch_args.scalar_count() != 2) {
        LOG_ERROR("urma_cq_sequence: expected 4 tensors and 2 scalars");
        return;
    }

    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &scratch = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &out = orch_args.tensor(2).ref();
    const simpler::tmr::Tensor &summary = orch_args.tensor(3).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(0));
    const uint64_t posts = orch_args.scalar<uint64_t>(1);

    TaskId previous = TaskId::invalid();
    for (uint32_t chunk = 0; chunk < kNTaskChunks; ++chunk) {
        CoreTaskArgsWithDeps<> sequence_args;
        sequence_args.add_input(input);
        sequence_args.add_input(scratch);
        sequence_args.add_inout(out);
        sequence_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
        sequence_args.add_scalar(posts);
        sequence_args.add_scalar(static_cast<uint64_t>(chunk));
        if (previous.is_valid()) {
            sequence_args.add_dep_wait(previous);
        }
        previous = rt_submit_aiv_task(0, sequence_args).task_id();
    }

    CoreTaskArgsWithDeps<> summary_args;
    summary_args.add_inout(summary);
    summary_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    summary_args.add_scalar(posts * kNTaskChunks);
    if (previous.is_valid()) {
        summary_args.add_dep_wait(previous);
    }
    rt_submit_aiv_task(1, summary_args);
}

}  // extern "C"
