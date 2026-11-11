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

// Three ways to drive the same sequence, selected by the mode scalar:
//
//   0  chain     one task at a time, chained by an ordering dependency, so the
//                shared QP sees one strictly monotone sequence.
//   1  notify    same chain, but every post is a native notify (two signalled
//                WQEs) registered as a deferred completion by the kernel.
//   2  parallel  one task per AIV, no ordering, so the completions of 16 tasks
//                become ready out of order and the ack prefix has to walk them.
//
// Every mode ends with a summary task that publishes the consumer cursor and
// checks both the counters and the payloads.

#include <stdint.h>

#include "orchestration_api.h"
#include "platform_comm/comm_context.h"

namespace {

constexpr uint32_t kFuncPosts = 0;
constexpr uint32_t kFuncNotify = 1;
constexpr uint32_t kFuncSummary = 2;

constexpr uint64_t kModeChain = 0;
constexpr uint64_t kModeNotify = 1;
constexpr uint64_t kModeParallel = 2;

constexpr int kTensorSend = 0;
constexpr int kTensorRecv = 1;
constexpr int kTensorTput = 2;
constexpr int kTensorSignal = 3;
constexpr int kTensorSummary = 4;
constexpr int kTensorStatus = 5;
constexpr int kScalarCommCtx = 0;
constexpr int kScalarMode = 1;
constexpr int kScalarPosts = 2;
constexpr int kScalarTasks = 3;

// Each task reports into its own four-word status slice (status, two details,
// and a "kernel entered" marker), so a later task cannot overwrite an earlier
// task's failure. The kernel addresses the slice as status + index * 4; slicing
// the tensor here would need a view per task, and views over a host staging
// tensor are not a supported path.
constexpr uint32_t kStatusWordsPerTask = 4;
// The parallel mode keeps one task id per chunk to hand to the summary; the
// chain modes only keep the previous one.
constexpr uint32_t kMaxParallelTasks = 64;
constexpr uint32_t kMaxChainTasks = 4096;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
rdma_cq_sequence_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 10};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return rdma_cq_sequence_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void rdma_cq_sequence_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() != 6 || orch_args.scalar_count() != 4) {
        LOG_ERROR("rdma_cq_sequence: expected 6 tensors and 4 scalars");
        return;
    }

    const simpler::tmr::Tensor &send = orch_args.tensor(kTensorSend).ref();
    const simpler::tmr::Tensor &recv = orch_args.tensor(kTensorRecv).ref();
    const simpler::tmr::Tensor &tput = orch_args.tensor(kTensorTput).ref();
    const simpler::tmr::Tensor &signal = orch_args.tensor(kTensorSignal).ref();
    const simpler::tmr::Tensor &summary = orch_args.tensor(kTensorSummary).ref();
    const simpler::tmr::Tensor &status_all = orch_args.tensor(kTensorStatus).ref();
    auto *comm_ctx = reinterpret_cast<CommContext *>(orch_args.scalar<uintptr_t>(kScalarCommCtx));
    const uint64_t mode = orch_args.scalar<uint64_t>(kScalarMode);
    const uint64_t posts = orch_args.scalar<uint64_t>(kScalarPosts);
    const uint64_t tasks = orch_args.scalar<uint64_t>(kScalarTasks);
    const uint32_t task_limit = mode == kModeParallel ? kMaxParallelTasks : kMaxChainTasks;
    if (tasks == 0 || tasks > task_limit) {
        LOG_ERROR("rdma_cq_sequence: task count out of range");
        return;
    }
    const uint32_t func = mode == kModeNotify ? kFuncNotify : kFuncPosts;

    // Chain modes keep exactly one task on the queue; the parallel mode leaves
    // every chunk independent so the scheduler spreads them across AIVs.
    TaskId previous = TaskId::invalid();
    TaskId submitted[kMaxParallelTasks];
    uint32_t submitted_count = 0;

    for (uint64_t chunk = 0; chunk < tasks; ++chunk) {
        CoreTaskArgsWithDeps<> task_args;
        task_args.add_input(send);
        if (mode == kModeNotify) {
            task_args.add_inout(tput);
            task_args.add_inout(signal);
        } else {
            task_args.add_inout(recv);
            task_args.add_inout(tput);
        }
        task_args.add_inout(status_all);
        task_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
        task_args.add_scalar(posts);
        task_args.add_scalar(chunk);
        if (mode != kModeParallel && previous.is_valid()) {
            task_args.add_dep_wait(previous);
        }
        const TaskId task = rt_submit_aiv_task(func, task_args).task_id();
        if (mode != kModeParallel) {
            previous = task;
        } else {
            submitted[submitted_count++] = task;
        }
    }

    CoreTaskArgsWithDeps<> summary_args;
    summary_args.add_input(recv);
    summary_args.add_input(tput);
    summary_args.add_input(signal);
    summary_args.add_inout(summary);
    summary_args.add_inout(status_all);
    summary_args.add_scalar(reinterpret_cast<uint64_t>(comm_ctx));
    summary_args.add_scalar(posts * tasks);
    summary_args.add_scalar(mode);
    // The summary reports into the slice after the last chunk.
    summary_args.add_scalar(tasks);
    if (mode == kModeParallel) {
        for (uint32_t i = 0; i < submitted_count; ++i) {
            summary_args.add_dep_wait(submitted[i]);
        }
    } else if (previous.is_valid()) {
        summary_args.add_dep_wait(previous);
    }
    rt_submit_aiv_task(kFuncSummary, summary_args);
}

}  // extern "C"
