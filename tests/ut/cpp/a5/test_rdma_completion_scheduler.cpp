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
#include <gtest/gtest.h>

#include "backend/rdma/rdma_completion_scheduler.h"

namespace {

using pto2::rdma_backend::Hns1825Cqe;
using pto2::rdma_backend::is_hns1825_cqe_owner_ready;
using pto2::rdma_backend::kCqeBytes;
using pto2::rdma_backend::poll_rdma_hns1825_cqe_record;

constexpr uint32_t kTestCqDepth = 1024;

struct alignas(kCqeBytes) TestRdmaCqe : Hns1825Cqe {};

inline bool test_cqe_ready_owner(uint32_t cqe_seq) { return (cqe_seq & kTestCqDepth) != 0; }

inline uint64_t cqe_addr(TestRdmaCqe &cqe) { return reinterpret_cast<uint64_t>(&cqe); }

inline void encode_test_cqe(TestRdmaCqe &cqe, uint32_t cqe_seq, uint32_t opcode) {
    constexpr uint32_t kOwnerShift = 31;
    constexpr uint32_t kCqeOpcodeShift = 27;
    const uint32_t owner = test_cqe_ready_owner(cqe_seq) ? 1u : 0u;
    cqe.owner_id_qpn = owner << kOwnerShift;
    cqe.op_sr_wqebb = opcode << kCqeOpcodeShift;
    cqe.syndrome = opcode == 0x1e ? 9 : 0;
}

inline void encode_test_cqe_with_owner(TestRdmaCqe &cqe, uint32_t owner, uint32_t opcode) {
    constexpr uint32_t kOwnerShift = 31;
    constexpr uint32_t kCqeOpcodeShift = 27;
    cqe.owner_id_qpn = owner << kOwnerShift;
    cqe.op_sr_wqebb = opcode << kCqeOpcodeShift;
    cqe.syndrome = opcode == 0x1e ? 9 : 0;
}

}  // namespace

TEST(A5RdmaCompletionScheduler, NullRecordAddressFails) {
    auto result = poll_rdma_hns1825_cqe_record(0, 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, OwnerReadyMatchesHns1825Backend) {
    EXPECT_TRUE(is_hns1825_cqe_owner_ready(false, 0, kTestCqDepth));
    EXPECT_FALSE(is_hns1825_cqe_owner_ready(true, 0, kTestCqDepth));
    EXPECT_TRUE(is_hns1825_cqe_owner_ready(true, kTestCqDepth, kTestCqDepth));
    EXPECT_FALSE(is_hns1825_cqe_owner_ready(false, kTestCqDepth, kTestCqDepth));
}

TEST(A5RdmaCompletionScheduler, InvalidExpectedOwnerFails) {
    TestRdmaCqe cqe{};
    encode_test_cqe_with_owner(cqe, 0, 0);
    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 2);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, OwnerNotReadyReturnsPending) {
    TestRdmaCqe cqe{};
    encode_test_cqe_with_owner(cqe, 1, 0);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, InvalidOpcodeReturnsPending) {
    TestRdmaCqe cqe{};
    encode_test_cqe_with_owner(cqe, 0, 0x1f);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, ErrorOpcodeFails) {
    TestRdmaCqe cqe{};
    encode_test_cqe_with_owner(cqe, 0, 0x1e);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, ReadyOpcodeReturnsReady) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, 0, 0);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, ReadyOpcodeAfterWrapMatchesPinnedOwnerPolarity) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, kTestCqDepth, 0);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 1);
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

static_assert(sizeof(TestRdmaCqe) == kCqeBytes);
