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

#include <cstdint>

#include "backend/rdma/rdma_completion_scheduler.h"

using pto2::rdma_backend::kRdmaCqeOpcodeError;
using pto2::rdma_backend::kRdmaCqeOpcodeInvalid;
using pto2::rdma_backend::poll_rdma_hns1825_cqe_record;

namespace {

constexpr uint32_t kSendOpcode = 0U;

// The poll reads the CQE's first 8 bytes as one little-endian word: dw0 in the low half (owner at
// bit 31) and dw1 in the high half (opcode at its own bit 27).
uint64_t make_cqe_dw01(uint32_t owner, uint32_t opcode) {
    const uint64_t dw0 = static_cast<uint64_t>(owner) << 31U;
    const uint64_t dw1 = static_cast<uint64_t>(opcode) << 27U;
    return dw0 | (dw1 << 32U);
}

}  // namespace

TEST(A5RdmaCompletionScheduler, PollsOneCqeOwnerAndOpcodeWithoutWritingIt) {
    alignas(CHIP_ALIGN_SIZE) uint64_t cqe = 0;

    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(/*cqe_addr=*/0, /*expected_owner=*/0).state, CompletionPollState::FAILED
    );
    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/2).state,
        CompletionPollState::FAILED
    );

    // An invalid opcode is a slot the NIC has not written yet, whatever the owner bit says.
    cqe = make_cqe_dw01(1U, kRdmaCqeOpcodeInvalid);
    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/1).state,
        CompletionPollState::PENDING
    );
    EXPECT_EQ(cqe, make_cqe_dw01(1U, kRdmaCqeOpcodeInvalid));

    // Owner still at the value the slot started the lap with: not done.
    cqe = make_cqe_dw01(0U, kSendOpcode);
    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/1).state,
        CompletionPollState::PENDING
    );

    // Owner flipped to the expected value with a valid opcode: done.
    cqe = make_cqe_dw01(1U, kSendOpcode);
    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/1).state,
        CompletionPollState::READY
    );
    EXPECT_EQ(cqe, make_cqe_dw01(1U, kSendOpcode));

    // Expected owner 0 is the mirror case, and it must not accept a slot stamped 1.
    cqe = make_cqe_dw01(1U, kSendOpcode);
    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/0).state,
        CompletionPollState::PENDING
    );
    cqe = make_cqe_dw01(0U, kSendOpcode);
    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/0).state,
        CompletionPollState::READY
    );

    // An error CQE is a completed-but-failed transfer, not a pending one.
    cqe = make_cqe_dw01(1U, kRdmaCqeOpcodeError);
    EXPECT_EQ(
        poll_rdma_hns1825_cqe_record(reinterpret_cast<uint64_t>(&cqe), /*expected_owner=*/1).state,
        CompletionPollState::FAILED
    );
    EXPECT_EQ(cqe, make_cqe_dw01(1U, kRdmaCqeOpcodeError));
}
