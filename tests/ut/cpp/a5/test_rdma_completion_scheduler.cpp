#include <gtest/gtest.h>

#include <cstring>

#include "backend/rdma/rdma_completion_scheduler.h"

namespace {

using pto2::rdma_backend::encode_rdma_event_handle;
using pto2::rdma_backend::is_rdma_error_handle;
using pto2::rdma_backend::kCqeBytes;
using pto2::rdma_backend::poll_rdma_event_handle;
using pto2::rdma_backend::RdmaCqCtx;
using pto2::rdma_backend::RdmaWqCtx;

constexpr uint32_t kTestCqDepth = 64;
constexpr uint32_t kTestCqeShift = 6;

struct alignas(kCqeBytes) TestRdmaCqe {
    uint32_t dw[16];
};

struct TestRdmaWorkspace {
    pto2::rdma_backend::RdmaInfo info;
    RdmaWqCtx sq[2];
    RdmaCqCtx scq[2];
    uint32_t sq_tail[2];
    uint32_t cq_tail[2];
    uint32_t cq_doorbell[2];
    TestRdmaCqe scq_entries[2][kTestCqDepth];
};

inline bool test_cqe_ready_owner(uint32_t cqe_seq) { return ((cqe_seq / kTestCqDepth) & 1u) == 0; }

inline uint32_t encode_test_cqe_dw0(uint32_t cqe_seq, uint8_t status, uint8_t substatus) {
    const uint32_t owner = test_cqe_ready_owner(cqe_seq) ? 1u : 0u;
    return (owner << 2) | (static_cast<uint32_t>(substatus) << 16) | (static_cast<uint32_t>(status) << 24);
}

inline uint32_t encode_test_pending_cqe_dw0(uint32_t cqe_seq) {
    const uint32_t owner = test_cqe_ready_owner(cqe_seq) ? 0u : 1u;
    return owner << 2;
}

struct SchedulerWorkspaceFixture {
    TestRdmaWorkspace ws{};

    SchedulerWorkspaceFixture() {
        std::memset(&ws, 0, sizeof(ws));
        ws.info.magic = pto2::rdma_backend::kRdmaWorkspaceMagic;
        ws.info.version = pto2::rdma_backend::kRdmaWorkspaceVersion;
        ws.info.backend = pto2::rdma_backend::kRdmaBackendHns1825;
        ws.info.qp_num = 1;
        ws.info.rank_count = 2;
        ws.info.sq_ptr = reinterpret_cast<uint64_t>(&ws.sq[0]);
        ws.info.scq_ptr = reinterpret_cast<uint64_t>(&ws.scq[0]);
        for (uint32_t rank = 0; rank < ws.info.rank_count; ++rank) {
            RdmaWqCtx &sq = ws.sq[rank];
            sq.wqe_shift_size = kTestCqeShift;
            sq.depth = kTestCqDepth;
            sq.tail_addr = reinterpret_cast<uint64_t>(&ws.sq_tail[rank]);

            RdmaCqCtx &cq = ws.scq[rank];
            cq.buf_addr = reinterpret_cast<uint64_t>(&ws.scq_entries[rank][0]);
            cq.cqe_shift_size = kTestCqeShift;
            cq.depth = kTestCqDepth;
            cq.tail_addr = reinterpret_cast<uint64_t>(&ws.cq_tail[rank]);
            cq.db_addr = reinterpret_cast<uint64_t>(&ws.cq_doorbell[rank]);
        }
    }

    uint64_t addr() { return reinterpret_cast<uint64_t>(&ws); }
};

}  // namespace

TEST(A5RdmaCompletionScheduler, ErrorHandleIsDetected) {
    EXPECT_TRUE(is_rdma_error_handle(encode_rdma_event_handle(0xffffffffu, 7)));
    EXPECT_FALSE(is_rdma_error_handle(encode_rdma_event_handle(1, 7)));
}

TEST(A5RdmaCompletionScheduler, ZeroHandleIsReady) {
    auto result = poll_rdma_event_handle(0, 0);
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, ErrorHandleFails) {
    auto result = poll_rdma_event_handle(encode_rdma_event_handle(0xffffffffu, 9), 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, NullWorkspaceFails) {
    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, InvalidWorkspaceMagicFails) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.info.magic = 0;

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, InvalidRankFails) {
    SchedulerWorkspaceFixture fixture;
    auto result = poll_rdma_event_handle(encode_rdma_event_handle(3, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, TailAlreadyAtTargetIsReady) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.cq_tail[1] = 7;

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 7), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5RdmaCompletionScheduler, OwnerNotReadyReturnsPending) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.scq_entries[1][0].dw[0] = encode_test_pending_cqe_dw0(0);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], 0u);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5RdmaCompletionScheduler, ReadyCqeAdvancesCqDoorbellAndSqTail) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.scq_entries[1][0].dw[0] = encode_test_cqe_dw0(0, 0, 0);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], 1u);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 1u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 1u);
}

TEST(A5RdmaCompletionScheduler, CqeErrorFailsAfterRetiringCqe) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.scq_entries[1][0].dw[0] = encode_test_cqe_dw0(0, 2, 9);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.ws.cq_tail[1], 1u);
    EXPECT_EQ(fixture.ws.cq_doorbell[1], 1u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 1u);
}

static_assert(sizeof(TestRdmaCqe) == kCqeBytes);
