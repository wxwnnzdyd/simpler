# A5 URMA Deferred Completion Flow

本文说明 A5 URMA deferred completion 的完整执行链路：从 Host 初始化
URMA workspace，到 AICore 提交 `TGET_ASYNC` / `TPUT_ASYNC`，再到 AICPU
scheduler 轮询真实 URMA CQE 并释放依赖任务。

核心目标不是单独写一个 demo，而是把 PTO-ISA 原生 URMA async event 接入
simpler 现有 deferred completion 框架。

## Problem

PTO-ISA URMA 可以在 AICore 上发起异步请求：

```cpp
event = TGET_ASYNC(...);
event = TPUT_ASYNC(...);
```

如果 AICore 立即调用 `event.Wait(session)`，AICore 会阻塞在硬件完成等待上。
这不符合 deferred completion 的目标：AICore 只提交请求，完成检测交给 AICPU
scheduler 统一管理。

本实现采用的链路是：

```text
AICore submits a URMA async request
  -> AICore records a deferred completion token
  -> AICore task reaches normal done
  -> AICPU scheduler polls the real URMA send CQ
  -> scheduler retires the task after the CQE is ready
  -> dependent consumer tasks become runnable
```

因此，任务完成语义是：

```text
AICore normal done + all deferred completions ready = task complete
```

## Host Setup

Host 通信初始化负责准备设备侧 URMA 能访问的上下文：

```text
allocate symmetric communication windows
  -> initialize HCCL communicator
  -> initialize PTO-ISA UrmaWorkspaceManager
  -> store workspace address in CommContext.workSpace
  -> copy CommContext to device memory
```

URMA workspace 包含 PTO-ISA 侧用于真实硬件通信的 ABI 数据：

```text
UrmaInfo
UrmaWqCtx
UrmaCqCtx
UrmaMemInfo
```

这些结构记录 rank 数、QP 数、SQ 地址、CQ 地址、CQE buffer、tail 地址、
doorbell 地址，以及 peer memory region 地址。

设备 kernel 后续通过 `comm_ctx->workSpace` 找到这些信息。A5 URMA 的远端地址
不是普通 ACL IPC VA，而是通过 URMA workspace 中注册的 peer memory region 派生。

远端地址计算规则是：

```text
local_offset = local_ptr - local_window_base
remote_ptr = peer_mr_base(workspace, peer) + local_offset
```

这样同一个 offset 可以映射到 peer rank 的对称窗口中对应位置。

## AICore Request Submission

AICore kernel 先确定 peer rank：

```cpp
uint32_t peer = (rank + 1) % rank_num;
```

然后计算本地 tensor 在当前 rank window 内的 offset：

```cpp
uint64_t local_offset =
    reinterpret_cast<uint64_t>(local_ptr) - comm_ctx->windowsIn[rank];
```

再从 URMA workspace 取 peer memory region base：

```cpp
uint64_t remote_base =
    pto2::urma_backend::peer_mr_base_addr(comm_ctx->workSpace, peer);
```

最终得到 peer 上对应位置的远端地址：

```cpp
float *remote_ptr = reinterpret_cast<float *>(remote_base + local_offset);
```

请求入口使用统一 descriptor：

```cpp
send_request_entry(
    async_ctx,
    UrmaTget(local_tensor, remote_tensor, workspace, peer)
);

send_request_entry(
    async_ctx,
    UrmaTput(remote_tensor, local_tensor, workspace, peer)
);
```

`UrmaTget` 和 `UrmaTput` 只负责构造请求描述：

```cpp
struct UrmaRequestDescriptor {
    UrmaOp op;
    DstTensor dst;
    SrcTensor src;
    uint8_t *workspace;
    uint32_t remote_rank;
};
```

这个 descriptor 说明本次请求的操作类型、源/目的 tensor、URMA workspace，
以及目标 peer rank。

## send_request_entry

`send_request_entry` 是 AICore URMA deferred path 的统一入口。它先进入
`submit_chunked_urma_request`，完成基本检查和可选切分：

```text
check src/dst are flat contiguous 1-D tensors
  -> calculate total transfer bytes
  -> submit once if total bytes <= 256 MiB
  -> otherwise split into 256 MiB chunks
```

`256 MiB` 来自 PTO-ISA URMA 单 WQE 最大传输限制。自动切分代码已经存在，
但当前硬件验收不依赖 `>256 MiB` smoke。大包 smoke 在当前 A5 环境中观察到
HCCL 初始化恢复问题，因此被归为后续 hardening 项。

实际提交一次 URMA 请求时，代码使用 PTO-ISA 原生 URMA async API：

```cpp
pto::comm::AsyncSession session;

BuildAsyncSession<pto::comm::DmaEngine::URMA>(
    workspace,
    remote_rank,
    session
);

pto::comm::AsyncEvent event;
if (op == UrmaOp::TGET) {
    event = TGET_ASYNC<pto::comm::DmaEngine::URMA>(dst, src, session);
} else {
    event = TPUT_ASYNC<pto::comm::DmaEngine::URMA>(dst, src, session);
}
```

这里故意不调用 `event.Wait(session)`。请求提交后，AICore 把 event 登记成
deferred completion：

```cpp
register_urma_async_event(async_ctx, event, session, workspace);
defer_flush(async_ctx);
```

## Deferred Completion Registration

PTO-ISA `AsyncEvent` 里有一个 `handle`。对 URMA 来说，这个 handle 可以理解为：

```text
remote_rank + target SQ head
```

也就是：scheduler 后续应该检查哪个 peer QP，以及 send CQ 至少要推进到哪个
head 才能认为该请求完成。

登记时，URMA event 被转换为 simpler 的通用 `CompletionToken`：

```cpp
CompletionToken token{
    event.handle,
    0,
    COMPLETION_ENGINE_URMA,
    COMPLETION_TYPE_URMA_EVENT_HANDLE,
    reinterpret_cast<uint64_t>(workspace),
};
```

字段含义是：

```text
addr            = event.handle
expected_value  = unused for URMA handle polling
engine          = URMA
completion_type = URMA_EVENT_HANDLE
backend_cookie  = workspace address
```

`backend_cookie` 必须保存 workspace 地址。AICPU scheduler 只有 event handle
无法找到 CQ/SQ 上下文；它需要 workspace 解析 `UrmaInfo`、`UrmaWqCtx` 和
`UrmaCqCtx`。

登记完成后，AICore 等价于向 scheduler 发布了这个条件：

```text
This task is not complete yet.
Wait for URMA event handle X by polling URMA workspace Y.
```

## AICPU Wait List

AICore 会向 completion mailbox 发布两类消息：

```text
CONDITION
TASK_NORMAL_DONE
```

AICPU scheduler drain mailbox 后，把它们合并成 `AsyncWaitEntry`：

```cpp
struct AsyncWaitEntry {
    PTO2TaskId task_token;
    PTO2TaskSlotState *slot_state;
    CompletionCondition conditions[MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count;
    int32_t waiting_completion_count;
    bool normal_done;
};
```

对一个 URMA deferred task，entry 的核心内容是：

```text
normal_done = true
conditions[0].engine = URMA
conditions[0].completion_type = URMA_EVENT_HANDLE
conditions[0].addr = event.handle
conditions[0].backend_cookie = workspace
waiting_completion_count = 1
```

如果一个 task 提交了多个 URMA 请求，entry 中会有多个 condition：

```text
conditions[0] = first URMA event
conditions[1] = second URMA event
waiting_completion_count = 2
```

只有所有 condition 都 ready，并且 `TASK_NORMAL_DONE` 已经到达，scheduler 才会
释放该 task。

## Completion Type Dispatch

通用 async wait 框架通过 completion type 分发到不同 backend：

```text
COMPLETION_TYPE_COUNTER
  -> counter_poll_op

COMPLETION_TYPE_SDMA_EVENT_RECORD
  -> sdma_event_record_poll_op

COMPLETION_TYPE_URMA_EVENT_HANDLE
  -> urma_event_handle_poll_op
```

URMA 条件最终调用：

```cpp
poll_urma_event_handle(cond.addr, cond.backend_cookie);
```

其中：

```text
cond.addr = event.handle
cond.backend_cookie = workspace address
```

这样 scheduler 不需要知道 URMA CQE 的细节。它只需要处理统一返回值：

```text
READY   -> condition satisfied
PENDING -> keep polling later
FAILED  -> set runtime error
```

## URMA CQ Polling

`poll_urma_event_handle` 是真实硬件完成检测的核心。输入是：

```text
event_handle
workspace_addr
```

第一步，解码 event handle：

```text
remote_rank
target_head
```

含义是：检查 `remote_rank` 对应 send CQ 的 tail 是否已经推进到 `target_head`。

第二步，从 workspace 读取 `UrmaInfo`：

```cpp
struct UrmaInfo {
    uint32_t qp_num;
    uint32_t local_token_id;
    uint32_t rank_count;
    uint64_t sq_ptr;
    uint64_t rq_ptr;
    uint64_t scq_ptr;
    uint64_t rcq_ptr;
    uint64_t mem_ptr;
};
```

基本校验包括：

```text
qp_num != 0
remote_rank < rank_count
sq_ptr != 0
scq_ptr != 0
```

第三步，根据 `remote_rank` 找到 SQ 和 send CQ 上下文：

```text
ctx_index = remote_rank * qp_num
cq_ctx = scq_ptr + ctx_index * sizeof(UrmaCqCtx)
wq_ctx = sq_ptr  + ctx_index * sizeof(UrmaWqCtx)
```

`UrmaCqCtx` 提供 CQE buffer、CQE size、depth、tail 地址和 doorbell 地址。
`UrmaWqCtx` 提供 SQ tail 地址。

第四步，读取当前 CQ tail：

```text
cur_tail = *cq_ctx.tail_addr
```

如果 `cur_tail` 已经到达 `target_head`，说明之前的 poll 已经推进过，直接返回
`READY`。

否则 scheduler 扫描 CQE：

```text
next_tail = cur_tail

while next_tail has not reached target_head:
    cqe_addr = cq_buf + cqe_size * (next_tail & (depth - 1))
    dw0 = load CQE dw0
    check owner bit
    break if CQE is not ready
    check status and substatus
    fail if status or substatus is non-zero
    next_tail++
```

如果发现新的 ready CQE，scheduler 会推进 CQ/SQ tail 和 CQ doorbell：

```text
*cq_tail_addr = next_tail
*cq_db_addr = next_tail & 0xFFFFFF
*wq_tail_addr = next_tail
```

最后返回：

```text
next_tail reached target_head -> READY
otherwise                     -> PENDING
```

这一步证明完成信号来自真实 URMA send CQ，而不是 fake counter。

## Scheduler Retirement

当 URMA poll 返回 `READY` 后，通用 wait list 会执行：

```cpp
cond.satisfied = true;
cond.retire();
entry.waiting_completion_count--;
```

URMA 的 retire 当前基本是 no-op，因为真实 CQ tail 推进已经在 poll 阶段完成。

当 entry 同时满足下面两个条件：

```text
entry.normal_done == true
entry.waiting_completion_count == 0
```

scheduler 才调用：

```cpp
on_task_complete(slot_state);
on_task_release(slot_state);
```

后续依赖这个 task 的 consumer task 才会变为 runnable。

## Demo Validation Semantics

真实 A5 deferred demo 不只是验证 URMA 请求能发出去，还验证 scheduler 是否真的
等到了硬件完成。

demo 每个 rank 都包含：

```text
send tensor
tget tensor
tput tensor
marker tensor
status tensor
```

TGET kernel 执行：

```text
pull peer send tensor into local tget tensor
write marker after request submission
```

TPUT kernel 执行：

```text
push local data into peer tput window
write marker after request submission
```

consumer kernel 依赖前面的 TGET / TPUT task。如果 deferred completion 过早释放，
consumer 会提前执行，并可能看到未完成的数据搬运。

consumer 检查：

```text
local TGET result equals peer expected data
remote TPUT slot contains expected local data
marker count matches submitted request count
```

典型通过状态是：

```text
rank=0 status=[0, 4096, 1, 4, 4, 4, 0, 0]
rank=1 status=[0, 4096, 0, 4, 4, 4, 0, 0]
```

含义是：

```text
status[0] = 0          -> no error
status[1] = elem_count -> validated element count
status[2]              -> peer/rank auxiliary value
status[3..5]           -> request / marker counters
```

## Relationship To SDMA

URMA 没有重写一套 scheduler，而是作为新的 completion backend 接入已有 async
wait 框架。

SDMA 和 URMA 的共同点是：

```text
AICore submits an async data movement request
  -> AICore records a completion condition
  -> AICPU scheduler polls the condition
  -> scheduler releases the task after the condition is ready
```

差异在 condition 的含义：

```text
SDMA condition:
    points to an SDMA event record / counter
    poll checks whether the event record is ready

URMA condition:
    addr stores PTO-ISA URMA event.handle
    backend_cookie stores URMA workspace address
    poll parses URMA SQ/CQ context and reads real send CQEs
```

因此可以把本实现理解为：

```text
URMA is implemented as a new deferred completion backend,
not as a separate scheduler path.
```

## Layering

整体实现分为四层：

```text
Host communication setup
    allocates windows, initializes URMA workspace, publishes CommContext

AICore submission API
    exposes send_request_entry, UrmaTget, and UrmaTput

Deferred completion registration
    converts PTO-ISA AsyncEvent into simpler CompletionToken

AICPU completion scheduler
    polls URMA CQEs and releases tasks after real hardware completion
```

各层职责边界是：

```text
Host knows how to create and publish the URMA workspace.
AICore knows how to submit URMA async requests.
Registration knows how to turn PTO events into completion tokens.
AICPU scheduler knows how to poll real hardware completion.
```

## Design Choices

没有选择 AICore 立即 `event.Wait(session)`，因为这会阻塞 AICore，无法体现
deferred completion。

没有选择 fake completion，因为 fake counter 只能证明框架能跑，不能证明真实 URMA
硬件完成。

最终设计是：

```text
AICore submits real PTO-ISA URMA async requests.
AICPU parses real URMA send CQEs.
simpler scheduler releases tasks through the common deferred completion path.
```

这个设计的收益是：

```text
uses real PTO-ISA URMA hardware semantics
reuses the existing scheduler and wait-list framework
keeps URMA aligned with the SDMA completion backend model
leaves room for event coalescing, larger transfer hardening, and diagnostics
```

## Current Scope

当前核心链路已经完成并在真实 A5 上验证：

```text
URMA TGET / TPUT can be submitted on real A5 hardware.
AICore does not synchronously wait for URMA completion.
Completion tokens enter the simpler deferred completion framework.
AICPU observes real URMA CQ readiness.
Dependent consumer tasks wait correctly and validate data.
```

当前不把 `>256 MiB` 大包 smoke 作为验收项。相关切分代码已存在，但大包 smoke
在当前 A5 环境中可能触发后续 HCCL communicator 初始化失败，应作为 Phase 5
hardening 单独处理。
