# simpler 中 URMA 的整体设计与实现

## 设计背景

PTO-ISA 已经提供 URMA async submit 能力：构造 `BuildAsyncSession<DmaEngine::URMA>(workspace, remote_rank, session)`，再通过 `TGET_ASYNC<URMA>` 或 `TPUT_ASYNC<URMA>` 提交 WQE，最后拿到 `AsyncEvent`。PTO-ISA testcase 里通常直接在 kernel 内 `event.Wait(session)`，这适合验证 URMA 数据通路，但不适合 simpler 的 task graph runtime。

simpler 的需求是：AICore kernel 可以提交异步通信并返回，但 task 不能立刻 retire；只有 AICPU scheduler 确认通信完成后，producer 的 fanout 才能释放，consumer 才能继续执行。

URMA 在 simpler 里本质上是仿照 SDMA deferred completion backend 做的。AICore 侧调用 PTO-ISA URMA async 接口，再把 completion 信息登记进 runtime 的 token/slab/mailbox 通路；AICPU 侧轮询这些 completion，等真正完成了再让 task retire。也就是说，SDMA 和 URMA 共用 runtime 运输链路，区别主要是 backend 怎么构造 session、token 里装什么、AICPU 去哪里 poll。

## Runtime 链路

URMA deferred completion 分成四段：AIV kernel 提交、AICore 登记、mailbox 转发、AICPU poll。Host 侧负责构建 runtime、初始化 workspace、准备 `CommContext` 并提交 orchestration；真正调用 `UrmaTget/UrmaTput` 的位置是在 device kernel 内。

第一段在 AIV kernel 中完成。kernel 从 `CommContext` 取到 URMA workspace，构造 URMA-valid 的 remote tensor 指针，然后调用：

```cpp
AsyncCtx async_ctx = get_async_ctx(args);
send_request_entry(
    async_ctx,
    UrmaTget(local_global, remote_global, workspace, peer_rank));
```

第二段在 `urma_completion_kernel.h` 中完成。`send_request_entry` 会构造 PTO-ISA URMA session，提交 `TGET_ASYNC<URMA>` 或 `TPUT_ASYNC<URMA>`，再把返回的 `AsyncEvent` 翻译成 simpler 的 `CompletionToken`。

第三段是 runtime 通用链路。这段和 SDMA 使用同一套机制：AICore 把 token 写入 `DeferredCompletionSlab`，FIN 处理逻辑发布到 `AICoreCompletionMailbox`，dispatch 线程再转成 `AsyncWaitList` 条件。generic runtime 不解释 `addr` 的含义，也不理解 SDMA event record 或 URMA CQ，只按 `completion_type` 分发给对应 backend。

第四段在 AICPU scheduler 中完成。`AsyncWaitList` 根据 completion type 调用 URMA backend poll 函数。poll 解码 event handle，读取 workspace 中的 SQ/CQ context，消费可见 CQE，并在 CQ tail 到达目标 SQ head 后返回 READY。只有这时 runtime 才认为 producer task 完成，并释放后续 fanout。

可以把主链路理解成下面这条顺序：

```text
AIV kernel
  -> PTO-ISA URMA submit
  -> AsyncEvent(handle)
  -> CompletionToken(handle + workspace)
  -> DeferredCompletionSlab
  -> AICoreCompletionMailbox
  -> AsyncWaitList
  -> AICPU poll send CQ
  -> task retire / fanout release
```

## Completion Token 设计

`CompletionToken` 是 AICore submit adapter 和 generic runtime 之间的统一 completion 描述。generic runtime 只透传这些字段，真正解释字段含义的是 backend：

```text
addr            backend-specific 地址或 handle
expected_value  counter 类 completion 使用的目标值，SDMA/URMA 不使用
engine          统计和诊断用的 engine id
completion_type  选择 AICPU poll/retire ops
backend_cookie  backend 私有上下文，generic runtime 只透传
```

SDMA 和 URMA 都写同一个 token ABI，但填法不同。SDMA 登记的是 event record 地址：

```text
addr            = event_record_addr
expected_value  = 0
engine          = COMPLETION_ENGINE_SDMA
completion_type = COMPLETION_TYPE_SDMA_EVENT_RECORD
backend_cookie  = 0
```

SDMA 的 AICPU poll 只需要读 event record 的 flag，所以不需要额外 cookie。URMA 不一样，event handle 只包含两个信息：

```text
high 32 bits: remote rank
low  32 bits: target SQ head
```

这个 handle 能表达“要等哪个 peer 的 QP 推进到哪个 SQ head”，但不能单独定位 CQ/SQ，所以 URMA 额外用 `backend_cookie` 携带 workspace：

```text
addr            = event.handle
expected_value  = 0
engine          = COMPLETION_ENGINE_URMA
completion_type = COMPLETION_TYPE_URMA_EVENT_HANDLE
backend_cookie  = reinterpret_cast<uint64_t>(workspace)
```

`pto_async_wait.h` 的 ops table 会根据 `COMPLETION_TYPE_URMA_EVENT_HANDLE` 调到 `poll_urma_event_handle(cond.addr, cond.backend_cookie)`。wait-list 不理解 URMA CQ 细节，只做分发。

## AICore Submit 侧

这部分可以理解成“把 PTO-ISA 的 URMA async 调用接进 simpler runtime”。AICore/AIV kernel 里通过 `UrmaTget/UrmaTput` 构造 descriptor，再调用 `send_request_entry(ctx, desc)`。这个入口内部会构造 URMA session，调用 PTO-ISA 的 `TGET_ASYNC/TPUT_ASYNC`，拿到 `AsyncEvent` 后不在 kernel 里自己等完，而是把 event 转成 simpler 的 `CompletionToken` 写进 deferred completion slab。

这样 runtime 就知道：这个 task 还有一个 URMA completion condition，不能因为 AICore kernel 返回了就马上 retire。如果当前不是 deferred context，代码仍然可以 fallback 到 `event.Wait(session)`，但正常 runtime 路径走的是 deferred completion。

这层还保留了基本的传输约束检查，例如 dtype/shape 要匹配、tensor 需要是 flat contiguous 1D、单个 URMA WQE 不超过 256 MB。超过 256 MB 时 backend 会拆成多个 submit，每个 chunk 各自登记 completion condition。

## AICPU Poll 侧

这部分可以理解成“runtime 真的去看 URMA 完没完成”。AICPU 不直接做 blocking wait，而是根据 token 里的 `event.handle` 和 `workspace` 去 poll URMA 的 send CQ。

具体来说，URMA backend 会从 handle 里解出 remote rank 和目标 SQ head，再通过 workspace 找到对应的 SQ/CQ 状态，检查 CQE 有没有 ready。ready 就返回 `READY`，没 ready 就返回 `PENDING`，有错误就返回 `FAILED`。这样 simpler scheduler 就能决定 task 什么时候真正完成。

这里和 PTO-ISA 原生 `Wait` 最大的区别是：`Wait` 更像在当前调用里等完；simpler 这里做成了非阻塞 poll。因为 runtime 本来就有调度循环，所以 backend 不应该一直卡住等完成，而是把“没完成”表达成 `PENDING`，让 scheduler 后面继续检查。当前 `retire_urma_event_handle` 是 no-op，因为 CQ/SQ tail 和 doorbell 更新已经在 poll 阶段完成。

## Remote Address Contract

URMA remote 地址不能默认沿用 SDMA 的 peer VA 构造方式。PTO-ISA URMA testcase 使用的是 `UrmaPeerMrBaseAddr(workspace, peer) + offset`，也就是从 URMA workspace 查 peer MR base，再加 symmetric window 内 offset。simpler backend 提供了两个 helper：

```cpp
peer_mr_base_addr(workspace, peer)
peer_mr_ptr<T>(workspace, peer, local_offset)
```

典型 AIV kernel 做法是：用本地 tensor data pointer 减本 rank window base 得到 offset，再调用 `peer_mr_ptr<T>(workspace, peer, offset)` 得到 remote MR 指针。这样可以避免把普通 imported peer VA 误传给 URMA WQE。

## Host Workspace 集成

真实 URMA 需要 host 侧 `UrmaWorkspaceManager`。A5 onboard host runtime 通过 `SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON` 启用 URMA workspace overlay，默认不启用；它和 SDMA workspace overlay 互斥，因为 `CommContext` 当前只有一个 `workSpace/workSpaceSize` 字段。

base window 路径中，`comm_alloc_windows` 在 symmetric window 分配完成后调用：

```cpp
UrmaWorkspaceManager::Init(hccl_comm,
                           rank,
                           nranks,
                           local_window,
                           win_size)
```

成功后，host runtime 把 workspace 地址写到 `CommContext::workSpace`，device kernel 后续就从这里取 workspace。

## Example 中的端到端语义

`urma_deferred_completion_demo` 展示了真实 A5 上的端到端用法：每个 rank 把 input 放在 HCCL/URMA communication window 内，producer 用 `peer_mr_ptr` 构造 peer input 的 remote pointer，再提交 `UrmaTget` 拉到本地 out；consumer 依赖 producer 的 out 并写 `result = out + 1`。

这个 demo 不只是校验 TGET 数据正确，也校验 runtime 语义：producer kernel 可以先返回，但 producer task 不能在 URMA completion READY 前 retire；consumer 只有在 fanout 被释放后才应该读到完整 out。

## 当前固定约束

- 真实 URMA submit 依赖 `PTO_URMA_SUPPORTED` 和对应 CANN/HCCL/PTO-ISA 环境。
- Host runtime 需要用 `SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON` 构建，且 SDMA/URMA workspace overlay 当前互斥。
- QP index 固定为 0；remote tensor 地址必须是 URMA MR-valid address，推荐通过 `peer_mr_ptr` 构造。
- CQE status/substatus 异常当前统一映射为 `PTO2_ERROR_ASYNC_COMPLETION_INVALID`。

## 总结

simpler 的 URMA backend 做的是 runtime 适配层：AICore 侧调用 PTO-ISA URMA 接口提交 async work，并把 `AsyncEvent` 转换成 runtime completion condition；AICPU 侧根据 event handle 和 workspace poll send CQ，推进 CQ/SQ tail 与 doorbell，并把 READY/FAILED/PENDING 反馈给 `AsyncWaitList`。

这样，URMA 的底层提交和 workspace 语义仍由 PTO-ISA 承担，simpler 只负责把它接进自己的 deferred completion、task retirement 和 fanout release 机制中。
