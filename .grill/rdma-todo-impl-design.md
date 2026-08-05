# Grill: RDMA todo 和 impl 文档设计
Date: 2026-07-29

## Intent
基于 `../rdma_design.md` 设计两份 RDMA 规划文档，职责类似现有 URMA 的
`../todo.md` 和 `../impl.md`。文档需要把 RDMA 工作拆成可分阶段完成的任务，
并为每个阶段定义足够明确的验收方式，避免后续阶段因为前置阶段验收不足而无法判断问题边界。

## Constraints
- 新建同级文件，不覆盖现有 URMA 的 `../todo.md` 和 `../impl.md`。
- `rdma_todo.md` 和 `rdma_impl.md` 严格分工：todo 只跟踪阶段 checklist、验收标准、命令参考；impl 说明实施设计、涉及文件、关键接口、失败路径和风险处理。
- RDMA 第一版直接面向真实 A5 onboard，不做 fake `a5sim` RDMA 链路。
- 第一版完成线是真实 A5 两 rank deferred RDMA demo，覆盖 TGET、TPUT 数据正确性和依赖 fanout 行为。
- RDMA IP、phyId、basePort 配置沿用 pto-isa rootinfo/env 约定，不新增 Python public API。
- 第一版只支持 `orch.allocate_domain()`，不支持 `comm_derive_context()`。
- SDMA/URMA/RDMA workspace overlay 保持互斥，因为 `CommContext` 当前只有一个 `workSpace/workSpaceSize`。
- 第一版不做 SDMA/URMA/RDMA 性能对比。

## Key decisions
- Decision: 不做 fake `a5sim` RDMA 阶段。Reason: deferred 框架形状已由 SDMA/URMA 验证，RDMA 风险主要在真实 HNS1825 bootstrap、workspace metadata、CQ polling 和 rank 映射。Alternative considered: 像 URMA 一样先做 fake sim Phase 1。
- Decision: 新文件命名为 `../rdma_todo.md` 和 `../rdma_impl.md`。Reason: 现有同级 todo/impl 是 URMA 语境，不应覆盖历史。Alternative considered: 覆盖或泛化现有文件。
- Decision: 第一版完成线设为真实 A5 deferred demo 通过。Reason: 仅 build 成功无法证明 event 注册、CQ polling 和 fanout release。Alternative considered: 停在 workspace 初始化或 standalone blocking wait。
- Decision: pto-isa 依赖和 pin 检查并入 Phase 1。Reason: RDMA headers 和 ABI contract 是后续所有阶段的稳定基准。Alternative considered: 单独拆 Phase 0。
- Decision: RDMA bootstrap 使用 pto-isa 现有 rootinfo/env。Reason: 在链路跑通前不扩大 simpler public API。Alternative considered: 给 `Worker` 或 `allocate_domain()` 增加 RDMA 配置参数。
- Decision: 多 workspace 同时开启作为 future work。Reason: 同时支持 SDMA/URMA/RDMA 需要 `CommContext` ABI 变更，不是 RDMA 链路跑通任务。Alternative considered: 放入 hardening。

## Surfaced assumptions
- `pto_isa.pin` 可以切到包含 MR 1374 RDMA/HNS1825 接口的 commit。
- 后续有可用的 A5 onboard + HNS1825/RoCE 配置环境执行硬件阶段。
- 硬件验收需要遵守仓库规则：A5 arch precheck、`task-submit` 锁设备、每次运行独立 `ASCEND_PROCESS_LOG_PATH`。
- `rdma_completion_scheduler.h` 会复制 pto-isa HNS1825 metadata 的必要子集，因此 ABI `static_assert` 是验收的一部分。

## Out of scope
- `a5sim` 或 fake RDMA backend。
- `a2a3`、`a2a3sim`、`a5sim`、`host_build_graph`。
- Remote-L3 bulk transport。
- `comm_derive_context()` RDMA workspace 支持。
- 同时启用 SDMA/URMA/RDMA workspace overlay。
- RDMA bootstrap 的新增 Python API。
- 与 SDMA/URMA 的性能对比。
