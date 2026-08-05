# 任务：rdma backend 开发

## 目标
在 `newtry` 分支完成 simpler 的 rdma backend，实现方式对齐已有 sdma/urma 架构，并按同级 `rdma` 目录的 design、todo、impl 文档验证。

## 待办事项
- [x] 切换到 `newtry` 分支并刷新仓库规则
- [x] 阅读同级 `rdma` 的 design/todo/impl 文档
- [x] 梳理现有 sdma/urma backend 架构与调用面
- [x] 找到 rdma 缺口并先补充/运行能暴露缺口的测试或构建检查
- [x] 实现 rdma backend 代码与构建注册
- [x] 运行格式化、构建和相关测试验证
- [x] 检查文档/已知问题并总结结果

## 进度
7/7

## 当前阻塞
当前 `pto_isa.pin`/`build/pto-isa` 不包含 PTO-ISA RDMA/HNS1825 headers，
因此 `SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON` 的真实 host build 和 A5 硬件
demo 还不能完成。CMake 已验证会在缺少
`include/pto/comm/async/rdma/rdma_workspace_manager.hpp` 时明确失败。
