from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RDMA_BACKEND_KERNEL = (
    REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/rdma/rdma_completion_kernel.h"
)
ASYNC_WAIT = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_async_wait.h"
MAILBOX_TYPES = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/aicore_completion_mailbox_types.h"
HOST_CMAKE = REPO_ROOT / "src/a5/platform/onboard/host/CMakeLists.txt"
HOST_COMM = REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp"
KERNEL_COMPILER = REPO_ROOT / "simpler_setup/kernel_compiler.py"
RDMA_SCHEDULER = (
    REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/rdma/rdma_completion_scheduler.h"
)
SCHEDULER_COMPLETION = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp"
RDMA_DEFERRED_DEMO = (
    REPO_ROOT
    / "examples/a5/tensormap_and_ringbuffer/rdma_deferred_completion_demo/test_rdma_deferred_completion_demo.py"
)
RDMA_DEFERRED_ORCH = (
    REPO_ROOT
    / "examples/a5/tensormap_and_ringbuffer/rdma_deferred_completion_demo/kernels/orchestration"
    / "rdma_deferred_completion_orch.cpp"
)


def test_rdma_kernel_backend_exposes_deferred_adapter_contract() -> None:
    backend = RDMA_BACKEND_KERNEL.read_text()

    assert "enum class RdmaOp" in backend
    assert "struct RdmaRequestDescriptor" in backend
    assert "RdmaTget" in backend
    assert "RdmaTput" in backend
    assert "DmaEngine::RDMA" in backend
    assert "TGET_ASYNC<pto::comm::DmaEngine::RDMA>" in backend
    assert "TPUT_ASYNC<pto::comm::DmaEngine::RDMA>" in backend
    assert "register_rdma_async_event" in backend
    assert "COMPLETION_ENGINE_ROCE" in backend
    assert "COMPLETION_TYPE_RDMA_EVENT_HANDLE" in backend
    assert "backend_cookie" not in backend or "reinterpret_cast<uint64_t>(workspace)" in backend


def test_rdma_kernel_backend_uses_peer_mr_base_not_windows_in() -> None:
    backend = RDMA_BACKEND_KERNEL.read_text()

    assert "peer_mr_base_addr" in backend
    assert "peer_mr_ptr" in backend
    assert "PeerMrBaseAddr(workspace, peer)" in backend
    assert "windowsIn" not in backend


def test_rdma_completion_type_is_registered_in_async_wait() -> None:
    wait_source = ASYNC_WAIT.read_text()
    type_source = MAILBOX_TYPES.read_text()

    assert "#define COMPLETION_TYPE_RDMA_EVENT_HANDLE 3" in type_source
    assert 'backend/rdma/rdma_completion_scheduler.h' in wait_source
    assert "rdma_event_handle_poll_op" in wait_source
    assert "rdma_event_handle_retire_op" in wait_source
    assert "COMPLETION_TYPE_RDMA_EVENT_HANDLE = 3" in wait_source


def test_a5_host_cmake_gates_rdma_workspace_overlay() -> None:
    cmake = HOST_CMAKE.read_text()

    assert "option(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE" in cmake
    assert "SIMPLER_ENABLE_PTO_URMA_WORKSPACE_DEFAULT OFF" in cmake
    assert "SIMPLER_ENABLE_PTO_URMA_WORKSPACE_DEFAULT ON" in cmake
    assert "Only one PTO async workspace overlay may be enabled" in cmake
    assert "PTO RDMA workspace overlay requires pto-isa RDMA headers" in cmake
    assert "PTO_RDMA_SUPPORTED" in cmake
    assert "PTO_RDMA_BACKEND_HNS_1825_SUPPORTED" in cmake


def test_a5_comm_hccl_keeps_urma_and_rdma_workspace_paths_macro_gated() -> None:
    source = HOST_COMM.read_text()

    assert "#ifdef SIMPLER_ENABLE_PTO_URMA_WORKSPACE" in source
    assert "#ifdef SIMPLER_ENABLE_PTO_RDMA_WORKSPACE" in source
    assert "#include \"pto/comm/async/rdma/rdma_workspace_manager.hpp\"" in source
    assert "std::unique_ptr<pto::comm::rdma::RdmaWorkspaceManager> rdma_workspace" in source


def test_a5_comm_hccl_uses_mr1374_rdma_host_api_shape() -> None:
    source = HOST_COMM.read_text()

    assert "#define __gm__" in source
    assert "aclrtGetPhyDevIdByUserDevId" in source
    assert "PTO_ROCE_PHYIDS" in source
    assert "PTO_ROCE_ROOTINFO" in source
    assert "kDefaultRdmaRootinfoPath = \"/etc/hccl_rootinfo.json\"" in source
    assert "kDefaultVirtualTopologyPath = \"/var/run/ascend-topologyd/virtualTopology.xml\"" in source
    assert "rdma_resolve_local_ip_from_virtual_topology(out.phy_id, out.local_ip)" in source
    assert "rdma_resolve_local_ip_from_env(base_rank, rank_count, out.local_ip)" in source
    assert "PTO_ROCE_LOCAL_IP" in source
    assert "PTO_ROCE_IPS" in source
    assert "rdma_resolve_local_ip_from_hccn_tool(device_id, out.local_ip)" in source
    assert "rdma_resolve_local_ip_from_rootinfo(out.phy_id, out.local_ip, rootinfo_path)" in source
    assert "s.find(\"\\\"CLOS\\\"\", dev_key_pos)" in source
    assert "hccn_tool -g -dev_info -i %d" in source
    assert "\\\"addr\\\"" in source
    assert "manager->Init(config)" in source
    assert "WorkspaceInitResult::READY" in source
    assert "RdmaBackend::HNS_1825" not in source


def test_kernel_compiler_forwards_rdma_workspace_macros_to_incore_builds() -> None:
    source = KERNEL_COMPILER.read_text()

    assert '"SIMPLER_ENABLE_PTO_RDMA_WORKSPACE"' in source
    assert '"-DPTO_RDMA_SUPPORTED"' in source
    assert '"-DPTO_RDMA_BACKEND_HNS_1825_SUPPORTED"' in source
    assert "cmd += self._incore_feature_defines()" in source
    assert "_pto_isa_include_dirs" in source
    assert '"pkg_inc"' in source


def test_kernel_compiler_adds_ascend_device_headers_for_rdma_backend_headers() -> None:
    source = KERNEL_COMPILER.read_text()

    assert "get_ascend_incore_include_dirs" in source
    assert 'root / "asc" / "include" / "interface"' in source
    assert 'root / "asc" / "include" / "basic_api" / "interface"' in source
    assert 'root / "ascendc" / "include" / "basic_api" / "interface"' in source
    assert '"kernel_operator_sys_var_intf.h"' in source
    assert '"kernel_operator_sys_var_intf_impl.h"' in source
    assert ".rglob(header)" in source
    assert "for inc_dir in self.get_ascend_incore_include_dirs():" in source


def test_rdma_scheduler_abi_matches_mr1374_workspace_and_hns1825_contexts() -> None:
    source = RDMA_SCHEDULER.read_text()

    assert "uint32_t rank_count;" in source
    assert "uint32_t reserved;" in source
    assert "uint32_t local_token_id" not in source
    assert "static_assert(sizeof(RdmaWqCtx) == 96" in source
    assert "static_assert(sizeof(RdmaCqCtx) == 64" in source
    assert "struct Hns1825Cqe" in source
    assert "static_assert(sizeof(Hns1825Cqe) == 32" in source
    assert "uint32_t cqe_size;" in source
    assert "1u << cq_ctx.cqe" not in source
    assert "db_sw_addr" in source
    assert "is_hns1825_cqe_owner_ready" in source
    assert "const uint32_t cq_ring = cq_ctx.depth;" in source
    assert "kHns1825CqeMaxGenNum" not in source
    assert "owner_id_qpn" in source
    assert "op_sr_wqebb" in source


def test_rdma_scheduler_stall_snapshot_reports_sq_and_cq_progress() -> None:
    source = RDMA_SCHEDULER.read_text()

    assert "load_device_u32_or_zero" in source
    assert "load_wq_ctx" in source
    assert "load_cq_ctx" in source
    assert "sq=0x%llx rq=0x%llx scq=0x%llx rcq=0x%llx" in source
    assert "%s wqn=%u head=%u tail=%u db_sw_be=0x%x" in source
    assert "head_addr=0x%llx " in source
    assert "tail_addr=0x%llx db_hw=0x%llx db_sw=0x%llx" in source
    assert "%s cqn=%u depth=%u cqe_size=%u cur_tail=%u target_head=%u" in source
    assert "db_sw_be=0x%x cq_buf=0x%llx" in source
    assert "log_rdma_wq_snapshot(\"rq\"" in source
    assert "log_rdma_cq_snapshot(\"rcq\"" in source


def test_rdma_domain_bootstrap_resolves_env_arrays_by_domain_rank() -> None:
    source = HOST_COMM.read_text()

    assert (
        "resolve_rdma_bootstrap(h, domain_rank, static_cast<uint32_t>(rank_count), myDevice, rdma_bootstrap)"
        in source
    )
    assert "resolve_rdma_bootstrap(\n            h, static_cast<uint32_t>(h->rank)" not in source


def test_a5_scheduler_invalidates_deferred_completion_slab_before_reading_count() -> None:
    source = SCHEDULER_COMPLETION.read_text()
    assert "volatile DeferredCompletionSlab *deferred_slab" in source
    assert "cache_invalidate_range" in source
    assert "sizeof(*deferred_slab)" in source
    assert source.index("cache_invalidate_range") < source.index("deferred_slab->error_code")
    assert source.index("cache_invalidate_range") < source.index("deferred_slab->count")


def test_rdma_deferred_demo_serializes_hns1825_sq_posts() -> None:
    py_source = RDMA_DEFERRED_DEMO.read_text()
    orch_source = RDMA_DEFERRED_ORCH.read_text()

    assert py_source.count("[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT") == 1
    assert py_source.count("[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT") == 1
    assert "serialize producer posts through marker dependencies" in orch_source
    assert "tget1_args.add_input(tget0_marker)" in orch_source
    assert "tput0_args.add_input(tget1_marker)" in orch_source
    assert "tput1_args.add_input(tput0_marker)" in orch_source
