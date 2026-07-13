# A5 URMA Deferred Completion Status

## Phase 1: Sim Deferred Completion

Status: complete.

Validated:

- `a5sim` URMA deferred completion demo builds and runs.
- Fake TGET and TPUT paths both publish a pending CQE before scheduler polling.
- Scheduler observes pending before ready.
- Dependent consumer runs only after the fake URMA completion becomes ready.

Scope note:

- Phase 1 uses fake copy and fake CQE production. It does not validate HCCL
  channels, memory registration, huge pages, doorbells, or real URMA WQ/CQ
  hardware behavior.

## Phase 2: Real A5 Build And Workspace

Status: complete; A5 hardware smoke passed through `task-submit`.

Validated:

- `pto_isa.pin` points at a PTO-ISA revision that contains URMA async support
  and `UrmaWorkspaceManager`.
- A5 onboard `tensormap_and_ringbuffer` runtime builds with CANN 9.1.
- A5 onboard host runtime can include PTO-ISA URMA host headers.
- A5 host link includes the HCOMM dependencies required by the URMA workspace
  manager.
- The selected `libhcomm.so` exposes:
  - `HcclCommMemReg`
  - `HcclChannelGetRemoteMems`
  - `HcclRankGraphGetLinks`
  - `HcclChannelAcquire`
- `comm_alloc_windows` initializes a base URMA workspace and mirrors
  `GetWorkspaceAddr()` into `CommContext::workSpace`.
- `CommContext::workSpaceSize` is populated with the expected workspace table
  size.
- Dense prefix derived contexts reuse the base URMA workspace.
- Non-dense derived contexts leave the URMA workspace disabled instead of
  reusing a mismatched base workspace.
- Dense dynamic-domain allocations initialize an independent workspace for the
  new symmetric buffer.
- Documentation records that the A5 URMA symmetric window currently keeps the
  HCCL Path-D allocation policy, `ACL_MEM_MALLOC_HUGE_FIRST`, and explains when
  to tighten it to `ACL_MEM_MALLOC_HUGE_ONLY`.

Hardware notes:

- The direct, unlocked run failed before workspace validation because
  `rtSetDevice` returned `507899`.
- The locked `task-submit --device auto --device-num 2` run passed the Phase 2
  hardware smoke: `1 passed in 17.37s`.

Acceptance decision:

- Phase 2 is hardware-accepted. Use `task-submit` for future A5 validation; do
  not treat the earlier unlocked `507899` run as a Phase 2 failure.

Required hardware acceptance command:

```bash
PYTHONPATH=$PWD:$PWD/python python -m pytest \
  tests/ut/py/test_worker/test_platform_comm.py \
  --platform a5 --device 0-1 -q
```

Expected result:

```text
1 passed
```

If this command fails at `stage 'init_device'` or reports
`simpler_init failed with code 507899`, the run has not reached Phase 2
workspace validation. That failure is at the `rtSetDevice`/device attach
boundary; first rerun through `task-submit` on healthy A5 devices and confirm
`npu-smi info` works for the test user.

Local non-A5 guard:

```bash
pytest tests/ut/py/test_urma_real_async_demo_source.py -q
```

This checks that the A5 platform comm smoke asserts the Phase 2 acceptance
points and that `comm_hccl.cpp` still initializes and propagates the URMA
workspace through base, dense-derived, and dynamic-domain contexts.

## Phase 3: Standalone Real URMA Correctness

Status: not complete; hardware probes show the current real URMA submit design
is unsafe on A5.

Goal:

- Prove real PTO-ISA URMA submit and data movement correctness on A5, without
  connecting the operation to AICPU deferred completion.

Implemented:

- Added `examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/`.
- The demo allocates all URMA source, destination, and signal buffers inside a
  two-rank dynamic communication domain.
- Host staging uses `orch.copy_to` to initialize each rank's registered
  symmetric window before submitting the AIV kernel.
- The AIV kernel reads `CommContext::workSpace` and can validate workspace,
  remote memory metadata, EID, queue indices, and WQE address calculation.
- The AIV kernel now fail-fasts known unsafe WQE write and submit probes with
  status code `60`, because hardware runs showed those paths can stall until
  the runtime op timeout.
- URMA remote addresses are derived from `UrmaPeerMrBaseAddr(workspace, peer) +
  offset`.
- The demo keeps the small case (`16` float32 elements) and page-spanning case
  (`4096` float32 elements) harness, but the full path now reports the unsafe
  submit status instead of attempting data movement.
- The default CLI/pytest entry now runs a focused safe probe suite: workspace,
  session, remote metadata, EID, and WQE address probes must return status `0`,
  while representative unsafe WQE/submit probes must return status `60` without
  touching the WQE ring.

Validated:

- `test_urma_real_async_demo.py` passes Python syntax compilation.
- The new A5 incore kernel and orchestration shim compile with the local CANN
  9.0.1 toolchain after the local sparse PTO-ISA checkout is populated with A5
  instruction headers.
- Pytest collection includes the demo for `--platform a5`.
- The hardware pytest entry validates the safe probe suite instead of treating
  real data movement as complete.
- Pytest collection deselects the demo for `--platform a5sim`, so sim batches do
  not accidentally run a real-URMA-only test.
- Local source regression confirms the kernel no longer bypasses rank 1 and no
  longer attempts the known unsafe direct WQE write variants.

Hardware findings:

- A real A5 run on 2026-07-13 reached URMA workspace setup and launched the
  demo, but timed out after about 45 s with `aclrtSynchronizeStreamWithTimeout
  (AICPU) failed: 507000` and `PTO2 scheduler timeout sub_class=S1:running-stalled`.
  The host status tensors remained zero because fatal runtime status skipped
  copy-back, so the next diagnostic step is the bounded-wait kernel above.
- Later probe runs narrowed the failing access to the SQ WQE ring buffer at
  `UrmaWQCtx::bufAddr`: `wqe_addr`, `remote_mem`, and `eid_read` passed, while
  `wqe_read`, `wqe_first_store`, `wqe_first_st_dev`, `wqe_mte_store`, and the
  original PTO-ISA-backed `tput_post` timed out. This proves that the current
  URMA SQ WQE ring address is not safely reachable from AICore, either through
  ordinary LSU stores, `st_dev`, or MTE copy. The demo now fail-fasts these
  known unsafe paths with status code `60` instead of hanging the device.
- CANN's public AIV `Hcomm<CommEngine::AIV, CommProtocol::ROCE>` path is not a
  drop-in replacement for this workspace: it expects a ROCE/RDMA `Channel`
  layout, while the HCCL channel conversion used here exposes a URMA/UB-JFS
  `ChannelEntity` layout.
- The host workspace setup now logs every `HcclRankGraphGetLinks` protocol, the
  selected channel protocol, the converted `ChannelEntity` protocol, and the SQ/CQ
  context ABI type. If A5 logs show only `UBC_CTP`/`UBC_TP` links and
  `UB_JFS`/`UB_JFC` contexts, the public HCOMM ROCE device API remains
  inapplicable. If a `ROCE` link/context appears, the next step is a minimal
  `Hcomm<AIV, ROCE>` post/wait probe.
- Device logs have not yet been inspected for URMA CQE status/substatus errors.

Required hardware acceptance command:

```bash
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1
```

This default command runs the focused probe suite. It should pass only if safe
probes return `0` and representative unsafe WQE/submit probes return `60`.

Diagnostic probe commands:

```bash
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage workspace
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage build_session
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage workspace_info
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage wq_ctx
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage queue_index_read
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage queue_index_ld_dev
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage wqe_addr
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage wqe_read --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage wqe_first_store --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage wqe_first_st_dev --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage wqe_mte_store --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage wqe_write_restore --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage remote_mem
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage eid_read
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage tget_post --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage tget_test_once --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage tput_post --expect-status 60
PYTHONPATH=$PWD:$PWD/python python \
  examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py \
  -p a5 -d 0-1 --probe-stage tput_test_once --expect-status 60
```

The probe stages run only the small case and return early after the named
operation. The WQE write probes now return status code `60` because the matching
hardware runs already showed they can stall the AICore until the runtime reports
`507000`. A future fix must replace the raw AICore WQE post mechanism rather
than try another AICore write variant to `UrmaWQCtx::bufAddr`.

Expected result for a completed Phase 3 implementation:

```text
[urma_real_async_demo] count=16 rank=0 status=[0, 16, 1, ...]
[urma_real_async_demo] count=16 rank=1 status=[0, 16, 0, ...]
[urma_real_async_demo] count=4096 rank=0 status=[0, 4096, 1, ...]
[urma_real_async_demo] count=4096 rank=1 status=[0, 4096, 0, ...]
```
