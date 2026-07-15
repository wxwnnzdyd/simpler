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
- `comm_alloc_windows` initializes a base URMA workspace through PTO-ISA's
  native `pto::comm::urma::UrmaWorkspaceManager` and mirrors
  `GetWorkspaceAddr()` into `CommContext::workSpace`.
- `CommContext::workSpaceSize` is populated with the expected workspace table
  size.
- Dense prefix derived contexts reuse the base URMA workspace.
- Non-dense derived contexts leave the URMA workspace disabled instead of
  reusing a mismatched base workspace.
- Dense dynamic-domain allocations reuse the base URMA workspace and derive a
  slice of the base symmetric window. They do not require a second ACL
  `ImportByKey` flow.
- Documentation records that the A5 URMA base symmetric window currently keeps
  the HCCL Path-D allocation policy, `ACL_MEM_MALLOC_HUGE_FIRST`, and explains
  when to tighten it to `ACL_MEM_MALLOC_HUGE_ONLY`.

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

Status: complete; A5 hardware probes passed with the native PTO-ISA URMA
workspace path.

Goal:

- Prove real PTO-ISA URMA submit and data movement correctness on A5, without
  connecting the operation to AICPU deferred completion.

Accepted implementation:

- `examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/` is the hardware
  smoke for standalone real URMA.
- Host allocation uses a two-rank dynamic communication domain, with send,
  TGET receive, TPUT receive, and signal buffers inside the registered
  symmetric window.
- The kernel derives remote addresses through
  `UrmaPeerMrBaseAddr(workspace, peer) + offset`; it does not use
  `windowsIn[peer]` as a URMA remote VA.
- The full path uses PTO-ISA `TGET_ASYNC<URMA>` and `TPUT_ASYNC<URMA>` followed
  by bounded in-kernel `event.Test`/`event.Wait`.
- The full probe uses the native root-only pattern and does not use
  `DeviceBarrierBounded`, `CommRemotePtr`, or peer-VA barriers.

A5 hardware accepted probes:

```bash
python examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py -p a5 -d 0,1 --build --probe-stage workspace_info
python examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py -p a5 -d 0,1 --build --probe-stage remote_mem
python examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py -p a5 -d 0,1 --build --probe-stage tget_post
python examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py -p a5 -d 0,1 --build --probe-stage tget_root_post
python examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py -p a5 -d 0,1 --build --probe-stage tput_root_post
python examples/a5/tensormap_and_ringbuffer/urma_real_async_demo/test_urma_real_async_demo.py -p a5 -d 0,1 --build --probe-stage full
```

Acceptance decision:

- Phase 3 is complete for small and page-spanning real-submit correctness.
- This proves real URMA WQE post and data movement under PTO-ISA's blocking
  wait path.
- This does not prove AICPU deferred CQ polling, CQ/SQ tail update, CQ
  doorbell update, or fanout release under deferred completion. Those are Phase
  4 acceptance criteria.

## Phase 4: Real A5 Deferred Runtime

Status: implementation in progress; hardware acceptance pending.

Goal:

- Connect real PTO-ISA URMA events to the deferred completion path already
  exercised by the Phase 1 a5sim demo.

Implemented wiring:

- `UrmaTget` / `UrmaTput` descriptors use `send_request_entry` in
  `src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/urma/urma_completion_kernel.h`.
- The real onboard path builds a PTO-ISA URMA session, submits
  `TGET_ASYNC<URMA>` / `TPUT_ASYNC<URMA>`, and registers the returned event
  instead of calling `event.Wait`.
- Completion tokens use:
  - `addr = event.handle`
  - `backend_cookie = workspace pointer`
  - `engine = COMPLETION_ENGINE_URMA`
  - `completion_type = COMPLETION_TYPE_URMA_EVENT_HANDLE`
- `poll_urma_event_handle(event_handle, workspace)` decodes the handle,
  consumes send CQEs, and updates CQ tail, CQ doorbell, and SQ tail.
- `examples/a5/tensormap_and_ringbuffer/urma_real_deferred_demo/` is the new
  real A5 deferred smoke scene. It posts real URMA TGET/TPUT from AICore,
  relies on AICPU polling for completion, and validates dependent consumer
  execution after deferred completion.

Validation commands:

```bash
python -m pytest tests/ut/py/test_urma_real_async_demo_source.py -q
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build --target test_a5_urma_completion_scheduler
ctest --test-dir tests/ut/cpp/build -R test_a5_urma_completion_scheduler --output-on-failure
```

Required A5 hardware acceptance command:

```bash
python examples/a5/tensormap_and_ringbuffer/urma_real_deferred_demo/test_urma_real_deferred_demo.py -p a5 -d 0,1 --build
```

Hardware acceptance criteria:

- no scheduler timeout;
- no URMA CQE status/substatus error;
- AICPU polling observes real CQ readiness and retires the task;
- dependent consumer sees correct TGET data;
- TPUT data lands in the peer registered symmetric window;
- status tensors report `status[0] == 0` on both ranks for all cases.

Open Phase 4 risk:

- PTO-ISA's blocking wait path updates CQ tail and doorbell from AICore with
  device load/store. The deferred path updates them from AICPU via the scheduler
  mirror. A5 hardware must confirm that these AICPU stores are sufficient. If
  not, use one of the fallback paths in `../impl.md`: AICore blocking smoke,
  AICPU device-store primitive, or an AICore helper task for CQ retire.
