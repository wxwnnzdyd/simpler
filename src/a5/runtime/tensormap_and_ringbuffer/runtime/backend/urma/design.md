# A5 URMA Deferred Completion Backend

## Purpose

This backend adapts PTO-ISA URMA async operations to simpler's deferred
completion model. AICore submits a real URMA WQE and records the returned event
handle in the deferred completion slab. The AICPU scheduler later polls the URMA
send CQ and retires the task only after the hardware completion is ready.

This is intentionally shaped like the SDMA deferred backend, but URMA completion
is not an event-record flag. URMA completion is represented by a PTO-ISA event
handle plus the native URMA workspace pointer needed to locate SQ/CQ context.

## Files

- `urma_completion_kernel.h`: AICore-side `UrmaTget`/`UrmaTput` descriptors,
  real PTO-ISA submit path, chunking, and deferred completion registration.
- `urma_completion_scheduler.h`: AICPU-side URMA workspace ABI mirror, CQ
  polling, and CQ/SQ tail/doorbell update.
- `pto_async_wait.h`: completion-type dispatch that routes
  `COMPLETION_TYPE_URMA_EVENT_HANDLE` to this backend.

## AICore Submit Contract

A kernel submits URMA through:

```cpp
send_request_entry(async_ctx, UrmaTget(dst, src, workspace, peer));
send_request_entry(async_ctx, UrmaTput(dst, src, workspace, peer));
```

The real A5 path requires:

- `PTO_URMA_SUPPORTED` is available in the PTO-ISA build;
- `workspace` points at PTO-ISA's native
  `pto::comm::urma::UrmaWorkspaceManager::GetWorkspaceAddr()` output;
- `peer` is the URMA workspace peer index;
- source and destination addresses passed to PTO-ISA are URMA MR-valid;
- real remote addresses are derived from the URMA MR base, not from SDMA-style
  peer VAs.

Use the backend helpers for remote MR address derivation:

```cpp
uint64_t base = pto2::urma_backend::peer_mr_base_addr(workspace, peer);
__gm__ float *remote =
    pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, offset);
```

When the task has a valid deferred `AsyncCtx`, the backend registers a
completion token instead of waiting inline:

- `addr = event.handle`
- `backend_cookie = workspace`
- `engine = COMPLETION_ENGINE_URMA`
- `completion_type = COMPLETION_TYPE_URMA_EVENT_HANDLE`

When the task has no deferred `AsyncCtx`, the backend falls back to
`event.Wait(session)`. This keeps the helper usable outside the deferred
runtime path.

## Scheduler Poll Contract

`poll_urma_event_handle(event_handle, workspace)` decodes the event handle as:

```text
remote_rank = event_handle >> 32
target_head = event_handle & 0xffffffff
```

The scheduler then:

1. loads `UrmaInfo` from the workspace;
2. locates SQ and send-CQ context for `remote_rank * qp_num`;
3. invalidates CQE cache lines before reading CQE DW0;
4. consumes ready CQEs until the CQ tail reaches `target_head` or the next CQE
   owner bit is not ready;
5. fails if CQE `status` or `substatus` is non-zero;
6. writes CQ tail, CQ doorbell, and SQ tail when progress is made;
7. returns `READY` only after the consumed tail reaches `target_head`.

A zero event handle is treated as already ready. A null workspace, invalid rank,
missing SQ/CQ pointers, unsupported CQ shape, or CQE error returns failed.

## Simulation Path

The fake a5sim URMA submitter was removed after the real A5 path passed
hardware validation. `send_request_entry` now reports an async-completion
error when compiled without `PTO_URMA_SUPPORTED`, rather than emulating URMA
data movement in software.

## Validated Status

The current real A5 deferred demo has validated:

- real TGET and TPUT submit through PTO-ISA URMA;
- AICPU polling of real send CQEs;
- CQ tail, SQ tail, and CQ doorbell updates from AICPU;
- dependent consumer execution after deferred completion;
- size sweep for `1/16/64/256/4096/16384` float elements;
- multiple URMA completion entries from one task;
- concurrent TGET and TPUT deferred tasks before one consumer.

One earlier direct run produced a single `count=16` TGET mismatch and later
retries did not reproduce it. Treat that as a low-frequency observation unless
it recurs during `--repeat` stress.

## Known Limits And Next Entry Points

The current backend is a first runnable real-hardware path. Phase 5 work should
start from these entry points:

- **Transfer splitting**: `send_request_entry` now chunks flat contiguous
  transfers larger than `kUrmaMaxTransferBytes` before submission. The next
  step is to validate the split path on hardware with
  `urma_real_deferred_big_demo` and extend it beyond the flat 1-D case if
  needed.
- **Event coalescing**: coalesce per-task/per-peer URMA events before writing
  deferred completion entries. URMA quiet semantics allow the largest target
  SQ head for a peer to cover earlier events on the same QP.
- **Address helper**: `peer_mr_base_addr` and `peer_mr_ptr` now centralize
  `UrmaPeerMrBaseAddr(workspace, peer) + local_offset` derivation. Future
  callers should use these helpers instead of calling PTO-ISA directly.
- **Non-dense rank mapping**: update host workspace ownership and peer-index
  mapping before enabling URMA for non-dense derived contexts.
- **CQE diagnostics**: extend `poll_urma_event_handle` error reporting beyond
  the current generic invalid-completion code if runtime status propagation can
  carry the extra detail.
- **DFX integration**: add log/parser support for URMA poll failures and CQE
  progress counters after the hardware behavior is stable.

Do not replace the native PTO-ISA `UrmaWorkspaceManager` with a private
workspace converter. The host platform path already wires the native workspace
through `CommContext::workSpace`; the backend depends on that ABI.
