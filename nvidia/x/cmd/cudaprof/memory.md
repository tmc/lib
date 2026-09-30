# Memory evidence

```
cudaprof memory capture.pprof
```

The JSON report retains capture identity, completeness and collector capability.
It lists memcpy records with their journal, device, stream, raw copy kind,
bytes, timestamps and correlation ID. Allocation API calls are counted by
API domain and name. These are call counts: arguments, return codes,
allocation pointers and lifetimes are not retained. Driver and runtime calls
may describe the same allocation and are reported separately.

| Evidence | Current collector/decoder | What it establishes |
| --- | --- | --- |
| Memcpy activity | Collected and decoded | Recorded transfer bytes and interval |
| Allocation API activity | Optional driver/runtime API collection | A call with an allocation API name occurred |
| Allocation success and lifetime | Not retained in API activity | Unavailable |
| Per-range current residency | No observation records | Unsupported by current collector |
| Managed-memory migration | Unified-memory counters not enabled or decoded | Unsupported by current collector |
| CPU/GPU page faults | Unified-memory counters not enabled or decoded | Unsupported by current collector |
| Current device/backend counter support | No probe performed | Unprobed, not hardware-unsupported |

Collector initialization enables memcpy activity in
`internal/cudainject/native/cupti_agent.c`. API collection is optional; the
capability receipt records `driver_api`, `runtime_api` and `api_records`.
The timeline decoder retains API names/timestamps, not parameters or results.
Missing transfers or allocation calls remain unavailable evidence rather
than becoming zero migration, successful allocation counts, or device residency.

CUDA exposes managed-memory counters, but configuration must precede activity
enablement and can fail for platform, device, or multi-device peer-access
reasons. A collector implementation must retain those failures and probe
results before claiming supported collection. See the
[NVIDIA CUPTI activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html).

The existing Go driver binding exposes `DevicePtr.MemRangeGetAttribute` and
`HasMemRangeGetAttribute`. Symbol availability does not establish device
support. Preferred-location attributes describe advice; last-prefetch
attributes describe a request and do not establish completion or current
location. The report therefore does not treat them as residency evidence.
See [NVIDIA unified addressing](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__UNIFIED.html).

## Controlled pre-window investigation

This is an investigation procedure, not a measured demonstration. Keep one
shared GPU-lock hold around the workload and controls. Run queries inside
the allocating process; a pointer from another process is not a valid query
handle.

1. Record device model, driver/CUDA/CUPTI versions, OS, and managed-memory
   device attributes. Probe unified-memory counter configuration separately
   before enabling it, retaining exact error codes. Current cudaprof does
   not implement this collection; use a supported external collector or a
   future collector implementation that preserves these receipts.
2. Allocate a small managed-memory region. Record allocation API, pointer,
   byte count, and success independently of CUPTI API activity. CPU-touch
   the region outside the timed operation window.
3. Before the window, query preferred location and last-prefetch location
   through `MemRangeGetAttribute` when available. Record query start/end,
   attribute, result, and error. Use a 32-bit result for those attributes;
   preserve invalid/mixed-range results. Label the values advice and prefetch
   history, never physical residency.
4. Run a GPU operation that touches the same region, bracketed by explicit
   operation markers. Validate deterministic output. Repeat with explicit
   GPU prefetch and stream synchronization before opening the window. These
   are different placement requests, not guaranteed residency controls.
5. If a supported collector emits unified-memory migration or fault records,
   retain raw address, size/value, processor IDs, timestamps and counter
   kind. Correlate only when the clock mapping and allocation identity are
   established. Compare observed evidence and errors between controls;
   absence of records under unsupported collection proves nothing.
6. Repeat without profiling to assess overhead before interpreting timing.
   Preserve failed completeness and stationarity gates. Only after the
   controlled records are validated should the same procedure be applied
   to Gemma4.

Existing Perfetto export already represents observed memcpy intervals. There
are no new residency, migration or fault tracks because the collector has no
such records. This command does not manufacture them or correlate separate
host-clock observations to GPU windows.

CPU-only checks:

```
cd nvidia
go test ./x/cmd/cudaprof -run TestMemory
```
