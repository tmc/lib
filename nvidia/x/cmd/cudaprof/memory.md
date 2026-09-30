# Managed-memory evidence

The optional native collector retains supported CUPTI UnifiedMemoryCounter3
migration/fault service records and managed-range query/advice/prefetch callbacks.
These are observations of service, policy and history. They do not observe
present physical page residency. Allocation type is not residency; last-prefetch
location is not current location; successful asynchronous prefetch is not
completion. UM service intervals do not contribute to recorded GPU occupancy.

The decoder qualifies CUDA 12.9 (CUPTI API 28) and CUDA 13.0–13.4 headers, checks
native struct size when the API is available, and bounds each record. Native
configuration return codes are retained per requested counter kind. ARM UM
activity collection remains refused by documented architecture policy even if
configuration succeeds. A measured native configuration refusal is separate
from that policy. Ordinary kernel capture remains available after optional
memory collection refusal. Neither an empty collection nor missing events proves
zero migration or faults.

Build a CLI and injection library using the normal project build. Under the
shared GPU lock, run a small controlled pre-window procedure:

```sh
cudaprof capture --memory-policy --unified-memory -o memory.pprof -- \
  cudaprof probe -placement -prefetch=device -advice=preferred-device
cudaprof memory memory.pprof > memory.json
cudaprof export -memory -o memory.pftrace memory.pprof
```

`probe -placement` allocates a 4096-byte managed range. It queries supported
scalar attributes before setup, before the controlled write scopes, and after
those scopes. Optional prefetch is followed by a separately reported native
context synchronization result. The two ordinary controls reset a sentinel and
verify output 7; optional counter replay verifies the same value. Query and
request receipts preserve native numeric errors and absent symbols separately.
Timing fields make full-stdout equality inappropriate for workload correctness.
The sentinel copies and output reads also touch the allocation; the procedure
records their context, not a guarantee that placement stayed unchanged.

Use `-prefetch=host`, `-advice=read-mostly`, or omit both actions for controls.
UM configuration-only `probe -um` runs separately from hardware range profiling.
No arbitrary captured operation is replayed by this memory procedure.

Perfetto memory tracks retain raw journal sequence, native value meaning,
virtual address and valid source/destination fields. The native UM stream field
is unusable and supplies no stream affiliation. Memory report operation windows
are only scoped timestamp overlaps when the collector retained a unified clock;
they establish neither submission identity nor causation. Unknown-clock host
policy events remain in JSON and are omitted from aligned memory tracks.
