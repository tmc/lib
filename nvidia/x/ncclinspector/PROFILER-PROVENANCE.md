# Example profiler v7 trace provenance

`profiler.go` reads the Chrome-event JSON array that the NCCL example profiler
(`plugins/profiler/example`) writes with `NCCL_PROFILE_DUMP_FILE`. It is a
source-selected host importer. No plugin was built or loaded, no native
capture was taken, and no authentic trace is in this repository.

## Selected producers

| Profile | NCCL tag | Commit | `print_event.cc` SHA-256 | `plugin.cc` SHA-256 |
| --- | --- | --- | --- | --- |
| `ProfilerNCCL2323` | v2.32.3-1 | `12df1a11afad322be5a204a2db890161cbf8131d` | `0c7e4424d630a2448622043bf8205e739dd4a8425d6af1caec5b9726b6640a4c` | `30b1315ee3447ad508ed6b71d17d671d252f6e2028dcbef2a2f55685c94b4335` |
| `ProfilerNCCL2312` | v2.31.2-1 | `7b83616df3ae082a1f32bb74c27458bfe8153a13` | `0c7e4424d630a2448622043bf8205e739dd4a8425d6af1caec5b9726b6640a4c` | `7fe7f2cf1fdc030fbc29edaa262a3a5cd1c390a6c99e8600d94e77c3c10d467e` |

The two writers are byte-identical, but the profiles differ:

- Channel bound: `MAX_CHANNELS` is 64 in 2.32.3 and 32 in 2.31.2
  (`plugins/profiler/example/event.h`).
- Phase gating: in 2.32.3 `src/plugin/profiler.cc` sets `op->workPhases` only
  for symmetric collectives, so a phase record for regular or point-to-point
  work contradicts the source and the trace is refused. In 2.31.2 regular work
  is not gated; its phase records are retained but not admitted as durations.
- API event ranks: 2.32.3 sets `eDescr.rank` for group, collective,
  point-to-point API and kernel launch events; 2.31.2 leaves it zero. The
  example writer prints no rank on those events in either version, so nothing
  is recovered or discarded; the result lists `api_event_rank` as unavailable
  with the per-version reason. COLL and P2P ranks are printed in both versions
  and are kept for both.

- Symmetric channel accounting also differs: 2.32.3 changes it, and in 2.31.2
  the `nChannels` advertisement is gated by the KernelCh mask, so `nChannels`
  can be 0 for a symmetric collective. The importer therefore never compares
  channel counts with `nChannels`, and the two profiles do not qualify
  equally; byte-identical printing is not a qualification transfer.

The trace has no in-band version. The caller selects the profile and supplies
the plugin build identity and the artifact SHA-256; none of these is inferred
from a file name, process ID or event presence. Any other profile is refused.

Source excerpts and hashes were read from
`/Users/tmc/tmp/lib-wt-next/20261018-nccl-profiler-contract` (`CONTRACT.md`,
`PINS.json`, `MANIFEST.json`) and rechecked against the two commits with
`git show`. The contract directory is not part of this repository.

## What is imported

- COLL, P2P, KernelCh and `ph:X` phase events with exact printed spelling:
  `%lu` values (hash, sequence, count, GPU timers) parse as `uint64`, never
  through `float64`; `%f` timestamps are kept as strings; `%d` values are
  checked for form and range.
- Phase names, IDs, channels and printed `dur` must agree with the GPU timer
  endpoints (`dur` is `(stop-start)/1000` in microseconds, or 0).
- Phase endpoints are the producer boundaries `begin -> after_open`
  (`initial_sync`), `after_open -> before_close` (`compute`) and
  `before_close -> end` (`final_sync`) from `src/include/device.h`.
- Proxy, network, API, group and copy-engine events are retained as raw
  evidence, not interpreted.
- Input order, raw lines and framing are kept. Bounds: `Limits` bytes, line
  and event count, and the profile's channel bound.

Refusals: unsupported profile, build or digest missing or wrong, truncation,
missing terminal `{}]`, data after it, duplicate or case-aliased keys, null
or wrong-typed or unexpected fields, unknown categories, inconsistent
process or thread, nested or mismatched scopes, repeated or out-of-order
channels and phases, phase name/ID/channel/duration disagreement, and a
phase record the profile's source cannot produce. Disagreeing shared GPU
boundaries withhold the channel instead (see below).

## Stale pool slots

The producer reuses finite pool slots (default 8) and clears only the proxy
counts, while `printEvent` walks every `MAX_CHANNELS` slot. A channel the new
operation never ran can therefore print the previous occupant's KernelCh
data and phases inside the new COLL. The importer withholds (retains, not
admits, and never refuses) any KernelCh and its phases whose host begin
precedes the enclosing COLL or P2P begin, comparing the printed decimals
exactly, before the 2.32.3 phase gating. `BindProfiler` additionally refuses a
bound channel that is not among the record's `kernel_events` channel IDs when
the record lists any. A channel the record does not list because it has no
kernel events cannot be bound.

## Shared GPU boundaries

`profilerFireKernelPhaseEvents` (`src/plugin/profiler.cc`, both commits)
prints phase `i` as `ts[i] -> ts[i+1]` of one per-work `timestamps` array,
after folding a missing intermediate boundary onto its outer neighbour and
skipping zero-duration pairs. The device (`src/device/symmetric/kernel.cuh`,
and in 2.31.2 `src/device/common.h`) writes `ts[BEGIN]` and `ts[END]` with
the same `globaltimer` values it writes to `workStarted` and `workCompleted`,
which become the KernelCh `StartGpuClk` and `StopGpuClk`. In the trace:

- consecutive printed phases share their boundary exactly
  (`initial_sync.StopGpuClk == compute.StartGpuClk`, and likewise for
  `compute -> final_sync`; with compute skipped, `ts[after_open] ==
  ts[before_close]`, so `initial_sync.StopGpuClk == final_sync.StartGpuClk`);
- each printed phase with a recorded stop has `StartGpuClk < StopGpuClk`.
  The writer skips only zero or equal pairs and does not enforce order; the
  device takes its stamps in program order, so a reversed pair means mixed
  arrays;
- the first printed phase starts at the KernelCh `StartGpuClk` and the last
  printed phase stops at the KernelCh `StopGpuClk`, whichever phases were
  folded or skipped (a skipped leading phase means `ts[after_open] ==
  ts[begin]`, a skipped trailing one `ts[before_close] == ts[end]`), when the
  KernelCh value is recorded (nonzero).

A trace that breaks any of these mixes boundaries from more than one array
(for example phase callbacks of two works landing in one per-channel
`kernelCh` slot, or a device slot rewritten before the host read it; neither
is observed here, both are allowed by the source). The importer cannot tell
which value is current, so it withholds that KernelCh and all its phases with
the reason, before admission. It does not refuse the trace: the other
channels and the COLL facts are separate printed records that the conflict
does not touch. `NeedPhases` then refuses a binding that includes the
channel. A missing phase is still not loss, and a stop the producer had not
recorded (zero) stays unreported, not reversed.

## Association

A phase belongs to the KernelCh scope and the COLL or P2P scope that enclose
it in the file. `kernelId` is never incremented, so every KernelCh and phase
ID may be 0, and `tid` is the constant 1; neither is used, nor are
timestamps. `BindProfiler` selects the COLL scope by the caller's full
communicator hash, rank, operation and sequence, requires the process,
host name, Inspector record and datatype/count-to-bytes correspondence to
agree (the trace carries no hostname; the host is asserted and compared with
the record's `hostname`), refuses traces not produced by `ReadProfilerTrace`, and takes
channels only from the caller's explicit list.

## Observed versus asserted

`ProfilerBound` keeps trace facts (`Collective`, channel results, raw lines,
artifact digest) apart from caller-asserted facts (`Asserted`: incarnation,
evidence, plugin build, the clock mapping, channel selection, host).
The incarnation is never verified by anything in the trace or record. The
observation is the unmodified `ObservationFromRecord` result. Phase times
do not replace or augment Inspector stages. A phase `ts` is the host callback
time, not the GPU start; reported durations come only from the raw
`StartGpuClk`/`StopGpuClk` nanoseconds on one GPU's timer.

`ncclinspector` does not import `nccljoin` (which imports it). Tests in
package `ncclinspector_test` run Inspector `Read`, `nccljoin.Join`,
`BindProfiler` and `TwoRankAttributor` over a constructed two-rank fixture.
There is no second join.

## Not established

- No authentic trace. Every test trace and Inspector record is constructed
  from the pinned printf formats and the existing constructed two-rank
  fixture.
- No same-run Inspector v5 plus example v7 capture. NCCL selects one profiler
  plugin, and the two plugins have not been run together.
- Loss: the example keeps finite pools and exports no drop, eviction or
  population counts. A closed array proves framing only. A missing phase is
  not loss: the producer skips absent endpoints and zero-duration boundaries.
- Completion: the COLL end is a printed scope, not device completion.

## Unmet original requirements

- Watchdog and progress: `ncclProgressCountersBlock`, its pinned host mirror
  and refresh/copy validity are not exported to this trace or to RAS JSON. No
  schema is invented. Progress age, stale-mirror versus observed no-progress,
  and watchdog fields remain unsupported until the producer owner selects an
  export. `NeedProgressAge` and `NeedLoss` are always refused.
- Cross-rank phase ordering: GPU timers have no supplied calibration to a
  reference clock, so `NeedCrossRankTimeline` is always refused. Clock
  transform composition belongs to the stall.go owner.
- Native callbacks (v7 descriptors, event masks, lifetimes), authentic
  healthy and loss captures, observer cost per plugin and mask, and
  multi-node and backend cells are not covered.
