# Inference UI and telemetry roadmap

This roadmap orders visualization work by evidence availability. A panel is not
complete when it can draw a chart; it is complete when the runtime supplies the
state and provenance needed to interpret that chart.

Status: partial implementation. Several dashboard panels exist, but some inputs
are modeled, configured, illustrative, or unavailable.

## State vocabulary

Every displayed value uses one of these states:

| State | Meaning |
| --- | --- |
| `observed` | Recorded by the runtime or device over a stated interval |
| `modeled` | Calculated from declared inputs and a versioned formula |
| `configured` | Supplied as a capability or policy value |
| `illustrative` | Example data that did not come from the active runtime |
| `unavailable` | No trustworthy value exists |

The UI must show the state near the value. It must not replace missing values
with plausible defaults. In particular, configured 40 TFLOPs or 800 GB/s
fallbacks are not observations and should not appear as active-device facts.

## Current surfaces and gaps

| Surface | Available today | Gap |
| --- | --- | --- |
| Request history and latency | Host-observed generation records | Needs explicit timing boundaries and plan identity |
| GPU timeline | FlightRecorder regions | Regions may cover whole forward or prefill work, not individual kernels |
| Roofline | API can represent unmeasured points; `gpuprof` emits analytical rates | No current cross-backend hardware-counter source for observed DRAM bytes and FLOPs |
| Graph plan | Small illustrative graph endpoint and a descriptive configuration plan | Neither is proof that extraction selected and executed a plan |
| Placement | GPU/ANE suggestions can be rendered | No ANE executor is wired; placement is advisory |
| Memory | Observed or configured allocation data and unavailable states exist | Scope and allocator coverage must be shown |
| Prefix cache | Metal-side prefix cache data exists | Not a portable cross-backend feature |
| Queue | Request admission and bounded scheduler state exist | Current execution is serial across sequences, not continuous batching |
| Mesh | Peer and configured capability data can be displayed | Discovery and configured ceilings do not prove route quality or usable throughput |
| Perfetto | Dashboard requests an embedded trace | Server route for `/debug/perfetto.pftrace` is not registered |

## Telemetry envelope

Each API payload should carry:

- schema version and collection time;
- source revision and dirty state;
- backend, device, model, and plan identity;
- scope: process, engine, request, phase, operation, sequence, or peer;
- state and provenance for every optional numeric field;
- start, end, and synchronization boundaries for observations; and
- coverage or reason when data is partial or unavailable.

Browser code should not infer provenance from a field name. Prefer a typed
value such as `{value, unit, state, source}` over a bare number.

## Delivery phases

### 1. Provenance and unavailable states

Before adding panels:

- remove or relabel synthetic hardware defaults;
- align server and `gpuprof` terminology;
- distinguish host wall time, device time, hardware counters, and formulas;
- make unsupported fields explicit rather than zero; and
- add response-schema tests for every state.

Acceptance: fixtures containing unavailable data render as unavailable and
never as zero utilization.

### 2. Executed-plan identity

Thread one immutable plan identity through compilation, engine storage,
scheduler dispatch, telemetry, and trace events. Keep four graph kinds distinct:
`illustrative`, `candidate`, `selected`, and `executed`.

Acceptance: the UI displays `executed` only when the backend records the same
identity. A configuration-derived operation list remains `descriptive` or
`illustrative`.

### 3. GPU timeline

Render exactly the regions the runtime records. Separate host work, queue or
stream work, transfers, synchronization, and readback. Show collection source
and coverage.

Acceptance: a trace with one whole-forward event displays one event. Empty or
unsupported traces do not produce a kernel waterfall. Recording overhead and
allocation behavior are benchmarked; the current FlightRecorder label copy is
not described as zero-allocation.

### 4. Roofline and workload model

Display two independent layers:

- modeled algorithmic work and model-weight traffic from declared model and
  dtype inputs; and
- observed device counters only when a named counter source covers the same
  measured interval.

Do not divide modeled bytes by time and label the result observed bandwidth.
Show formula version, exclusions, configured peak source, and whether embeddings
are tied.

Acceptance: no observed roofline point is emitted from FlightRecorder timing
alone. Counter loss or partial queue coverage produces `unavailable` or
`partial`, not a fallback formula.

### 5. Memory and KV state

Show device allocations, allocator coverage, KV pages by sequence, prefix-cache
entries where implemented, and high-water marks. Mark modeled tensor sizes as
modeled and allocator-reported residency as observed.

Acceptance: two interleaved sequences display disjoint KV ownership; reset,
cancellation, and capacity reuse update the view without stale ownership.

### 6. Scheduler and batching

Until real batching exists, show the current queue and serial step transitions.
After batched APIs exist, add batch composition, wait time, phase, bucket,
sequence position, padding, and reason for exclusion from a batch.

Acceptance: the UI's batch size equals the backend launch's sequence dimension.
Queue length or scheduler capacity is not reported as executed batch size.

### 7. Mesh and distributed plans

Show membership epochs, node capabilities, observed link probes, model
availability, and distributed plan placement. Configured TFLOPs and bandwidth
remain configured. A discovered peer is not displayed as a proven direct route.

Acceptance: topology transitions expose prepare, state-transfer, activate,
drain, and failure states. Degraded transitions are visually and semantically
distinct from exact ones.

### 8. PGO and candidate search

Display workload buckets, candidate outcomes, correctness gates, distributions,
selected plan, publication epoch, and rollback. Do not animate a live tuning
loop before immutable plan publication and retained calibration records exist.

Acceptance: rejected and timed-out candidates remain visible; the UI can explain
why the selected plan was eligible and when its measurement was collected.

## Visual contracts

### Graph explorer

- Color encodes semantic state, not decoration.
- Suggested ANE or GPU placement is labeled `suggested` until executed.
- Candidate rejection reasons are visible.
- Tensor shapes and dtypes come from the plan schema, not browser guesses.

### Timeline

- Time axes state their clock source.
- Gaps and synchronization are visible.
- Partial coverage is shaded or labeled.
- Plan and operation links require matching identities.

### Roofline

- Modeled and observed series use different marks and legends.
- Peaks identify their configured source.
- Missing hardware counters remove the observed point.
- Example values in screenshots or fixtures are labeled `illustrative`.

### Queue and mesh

- Admission, waiting, executing, cancelled, and failed are separate states.
- Capacity is not utilization.
- Peer capability is not link performance.
- Exact and degraded distributed plans are separate modes.

## API and browser constraints

- Bound response and trace sizes.
- Escape all runtime strings.
- Protect diagnostic endpoints with the server's authorization policy.
- Avoid prompts, generated text, tensor contents, local paths, and credentials
  in telemetry by default.
- Validate iframe source and origin for embedded Perfetto messages.
- Provide accessible text and tabular fallbacks for charted data.

## Milestones

1. Telemetry state and provenance schema; remove misleading defaults.
2. Executed-plan identity and truthful graph kinds.
3. Trace endpoint and exact-region timeline.
4. Modeled roofline, followed separately by observed counters where supported.
5. KV ownership and memory views.
6. Real batch visualization after batched execution lands.
7. Distributed epoch and transition view after exact distributed plans exist.
8. PGO candidate and publication view after bounded calibration exists.

This ordering makes the UI a diagnostic surface for real runtime behavior. It
does not block backend development, but it prevents visualization work from
creating stronger claims than the engine can support.
