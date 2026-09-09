# Late activation and attachment for running Python CUDA programs

Status: proposed / TODO. Date: 2026-09-09. Planning only; no attachment mechanism or hardware behavior was implemented or qualified for this document.

## Outcome

Capture useful Python and CUDA profiling data from a job that is already running, without requiring a restart. Prefer enabling an agent that was loaded at startup, then support cooperative late loading, and qualify unprepared process attachment separately. Reuse [spec 005](005-third-party-cuda-capture-spec.md) for collector orchestration, typed observations, process identity, portable recordings and offline viewing.

The main use case is a long-running training job: request a window covering the next few complete steps, correlate Python/operator origins to CUDA work, and inspect the resulting standalone pprof capture. This can provide strong evidence about steady-state execution even when startup was not recorded. It cannot recreate past events or prove why an allocation made hours earlier occurred.

Keep `cudaprof` in `nvidia/cmd/cudaprof`. Reuse plan 003's CUPTI collectors; deeper CUDA samples remain owned by session `6DE5`. This plan does not change that assignment or the separate Metal work. Implementation is not authorized by this planning file alone.

## Three distinct modes

| Mode | Mechanism | Expected strength | Main limitation |
| --- | --- | --- | --- |
| Preloaded agent, late activation | Load a small agent at startup; activate selected collectors through a control endpoint | Most reliable control and optional retained module/allocation history | Agent presence, subscriptions and metadata retention still have overhead and may conflict with other profilers |
| Cooperative late attach | Target explicitly loads/starts the agent through a Python package or existing control hook | Good future coverage, especially at a training-step boundary | Earlier code/resource/task history must be reconstructed where possible or marked unknown |
| Unprepared PID attach | Discover a process, sample externally where supported, and separately attempt qualified native/Python agent loading | Useful incident investigation without prior setup | Injection, interpreter versions, permissions and runtime state can prevent full collection |

Do not present these as interchangeable implementations of the same coverage guarantee. CUPTI's dynamic attach means initializing CUPTI inside the target process; it does not supply a general external PID injector. NVIDIA describes this distinction through its lazy initialization and `cuptiFinalize` lifecycle in [Dynamic Attach and Detach](https://docs.nvidia.com/cupti/13.0.0/main/main.html).

## Proposed interface

These commands define intended behavior, not existing commands:

```sh
# Start a job with an agent available for later activation.
cudaprof capture --armed --history=metadata -- python train.py

# Discover agents belonging to the current user.
cudaprof agents

# Collect from a cooperating agent at a complete step boundary.
cudaprof attach --pid 1234 --window=nvtx:train-step --ranges=5 -o steps.pprof

# Request only externally sampled Python stacks from an unprepared process.
cudaprof attach --pid 1234 --mode=external --duration=30s -o cpu.pprof
```

Default `attach` MUST use a verified cooperating endpoint and fail clearly when none exists. Native injection into an unprepared process requires an explicit mode and separately qualified implementation; no silent debugger-based fallback. External Python sampling MUST be labeled CPU sampling and MUST NOT claim exact GPU launch attribution.

Process selection MUST verify user, PID generation/start identity and executable/interpreter identity. Local endpoints require peer authentication and permissions restricted to the owning user. Endpoint messages MUST NOT provide arbitrary code execution or shell-command APIs. Do not signal an arbitrary process to activate an agent unless that process has registered the protocol; an unhandled signal can terminate it.

Specify duration, matching-range count, maximum wait for the start boundary, buffer/storage budgets and requested collectors. If no range matches before the wait bound, return `window_not_observed`, not an empty successful GPU profile. Do not assume every program emits a marker named `train-step`; support explicit user-selected NVTX or cooperative boundaries and a separately labeled time-window mode.

## Information retained, recovered and lost

| Information | After activation | Historical limitation |
| --- | --- | --- |
| CUDA APIs, kernels, copies and synchronization | Collect supported future activities and correlations | Events before the observation boundary are absent; in-flight work may have incomplete origin/start data |
| Python launch stacks | Capture at future submission points when integration supports them | Current stacks cannot reconstruct earlier submission stacks |
| Framework/operator origins | Propagate newly observed operator/task IDs across asynchronous boundaries | Existing tasks and autograd work may have unknown origins |
| PC, PM, SASS and range measurements | Enable supported collectors at a qualified boundary | Initialization constraints and conflicts are collector-specific; availability is not assumed merely because activity tracing works |
| Allocations | Observe future alloc/free; record available current-state snapshots | Existing live bytes do not reveal allocation time, stack, total historical allocation volume or historical peaks |
| Code images and symbols | Capture new modules and attempt supported discovery of already-loaded images | Missed module callbacks may leave missing images/source maps; names alone are insufficient for PC attribution |
| CUDA Graphs | Record future launches and available node/graph metadata | Construction history, original Python frames and prior mutations may be absent |
| Device telemetry | Collect future samples and supported current state | No reconstruction of previous power, clocks, utilization or throttling |
| Other workers/ranks | Activate individually through verified agents | Attaching to one Python PID does not establish distributed-job coverage |

Recoverable state MUST be marked as a snapshot or reconstruction with its mechanism and timestamp. It MUST NOT be relabeled as an observed historical event. Unknown allocation size or origin remains unknown when a free occurs; do not make the live-byte ledger negative or fabricate a zero-byte allocation.

## Python stack and CUDA correlation

Support separate producers for statistical Python CPU stacks, stacks captured at CUDA submission, and framework-origin stacks propagated across worker queues. Store Python function, file, line, interpreter identity and an immutable interned stack ID. Collect source only under the explicit artifact policy from spec 005; do not capture locals, credentials, tensors or arbitrary object representations by default.

Associate launch stacks with CUPTI correlation IDs and stable process/thread identities. Framework task/operator IDs are required when work crosses threads or executes later. Do not attach the Python stack visible at GPU completion to that kernel. Compiled/fused operators, autograd workers and CUDA Graph replay require explicit attribution modes; a graph replay invocation stack and its original graph-construction stack are different records.

An agent loaded mid-call may first observe a return, nested native work or a continuation without the original call event. Mark this boundary and resume complete attribution at a known valid point. Existing active NVTX ranges or Python frames may be snapshotted if supported, but their true entry time remains unknown.

CPython version, extension/runtime compatibility, subinterpreters and free-threaded builds require an explicit support matrix. Never acquire the GIL or walk mutable Python frames arbitrarily from CUPTI callbacks without a proven safe protocol. Prefer immutable stack/task state maintained by the language agent at safe points. A Python integration failure must not be silently converted into exact native-to-Python attribution.

## Dormant agent and retained history

Offer explicit history policies:

- `none`: agent control only; do not promise prior module or allocation knowledge.
- `metadata`: bounded module/resource identities and selected task/range metadata.
- `flight`: a bounded event history in addition to metadata, with independently selected collectors.

“Armed” is not zero overhead and does not mean CUPTI is unsubscribed. State which subscriptions and buffers remain active. Measure idle-agent and history-policy overhead on real workloads, including memory use, module churn, throughput and tail latency. Make conflicts with framework/Nsight subscribers visible before activation.

History must have byte/time bounds and explicit oldest retained time, evictions and loss counts. A rolling event buffer alone cannot supply a correct live-allocation inventory after allocation records are evicted; maintain a bounded state snapshot or mark that inventory incomplete. Record overflow instead of silently discarding identity state. Retain code bytes only under the chosen artifact/size policy.

## Activation, stopping and ownership

Use a state machine with acknowledged transitions:

`available -> preparing -> armed-for-boundary -> collecting -> draining -> finalized`

Failures produce an explicit failed/partial state, including the last acknowledged transition. Assign new run and collector epochs on each activation. Allow repeated windows without merging them into one timeline. Prevent conflicting controllers from reconfiguring the same agent concurrently.

Preparation inventories runtime capabilities, existing contexts/devices/modules where possible, current profiler ownership and requested collector compatibility. Initialization MUST occur through safe target-side control flow. No CUPTI initialization/finalization, Python runtime calls or buffer flushes from arbitrary asynchronous signal handlers.

A requested clean step boundary is preferable to interrupting active work. If establishing the boundary requires synchronization or serialization, record the intervention and obtain it through the selected explicit collection policy. Do not globally synchronize other ranks by assumption. Work crossing the start/stop boundary must carry partial-interval/origin status rather than invented complete duration.

Stop disables future collection, drains supported buffers and preserves final loss/coverage metadata. Distinguish **stop recording**, **detach CUPTI**, and **unload agent**. The default SHOULD stop recording while leaving a safe dormant agent loaded. Detach requires CUPTI-safe synchronization and lifecycle handling; unloading native/Python code while callbacks or threads can reference it is prohibited. Never finalize another tool's CUPTI session.

The controller owns only its session and verified helper processes. It MUST NOT kill the target or unrelated profiler processes when an attach operation times out. If shutdown or writer ownership is unresolved, retain scratch and report the pending state. Existing training must continue unless the user explicitly requested an intervention requiring a pause or termination.

## Capture coverage contract

Extend spec 005's manifest with:

- Attach mode, target generation identity, agent load time, attach request/acknowledgment and observation start/end for each clock domain.
- Requested versus actual start/stop boundary, in-flight work policy and interventions.
- History policy, retention start, snapshot timestamps, dropped/evicted record counts and known missing prehistory.
- Per-family baseline state: observed since startup, recovered snapshot, observed since activation, or unknown.
- Python attribution mode and missing-origin counts; process/rank/device activation coverage.
- Per-collector initialization, collection and drain results, conflicts and unresolved ownership.

Viewer profiles MUST state their observation window. Missing historical allocation origins, graph origins and code maps must be visible at affected records, not only in a summary footer. Do not emit historical peaks or lifetime totals from a late snapshot. Preserve partial boundary records in the event view even if excluded from an additive pprof projection.

Late capture can be highly informative for repeated steady-state steps. It MUST NOT claim representativeness of unobserved steps or phases without evidence. No percentage-of-information claim is defined: coverage is assessed by family, history and attribution quality.

## Implementation sequence and acceptance

1. **Preloaded control path:** authenticate discovery and repeated start/stop windows; preserve ordinary target behavior. Test no-CUDA targets, PID reuse, stale endpoints, concurrent controllers and conflict rejection.
2. **Mid-run boundary witness:** start a deterministic training-like program, activate after several completed steps, capture the next complete steps and compare their numerical results and event counts with a from-launch recording. Account for instrumentation perturbation; do not compare timings for equality.
3. **Historical gap controls:** allocate memory, load/unload code, enqueue work and construct a CUDA Graph before activation. Assert correct snapshot/unknown states, no invented origins, and correct treatment of boundary-crossing events. Test history eviction and allocation-address reuse.
4. **Python attribution:** nested Python calls and asynchronous worker execution with known task IDs; verify future kernel launch stacks and explicitly unknown pre-activation origins. Test a compiled/fused path and graph replay separately. Sampling stacks must not pass as launch-correlation evidence.
5. **Cooperative late loading:** qualify actual supported CPython/PyTorch versions, late module discovery, existing contexts and safe repeated activation. Fail unsupported initialization cleanly without damaging the running job.
6. **Distributed activation:** two real GPUs/ranks, with one deliberately unavailable agent. Verify full success only with all requested coverage and explicit partial result otherwise; retain clock uncertainty and per-rank boundaries.
7. **Deep collectors:** qualify each plan 003 collector in late mode, not just launch mode. Confirm initialization constraints, profiler conflicts, sampling loss and stopped-window flushing. No implicit kernel/range/application replay of a live training job.
8. **Unprepared attachment:** implement external Python sampling first. Native injection requires its own design and subprocess witnesses for permissions, incompatible interpreters, blocked threads, fork/exec and cleanup. Existing lower-coverage modes must remain usable when injection is unavailable.
9. **Offline and failure gates:** copied single-file capture renders window/coverage/source information on Darwin and Linux. Target/controller crashes, disk-full, lost IPC and timeout preserve explicit partial data. A negative control proves unrelated same-input processes survive all cleanup paths.

All native gates require retained commands, tool/source identities, target output checks, timestamps, status/loss records and artifact hashes under `~/tmp`. Software tests and skipped hardware tests do not qualify attachment. Record performance overhead separately for dormant, metadata-history, CPU-sampling and deep-collection modes. Implementation can deliver the preloaded/cooperative modes before unprepared injection, but must report those as distinct milestones.

## Completion boundary

This plan is complete only when the selected mode's supported-platform matrix and acceptance gates are documented with executable evidence. A successful late activity window does not establish recovered startup history, full Python attribution, safe detach or support for every hardware collector. The capture should make those differences inspectable without relying on the person who recorded it to explain them.
