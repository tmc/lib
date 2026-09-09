# Third-party CUDA capture specification

Status: proposed, not implemented or qualified. Date: 2026-09-09.

## Purpose and definition of completeness

Provide `cudaprof capture -- command args...` for unchanged CUDA applications, including Python training, native executables and Go programs. Produce a portable, inspectable recording containing the fullest defensible account of execution, measurements, code and attribution attainable under the selected collection policy.

“Complete” means every requested measurement and artifact has a recorded disposition, every retained value has provenance and scope, all observed processes/devices are accounted for, and loss is disclosed. It does not mean all counters can run simultaneously, every binary contains source/PTX, or an arbitrary training process can be replayed correctly. The maximum capture is a collection of explicitly related runs and homogeneous profiles, not one flattened vector of all metrics.

This specification extends plans 001–003. Plan 003 and session `6DE5` own the underlying CUPTI collector and sample-type work. Reuse those implementations; this specification adds third-party launch/injection, attribution, process ownership, session orchestration and the portable product contract. Plan 004 remains separate Metal work. This document does not authorize an implementation handoff, cloud provisioning, commits or changes to other agents' assignments.

MUST, SHOULD and MAY below describe required behavior, preferred behavior and optional extensions. Claims of completion require the acceptance evidence at the end.

## User contract

Proposed CLI; these commands are specification examples, not currently implemented commands:

```sh
cudaprof capture -o train.pprof -- python train.py
cudaprof capture --preset=deep --window=nvtx:train-step -o step.pprof -- python train.py
cudaprof capture --preset=exhaustive --replay=application --max-runs=8 -o study.pprof -- python train.py
cudaprof capture --follow=children -o distributed.pprof -- torchrun --nproc-per-node=2 train.py
cudaprof inspect --coverage train.pprof
cudaprof inspect --profiles train.pprof
cudaprof -http=127.0.0.1:8087 train.pprof
```

The launcher MUST pass argv directly without an implicit shell, preserve stdio/cwd and ordinary environment behavior, and apply instrumentation variables only to owned child processes. It MUST record explicit injection success/failure per process. A command that never initializes CUDA gets a valid `no_cuda_observed` result, not fabricated zero GPU samples.

| Preset | Intent | Required policy |
| --- | --- | --- |
| `trace` (default) | Useful unchanged-program execution record | Activity/callback correlation, code discovery and available markers; single execution; no implicit replay or forced serialization |
| `deep` | Locate instruction, scheduler and memory bottlenecks | Add selected supported PC/PM/SASS/range collectors; disclose incompatible requests and perturbation; replay stays off unless explicitly selected |
| `exhaustive` | Account for every discoverable profiling family/metric | Enumerate and plan all applicable requests; execute only allowed modes within budgets; leave incompatible/unexecuted requests visible and mark incomplete |

`--collect` selects native IDs or named families; `--exclude` narrows them. `--replay=off|kernel|range|application` is independent of preset. `--window` selects supported time, launch-index, NVTX or cooperative ranges. A window that never matches MUST be reported. `--max-runs`, duration, buffering, output size and overhead policies MUST be explicit and retained. Reaching a budget preserves partial data and records what was omitted; it never converts an exhaustive request into an undocumented subset.

Before launch, show the planned collector groups, permissions/capabilities, replay count estimate when knowable and conflicting profiler state. No silent change to driver settings, clock settings or privilege configuration. `--require-complete` MUST reject a known unsatisfied plan before starting the workload; unforeseen runtime loss still yields a partial recording.

A successfully launched target's nonzero exit code MUST be preserved. On target success, return 0 only if the requested capture policy succeeds; use documented distinct nonzero codes for capture failure, incomplete required coverage and launcher failure. Store target exit/signal and capture outcome separately. Signal forwarding MUST affect only verified owned processes. A timeout does not prove all GPU or process work stopped.

## Architecture

1. **Go launcher/controller:** subprocess lifecycle, per-process handshake, capability negotiation, collection windows, signal forwarding, bounded ingestion, durable checkpoints and final packaging.
2. **Minimal injected native agent:** initialized inside each CUDA process through the qualified injection mechanism; CUPTI subscription, activity buffers, callbacks, safe module-byte capture and transport. Avoid blocking I/O, recursive CUDA calls, language-runtime initialization or unbounded allocations in callbacks. Callback legality and flush timing MUST follow the selected CUPTI version.
3. **Existing public NVIDIA packages:** collector setup, native decoding, correlation, typed observations, pprof projections and `x/cudacapture` transport. Keep native injection separate from the pure offline viewer. Prefer a small C ABI shim over embedding another Go runtime into arbitrary third-party Go processes; any alternative needs explicit loader/fork/runtime qualification.
4. **Optional language/framework agents:** Python frames, framework operator ranges, Go trace/pprof labels and NCCL metadata. These are separately enabled and versioned; their absence cannot invalidate baseline CUDA activity capture.

Initial native targets are Linux amd64 and arm64 with actual CUDA execution qualification. Offline reading MUST work on Darwin and Linux without CUDA, Python, the original application or private paths. Windows injection and running-process attachment are later independently qualified capabilities; launch support must not imply attach support.

## Command and package placement

Keep the executable at `nvidia/cmd/cudaprof`, with install path `github.com/tmc/lib/nvidia/cmd/cudaprof`. Keep reusable APIs under `nvidia/x`; do not introduce `nvidia/x/cmd`. The command owns CLI/process policy; packages own collectors, records and transport. Internal injection/IPC implementation may live under `nvidia/internal` if it is used only by this command, while externally usable collector APIs remain public. Native agent build artifacts must have explicit target architecture and ABI/version identity and must not be committed as binaries.

Preserve current pprof flag invocation and `-pack`/`-extract` behavior while adding subcommand dispatch. Parse capture options only before `--`; all following arguments belong to the target, including flags named like cudaprof flags. Missing commands and invalid collector policies must fail before spawning anything.

## Required measurement inventory

Inventory installed headers, binding coverage, driver/CUPTI versions, device support and runtime-discovered native metrics. Record each feature's collector, decoder, exporter, support predicate, source definition and acceptance witness. An API declaration is not a working collector.

| Family | Maximum desired capture | Attribution and qualifications |
| --- | --- | --- |
| Runtime/driver activity | API entry/exit, kernel launches, copies, memsets, synchronization, errors and overhead | Process/thread, callback and activity correlation, context/stream; preserve nesting without double-counting driver/runtime calls |
| GPU execution | Start/end, queue/submit observations where exposed, launch dimensions, resources, graph execution/node identities | Distinguish observed enqueue delay from unexplained gaps; API exit is not universally a valid submission timestamp |
| Allocation and memory lifecycle | Alloc/free, async pools, reservations/commitment, residency/migration/page-fault records where exposed | Allocation generation and address reuse; separate allocation volume, live bytes, peak bytes and physical traffic |
| Copy and topology | H2D/D2H/D2D/P2P direction, endpoints, bytes, duration, physical topology and supported link counters | Logical copied bytes are not necessarily physical PCIe/NVLink traffic; topology alone is not traffic |
| PC sampling | Native PCs, counts, all discovered stall reasons and latency/not-issued distinctions, sampling configuration and dropped counts | Exact code image/offset; sampling is statistical, not an instruction execution census |
| Scheduler/occupancy | Active/eligible/issued warps, issue/active cycles, occupancy and launch resource limits | Preserve counter windows/denominators; distinguish theoretical occupancy from measured active occupancy |
| Executed instruction/SASS | Supported executed/issued instruction and thread counts, predication, branch behavior, pipeline/type breakdown, tensor activity | Distinguish static binary counts, warp instructions, thread instructions and executed arithmetic operations |
| Memory efficiency | DRAM/L2/L1 traffic, requests/sectors/hits/misses, local-memory/spill traffic, shared accesses/conflicts, atomics | Native metric definitions; avoid inferring bandwidth, cache efficiency or transaction amplification from incompatible windows |
| Range counters | All reachable metric configurations, pass groups, range trees and raw counter images | Hardware limits may require multiple passes; retain pass completion and evaluated metric dependencies |
| PM sampling | Time-resolved supported hardware counters and sample intervals | Device/partition scope and clock domain; never assign an overlapping device-wide sample wholly to one kernel |
| Device telemetry | Supported clocks, power, temperature, utilization, memory occupancy and throttling via an explicit telemetry backend | Record shared-device scope, polling interval, units and validity; device power is not per-kernel energy |
| Framework/operator | NVTX domains/ranges/marks; optional Python/native stacks and operator hierarchy | Explicit correlation or bounded temporal association, never guessed parentage presented as exact |
| NCCL/distributed | Rank/communicator/collective sequence, operation, dtype/count, per-rank intervals, transport observations and overlap | CUDA kernels alone do not identify every collective; use available library hooks/markers and preserve unknowns |
| Collection quality | Dropped records, buffer use, decode failures, timestamps, conflicts, permission errors, collection overhead and termination | Quality is first-class capture data, including partial process and device coverage |

The familiar nine fields (`samples`, `vram_alloc`, `dram_read`, `dram_write`, `shared_mem`, `tensor_warps`, `stall_scoreboard`, `stall_wait`, `gpu_time`) are optional projections, not a schema ceiling or evidence of collection. Each MUST have a documented producer and unit. `samples` MUST distinguish dispatch counts from PC samples; memory allocation MUST not stand in for memory traffic. Uncollected fields MUST NOT appear as measured zeros.

## Code and host-source attribution

Capture module load/unload generations, function identities, available CUBIN/fatbin/PTX, code hashes, architecture, toolchain information and dynamic/JIT lifecycle. Copy callback-owned binary data before its documented lifetime ends. Preserve original mangled and display names. CUDA Graphs, module reloads and same-name specializations MUST not collide.

SASS MUST be generated from the captured matching code image, with disassembler version/options retained. PC coordinates MUST identify their address space and module generation; logical PTX/text offsets and navigation anchors MUST never be called device PCs. A captured PTX artifact is not proof of final SASS identity without a recorded relation to the loaded image.

Available debug maps, source files and inline chains SHOULD be captured with hashes and path mappings. Source collection requires an explicit inclusion policy. An unavailable third-party source, missing PTX, stripped line table or unavailable binary gets a reason, not a guessed replacement. CUBIN-only library kernels remain useful through names and actual SASS. Compile-only resource estimates MUST stay distinct from measured execution.

CUDA API callbacks do not automatically supply Python stacks or Go runtime traces. Native stack unwinding, Python frame/operator integration and cooperative Go `runtime/trace` each require their own tested producer. Optional agents MUST report version support, overhead and missing frames. CPU sampling profiles are separate from GPU profiles, linked through correlation rather than values added together.

## Process trees, distributed execution and clocks

Every record MUST have session, run, host, process-generation and collector identity, plus relevant thread/context/stream/device/partition/module/pipeline/dispatch identities. OS PID, device ordinal and function name alone are insufficient. Preserve device UUID and partition identity when available. Child inheritance, exec and supported spawn paths need independent handshakes; fork-only CUDA use must not be represented as supported without native qualification.

`--follow=children` records observed children and explicitly missing injection. Finish policy must cover surviving workers, orphaned children and launcher exit; preserve partial streams on crash or forced termination. Concurrent sessions MUST have isolated transport and scratch roots. Never terminate processes by name or input-path substring. Use owned handles; retain scratch when writer termination is unresolved.

Multi-host capture is an explicit coordinator/worker mode with shared session ID, per-host streams and a final merge. No automatic SSH or remote execution is implied by a local launcher. Store clock domains and measured calibration/uncertainty; do not subtract unrelated GPU/host clocks. Rank skew and overlap comparisons across hosts require qualified synchronization or an uncertainty interval.

## Replay and compatibility rules

Default capture MUST execute the application once. Enabling a preset does not authorize rerunning training, network requests, checkpoints or external writes. Explicit application replay means launching separate recorded runs; automatic equality of seeds or argv is not proof of equivalent execution.

Kernel/range replay is allowed only when the selected collector supports it and resource/state fidelity is established. GPU-memory checkpoint restoration does not restore files, sockets, host state, external services or other ranks. NCCL/IPC/cooperative multi-device work MUST NOT be replayed one rank/kernel at a time without a proven coordinated protocol. Reject unsafe/incompatible combinations with a precise reason and keep other collection groups available.

Record live versus replay, run/pass IDs, input/config hashes where available, range boundaries, warmup inclusion, cache policy, serialization, counter configuration, sampling period and validation result. Do not merge pass measurements into one physical timeline. Derived multipass metrics require matching ranges and documented evaluation rules. Distinct-run evidence remains distinct even when visualized together.

Existing CUPTI subscribers, Nsight tools, framework profilers, MIG/MPS, permission settings and driver versions MUST participate in capability negotiation. Multi-subscriber availability is version/configuration dependent and does not establish that every counter/sampling mode can coexist.

## Recording model and single-file delivery

Use `x/cudacapture` with an explicitly versioned extension. Preserve supported existing recordings. The ordinary outer gzip pprof is the default live GPU-duration projection; additional homogeneous profiles, typed observations, event streams and assets are embedded with a manifest. Attachments carried in pprof comments are a project convention: test pinned parse/write preservation, and document that third-party transformations may strip them while leaving the ordinary profile readable.

Each **metric descriptor** MUST include native namespace/ID, name, definition/source, unit and scaling, native numeric representation, aggregation, scope, denominator/window, collector/version and support predicate. Each **observation** MUST include descriptor ID, stable entity/run/pass identity, interval or instant, native value and availability. Support lossless int64/uint64, finite floating-point values and structured distributions; distinguish absent, invalid and measured zero. Preserve raw payloads or normalized exports with enough provenance for independent reconstruction.

Pprof projections MUST be explicit: additive integer values only, checked overflow/conversion, consistent units, and no duplicate accounting from alternate PTX/SASS/source views. Gauges, percentages, rates, distributions and compiler metadata remain typed data or are derived from retained compatible numerators/denominators. Device-wide data without kernel attribution gets a device-scoped profile/view, not invented kernel frames.

The manifest MUST contain requested/resolved configuration, discovered feature ledger, process/device coverage, collector outcomes, loss counts, profile inventory, artifact roles and hashes, code relationships, tool versions, target termination and capture completion state. Command/env metadata MUST follow an allowlist or explicit opt-in; do not copy the entire environment, tensor contents, model weights or arbitrary files by default. Users must be able to see which source/code artifacts will be shared.

Support `compact` and `full` artifact retention. Compact includes everything needed for displayed measurements, code views, coverage and provenance; full adds raw/native replay material when selected and within limits. Full does not guarantee a replayable application checkpoint. A single file may be large; expose estimated/final sizes and omissions. Never silently truncate assets.

Bound native buffers, IPC messages, decompression, individual assets and total decoded size. Reject duplicate members, traversal, unsafe extraction links, case collisions, incompatible schemas and checksum mismatches. Preserve unknown supported extension data without executing it. Finalize atomically; interrupted sessions retain a recoverable journal marked partial. Readers MUST work without launching captured code or fetching original paths. Scratch belongs under `~/tmp` by default.

## Viewer and completeness semantics

The viewer MUST offer run/process/device/rank/collector/profile/window selection, source/PTX/SASS navigation, CPU/GPU timeline correlation, typed metric inspection, code provenance, artifact inventory and a coverage/loss report. Include `why unavailable` explanations. Missing metrics and empty/zero profiles MUST render without NaN percentages; invalid sample selection must not silently substitute a different metric.

Availability states are `collected`, `not_requested`, `unsupported`, `permission_denied`, `conflict`, `not_observed`, `not_implemented`, `failed`, `budget_exhausted` and `lost_or_incomplete`. Collection status and attribution confidence are separate. Completeness is evaluated per requested family, process/device, run and window; no single global “all metrics” badge may hide gaps.

“Full on this configuration” requires every requested applicable collector to complete within the approved run policy with required artifacts and bounded disclosed loss. “Exhaustive inventory complete” only means all feature dispositions are known. “Instruction attribution qualified” additionally requires real PC/SASS measurements joined to the exact code image. These are different claims.

## Implementation and acceptance sequence

1. **Injection witness:** unchanged native CUDA and Python programs on Linux amd64/arm64; collector handshake before first observed CUDA work; clean no-CUDA and injection-failure outcomes. Qualify subscriber conflicts and target exit/signal behavior.
2. **Trace product:** kernels/copies/API/NVTX correlation, allocations and module bytes; standalone profile from an unchanged PyTorch script. Compare record counts and intervals against retained raw data and an independent supported NVIDIA-tool run, accounting for perturbation.
3. **Process tree:** `torchrun` two ranks on two actual GPUs, child exec and concurrent sessions. Verify complete per-rank output checks and recording identity; prove cancellation leaves unrelated same-input processes alive. No physical multi-GPU qualification from mocks.
4. **Code and PC:** two same-name specializations, unload/reload, JIT code and a stripped library. Demonstrate real sampled PCs resolve to matching SASS; missing source/PTX remain explicit. Keep sampling loss and instruction-anchor-only modes distinguishable.
5. **Deep counters:** reuse plan 003 collectors. Exercise range/PM/SASS groups, zeros, unsupported metrics, conflicting modes, dropped buffers, overflow and native uint64 values above 2^53. Reconcile projected totals to native observations. Test duplicate pipeline names and overlapping device-scoped samples.
6. **Controlled replay:** deterministic qualified workload with output verification; deliberately mismatched replay rejected. NCCL rank-isolation and external-side-effect controls prevent inappropriate replay. Retain all failed runs rather than widening budgets after failure.
7. **Offline delivery:** copy only the recording to Darwin and Linux machines without the target source; actually run stock pprof and custom views, inspect code and typed metrics, test parse/write and extraction, verify zero/missing views through Brave AXMCP + CDP. Separate cross-compilation from actual execution.
8. **Robustness:** target crash/SIGKILL, collector crash, disk full, corrupt IPC/archive, lost module bytes, late child, unmatched range and exhausted budgets produce correct partial states. Test callbacks under contention and ensure shutdown does not free resources still referenced by CUPTI.

Every acceptance receipt MUST retain commands, target/tool/source identities, configuration, expected/observed counts, native outputs, losses, exit statuses and artifact hashes. Hardware tests skipped for lack of resources remain unqualified. Performance overhead needs paired instrumented/uninstrumented workload runs and numerical agreement; no fixed universal overhead claim is permitted. Full-plan completion requires the matrix, not simply an impressive example capture.

## Source basis

Inspected local boundaries: `nvidia/cmd/cudaprof/main.go` currently provides viewing/pack/extract, `nvidia/x/cuptiactivity/native.go` provides in-process activity setup, and `nvidia/x/cudaprof/profile.go` provides caller-facing profile sessions. A launcher alone is not cross-process collection.

NVIDIA documents the distinct Activity, Callback, Range, PC, SASS, PM and Checkpoint families in the [CUPTI overview](https://docs.nvidia.com/cupti/). Its [usage guide](https://docs.nvidia.com/cupti/13.0.0/main/main.html) describes injection examples, code/source correlation and collection mechanisms. [CUPTI 12.9 usage](https://docs.nvidia.com/cupti/12.9.1/main/main.html) documents finite hardware counter capacity and checkpoint-based device-memory restoration; these motivate separate passes and the replay-state boundary above. The [multi-subscriber tutorial](https://docs.nvidia.com/cupti/tutorial/tutorial.html) documents version/configuration requirements. Runtime qualification, not documentation alone, decides support on each target.
