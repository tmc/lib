# Implementation plans

Plans describe proposed work. They do not indicate implementation is complete.

| Plan | Priority | Effort | Dependencies | Status |
| --- | --- | --- | --- | --- |
| [001: Make the PTX/SASS trace example a public API example](001-ptx-sass-trace.md) | P1 | 2–4 days, plus GPU verification | None | COMPLETED |
| [002: Optional multi-GPU NCCL capture](002-optional-nccl-capture.md) | P2 | Separate AllReduce and AllGather increments, plus two-GPU verification | Plan 001 public capture API | COMPLETED (software/oracle; 2-GPU deferred) |
| [003: CUPTI features and all representable measurements](003-cupti-feature-coverage.md) | P1 | Multi-stage; estimate after inventory | Inventory/schema before 001 API freeze; collectors build on its capture boundary | COMPLETED (census/schema/audit) |
| [004: Standalone Metal/Darwin profiling example](004-metal-darwin-profile.md) | P1 | Incremental; estimate after capability investigation | Independent; transport lessons from 001–003 | IN PROGRESS (Slices A-C completed & verified on hardware; D partially qualified / blocked on platform native ISA; E deferred) |
| [005: Third-party CUDA capture specification](005-third-party-cuda-capture-spec.md) | P1 | Multi-stage; scope defined by capability and replay policy | 001 transport and 003 collectors; command stays in nvidia/cmd | PROPOSED |
| [006: Running Python capture and late activation](006-running-python-capture.md) | P2 | Preloaded, cooperative and unprepared attach milestones | 005 session/transport contract and 003 collectors | PROPOSED |

All three plans have been implemented and verified. Plan 001 was verified end-to-end on real hardware (NVIDIA GB10, sm_121, CUDA 13.0) and offline on darwin/arm64. Plan 002 software, CLI validation, CPU oracles, error paths, and pprof labels were verified (physical 2-GPU collective execution deferred due to host `dgx` having 1 physical GPU). Plan 003 completed the CUPTI 13.0 header survey, exported logical image base semantics, corrected `RecordSASSMetric` tensor and warp attribution, and verified extensible dynamic sample schemas.

Plan 002 adds explicitly selected, single-host NCCL operations after plan 001.
It preserves the default single-GPU workload and the single-file recording.
Its GPU gates are separate. Hardware-counter and PC-sampling collection belong
to plan 003; plan 002 specifies coordinated replay and rank-aware attribution
when those collectors are used with NCCL. These plans remain proposed.

The writeup was sent to iTerm session `A7B06C8E-D7F5-4431-8392-416EB0F7C5EA` on 2026-09-09 at the user's request. The submitted message and recipient's active turn were verified. This records delivery of the proposal; implementation remains TODO.

Considered and rejected:

- Calling the existing internal import illegal: it is allowed inside the NVIDIA source subtree. The problem is copying the example to an unrelated module.
- Moving `internal/profilebundle` unchanged into `x/`: this would expose example filenames, workload parameters, and viewer temporary-directory policy as a public API.
- Copying the internal package into the example: this would duplicate the capture reader/writer and assembly validation.
- Hiding the workload behind a generic runner or CUDA backend interface: that would obscure the operations this example should teach.
- Removing output validation, real disassembly, or portable captures to shorten the example: those are useful behavior to preserve.

Plan 003 adds a source-backed CUPTI family/metric coverage plan, including the existing activity/range collectors, missing native sampling collectors, extensible sample schemas and generated-binding gaps. Its [binding snapshot](003-cupti-bindings.tsv) records declarations, not runtime qualification. Remote hosts and GPU performance were not audited.

Plan 004 is a separate proposed Metal/Darwin development effort: a public example, portable capture/viewer packages, runtime counters, replay and source/ISA attribution, plus optional multi-device and MPS workloads. NVIDIA implementation status above does not apply to plan 004.

Spec 005 defines third-party launch/injection, process-tree and distributed capture, exhaustive measurement coverage, replay policy, code attribution and portable offline delivery. It is proposed; the implementation statements above do not apply to it.
