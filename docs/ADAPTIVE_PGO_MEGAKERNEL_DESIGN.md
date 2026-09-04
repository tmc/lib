# Adaptive profile-guided plan selection

This document proposes bounded profile-guided optimization for the Metal and
CUDA inference backends. It does not specify compilation on live request paths,
and it does not promise a latency improvement.

Status: proposal. The source snapshot used for this revision was commit
`501e487`.

## Current substrate

The repository contains useful pieces, but not the closed-loop system described
here:

| Component | Current behavior | Limit |
| --- | --- | --- |
| `DispatchTuner` | Compares threadgroup widths 64 and 128 after a small number of samples | It does not compile plans, extract top-K candidates, or hot-swap a model plan |
| `PGOFeedbackCostModel` and `PGOTracker` | Store timing-oriented feedback and trial records | Recording feedback is not plan selection or execution |
| Metal `FlightRecorder` | Retains bounded dispatch timing records | Label recording allocates; timing does not provide observed DRAM bytes or FLOPs |
| Metal sample timestamps | Resolve device timestamps when supported | The fallback synthesizes timestamps from host time and must not be called a hardware observation |
| `metalbench.DynamicHarness` | Exercises an experimental batch-threshold choice | It is not a serving-runtime policy |
| Backend fused paths | Offer selected handwritten fused kernels | Selection is configuration-driven, not extracted from an e-graph plan |

No current `OffloadLayer` API, background megakernel compiler, atomic plan-set
swap, or cross-backend calibration cache should be inferred from these pieces.

## Goal

Select a previously validated execution plan for a stable workload bucket using
measurements from the same deployment path that will execute it.

The first useful version is offline or startup calibration. It answers a narrow
question: among semantically equivalent plans already admitted for this device
and bucket, which one has the best robust device time without violating memory
or latency limits?

Live adaptation is a later option. It must not compile unbounded candidates or
experiment with unvalidated plans on user requests.

## Plan and bucket identity

A plan identity includes:

- semantic graph and model revision;
- backend, device capability, driver, and compiler version;
- phase: prefill or decode;
- batch, prompt-length, and context-length bucket;
- dtype and tensor layouts;
- operation variants and launch geometry;
- stable buffer layout; and
- compiler and rewrite versions.

A calibration result is reusable only when every compatibility field matches.
Unknown fields cause a cache miss.

## Candidate pipeline

```text
generate legal alternatives
        |
        v
compile and resource-check
        |
        v
compare with semantic reference
        |
        v
warm the deployment path
        |
        v
collect repeated device and wall observations
        |
        v
reject unstable or constraint-violating candidates
        |
        v
select and persist an immutable plan set
```

Generation may use e-graph extraction, explicit templates, vendor libraries, or
all three. Graph matching and fusion selection belong in the graph rewrite and
extraction layer; the runtime validates and profiles the resulting candidates.

Every candidate has a recorded outcome: `accepted`, `rejected`, `invalid`,
`timed_out`, or `unsupported`. Empty candidate sets and compilation failures are
errors.

## Measurement

Use device timestamps for candidate timing when the backend provides them, with
host wall time as a separate field. A robust comparison needs warmup, repeated
samples, and an estimator resistant to occasional scheduling noise. Record the
sample count and distribution; do not retain only the winner.

Measurements include:

- plan and bucket identity;
- device and wall durations;
- allocation high-water mark;
- compilation and graph-materialization cost;
- synchronization boundaries;
- correctness result and tolerance; and
- telemetry provenance.

Analytical model-weight traffic and dense FLOPs may help explain a result, but
they are `modeled`. A timing record does not turn them into observed counters.

No fixed limits such as “less than 5 ms JIT,” “less than 50 ns lookup,” or
“zero stall” are architectural guarantees. Budgets must be chosen from a
measured serving objective and enforced as acceptance tests.

## Selection policy

Selection is constrained before it is optimized. A candidate is ineligible if
it:

- fails the semantic comparison;
- exceeds device resource or arena limits;
- changes output dtype or model semantics without an explicit policy;
- requires unsupported alignment, shape, or alias behavior;
- increases a configured tail-latency or memory limit; or
- has insufficient or unstable samples.

Among eligible candidates, optimize a declared objective such as median device
time with a p95 guard. Do not combine incomparable prefill and decode samples
into one score.

## Publication and lifetime

Selected plans are immutable. Publish a complete compatible plan set with one
atomic pointer update; never mutate an executing plan in place. The runtime
holds an epoch or reference for the duration of a step. Old code objects and
buffers are released only after their last user exits.

If validation, compilation, or publication fails, keep the last admitted plan.
If no admitted plan exists, return an error or use a separately validated
reference plan. Do not silently install a synthetic or partially compiled plan.

## Live adaptation, if enabled

Live adaptation is opt-in and bounded:

- only prevalidated candidates may receive user traffic;
- exploration has a fixed request and time budget;
- samples are tagged by bucket and current thermal/power context when known;
- regressions trip an automatic rollback to the last admitted plan; and
- background compilation has bounded CPU, memory, and compiler concurrency.

Prefer shadow or canary workloads when a plan can affect correctness. Offline
or startup calibration remains the default until live exploration demonstrates
a benefit under production-like load.

## Memory pressure

Memory recovery is separate from plan optimization. Safe recovery may evict
unused compiled plans, calibration entries, or reclaimable cache pages according
to explicit lifetime rules.

Changing weight precision, dropping cache state, or spilling active tensors to
storage changes accuracy or latency and requires a distinct degraded-service
policy. It is not an automatic OOM recovery action.

## Backend symmetry

Metal and CUDA share plan, bucket, observation, and publication contracts. They
do not need identical implementation mechanisms:

- CUDA may compile with NVRTC, time with CUDA events, and materialize a CUDA
  Graph after stable bindings exist.
- Metal may compile MSL libraries, time supported command-buffer intervals, and
  reuse pipeline and binding state appropriate to Metal.

A result from one backend is not portable to the other. Cache keys include the
backend and exact device capability.

## Acceptance plan

1. Define immutable plan and bucket identities.
2. Add a calibration file format with strict compatibility checks and atomic
   replacement.
3. Calibrate a small, fixed set of already-correct launch variants offline.
4. Prove selection with a deliberately slower valid candidate and a mutation
   control that makes an invalid candidate fail.
5. Re-rank through the actual stable-buffer deployment path.
6. Add concurrent readers, publication, invalidation, and reclamation tests.
7. Measure startup cost, steady-state benefit, cache-hit behavior, and rollback.
8. Consider bounded live adaptation only after the offline path is useful.

This work follows authoritative plan extraction, native dtype support, and real
prefill/decode batching. Before those foundations exist, tuning launch details
has a smaller and less stable performance ceiling.
