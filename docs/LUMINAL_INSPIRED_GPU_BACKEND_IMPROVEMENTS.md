# Luminal-inspired GPU backend improvements

> Historical review: this file describes `lib` commit `501e487`, not the
> current implementation. Its status table is retained as a dated design input;
> later continuous-batching and backend work supersedes several rows.

This document records design lessons from Luminal that apply to
`examples/llama-infer`. It is a source review, not a claim that either engine is
faster. A head-to-head result requires the same model revision, weights, prompt
tokens, generation length, dtype, correctness gate, device, and measurement
protocol.

Review snapshot: `lib` commit `501e487` and Luminal commit `598e5ca7`, inspected
on 2026-09-01.

## What to copy from Luminal

The highest-value lesson is the runtime boundary:

```text
semantic alternatives
    -> shape buckets
    -> hard candidate validation
    -> correctness screening
    -> deployment-path profiling
    -> immutable selected plans
    -> stable storage and repeated execution
```

Luminal separates graph construction and equality saturation from runtime
search, profiling, and loading. Its CUDA runtime can reject candidates, compare
finalists through the deployment path, and install compatible plans by bucket.
`lib` should adopt that ownership model. Copying more rewrite rules before the
execution and validation boundary exists would have little leverage.

## Present `lib` status

| Capability | Metal | CUDA | Admission |
| --- | --- | --- | --- |
| GPU-resident weights and scratch | Implemented | Implemented | Source verified |
| Per-sequence KV object | Implemented | Implemented | Source verified and covered by local tests |
| Bounded scheduler step | Implemented | Implemented | Serial across sequences |
| Distinct fused QKV and SwiGLU branches | Implemented | Implemented | Selected by `FuseEGraph`, not by an extracted plan |
| Native 16-bit weight retention | Half-weight kernels and pipeline states compile, but the active load and dispatch path remains float32 | Not established | Not yet an executed cross-backend contract |
| Matrix-oriented batched prefill | Not implemented | Not implemented | Prefill repeats token steps |
| Multi-sequence decode launch | Not implemented | Not implemented | Scheduler calls one sequence at a time |
| Reusable deployment graph | Not established | Not implemented | No CUDA Graph capture/replay in the unified engine |
| Hardware DRAM/FLOP counters | Unavailable in the current contract | Not wired into `gpuprof` | Estimates must remain labeled modeled |

The engine exposes a descriptive `ExecutionPlan`. It mirrors the configured
operation sequence, but it does not yet drive backend dispatch. It is therefore
an inspection artifact, not the extracted plan described below.

## Correctness work that is complete

The current tree includes:

- synchronized CUDA token readback with propagated errors;
- fail-closed real-model loading and explicit test-mode synthetic weights;
- reference-vs-GPU logit checks with declared absolute and relative tolerances;
- a same-shape uploaded-weight mutation and restore control;
- distinct fused and unfused branches on Metal and CUDA;
- per-sequence KV storage and a two-sequence interleaving test; and
- scheduler callbacks outside scheduler locks, including a real-engine stepping
  regression for the earlier recursive-mutex deadlock.

These gates establish useful local invariants. They do not establish portable
CUDA correctness on every advertised compute capability, true batching, or
performance superiority.

## Remaining correctness blockers

### CUDA prompt staging

`GPUForwardPassStep` writes a token to the same one-element `HostTokenIn` slice
before each `CopyFromHostAsync`. The copy helper pins the supplied host slice;
it does not create immutable staging storage for every queued value. A prefill
loop can therefore overwrite the source before an earlier transfer consumes
it.

Use one of these contracts:

- upload the prompt into persistent device storage once;
- use a pinned staging ring whose slots are not reused until their completion
  event; or
- synchronize each transfer, accepting the performance cost.

Admission requires a CUDA differential using several distinct prompt token IDs
and the same asynchronous path used by serving.

### Error and state boundaries

All convenience wrappers used by benchmarks and serving must return forward
errors. Allocation, compilation, launch, synchronization, and readback failures
need injection tests. Synthetic weights or CPU simulation cannot produce an
eligible benchmark record.

### Fused-kernel resource behavior

Mathematical equivalence does not prove that a fusion is efficient. In the
current Metal fused QKV organization, each output-row threadgroup performs its
own RMS reduction over the input. That duplicates normalization work across Q,
K, and V rows. Treat the kernel as a correctness implementation until a device
profile shows that the saved intermediate traffic outweighs the repeated
reduction and resource cost.

Candidate validation must check tail dimensions, alignment, registers,
threadgroup or shared memory, grid limits, aliasing, and buffer capacity before
timing.

## Architecture changes

### Make selected plans authoritative

Today, both backend forward loops select fused branches with Go conditionals.
The e-graph does not control execution.

An extracted plan should contain explicit backend operations and their dtype,
layout, alignment, alias, shape, resource, and buffer-lifetime requirements.
The executor consumes that plan without matching operation patterns again.
Start with narrow alternatives:

- RMSNorm plus Q/K/V projection;
- RoPE plus KV-cache store;
- output projection plus residual;
- SwiGLU plus down projection plus residual; and
- final norm, LM head, and sampling where resource limits allow.

Keep an unfused plan as the semantic reference. If two selectable plans disagree,
fix the equivalence or lowering defect; do not tune around it.

### Separate prefill from decode

Both current backends implement prefill as repeated single-token steps. A true
prefill plan carries a token dimension through projections and causal attention
and writes all prompt KV entries in a bounded number of matrix-oriented
launches. Bucket it by prompt length and context range.

Decode carries a batch dimension plus per-sequence positions and KV-page
metadata. The scheduler may group only sequences whose plan, dtype, layout, and
shape bucket are compatible.

### Stabilize memory before graph replay

The engine already owns persistent allocations, but CUDA Graph replay requires
stable bindings and explicit invalidation. Assign intermediates from live ranges
into a runtime-owned arena, capture a selected fixed-shape plan once, and reuse
it while its binding contract remains valid. Re-rank finalists through this
captured path because direct-launch rankings may not survive deployment.

Metal command reuse should be evaluated against the same plan contract, without
assuming that CUDA Graph mechanics map directly to Metal.

### Preserve checkpoint dtype

Make dtype part of tensor identity. Upload supported FP16 or BF16 checkpoint
storage without an unconditional host expansion and use FP32 accumulation where
required. Admission needs:

- dtype-aware shape and size validation;
- device allocation evidence showing compact storage;
- logits and cached-decode comparison with a float32 reference; and
- end-to-end timing on the real model, not only a microbenchmark.

Quantization is a separate semantic and accuracy contract. Do not treat it as an
automatic memory-pressure fallback.

### Search with hard refusals

A candidate is eligible for profiling only after it compiles and passes layout,
resource, capacity, alias, and correctness checks. Record rejected, invalid,
timed-out, and accepted outcomes. An empty candidate set is a compilation
failure, not an optimized plan.

Search end-to-end phase plans rather than isolated kernels. Use fixed seeds and
workload buckets, then measure finalists through the same buffer and graph path
used for serving.

## Measurement contract

`gpuprof` computes workload estimates from model dimensions, declared precision,
elapsed time, and configured hardware ceilings. Its
`model-weight-GB/s`, `assumed-peak-bw-pct`, and
`dense-model-TFLOPs/s` values are modeled rates. They do not observe DRAM
transactions, cache reuse, executed instructions, activation traffic, or
attention's context-dependent work.

Keep the server's fail-closed telemetry rule authoritative:

- wall time is an observation;
- backend events may provide a device-time observation;
- a named, coverage-bounded hardware counter may provide observed bytes or
  operations;
- formulas remain `modeled`; and
- missing counters remain `unavailable`.

Metal timestamp fallback values generated from host time must be labeled as a
fallback, not as device counter data. CUDA event and CUPTI support are separate:
events time intervals; CUPTI may provide counters if collection coverage is
proved.

Retain each performance run with:

- full source revision and dirty state;
- model revision, file hashes, dtype, and tied-embedding configuration;
- prompt tokens, context length, generation length, and plan identity;
- device, OS, driver, compiler, and toolkit versions;
- exact command and raw stdout/stderr; and
- correctness output from the same binary.

Agent-reported M4 and GB10 values without that bundle are `UNVERIFIED`. Existing
benchmark files with different prompt templates are not a controlled Luminal
comparison.

## Priority order

1. Fix CUDA asynchronous prompt staging and finish error-injection gates.
2. Preserve validated native 16-bit storage and measure its memory reduction.
3. Implement matrix-oriented batched prefill.
4. Implement real batched decode over isolated per-sequence state.
5. Make extracted plans authoritative and allocate stable plan arenas.
6. Capture and reuse CUDA Graphs, then compare Metal command-reuse strategies.
7. Add bounded e-graph candidate search and deployment-path re-ranking.
8. Add distributed execution only after the single-node contracts are stable.

Improved e-graph lowering becomes high leverage at step 5. Before then, dtype,
prefill, batching, and stable execution plans have clearer performance ceilings
and smaller correctness surfaces.
