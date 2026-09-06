# Luminal-inspired GPU backend improvements

> Re-verified against `lib` main on 2026-09-05. The status table, the
> correctness blockers, and the priority order below describe the tree as it
> stands, each row cited to the source that establishes it. The design
> discussion after the table is the original Luminal review and is unchanged;
> where it says "today" it means the 2026-09-01 snapshot, and the table is
> authoritative where the two disagree.

This document records design lessons from Luminal that apply to
`examples/llama-infer`. It is a source review, not a claim that either engine is
faster. A head-to-head result requires the same model revision, weights, prompt
tokens, generation length, dtype, correctness gate, device, and measurement
protocol.

Original review snapshot: `lib` commit `501e487` and Luminal commit `598e5ca7`,
inspected on 2026-09-01. Status re-verified against `lib` main `0b1e116` on
2026-09-05.

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

Re-verified 2026-09-05 against `lib` main. Every row cites the source that
establishes it; a row is Implemented only where a named symbol carries the
behavior, not where a helper exists unwired.

| Capability | Metal | CUDA | Evidence |
| --- | --- | --- | --- |
| GPU-resident weights and scratch | Implemented | Implemented | `NewGPUScratchpad` allocates every activation buffer in a tracked arena. |
| Per-sequence KV object | Implemented | Implemented | `SequenceKVCache`; `cudakv.Cache.Layer`. |
| Bounded scheduler step | Batched | Batched | `scheduledEngineAdapter.DecodeBatch` and `cudaServingBackend.DecodeBatch` take a slice of states. No longer serial across sequences. |
| Distinct fused QKV and SwiGLU branches | Selected by an extracted plan | Selected by `FuseEGraph` | `ExecutePlan` consumes an immutable `ExecutionPlan` on Metal; the CUDA loop still branches on `cfg.FuseEGraph` in `gpuForwardPassStep`. |
| Native 16-bit weight retention | Half-weight kernels, half-weight batched embedding, and `IsF16` over a `f16` plan dtype | Not established | `HalfWeightKernelsMSL` is compiled into the Metal library; no half path exists in `gpu_model_cuda.go`, `gpu_batch_cuda.go` or `forward.cu`. |
| Matrix-oriented batched prefill | Implemented | Implemented | `RunLeasedBatchedLayerChain` over packed segments; `gpuBatchedPrefillPassKVMeasured` advances a causal history in prompt tiles, computing each layer for all rows before the next. Prefill no longer repeats token steps. |
| Multi-sequence decode launch | Implemented | Implemented | `RunLeasedBatchedDecode`; `GPUForwardBatchKV` submits one shared-weight `[B,D]` transaction. |
| Reusable deployment graph | Immutable plan, replayed | Not wired | `ExecutePlan` is authoritative on Metal. `nvidia/x/cudagraph` exists as a package but nothing in `examples/llama-infer` imports it. |
| Hardware DRAM/FLOP counters | Unavailable in the current contract | Not wired into `gpuprof` | `gpuprof` computes estimates from declared dimensions and marks counter-backed observations unmeasured when absent. |

Batched decode is covered by equivalence tests against the single-sequence
path, not only by construction: `TestMultiSequenceBatchedDecodeEquivalence`,
`TestBatchedDecodeScratchIsolation` and `TestBatchedDecodeChainTailMatchesSeparate`
each compare batched output token-for-token with separate decodes and pass.

The two rows that remain genuinely open are CUDA-side: native 16-bit retention,
and a reusable deployment graph. Both are CUDA-only gaps against a Metal path
that already has the capability.

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

### CUDA prompt staging (resolved 2026-09-05)

The original blocker was that `GPUForwardPassStep` wrote each token to the same
one-element `HostTokenIn` slice before `CopyFromHostAsync`, so a prefill loop
could overwrite the source before an earlier asynchronous transfer consumed it.

`gpuForwardPassStep` now allocates isolated per-step storage instead:

```go
hostTokenIn := []int32{tokenID}
if err := sp.TokenIn.CopyFromHostAsync(hostTokenIn, stream); err != nil {
```

No slot is ever reused, which removes the overwrite hazard, and the lifetime is
held by the copy helper rather than by the caller: `CopyFromHostAsync` pins the
backing array with a `runtime.Pinner` and releases it from `stream.AddCleanup`,
so the storage outlives the queued transfer. That is the third contract listed
below, without its synchronization cost.

The `GPUScratchpad.HostTokenIn` field survives from the original design. Nothing
reads or writes it now; it is the shared slot the blocker described, and it
should not be reintroduced into the copy path.

The contracts originally offered were:

- upload the prompt into persistent device storage once;
- use a pinned staging ring whose slots are not reused until their completion
  event; or
- synchronize each transfer, accepting the performance cost.

Admission still requires a CUDA differential using several distinct prompt token
IDs over the same asynchronous path used by serving. `gpu_prefill_linux_test.go`
exercises the tiled prefill path but the differential is a CUDA-host gate and has
not been run from this review.

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

> Superseded on Metal as of 2026-09-05: `ExecutePlan` consumes an immutable
> `ExecutionPlan`. The paragraph below still holds for CUDA.

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

> Done as of 2026-09-05 on both backends; retained for the bucketing and
> admission requirements below, which still apply.

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

Re-ordered 2026-09-05. Steps 1, 3 and 4 of the original list are complete; what
remains is CUDA-side parity with the Metal path, then the search work that the
original review placed after stable execution.

1. Preserve validated native 16-bit storage on CUDA and measure its memory
   reduction. Metal already retains half weights; CUDA has no half path at all,
   so this is the widest remaining backend asymmetry.
2. Make extracted plans authoritative on CUDA. Metal's `ExecutePlan` consumes an
   immutable plan; the CUDA loop still branches on `cfg.FuseEGraph`. Allocate
   stable plan arenas as part of the same change.
3. Capture and reuse CUDA Graphs in the engine. `nvidia/x/cudagraph` exists but
   is unwired; compare against Metal command-reuse strategies once it is.
4. Add bounded e-graph candidate search and deployment-path re-ranking.
5. Add distributed execution only after the single-node contracts are stable.

Completed since the original review:

- CUDA asynchronous prompt staging (isolated per-step storage).
- Matrix-oriented batched prefill on both backends.
- Real batched decode over isolated per-sequence state, covered by equivalence
  tests against the single-sequence path.

Improved e-graph lowering becomes high leverage at step 2. Before then, CUDA
dtype and stable execution plans have clearer performance ceilings and smaller
correctness surfaces.
