# llama-infer design review and improvement plan

> Historical review: this file describes commit `4b728c2`, not the current
> implementation. Later work added production continuous batching, packed
> prefill, and shared batched decode. Recheck every status claim against the
> current tree before using it as an implementation plan.

This document audits the design of `examples/llama-infer` as built and proposes
the changes needed to turn it into the Luminal-shaped runtime the earlier design
notes describe. It supersedes the status tables in
`UNIFIED_INFERENCE_ENGINE_ARCHITECTURE.md` and
`LUMINAL_INSPIRED_GPU_BACKEND_IMPROVEMENTS.md`, both of which were written
against commit `501e487` and are now stale in several rows.

Review snapshot: `lib` commit `4b728c2` (2026-09-01), Luminal commit
`598e5ca7`. The tree was receiving commits from another agent session while
this review ran; five llama-infer commits landed between `501e487` and the
snapshot. Line references are to `4b728c2`.

Verification performed on this snapshot, on an Apple Silicon host:

| Check | Result |
| --- | --- |
| `go build ./llama-infer/` | ok |
| `go vet ./llama-infer/` | ok |
| `staticcheck ./...` | 4 findings: unused `restorePrefix`, three dead assignments |
| `go test . ./gpuprof/` (Metal, synthetic weights) | pass at `4b728c2`; `TestAuthoritativePlanDispatch` failed on the working tree mid-commit because `GPUForwardPass` swallows encode errors and returned stale logits |
| CUDA path | not built or run here; findings are from source only |

## Summary

The example contains a real Metal decode path with a CPU reference oracle, a
useful set of differential tests, and per-sequence KV isolation. Everything
above that layer is thinner than its names suggest. Ranked by impact:

1. **The compiled-plan layer is a descriptor, not a compiler.** `ExecutionPlan`
   is regenerated from `Config` and the executor binds buffers by op name, so
   the plan cannot express anything the executor did not already hardcode. The
   e-graph runs on a five-node toy expression and its result is never consulted.
2. **"Batched decode" is serial replay.** `GPUBatchDecodeStep` encodes B full
   single-token forward passes back to back in one command buffer. Every
   sequence re-streams all weights, so per-token cost grows linearly with batch
   size. This is the single largest performance gap versus any real server.
3. **The continuous-batching scheduler is not connected to the server.**
   `/v1/chat/completions` runs `engine.Generate` under a one-slot admission
   queue. `Scheduler` and `Sequence` are only exercised by tests.
4. **KV storage is dense float32 and per-sequence.** Each sequence reserves
   `MaxSeqLen` positions in every layer at 4 bytes per element regardless of
   weight dtype. The prefix cache copies KV through host memory instead of
   sharing device blocks. Paged and radix building blocks already exist in
   `metalkv` and `cudakv` and are unused.
5. **Prefill is not matrix-shaped on the GPU.** `BatchedGEMM` launches one
   128-thread threadgroup per output element and never touches
   `simdgroup_matrix`. `BatchedRoPE` uses one thread per token with nested head
   loops. `BatchedCausalFlashAttention` walks positions serially with two
   barriers each. CUDA prefill is a token loop.
6. **The two backends share no interface.** Thirty-odd package-level functions
   are duplicated under build tags with diverging semantics: CUDA ignores
   `startPos`, does not execute the plan, has no f16 path, and its scheduler
   step is a host-synchronized loop.
7. **Generation semantics are incomplete for serving.** No EOS stop, argmax
   only, `temperature` ignored, wrong Llama 3 chat template, prompt token count
   estimated as bytes/4, per-token BPE decode that splits UTF-8 sequences, and a
   whitespace-splitting tokenizer.
8. **Telemetry mixes measured and invented values.** TFLOPs and bandwidth are
   hardcoded per OS, one FlightRecorder event per forward pass carries a fixed
   "32L" name, and the CLI prints a constant tensor count.
9. **Error handling leaks.** Metal allocation and pipeline errors are discarded
   at engine build, `GPUForwardPass` and `GPUPrefillPass` drop encode errors,
   and the Metal engine cache key omits the model path.
10. **Startup loads the model twice.** `main` calls `LoadModel` to read
    `config.json`, which decodes every tensor to host float32, discards the
    result, then the engine opens the safetensors again.

The recommended direction is to stop layering validation on top of the
descriptor plan and instead build the four missing structural pieces in order:
a backend interface, paged KV with a device-resident prefix cache, a batched
decode kernel set, and a plan IR that owns buffer slots and launch geometry.
Sections 4 and 5 give the target design and the phased plan.

## 1. As-built architecture

### Files and responsibilities

| File | Lines | Role |
| --- | --- | --- |
| `main.go` | 253 | flags, `Config`, model presets |
| `model_ref.go` | 513 | `ModelConfig`, safetensors to host float32, CPU reference `Forward` |
| `engine_types.go` | 647 | `Sequence`, `Scheduler`, `ExecutionPlan`, `ValidatePlan` |
| `engine_darwin.go` | 3035 | Metal engine: load, two plan interpreters, batched prefill, batch decode, `Generate`, bench |
| `engine_other.go` | 719 | CUDA engine: load, wrappers, `Generate`, bench |
| `gpu_model_cuda.go`, `gpu_kernels_cuda.go`, `forward.cu` | 960 | CUDA weights, scratch, NVRTC kernels, per-token step |
| `prefix_cache_{darwin,other}.go` | 284 | host-side KV snapshot cache, radix index on darwin |
| `server.go` | 2279 | telemetry JSON, request slot queue, mesh endpoints, embedded dashboard HTML |
| `api_openai.go` | 392 | `/v1/models`, `/v1/chat/completions` |
| `cluster_node.go` | 829 | go-iroh mesh, hfsync, role inference, remote generate, draft/target comparison |
| `tokenizer.go` | 249 | HF `tokenizer.json` BPE approximation, byte fallback |
| `pgo_tuner.go`, `runtime_metrics.go` | 235 | GEMV width chooser, bounded metrics store |
| `gpuprof/` | 191 + tests | modeled workload estimates, benchstat lines |

Kernels come from `apple/x/metal/metalmegakernel` (decoder and half-weight
MSL), `metalattn` (decode attention), `metalsample` (argmax) and the embedded
`forward.cu` plus `nvidia/x/cudaattn` and `cudasample`.

### Two execution paths

The engine exposes two independent ways to run a request. Only the first is
served.

| Path | Entry | KV | Prefill | Decode | Prefix cache | Used by |
| --- | --- | --- | --- | --- | --- | --- |
| Single request | `ModelEngine.Generate` (`engine_darwin.go:2951`) | engine-owned `kCacheBuf` | batched prefill kernels, single command buffer | one command buffer per token | yes, restores into engine KV | HTTP handlers, mesh remote generate, CLI |
| Scheduler | `Scheduler.Step` (`engine_types.go:199`) | per-sequence `SequenceKVCache` | one sequence at a time, blocking | `GPUBatchDecodeStep`, one command buffer for all | no | tests only |

`Generate` holds `generateMu`; `Scheduler.Step` does not, and both write the
same engine scratch buffers (`xBuf`, `normXBuf`, `tokBuf`, prefill arena). A
process that used both concurrently would corrupt activations. Today nothing
does, but the design permits it.

### Decode step, Metal

For each token: `GPUForwardPass` creates a command buffer and one compute
encoder, calls `ExecutePlanWithLogits`, which validates the plan
(`engine_darwin.go:944`) and then dispatches roughly 12 kernels per layer plus
3 epilogue kernels, commits, and blocks in `WaitUntilCompleted`. The sampled
token is read from a shared buffer. Per-pass GPU time is recorded and fed to
`DispatchTuner`, which alternates between 64- and 128-thread GEMV variants
until it has three samples of each and then locks the faster one forever
(`pgo_tuner.go:52-89`).

### Prefill, Metal

`GPUBatchedPrefillPassKV` grows an arena sized to the prompt
(`ensurePrefillArena`, `engine_darwin.go:2024`) and encodes
`encodeBatchedPrefillWithKV`, which runs the whole prompt through per-layer
"batched" kernels and computes logits only for the last row. This is a real
improvement over the earlier token loop and the differential test
`TestBatchedVsSequentialPrefillEquivalence` covers it.

### CUDA

`GPUForwardPassStep` (`gpu_model_cuda.go:257`) launches about 14 kernels per
layer on one stream, copies the token in from a fresh host slice each step, and
synchronizes the stream to read the sampled token back. Prefill is a loop over
this step. The `ExecutionPlan` is constructed (`engine_other.go:288`) but never
executed; fusion is selected by `cfg.FuseEGraph` in Go.

### Serving

`handleV1ChatCompletions` formats messages as `role: content\n`, acquires the
single request slot, and streams `Generate` output as SSE. Remote execution
sends the whole prompt to a peer over iroh and waits for the complete result.

## 2. Comparison with Luminal

Luminal's contract (its `spec.md`) is:

```text
frontend program -> HLIR (15 primitive ops) -> loop rolling
  -> egglog saturation with backend rewrites
  -> SearchSpace: one saturated e-graph per dynamic-dim bucket
  -> Runtime::compile: search, candidate validation, profiling, one LLIR per bucket
  -> load_llir -> execute(dyn_map)
```

with three rules that matter here: every extractable graph must be a valid,
semantically equivalent program; all pattern matching happens in the e-graph
and never in post-passes; and the reference runtime is the semantic oracle.

| Luminal concept | llama-infer analogue | Gap |
| --- | --- | --- |
| HLIR graph built from a tensor API | none; `BuildExecutionPlan` emits a fixed op list from `Config` | no graph exists to rewrite |
| Egglog rewrites produce LLIR ops | `egraph.Saturate` on `(gemv (rmsnorm x w) wq)`; result discarded (`engine_darwin.go:356-370`) | rewrites do not reach the plan; `FuseEGraph` is a boolean read in Go |
| Extraction with a cost model | `egraph.ExtractBest` in `server.graphPlan`, for display only | cost model never sees measured kernel times |
| Dynamic dims and shape buckets | none; prefill arena resized on demand, decode fixed at one token | no bucket concept, no per-shape specialization |
| Candidate validation (resources, aliasing, layout) | `ValidatePlan` checks op vocabulary and ordering of a plan the executor itself generated | validates grammar, not resources; runs per dispatch |
| Profiling search over candidates | `DispatchTuner`: 2 variants, 3 samples, whole-pass timing, then locked | no per-kernel timing, no re-ranking |
| Arena from live ranges, stable bindings | fixed engine buffers plus per-step `mempool.Alloc` in batch decode | no lifetimes; allocation in the decode loop |
| CUDA Graph replay | none | every token re-encodes and re-launches |
| Reference runtime | `model_ref.go` CPU `Forward` with differential tests | present; keep |
| KV as graph tensors with scatter/gather and a page table (`examples/paged_llama`) | dense per-sequence buffers, host-side snapshot copies | no paging, no sharing |
| Runtime trait per backend | build-tagged duplicate functions | no interface; semantics diverge |

The honest description of the current state is "a hand-scheduled Metal decoder
with a plan-shaped JSON description attached." That is a reasonable place to
be for an example, but the names (`ExecutePlan`, "E-Graph Megakernel",
"Adaptive JIT", "batched decode") describe a system that does not exist yet.
Section 4 proposes either building the pieces or renaming the surfaces.

## 3. Findings

Severity: **S1** wrong results or resource exhaustion in normal use; **S2**
design blocks the stated goal; **S3** performance or hygiene.

### 3.1 Plan and compiler layer

**F1 (S2) The plan cannot change execution.** `OpDispatch.Inputs`, `Outputs`,
`Threadgroup` and `Grid` are populated by name strings or left zero and are
never read by `ExecutePlanWithLogits`, which switches on `op.Name` and binds
`eng.xBuf`, `eng.qBuf` and friends directly (`engine_darwin.go:979-1391`).
`BufferAllocation` entries are never used to allocate anything; the engine
allocates the same buffers separately (`engine_darwin.go:438-455`). A plan that
renamed a buffer or changed a threadgroup size would execute identically.

**F2 (S3) Two interpreters, one dead.** `ExecutePlanWithLogits` and
`encodeForwardStepWithKVAndLogits` are each about 500 lines and encode the same
kernels. The second is reachable only when `eng.Plan == nil`
(`engine_darwin.go:1403`), which never happens outside a test that sets it. The
two also disagree: RoPE uses a 128-wide threadgroup in the plan path and 64 in
the legacy path (`engine_darwin.go:1162-1165` versus `1709-1712`).

**F3 (S3) Validation runs per dispatch.** `ValidatePlan` walks every op on
every token and on every sequence within a batch step (`engine_darwin.go:944`).
The plan is immutable after construction; validate once at compile time.

**F4 (S2) "Fused" ops are unfused in f16.** In the f16 branch,
`FusedRMSNormQKV` dispatches four kernels and `FusedSwiGLUDown` dispatches two
(`engine_darwin.go:998-1043`, `1265-1287`), while the plan still marks them
`Fused: true`. The plan lies about what ran. The f32 fused QKV kernel also
recomputes the RMS reduction in every output-row threadgroup
(`llama_kernels.metal:300-409`), which the earlier review already flagged.

**F5 (S2) The e-graph is decorative.** The engine builds a five-node e-graph
with two rewrites and keeps only the saturation count for the dashboard
(`engine_darwin.go:356-370`). `server.graphPlan` rebuilds the same toy graph on
each request. `egraph.GenerateMSL` (`apple/x/metal/egraph/codegen.go`) exists
and is unused. Luminal's rule is that fusion decisions are made only by
rewrites and extraction; here they are made by `if cfg.FuseEGraph`.

**F6 (S3) Three overlapping configuration types.** `Config` (`main.go:18`)
mixes model hyperparameters with CLI, serving, mesh, and test options.
`ModelConfig` (`model_ref.go:14`) is the HF schema. `ExecutionPlan` repeats
five of the dimensions. Conversion code appears in `main.go:166-179`,
`engine_other.go:183-206`, `engine_darwin.go:618-631`, and the differential
tests.

### 3.2 Batched decode

**F7 (S2) Batch decode is B serial passes.** `GPUBatchDecodeStep`
(`engine_darwin.go:2668-2746`) loops over sequences and calls the full
single-token encoder for each, sharing `xBuf` and the other scratch buffers.
Metal's serial compute encoder orders the dispatches, so the result is correct
(the tests confirm it), but each sequence streams every weight matrix again.
For LLaMA 3.2 1B in f16 that is about 2.5 GB of weight reads per sequence per
token. The `MemoryBarrierWithScope` call at `2736` is redundant under serial
dispatch. On CUDA the loop additionally synchronizes the host after every
sequence (`engine_other.go:441-456`).

Order-of-magnitude cost model, LLaMA 3.2 1B, f16 weights, unified memory at a
nominal 400 GB/s (a modeled figure, not a measurement):

| Batch | Current design, weight bytes per step | Real batched GEMV, weight bytes per step |
| --- | --- | --- |
| 1 | 2.5 GB | 2.5 GB |
| 8 | 20 GB | 2.5 GB |
| 32 | 80 GB | 2.5 GB |

**F8 (S3) Allocation inside the decode loop.** Each batch step allocates a
token buffer and B logits buffers with `mempool.Alloc` and frees them on
return (`engine_darwin.go:2681-2701`). `mempool` is a thin alias over
`metalmem.Alloc`, so this is a real `MTLBuffer` allocation per step.

**F9 (S2) Per-sequence KV is dense and float32.** `NewSequenceKV` allocates
`MaxSeqLen * NumKVHeads * HeadDim` float32 per layer for K and V
(`engine_darwin.go:847-870`). At the `MaxSeqLen` the loader clamps to (2048,
see F19) that is 128 MB per sequence for the 1B model; at the preset 8192 it is
512 MB. `Scheduler` defaults to 32 slots. KV dtype does not follow weight
dtype, so an f16 model pays 2x on cache traffic in attention.

**F10 (S2) Prefix cache round-trips through the host.** `capturePrefix` copies
KV rows into Go slices and `restorePrefixPartial` copies them back
(`engine_darwin.go:255-300`). Hits therefore cost a host memcpy of the whole
prefix per layer instead of sharing device blocks. The darwin variant wraps a
`metalkv.RadixTree` with one token per block and allocates fresh block IDs on
every store (`prefix_cache_darwin.go:138-148`); eviction removes map entries
but never calls `RadixTree.Evict`, so the tree fills to `maxItems*4` nodes and
later `Insert` calls fail silently (`prefix_cache_darwin.go:100-109`, `147`).
The prefix cache is only reachable from `Generate`, not from the scheduler.

### 3.3 Prefill kernels

**F11 (S2) `BatchedGEMM` is a GEMV loop.** One 128-thread threadgroup computes
one output element `(token, out_col)` with a dot product over `in_dim`
(`llama_kernels.metal:526-565`). Each weight row is re-read once per token.
The file includes `<metal_simdgroup_matrix>` but nothing uses it. A tiled
`simdgroup_matrix` kernel, or MPS matrix multiplication, is the standard fix.

**F12 (S3) `BatchedRoPE` is single-threaded per token.** The kernel is
dispatched with a threadgroup width of 1 (`engine_darwin.go:2265-2268`) and
loops over all heads and dimensions in one thread, recomputing `pow`, `cos`
and `sin` for every element (`llama_kernels.metal:608-674`). The decode RoPE
kernel does the same trigonometry per element per token; both should read a
precomputed `cos/sin` table.

**F13 (S3) Prefill attention is serial over positions.**
`BatchedCausalFlashAttention` keeps `acc_v[128]` in registers per thread,
iterates `p = 0..actual_pos` with two threadgroup barriers per position, and
has no guard for `head_dim > 128` (`llama_kernels.metal:676-750`). The decode
attention kernel in `metalattn` tiles by 256 and is the better template.

**F14 (S3) CUDA attention runs one thread per head.** The launch is
`Grid(qHeads).Block(1)` (`gpu_model_cuda.go:364`) against a kernel that strides
positions by `blockDim.x`, so each head's whole context is processed by a
single thread.

### 3.4 Scheduler and serving

**F15 (S2) `Scheduler` is not wired to the server.** `NewScheduler` has no
callers outside tests. The server's admission is a one-slot FIFO
(`server.go:275`) and every request runs `Generate` to completion. Streaming
concurrency is therefore one.

**F16 (S2) Prefill in the scheduler blocks decode.** `Scheduler.Step` runs
each pending prefill to completion, one sequence at a time, before the decode
batch (`engine_types.go:231-266`). `metalkv.ChunkedPrefillScheduler` exists and
is unused.

**F17 (S1) No end-of-sequence handling.** Neither `Generate` nor
`Scheduler.Step` stops on an EOS token; generation always runs to `maxTokens`
and `finish_reason` is always `"stop"` (`engine_darwin.go:3009-3022`,
`api_openai.go:319`, `380`). The scheduler also emits token text as the decimal
token ID (`engine_types.go:265`, `293`, `323`).

**F18 (S1) Chat and tokenizer semantics.** The prompt is `role: content\n`
rather than the Llama 3 header format (`api_openai.go:241-244`);
`temperature` is parsed and ignored (argmax only); `usage.prompt_tokens` is
`len(prompt)/4` (`api_openai.go:384-386`). `BpeTokenizer.Encode` splits on
`strings.Fields`, which drops newlines and repeated spaces, has no
pre-tokenizer regex, and cannot see special tokens such as
`<|begin_of_text|>` in the text (`tokenizer.go:144`). Streaming decodes one
token at a time, so byte-fallback tokens that form one UTF-8 character arrive
as broken bytes (`engine_darwin.go:3017`).

**F19 (S1) Context length is silently clamped.** `LoadModel` sets
`MaxSeqLen = 2048` whenever `config.json` reports more than 4096
(`model_ref.go:128-130`). LLaMA 3.2 reports 131072, so every real model gets a
2048 context and attention past that position writes out of the cache's
logical range (the kernel reads `min(cur_pos+1, max_seq_len)` but the RoPE
store does not bound `pos`).

**F20 (S3) Startup loads the model twice.** `main` calls `LoadModel` only to
read dimensions (`main.go:161`), which decodes every tensor to host float32
(about 5 GB for the 1B model) and discards it; the engine then opens the
safetensors again. Read `config.json` alone.

### 3.5 Backend contract

**F21 (S2) No interface between backends.** Every engine function is a
package-level function duplicated under `//go:build darwin` and `!darwin`. The
CUDA copies have already diverged:

| Function | Metal | CUDA |
| --- | --- | --- |
| `GPUBatchedPrefillPassKV(..., startPos)` | honors `startPos` | ignores it (`engine_other.go:431-433`) |
| `GPUBatchDecodeStep` | one command buffer | host sync per sequence |
| `ExecutionPlan` | executed | built, never executed |
| f16 weights | supported | not supported |
| `MutateWeightForTest` | in-place | copies the whole tensor to host and back |
| `ResetKV` | returns nothing | returns `error` |
| engine cache key | omits `ModelPath` (`engine_darwin.go:346`) | includes it |

The missing `ModelPath` in the Metal key means two model directories with the
same name and dimensions share one engine.

**F22 (S1) Errors are dropped at build and dispatch.** In
`getOrCompileDarwinEngine`, all half-precision pipeline creations and all
scratch and KV allocations discard their errors (`engine_darwin.go:402-407`,
`438-455`); a failed allocation yields a zero `Slice` that panics later.
`GPUForwardPass` ignores the encode error and returns whatever `tokBuf` held
(`engine_darwin.go:1912`, `1944`); `GPUPrefillPass` does the same
(`1948-1960`). The mid-commit test failure observed during this review was
exactly this path: validation rejected the plan, nothing was encoded, and the
previous logits were reported as the result.

**F23 (S3) Engine lifecycle.** Engines live forever in a global map keyed by a
format string; `Free` releases buffers but leaves the map entry, so a later
lookup returns a freed engine. `runtime.LockOSThread` in `init` pins only the
main goroutine while HTTP handlers drive Metal from other goroutines; Metal is
thread-safe so this is harmless, but the call is misleading.

### 3.6 Telemetry and distributed features

**F24 (S3) Invented hardware numbers.** `probeSelfHardware` returns fixed
TFLOPs, bandwidth and 16 GB free VRAM per OS (`cluster_node.go:575-602`), the
benchmark profiles are labeled `assumed-m4-max` and `assumed-gb10`, and the
telemetry fallback node advertises 40 TFLOPs and 800 GB/s (`server.go:834-851`).
The CLI prints `Ingested 146 tensors` as a constant (`engine_darwin.go:2762`).
`FlightRecorder` receives one event per forward pass named
`LLaMA_Forward_Pass_32L` for every model (`engine_darwin.go:1938`), so the
dashboard's "kernel events" and "dispatches" are pass counts.

**F25 (S3) Distributed roles are advisory.** `InferRoles` picks prefill and
decode nodes with a cost model but nothing transfers KV between nodes;
`GenerateRemote` ships the whole request. `Speculate` compares a local draft
against a remote target after both have finished and records the trace; it
does not accelerate anything. The dashboard labels these honestly, but the
code and flags (`-peer`) suggest disaggregated serving that does not exist.

### 3.7 Tests

The differential tests are the strongest part of the tree. Gaps:

- All GPU tests share one cached engine and mutate its `Plan`, tuner, KV and
  prefix cache, so results depend on test order. The observed
  `TestAuthoritativePlanDispatch` failure was order-dependent.
- No test exercises the server through the scheduler, EOS, cancellation during
  GPU work, or a batch larger than four.
- No test runs a real checkpoint; correctness against a known-good decode of
  LLaMA 3.2 1B for a fixed prompt is not asserted anywhere.
- CUDA tests cannot run on the development host and there is no CI signal.

## 4. Target design

The goal is Luminal's ownership model in Go, scoped to one architecture family
and two backends, without pretending to have a general compiler. Keep the CPU
reference as the oracle for everything below.

### 4.1 Package layout

Split the single `main` package into internal packages with one concern each:

```text
examples/llama-infer/
  main.go                      flags only
  internal/llamamodel          ModelConfig from config.json, tensor schema, presets
  internal/llamatokenizer      tokenizer.json with pre-tokenizer, special tokens, chat template
  internal/llamaref            CPU reference Forward (moved from model_ref.go)
  internal/llamaplan           Plan IR, buckets, validation, lowering
  internal/llamabackend        Backend and Executable interfaces
  internal/llamabackend/metal  Metal implementation
  internal/llamabackend/cuda   CUDA implementation
  internal/llamakv             paged KV, block tables, radix prefix cache
  internal/llamasched          Sequence, Scheduler, batch builder, sampling
  internal/llamaserve          OpenAI handlers, SSE, admission
  internal/llamatelemetry      metrics with provenance, dashboard
  internal/llamamesh           iroh, hfsync (unchanged), no role inference
```

Model and CLI configuration stop sharing a struct: `llamamodel.Config` holds
architecture only; `main` holds serving flags.

### 4.2 Backend interface

```go
package llamabackend

type Backend interface {
    Name() string
    Capabilities() Capabilities            // dtypes, max threadgroup, unified memory
    LoadWeights(ctx, *llamamodel.Weights) (WeightSet, error)
    Compile(*llamaplan.Plan, WeightSet) (Executable, error)
    NewKVPool(spec llamakv.Spec) (llamakv.Pool, error)
}

type Executable interface {
    Prefill(ctx, PrefillBatch) (PrefillResult, error)
    Decode(ctx, DecodeBatch) (DecodeResult, error)
    Close() error
}

type DecodeBatch struct {
    Tokens    []int32          // one per sequence
    Positions []int32
    BlockTables [][]int32      // per sequence, from the KV pool
    Sampling  []SamplingParams
}
```

Every function currently duplicated under build tags becomes a method on one
of these two types. The reference backend implements the same interface on the
CPU so the differential tests compare `Executable` against `Executable`.

### 4.3 Plan IR

Replace the name-keyed descriptor with an IR the executor can only execute
through:

```go
type Plan struct {
    Model   llamamodel.Config
    Phase   Phase                 // Prefill or Decode
    Bucket  Bucket                // {MaxBatch, MaxTokens, MaxContext}
    DType   DType                 // weights, activations, KV
    Slots   []BufferSlot          // index, size, dtype, lifetime (first,last op)
    Ops     []Op                  // Kernel, Args []SlotRef, Grid, Threadgroup
    Sampler SamplerOp
}
```

Rules, borrowed from Luminal's contracts:

- `Compile` validates once: op vocabulary, slot dtypes and sizes, alias and
  lifetime overlap, threadgroup limits from `Capabilities`. Execution never
  re-validates.
- The executor binds buffers by `SlotRef` index. It does not know op names.
- An unfused plan is always constructible and is the semantic reference for
  any fused plan of the same bucket.
- Fusion is a lowering choice recorded in the plan (`Op.Kernel =
  FusedRMSNormQKV`) and a plan with `Fused` set must dispatch a fused kernel or
  fail to compile.
- Plans are cached by `(model hash, backend, phase, bucket, dtype)`.

Keep the e-graph out of the critical path until the IR exists. Once it does,
the rewrite rules can produce `Op` sequences directly and the extractor's cost
model can consume measured per-kernel times from the profiler. If that work is
not scheduled, remove the e-graph from the engine and the dashboard rather than
reporting saturation counts from a toy graph.

### 4.4 Batched decode

A decode plan for bucket `MaxBatch = B` uses kernels that carry a batch
dimension:

- `GEMV` becomes a multi-vector kernel: each threadgroup owns a tile of weight
  rows, reads them once, and produces outputs for all B activations. This is
  the change that restores the weight-bandwidth ceiling in the table under F7.
- Attention takes a block table per sequence and computes each `(sequence,
  head)` in its own threadgroup. `cudaattn.BatchedPagedAttentionTemplate`
  already has this shape; write the Metal counterpart from
  `metalattn/attention.metal`.
- RMSNorm, RoPE, SiLU and residual kernels index by `batch * dim`.
- Sampling runs on device per sequence with temperature, top-k, top-p and
  repetition penalty; argmax is the `temperature == 0` case.

Decode reads B tokens from one input buffer and writes B tokens to one output
buffer; there are no per-step allocations. On CUDA, capture the fixed-bucket
decode into a CUDA Graph after the first launch and replay it while the block
tables fit the captured capacity.

### 4.5 Paged KV and prefix sharing

`llamakv.Pool` owns one device allocation per layer in the model's KV dtype,
split into blocks of 16 or 32 tokens. Sequences hold block tables. The prefix
cache becomes a radix tree over token blocks whose leaves are block IDs with
reference counts, using `metalkv.RadixTree.Acquire`, `Release` and `Evict` as
they exist today. A prefix hit is then a block-table copy, not a host memcpy,
and the scheduler can share a system prompt across every active sequence.

Position bounds are enforced in one place: appending past the pool's capacity
returns an error to the scheduler, which finishes the sequence with
`finish_reason = "length"`.

### 4.6 Prefill

Prefill plans are bucketed by token count (for example 32, 128, 512, 2048) and
use tiled GEMM kernels. On Metal, `simdgroup_matrix` 8x8 tiles or MPS
`MPSMatrixMultiplication`; on CUDA, cuBLASLt through the existing
`nvidia/cublaslt` binding. RoPE and attention use a precomputed `cos/sin`
table stored with the weights. The scheduler splits long prompts into chunks
of one bucket size and interleaves a chunk with each decode step, so a long
prompt no longer stalls active streams.

### 4.7 Scheduler and serving

One goroutine owns the executable. Handlers submit `Sequence` values and
receive tokens on a channel; they never touch the backend. Each step:

1. Admit new sequences while KV blocks are available.
2. Take up to one prefill chunk per pending sequence, bounded by a token budget.
3. Build the decode batch from all decoding sequences.
4. Run prefill then decode, apply sampling, append tokens, detect EOS.
5. Deliver tokens outside the lock; free blocks for finished sequences.

The OpenAI layer renders the Llama 3 chat template from the tokenizer's
special tokens, counts prompt tokens by encoding, honors `max_tokens`,
`temperature`, `top_p`, `stop`, returns `finish_reason` of `stop` or `length`,
and streams text through an incremental UTF-8 detokenizer.

### 4.8 Telemetry

Keep the provenance vocabulary from `UNIFIED_INFERENCE_ENGINE_ARCHITECTURE.md`
and enforce it at the type level: every number the dashboard shows carries
`Provenance`. Replace the hardcoded hardware profiles with queried values
(`MTLDevice` name and recommended working set on Metal; NVML on CUDA) labeled
`configured`, and with `unavailable` when the query fails. Record per-kernel
device time in a profiling mode that encodes one kernel per command buffer, so
FlightRecorder events are kernels, not passes. Remove the constant tensor
count and the per-OS TFLOPs table.

### 4.9 What to delete

- The legacy interpreter `encodeForwardStepWithKVAndLogits` and the `Plan ==
  nil` path.
- `ExecutionPlan.Buffers`, `OpDispatch.Inputs/Outputs` as strings, and the
  per-dispatch `ValidatePlan` call.
- `DispatchTuner` in its current form; per-kernel profiling in the plan
  compiler replaces it.
- `InferRoles`, `NodeRouting`, and the paired `Speculate` endpoint until a KV
  transfer path exists.
- `GPUForwardPass`, `GPUPrefillPass` and the other error-dropping wrappers.
- The double model load in `main`.

## 5. Phased plan

Each phase keeps the previous path as the correctness reference until the new
path passes the same differential gates. Phases are ordered so that each one
is useful on its own.

### Phase 0: stop the bleeding (days)

- Return errors from every wrapper; fail engine construction on any nil
  pipeline or allocation.
- Include `ModelPath` in the Metal engine key; remove freed engines from the
  cache.
- Read `config.json` without decoding tensors; stop clamping `MaxSeqLen`
  silently (make it a flag with an explicit default and reject positions past
  it).
- Add EOS detection and `finish_reason = "length"`.
- Make GPU tests build their own engine or reset shared state in a helper.
- Remove the constant tensor count, the fixed FlightRecorder name and the
  hardcoded TFLOPs table.

Gate: existing differential tests pass in any order; `staticcheck` is clean.

### Phase 1: backend interface and package split (1 to 2 weeks)

- Introduce `llamabackend.Backend` and `Executable`; move the Metal and CUDA
  code behind them; add the CPU reference as a third backend.
- Split packages as in 4.1. Tokenizer gains a pre-tokenizer, special tokens,
  chat template and incremental decode.
- Delete the legacy interpreter and the build-tagged function duplicates.

Gate: the differential suite runs against the interface and passes for the
reference and Metal backends; the CUDA backend compiles under the interface.

### Phase 2: paged KV and device prefix cache (1 to 2 weeks)

- `llamakv.Pool` with block tables in the model's KV dtype on both backends.
- Radix prefix cache over device blocks with reference counting and eviction.
- Sequence lifecycle tests: capacity exhaustion, shared prefix across
  sequences, eviction under pressure, restore after partial match.

Gate: two interleaved sequences with a shared prefix produce the same tokens
as serial runs with independent caches; prefix hits perform no host copies.

### Phase 3: batched decode and scheduler wiring (2 to 3 weeks)

- Multi-vector GEMV, paged batched attention, batched norm/RoPE/SiLU kernels on
  Metal; the CUDA counterparts using the existing paged attention template.
- Device sampling with temperature, top-k, top-p and repetition penalty.
- Scheduler goroutine owning the executable; OpenAI handlers submit sequences.
- Chunked prefill interleaved with decode.

Gate: batch of 8 sequences matches 8 serial runs token for token at
`temperature = 0`; measured per-token time at batch 8 is under twice the
batch-1 time on the development host; streaming concurrency test at 16
clients.

### Phase 4: plan IR, buckets and compile-time validation (2 weeks)

- `llamaplan.Plan` with slots, lifetimes and launch geometry; `Compile`
  validates once; executors bind by slot.
- Prefill buckets with tiled GEMM kernels; decode buckets by batch size.
- Plan cache keyed by model hash, backend, phase, bucket and dtype.
- CUDA Graph capture for fixed-bucket decode.

Gate: a fused plan and its unfused reference for the same bucket agree within
declared tolerance; a plan whose slot lifetimes overlap fails to compile;
prefill of a 512-token prompt on the 1B model is measured and retained with the
evidence bundle described in `LUMINAL_INSPIRED_GPU_BACKEND_IMPROVEMENTS.md`.

### Phase 5: search (optional, after Phase 4)

Only once plans are real: have `egraph` rewrites emit `Op` sequences, extract
candidates per bucket, validate them through `Compile`, profile finalists
through the deployment path, and select per bucket. Retain rejected and
measured outcomes as Luminal does. If this phase is not scheduled, remove the
e-graph from the engine.

Distributed execution stays out of scope until Phase 4 is complete; hfsync and
the mesh join endpoint can remain as they are.

## 6. Notes for the concurrent workstream

Recent commits (`4763e0f`, `a176041`, `4b728c2`) added dimension, memory and
per-layer grammar validation to `ValidatePlan`. That validation is checking a
plan that the same code generated one call earlier, and it runs on every
dispatch. Further work on the descriptor plan has low leverage; the structural
items in Phase 0 and Phase 1 unblock everything else. The two earlier design
documents should have their status tables replaced by section 1 of this
document, or be folded into it.

## Appendix: findings index

| ID | Severity | Area | Location |
| --- | --- | --- | --- |
| F1 | S2 | plan | `engine_darwin.go:979-1391`, `engine_types.go:372-403` |
| F2 | S3 | plan | `engine_darwin.go:1395-1905` |
| F3 | S3 | plan | `engine_darwin.go:944` |
| F4 | S2 | plan | `engine_darwin.go:998-1043`, `1265-1287` |
| F5 | S2 | plan | `engine_darwin.go:356-370`, `server.go:526-564` |
| F6 | S3 | config | `main.go:18-59`, `model_ref.go:14-29` |
| F7 | S2 | decode | `engine_darwin.go:2668-2746`, `engine_other.go:441-456` |
| F8 | S3 | decode | `engine_darwin.go:2681-2701` |
| F9 | S2 | kv | `engine_darwin.go:847-870`, `engine_other.go:300-310` |
| F10 | S2 | kv | `engine_darwin.go:255-300`, `prefix_cache_darwin.go:100-149` |
| F11 | S2 | prefill | `llama_kernels.metal:526-565` |
| F12 | S3 | prefill | `llama_kernels.metal:608-674`, `engine_darwin.go:2266` |
| F13 | S3 | prefill | `llama_kernels.metal:676-750` |
| F14 | S3 | cuda | `gpu_model_cuda.go:364` |
| F15 | S2 | serving | `server.go:275`, `api_openai.go:292`, `347` |
| F16 | S2 | scheduler | `engine_types.go:231-266` |
| F17 | S1 | serving | `engine_darwin.go:3009-3022`, `engine_types.go:265` |
| F18 | S1 | serving | `api_openai.go:241-251`, `384-386`, `tokenizer.go:144` |
| F19 | S1 | model | `model_ref.go:128-130` |
| F20 | S3 | startup | `main.go:161` |
| F21 | S2 | backend | `engine_other.go:431-433`, `engine_darwin.go:346` |
| F22 | S1 | errors | `engine_darwin.go:402-455`, `1912`, `1948-1960` |
| F23 | S3 | lifecycle | `engine_darwin.go:319-347`, `2126` |
| F24 | S3 | telemetry | `cluster_node.go:575-602`, `engine_darwin.go:1938`, `2762` |
| F25 | S3 | mesh | `cluster_node.go:452-499`, `738-801` |
