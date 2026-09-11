# Prefill execution audit

The port's `EvalBatch` name does not imply batched matrix execution. At baseline
`679b74a`, both backends traverse all 43 layers separately for every token. Only
intermediate output projections are skipped. This was also true in the shared
`df5a09c` checkout inspected on 2026-09-07, including its in-flight changes.

This is source evidence, not a throughput measurement. No model was loaded by
this audit. Source identities and test receipts are retained under
`/Users/tmc/tmp/ds4-lib-workflow/evidence/worker-CODEX6E6A/iteration1/`.

## CUDA width-eight state at 00263d9

CUDA groups up to eight tokens and batches three Q/KV projections and two
activation quantizations across each group. Attention, cache updates, attention
output projections, routing, and FFN remain token ordered inside each layer.
This is not a 2,048-row matrix implementation.

Commit `00263d9` queues compressor updates and odd-layer mask fills on the
default stream instead of synchronizing once per compressed layer. A qualified
n=64 comparison preserves both full-logit vectors bit for bit. Operation spans
move `stream.sync` from 580 to 6, exactly 14 evaluated tokens times 41
compressed layers, while every other recorded operation count remains equal;
both arms report zero dropped spans. Ratio-4 and ratio-128 synthetic controls
also match a synchronized GPU baseline through emission and following-token
reuse, and queued-arm mutations make the comparisons fail.

The removed waits do not close the end-to-end workload. Both the preceding
source and `00263d9` time out at the registered 29-minute deadline on the same
2,048 captured oracle token IDs. Both terminal stacks show position 1,254; the
earlier sample is at layer 34 in compressor synchronization and the successor
sample is at layer 36 in RMS launch. Each stack is one location sample and does
not identify where time was spent, establish a rate, or distinguish slow
progress from a stall. Neither run writes a result or completion marker.
The run wrapper qualified CPU load but did not record CUDA device health. A
later check found the device unavailable, but no retained evidence dates when
that state began. After reboot, the functional CUDA probe succeeded while the
reported PCIe link remained 2.5 GT/s x1 and the procfs model remained unknown,
so neither field diagnoses the earlier failure. The receipts do not establish
device health during either run and are not qualified performance results.

Evidence is in
`../evidence/worker-BABFB352/cuda-compressor-defer-n64-20260907/` and
`../evidence/worker-BABFB352/cuda-compressor-defer-workload-20260907/`.
The remaining performance work must batch more of the token-ordered layer body
or otherwise reduce its device work, then run under a device-health precondition.
Another unchanged 29-minute run is not a measurement plan.

## Early-position CUDA phase diagnostic

An event-instrumented `EvalBatch` over positions 0 through 7 preserves the
uninstrumented full-logit result at exact float32 bits. Default-stream elapsed
time ranks attention at 44.18%, expert FFN at 17.80%, attention output at
15.16%, the batched Q/KV front at 9.53%, FFN pre at 6.58%, shared FFN at
5.75%, and the remaining phases below 0.4% each. Counts reconcile to 2,452
intervals and the result passed functional device checks before and after.

These are instrumented stream-event intervals, including device idle time, at
positions 0 through 7 only. They are not kernel-busy time, throughput, Metal
comparison, or evidence about position 1,254. In particular, the result does
not transfer the older unretained short-decode claim that MoE dominates to the
2,048-token prefill workload. The retained receipt is
`../evidence/worker-BABFB352/cuda-phase-profile-v3-run-20260907/`.

A subsequent instrumented 2,048-token call retained monotonic progress every
128 completed tokens. It reached the 1,152 checkpoint before the same
29-minute timeout. The nine interval durations increase from 60.94 seconds to
243.97 seconds, consistent with growing-context work and inconsistent with a
constant per-token cost. The terminal stack was at position 1,259, so this run
also passed the position 1,254 sampled by both earlier timeouts; a fixed stall
at exactly 1,254 is ruled out for this run.

The progress records include hook and fsync overhead. They do not establish a
rate, completion time, absence of a later stall, or uninstrumented throughput.
The closed receipt is
`../evidence/worker-BABFB352/cuda-prefill-progress-v3-run-02-20260907/`.

## Integrated state at f4577d6

CUDA now processes longer input as ordered adjacent pairs. Each pair batches
Q8_1 activation quantization and the Q/KV dense projections; attention, cache
updates, routing and FFN remain token ordered. Intermediate pairs omit the
unused vocabulary projection. This is two-row matrix work, not a single
2048-row layer-major matrix operation.

Retained warmed/reset comparisons pass for both prefill and fixed continuation
at n=2 and n=64, with all 129280 float32 values byte-identical. Token-row and
adjacent-pair-order mutations produce captures that the exact comparator
rejects. These gates do not establish n=2048 identity or throughput.

Metal encoder sharing at `be98972` preserves all ten warmed prompt-A state
files against `72a645d`, matching registered concat `db6f6125aace1267...`.
Independent inspection matches all 94 source files per arm to their revisions;
only `metal_batch_darwin.go` and `metal_hc_darwin.go` differ. Generated-text
identity alone was insufficient; the full-logit capture closes that earlier
evidence gap. Submission counts describe a mechanism, not measured attribution
of the throughput difference.

A separate CUDA GB10 reference run on spark-31db measured 661.18 tokens/s for
2048-token prefill and 17.73 tokens/s for 128 generated tokens (17.80 after the
first token). The build is `b0a147a` **plus `ds4-probed.patch`**, using the ordinary
single-device CUDA path, raw `promessi_sposi.txt`, context allocation 65665 and
prefill chunk 4096. It is one first-frontier observation, with no separate
prefill warmup. Startup repacking of 474 aligned artifacts precedes timing.
Host load qualified at 1.00 before and 1.77 after, against limit 5.00. All 2131
archived source files match before and after, allowing the registered patch.
This is a qualified reference, not a port parity or significance result.

Evidence locations relative to the primary worktree:

- CUDA n=2: `../evidence/worker-BABFB352/prefill-dense-n2-20260907/`.
- CUDA n=64: `../evidence/worker-BABFB352/prefill-dense-n64-20260907/`.
- Metal full logits: `evidence/worker-DF6906F6/metal-encoder-merge-logits-20260907/`.
- Qualified CUDA reference: `../evidence/worker-CODEX6E6A/oracle-bench-2048/RESULT.md`.

## Historical integrated state at 6b0bfcb

CUDA uses layer-major traversal only when `len(tokens) == 2`
(`cuda_session_linux.go:277-281,310-315`). Each row has separate transient
activations; layer history is shared. The arithmetic kernels remain single-row.
The 64- and 2048-token paths still use the original token-major loop.

The n=2 candidate captured at `f59c503` matches its baseline exactly for both
129280-float prefill and fixed-continuation vectors. Reversing token order within
each layer produces a valid capture that the comparator rejects. These are
warmed, reset, position-zero captures for tokens `[0,3476]`, continuation `0`,
context 2049. They do not establish arbitrary prefixes or throughput. The
successor `cb52b99` (integrated as `6b0bfcb`) makes extra-row allocation
transactional; its model-free failure/retry/reuse controls pass. That successor
has source-equivalent arithmetic, but was not the captured executable.

Metal remains token-major (`metal_runtime_darwin.go:561-567`). Router integration
`1842667` leaves three explicit layer waits at lines 665, 783 and 843 and removes
the separate synchronous router helper. For 2048 tokens this is
`2048 * 43 * 3 = 264192` explicit layer waits, excluding other operations.
Both arms built at the DataOffset baseline `02ef5c7` produce identical prompt-A
state files; their ordered concatenation SHA-256 is
`db6f6125aace12670703a3b2a422c86655a9938fc1335f718657d2ab4913a6fe`.
The model-enabled operation-count test passes with 15 encoded operations and
fails when the runtime is reverted to 14. Neither result measures throughput.

Current receipts:

- CUDA n=2: `evidence/worker-BABFB352/prefill-n2-layer-major-20260907/`.
- CUDA lifecycle: DGX `/home/tmc/tmp/ds4-prefill-n2-lifecycle-test-20260907/`.
- Metal router: sibling `../evidence/worker-CODEX6E6A/router-integration/`.

The remaining performance work is larger token-row execution and batched matrix
kernels, with exact 64/2048 prefill and continuation gates before rate claims.
Qualified 2048-token port/oracle rates on both hosts remain missing. The older
source counts below describe their named baselines, not the integrated code.

## Execution shape

Paths below are relative to `examples/ds4-min` at `679b74a` unless noted.

| Mechanism | Port source | Oracle b0a147a source |
| --- | --- | --- |
| Token-major prefill | `metal_runtime_darwin.go:556`, `cuda_session_linux.go:250` | `ds4.c:36091-36118` uploads token rows then loops layers with `n_tokens` |
| Metal submission | Six explicit layer waits in `metal_runtime_darwin.go:666,748,774,808,850,866`; commit/wait implementation at `metal_batch_darwin.go:463` | `ds4.c:36058-36062` controls split conditions; unsplit execution ends at `36150` |
| CUDA synchronization | Per-token at `cuda_session_linux.go:296`, compressor at `443` | Batch matrix dispatch operates over token rows |
| CUDA dense projection | Single-vector API at `cuda_runtime_linux.go:888` | `ds4_cuda.cu:14763-14793` selects MMQ or cuBLAS for multiple tokens |
| CUDA routed experts | Single-token workspace in `cuda_session_linux.go:383-418` | `ds4_cuda.cu:24439-24479` processes token/expert assignments |
| CUDA mixed attention | One block of 64 threads at `cuda_compressed_linux.go:73`; serial dot/value loops at `218-235` | Requires a matched live probe before assigning a measured share of the gap |

For 2048 tokens at this baseline, the explicit Metal layer waits total
`2048 * 43 * 6 = 528384`. CUDA stream synchronizations total
`2048 * (41 + 1) = 86016`. These are source-derived call counts, excluding helper
operations. They are not timings, and the baseline Metal count must not be
ascribed to a newer batch-collapse candidate.

CUDA indexer selection uses a strict `count > 512` condition. A fresh 2048-token
prefix reaches exactly 512 ratio-four rows, so that top-k cliff cannot explain
this prefix's slowdown. Odd layers still allocate and upload all-ones masks.

## Reconciliation with concurrent work

Shared CUDA at `df5a09c` retains token-major traversal and all the above explicit
stream synchronizations, now through the `syncStream` tracing wrapper. Dirty
CUDA changes already cover half-rounded Q, pinned cuBLAS scalar pointers, and
model first touch. Those files belong to peers. Source presence does not certify
their numerical or performance gates.

The Metal candidate inspected during the original audit included a peer-owned batch collapse. It has three
explicit layer waits (`metal_runtime_darwin.go:660,768,845`) plus a synchronous
router helper (`metal_ops_darwin.go:695-696,736-737`), or 352256 layer-associated
commit/waits per 2048 tokens before other helpers. Its measured impact and exact
output gate must be taken from its receipts, not inferred from these counts.

## Numerical contract before matrix batching

Oracle `ds4_cuda.cu:1081-1084` explicitly says MMQ changes FP32 reduction order,
and that its prefill was validated with continuation vectors rather than byte
comparisons. Its eligibility is conditional (`1095-1112`). Therefore simply
porting the oracle's fastest dispatch is not evidence of compatibility with the
registered bit-identical warmed-logit gate.

The coordinator identified the untracked CUDA diagnostic
`cuda_single_token_repeat_linux_test.go`, independently opened here: it evaluates
token 0 at position 0, context 1024, twice with Reset. The 129280 float32 logits
are hashed as little-endian bytes. The test explicitly disclaims oracle
equivalence. It cannot detect any multi-token batch error. A new batch gate must
compare old/new EvalBatch at 2, 64 and 2048 tokens, with the same input and lifecycle,
and inspect continuation logits to test the resulting cache. Preserve the exact
registered hash. Do not widen tolerances or replace it with plausible output.

An implementation plan for true prefill needs token-row workspaces, layer-major
execution, batched dense and routed projections, and causal raw/compressed-cache
updates. Removing waits alone does not implement that plan.

## Measurement receipt status

The previously reported DGX long run is complete with exit 2. Its stderr says
`test timed out after 55m0s`; the stack is inside `EvalBatch` at position `0x6ba`
(1722), layer `0x27` (39). Its empty stdout contains no completed throughput
result. The remote receipt is:

`/home/tmc/tmp/ds4-lib-workflow/evidence/worker-BABFB352/prefill2048-long-20260907T003836Z`

A separate peer-owned final run was active during inspection. No second model
process was started. Published oracle CSV rates remain reference measurements
from another run, not host-qualified measurements produced here.

## Reusable controls

`scripts/qualified-run_test.py` tests the authoritative external wrapper through
`QUALIFIED_RUN`. The copied implementation was removed at the owner's request.
The first authoritative run passed six of nine controls; remaining failures
were missing identity inputs, receipt overwrite, and invalid repetition input.
See the retained log for its exact wrapper identity and later receipts for fixes.

The authoritative wrapper moved to the primary checkout's
`scripts/qualified-run.sh`. Its final inspected hash is
`93e8ee3f22edc4bb57f2093e0e5bc2e202aa54e9ea6bae864e04be75278b8fa4`.
All 11 controls pass there (`authoritative-tool-tests-v5.log`). Actual hashing
failure is now fatal as well as unreadable/missing identity input. The earlier
v4 run failed because it targeted the old path during this move; no tool verdict
was obtained in that run.

## Metal widening surface at f9d11ce9

What it would take to make the Metal runtime evaluate T tokens per dispatch
instead of one. This is source evidence across every encoder in
`examples/ds4-min`, not a measurement; no model was loaded to produce it.

The encoders fall into four classes, and only one of them is a real obstacle.

**Already token-aware.** `EncodeMatmulF16`, `EncodeMatmulQ80`,
`EncodeMatmulQ2K`, `EncodeMatmulIQ2Selected` and `EncodeMatmulQ2KSelected` take
a `tokens` parameter. `EncodeHCPre`/`EncodeHCPost` carry a token dimension as of
`953109fc`; `EncodeRouterHashTokens`, `EncodeRouterBiasedTokens` and
`EncodeSumVectorsTokens` as of `c22fdb9b`. The runtime still calls the
single-token forms, so the work at these sites is changing call sites, not
kernels.

**Free, and now asserted.** `EncodeRMS`, `EncodeClamp10`, `EncodeSwiGLU`,
`EncodeSwiGLUWeightedGPU` and `EncodeSwiGLUWeightedSlots` need no change at all.
`rms_norm` offsets by `group*width` and reduces within one group; the other
kernels index a flat thread position. A T-row call is already T independent row
computations. `TestMetalElementwiseTokenIndependence` (`f9d11ce9`) pins each row
as bit-identical dispatched alone and dispatched as one row of five, and each of
the four kernels was mutated individually to confirm the assertion turns red.

**Widenable, decomposition already correct.** `EncodeRopeAttentionMixed` is four
dispatches -- `RopeQPrepare`, `RopeKVCommit`, `RopeAttentionWide`,
`RopeOutUnrotate` -- with encoder ordering supplying the dependency that a
device-scope barrier used to. That split is exactly what a batched form needs:
commit all T rows of the KV cache, then let all T tokens attend causally over a
cache that already contains their in-batch predecessors. What is missing is a
per-token `position` rather than the single scalar, and the fixed
`out.Len() != 32768+512+32768` guard.

**The one real scan.** `compressor_pool` writes row `position%ratio` of a
scratch window and pools when the window closes, and the two ratios differ in
kind. At ratio 128 the indexer pools and keeps nothing, so windows are
independent and batch cleanly. At ratio 4 the attention compressor pools across
a `2*ratio` sliding buffer and shifts the upper rows down, so each window reads
the previous one: T tokens are T/4 dependent steps, not one dispatch. This is
the only place in the Metal path where a token loop survives widening.
`cudaPrefillBatchWidth = 8` is two such steps.

It bounds the shape of a batched prefill without bounding its value. The scan
moves state buffers, not weights, so it is not what the streaming cost is made
of.

**Separate blocker, producer side.** `metalSession.routedOut` and `sharedOut` are
stored sub-slices of `ffnBlocks` at `[0,4096)` and `[4096,8192)`. The consumer is
already handled -- `EncodeSumVectorsTokens` reduces across the pair at a
`[token][routed|shared]` layout -- but `tokenRow` is contiguous, so a token-major
`routedOut` would place token t at `4096*t` where the layout needs `8192*t`.
These two fields should be deleted in favour of per-token windows rather than
widened. Recorded at `a038ec77`.

### Not established here

No throughput claim. Whether widening moves the registered oracle rate is a
measurement nobody has taken, and the one prior structural change of this family
-- 44 command buffers per token down to 2.1 -- bought 6 percent
(`ds4-prefill-gap-is-batching`). The F16 lineage defect and the hyper-connection
`sqrt`/`rsqrt` selector both remain open and both move
`metalRegionProfileLogitsSHA256` if resolved, so a widening that changes either
needs the digest re-registered deliberately rather than adjusted.

### Is the swap numerically free?

Swapping a call site from `EncodeMatvecX` to `EncodeMatmulX(..., tokens)` is only
safe if the batched kernel at T=1 reproduces the serial one bit for bit. The
runtime calls exactly four matvec forms. Three are settled, one is not.

`EncodeMatvecQ80` and `ops.MatvecQ80` dispatch `kernels.MatvecQ80` with the same
grid `(rows+1)/2`, the same `32x4` threads, the same `2*32*4` threadgroup bytes
and the same scalars; they differ only in submission. So
`TestMetalMatmulQ80PreservedBitIdentity`, which compares the batched `Mul`
against per-token `ops.MatvecQ80` at T of 1, 3, 7, 8, 13, 32 and 64 -- and since
`8c3cf6ca` at the seven production Q8_0 shapes -- transitively covers the
runtime's encoder.

`ops.MatvecIQ2Selected` and `ops.MatvecQ2KSelected` are thinner still: they
allocate, then call `EncodeMatvecIQ2Selected` and `EncodeMatvecQ2KSelected`
themselves rather than restating a dispatch. The equivalence there is
definitional. `cf0bc5c8` added the two production `(cols,rows)` pairs.

`EncodeMatvecF16` is the exception, and it is the known one.
`TestMetalMatmulF16AtRuntimeShape` asserts that `EncodeMatmulF16` -- a clone of
the output-head SIMD kernel -- and `matvec_f16` disagree on about 97% of outputs
at both runtime shapes, at T=1 as much as at T=8, with worst relative
differences up to 1.6e-2 against a registered tolerance of 1e-4.

So the F16 decision is not one blocker among many. It is the only numerics
blocker on the matmul call sites, and it gates exactly five of them: the
attention compressor KV and gate, the indexer compressor KV and gate, and the
router. The other three forms can be swapped without moving a bit.

## The GB10 decode floor

Measured 2026-09-11 on the DGX host by session `094936CE` with a compiled
`nvcc -O3 -arch=sm_121` benchmark. This is a peer's run, recorded here because
it bounds the parity target; it was not reproduced on this side.

GB10 is an integrated 256-bit LPDDR5x part, not HBM: ~273.06 GB/s theoretical,
**247.89 GB/s measured streaming read** over a 2 GB buffer at 20 iterations
(D2D kernel copy 225.41 GB/s, `cudaMemcpyDefault` 242.46 GB/s). An earlier
analysis had assumed ~1,000 GB/s.

A token must read about 8.5 GB of weights -- 43 layers of dense projections plus
six selected experts, not the 86.72 GB model, because `cuda_experts_linux.go`
computes its weight base from `ids[slot]` over a six-element buffer. So

	8.50 GB / 247.89 GB/s = 34.29 ms/token  ->  29.16 tok/s

is a ceiling for any implementation on this host that reads the weights it
needs.

State CUDA decode goals as a distance from this floor rather than as a multiple
of the oracle. A multiple moves when the oracle does and implies headroom that
does not exist; the floor is fixed for a given context.

### The floor is not one number: it has a KV term

Weights are the whole story only at short context. The KV cache adds `ds4RawCap`
raw entries (128) plus `ctx/4+2` compressed, each `ds4KVWidth` = 512 floats, per
layer over 43 layers -- see `metal_runtime_darwin.go:24-25,385,450,464` and the
matching `cuda_session_linux.go:135-170`. That term is negligible in the regime
the port has been measured in and is not negligible where the oracle was
measured:

	ctx  1,024:    128 +    258 entries   0.034 GB/token   floor 34.43 ms
	ctx 65,665:    128 + 16,418 entries   1.457 GB/token   floor 40.17 ms

So 34.29 ms is a short-context floor. Quote the floor with the context it was
computed for, and compare two engines only at the same one.

This matters immediately, because the two figures that have been compared all
week were taken at different contexts: the oracle's 56.40 ms at context 65,665
positions 2048..2176, our port's 111.25 ms at context 1024 positions 0..5
(disclosed by `094936CE`, 2026-09-11). Against its own floor the oracle is 1.40x,
not the 1.64x that a short-context floor implies -- it is nearer saturation than
the earlier framing said. The port was measured in the easier regime, so the real
gap is wider than the ratio of the two numbers, not narrower. Every fraction
downstream of that pair is unsettled until both are re-run at one context.

### What this does not establish

Not a mechanism, from the fractions. Dividing both engines by the same assumed
8.5 GB makes "percent of achievable bandwidth" the time ratio rescaled --
`(8.50/0.0564)/(8.50/0.1180)` and `0.1180/0.0564` are both 2.0922 -- so those
fractions cannot separate poor coalescing from launch serialization. Both
mechanisms produce them identically. The mechanism was settled instead by direct
measurement, below.

And not that decode is bandwidth bound. It is not. `094936CE` measured GPU stream
time at 99.99% of wall and the per-phase sum at 96.51% of that, which kills the
launch-serialization hypothesis; then `ncu` put the dominant kernel,
`ds4_iq2_xxs_pair_selected_q8_1` (542.2 us/layer), at 89.1% SM compute against
10.0% memory. Decode is ALU bound inside IQ2 dequantization. The mechanism is
visible in the source: our inner loop
(`examples/ds4-min/cuda_experts_q81_linux.go:25-33`) does 32 scalar
shift/mask/branch/multiply steps per 32 weights where the oracle
(`ds4_mmq.cu:2594-2615`) issues 8 `__dp4a` with `__vcmpne4`/`__vsub4` for
branchless signs.

The floor still binds -- it is a bound, not a prediction -- but a workload can sit
far above it for reasons that have nothing to do with bandwidth, and this one does.

### Where decode parity actually lives

What the floor does say, and what the phase table says with it, is that the
expert kernel cannot carry decode parity on its own. Zeroing the whole
`expert-ffn` phase leaves `111.25 - 23.22 = 88.03 ms`, still 1.56x the oracle's
56.40. The other 88 ms is attention (22.57), attention-output (20.50, at 0.89
waves/SM -- an occupancy problem, not an ALU one), ffn-post (14.81), compression
(10.40), ffn-pre (8.34) and shared-ffn (7.52).

### Batching does not amortize dequantization by T

Worth stating because it is the natural assumption and it is wrong by about 40x.
A dequantized weight block is reusable only across tokens that routed to that
same expert. With 6 of 256 experts per token, a chunk of W tokens shares an
expert about `W*6/256` times, so the amortization factor is that, not W -- 48 at
W=2048, and below 1 at the port's current `cudaPrefillBatchWidth` of 8, where a
dedup kernel would do nothing at all.

Adding a token axis to the existing kernel does not help either. Metal already
has the batched variant and it carries `tok` as a grid dimension only
(`examples/ds4-min/metal_matmul_expert_darwin.go:390`): every `(row, slot, tok)`
threadgroup re-reads the weight bytes and redoes the grid lookup, the sign lookup
and the multiply-accumulates. It is T independent copies fused into one dispatch.
The shape that does amortize is expert-major with first-owner dedup, which the
oracle implements at `ds4_mmq.cu:2656-2760` -- gather the slots that chose this
expert, dequantize once, accumulate against each. Its `MAXM` bound and the chunk
width are what set the real factor.
