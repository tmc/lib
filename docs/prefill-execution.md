# Prefill execution audit

The port's `EvalBatch` name does not imply batched matrix execution. At baseline
`679b74a`, both backends traverse all 43 layers separately for every token. Only
intermediate output projections are skipped. This remains true in the shared
`df5a09c` checkout inspected on 2026-09-07, including its in-flight changes.

This is source evidence, not a throughput measurement. No model was loaded by
this audit. Source identities and test receipts are retained under
`/Users/tmc/tmp/ds4-lib-workflow/evidence/worker-CODEX6E6A/iteration1/`.

## Integrated state at 6b0bfcb

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
