# ds4-min `-egraph`: an opt-in, CUDA-only e-graph fusion mode

Date: 2026-09-13. Status: **deferred** — see below. No implementation exists; the
draft started during review was discarded.

## Deferred, 2026-09-13

Not scheduled. The decision, from the repository owner: e-graph support for
ds4-min waits until `nvidia/x/egraph` covers much more of the model than it does
today. Nothing in this document is withdrawn — the audit below is what motivates
the deferral rather than contradicting it. The shortest statement of the reason:
of the four weight formats in the region this plan targets, `main`'s generator
emits two (`q8_0`, `q2_k`), has no IQ2_XXS kernel at all (what was briefly named
`iq2_*` is unrelated packed 2-bit, renamed `u2_*`), and three of the four rewrite
outputs `BatchGEMMRules` can produce are refused by `GenerateCUDA` while the cost
model prefers them. Even Stage 1, which is pure planning and generates nothing,
would print a plan for regions the tier below cannot build.

An IQ2_XXS kernel landed on `main` in `45dd2733` after this deferral was written
and was relayed as unblocking the plan. It does not: its codebook is llama.cpp's
quantizer-side `kgrid_2bit_256` ({1, 3, 5, 7}) rather than the dequant table
`iq2xxs_grid` ({8, 25, 43, 60} — held verbatim in `lm/quant/iq2xxs_tables.go`).
Compared entry by entry, the ordering is identical in all 256 entries and the
complete value mapping is {(1,8), (3,25), (5,43)}, which is not a scale factor
(8.0, 8.33, 8.6), so groups mixing two codes get the wrong relative weights. Its
device test cannot see this because `iq2xxsMatvecReference` calls
`egraph.DequantizeIQ2XXS`, which reads the same table the kernel embeds. Q8_1 is
also still absent — `q8_1_matvec` was removed in `adaf89a1` — so the region's four
formats are covered by two correct kernels, one wrong one, and one missing one.
Details: `~/tmp/agent-collab/lib/20260913-E4139097-iq2xxs-uses-the-quantizer-grid-not-the-dequant-grid.md`.

Revisit when `GenerateCUDA` covers the region's formats with device value oracles
for each — the gating pattern now exists (`q2k_cuda_test.go`, `u2_cuda_test.go`)
and is the precondition, not the blocker — and when the rewrite set and the
emittable set are no longer disjoint in the middle.

## Provenance: every path below is read on `codex/ds4-min`, not `main`

This matters more than it usually would, because the plan's own subject is
branch-only. Verified with `git ls-tree -r` at the time of writing
(`main` is 2 commits ahead of, and 301 behind, `codex/ds4-min`):

| Path | `main` | `codex/ds4-min` |
| --- | --- | --- |
| `examples/ds4-min` | absent | 178 files — the subject of this plan |
| `nvidia/x/cudamega` | absent | 3 files |
| `nvidia/x/cudafuse`, `x/cudair`, `x/cudatensor` | absent — deleted in `f824557c`, briefly resurrected by `489ebc16` today and reverted in `b539ebb5` | 3, 4, and (cudatensor) 0 files; `cudafuse`/`cudair` are divergent copies |
| `lm/quant` | 4 files: `doc.go`, `example_test.go`, `quant.go`, `quant_test.go` | 9: adds `kquant.go`, `iq2xxs_tables.go`, `kquant_test.go`, `testdata/ds4_golden.{c,txt}` |
| `nvidia/x/egraph`, `x/cudaoccupancy`, `x/cudagraph`, `x/cudaquant` | present, same file count | present |
| `nvidia/x/cudasass` | 18 files | 14 — `dot.go` and `pressure.go` and their tests are on `main` only |

`x/cudafuse` and `x/cudair` have a history worth stating, because both a peer
and I got it wrong in opposite directions. They were added on `main` in
`d4f2f74e` (2026-08-31), **deleted** from `main` in `f824557c` (2026-09-04,
"nvidia/x: remove unexecuted tensor DSL" — a deliberate removal of a
"three-package proof island" with no production consumer), and **re-added** in
`489ebc16` today by session 9980FF84, whose subject line is "reject
non-elementwise operators" and whose diff is 584 insertions and zero deletions
across seven files. So: my earlier reading of them as absent from `main` was
correct for the tree I read; 3CE9C186's ancestry argument (`merge-base
--is-ancestor d4f2f74e main`) was the wrong instrument, because ancestry of an
adding commit says nothing about a later deletion. Two details matter for whoever
reconciles it (found by 3CE9C186, verified here):

- `f824557c` removed **three** packages and their consumer, not two:
  `x/cudafuse`, `x/cudair` and `x/cudatensor`, plus
  `examples/nvidia/cuda-tensor-dsl/main.go` — 11 files, 1197 deletions, authored
  by Travis Cline. `489ebc16` resurrected two of the three; `x/cudatensor` is
  still absent from `main` (it survives on `codex/ds4-min`, which is 301 commits
  behind `main` and simply never took the deletion). So the question "should
  these be on `main`" covers a third package, and the example that consumed the
  DSL is gone on both refs.
- `489ebc16` is not a revert. The resurrected files are smaller rewrites:
  `fuse.go` 146 lines against 166 deleted, `ir.go` 196 against 325,
  `ir_test.go` 55 against 157, `fuse_test.go` 87 against 138, with no shared
  history to diff. Reconciliation is therefore a three-way read — the 2026-09-04
  deletion, `main` today, and `codex/ds4-min` — not a choice between two copies,
  and there is no "restore the deleted version" that keeps what `489ebc16`
  added.

The first question is whether the packages belong on `main` at all — a human
removed them nine days ago — and only then which text wins.

Apart from that, a reader checking this plan against `main` finds neither the
example nor the Q2_K/IQ2_XXS oracle. State the branch beside a claim
rather than saying "in the repo": the same mistake was made today in the other
direction, reporting branch-only packages as untracked from checkouts that could
not see the branch (noted by 3CE9C186; see
`memory/review-the-tree-the-package-lives-in.md`).

Origin: a proposal (retained at `/tmp/ds4-plan.txt`) to rebuild `examples/ds4-min`
on a Luminal-style three-tier compiler — e-graph saturation, megakernel codegen
behind a CPU-side occupancy pre-flight, and CUDA Graph packaging. This plan
specifies the first increment of that: a flag that turns the planning tier on,
reports what it decided, and changes nothing about what the GPU runs. It also
records which of the proposal's claims the repository supports today and which
it does not.

## Scope of the flag

```
  -egraph
        plan the routed layer's fusion regions with equality saturation
        and report the extracted plan; requires -backend=cuda
```

Normative behavior:

1. `-egraph` is a boolean, default false. Absent, no code path in this plan runs
   and no output changes. The flag is registered unconditionally in
   `parseConfig` (`examples/ds4-min/main.go:309` registers the adjacent
   `-cuda-profiler-stop` the same way), so `-h` lists it on every platform.
2. `-egraph` with a resolved backend other than `cuda` is a usage error, named
   as such: `-egraph requires -backend=cuda`. Resolution happens in
   `selectBackend`, so the check goes after it. On darwin/arm64 and on
   `!linux && (!darwin || !arm64)` `selectBackend` already rejects `cuda`
   (`examples/ds4-min/backend_darwin.go:17`, `backend_other.go:15`), which is
   what makes the mode CUDA-only without a build tag on the flag itself.
3. With `-egraph -backend=cuda`, the planning pass runs once, after the model
   profile is validated and before the backend is opened, and writes its report
   to stderr. A planning failure (below) is a run failure: non-zero exit, no
   tokens generated.
4. The planning pass must not allocate device memory, open a CUDA context, or
   compile anything. It is pure Go over a DAG of operator names.
5. Token output, logits digests, span sidecars, and every gate in
   `examples/ds4-min/manual-gates.md` are bit-identical with and without the
   flag. This is the increment's central constraint and its acceptance test:
   the same `-expect-final-logits` digest must pass both ways.

## What the planning pass plans

The DAG is the feed-forward region of a routed layer (index >= 3), with operator
names taken from the encode steps `(*cudaRuntime).evalLayerProjected` runs, in
that order (`examples/ds4-min/cuda_session_linux.go:645`):

```
ffn_norm  = (hc_pre hc w_ffn_norm)
ids       = (router_biased (f16_matvec ffn_norm w_router) w_expert_bias)
shared    = (q80_matvec (swiglu (q80_matvec ffn_norm w_shared_gate)
                                (q80_matvec ffn_norm w_shared_up)) w_shared_down)
routed    = (sum_slots (q2k_selected (quant_q81 (swiglu_slots
               (iq2_selected (quant_q81 ffn_norm) w_expert_gate ids)
               (iq2_selected (quant_q81 ffn_norm) w_expert_up ids)
               w_expert)) w_expert_down ids))
out       = (hc_post hc (add routed shared))
```

Three rewrites, saturated with `egraph.Saturate` and extracted with
`egraph.FusedMegakernelCost`:

| Rewrite | Collapses | Region it names |
| --- | --- | --- |
| `fuse_shared_gate_up_swiglu` | `(swiglu (q80_matvec ?x ?wg) (q80_matvec ?x ?wu))` | `fused_shared_gate_up_swiglu` |
| `fuse_routed_gate_up_swiglu` | the same shape over `iq2_selected` with a slot weight | `fused_routed_gate_up_swiglu` |
| `fuse_quant_into_routed_experts` | `(fused_routed_gate_up_swiglu (quant_q81 ?x) ...)` | `fused_quant_routed_gate_up_swiglu` |

The first two are the gate/up pair `encodeSwiGLU` and `encodeSwiGLUSlots`
currently consume from DRAM (`cuda_session_linux.go:726`, `:748`). The third
folds the Q8_1 quantization at `cuda_session_linux.go:736` into the expert
region that reads it.

The pass asserts that the extracted plan selects `fused_shared_gate_up_swiglu`
and `fused_quant_routed_gate_up_swiglu`, and fails with the extracted
S-expression in the error otherwise. Without that assertion the pass prints a
plan whether or not any rule matched, which is the failure mode recorded in
`memory/gate-fail-open-pattern.md`: a rename in the DAG would silently stop all
three rewrites from firing and the report would still look like success. The
extraction is deterministic, so the assertion is a unit test, not a GPU gate.

Report format (stderr, four lines, prefix `ds4-min: egraph:`): saturation
iterations and merges; plan cost and the fused region names; the extracted
S-expression; and the sentence that the run launches the hand-written kernels
regardless. The last line is not decoration — see the next section.

## Status of the tier-2 gaps as of 2026-09-13, late

Three of the gaps below were addressed on `main` today by session 9980FF84,
after this plan was drafted. The table that follows documents the state on
`codex/ds4-min`, which is still the state of the branch ds4-min lives on. On
`main`:

- `92b3cbd4` extends `GenerateCUDA` to `half`/`float16` (via `uint16_t` storage
  and inline PTX `cvt`, avoiding the missing-header problem) and to `q8_0`,
  `q8_1`, `iq2_xxs` and `q2_k`, with ten operator arms including
  `fused_swiglu_q8_0`, `q8_0_matvec`, `q8_1_matvec`, `iq2_simd`, `iq2_scalar`
  and `q2_k_matvec`.
- `489ebc16` removed the `cudafuse` placeholders on `main`'s copy, and was
  **reverted** in `b539ebb5` once it turned out to have resurrected two deleted
  packages to do it (see the provenance note). `main` now has no `x/cudafuse`,
  `x/cudair` or `x/cudatensor` at all, so Gap 2 is live on `codex/ds4-min`, which
  is the only ref where the package exists and the only ref where ds4-min is
  built. The fix belongs there.
- `65d7fde2` makes `qwen-egraph-infer` synthesize its source from the extracted
  AST instead of compiling the static constant.

Three of the new kernels initially disagreed with the reference, were reported by
3CE9C186 and me, and were fixed or withdrawn in `adaf89a1`:
`q2_k_matvec`'s arithmetic and element map now match `lm/quant/kquant.go:73`
exactly (verified here line by line: `base = 32*(group/8) + 16*(group&1)`,
`shift = ((group/2)&3)*2`, `d*(sc&0x0f)*q - dmin*(sc>>4)`); `iq2_simd`/`iq2_scalar`
were renamed `u2_simd`/`u2_scalar`, dropping a false IQ2_XXS conformance claim for
what they are — packed unsigned 2-bit with no codebook, signs or per-group scale;
and `q8_1_matvec` was removed. The emittable precisions on `main` are now
`float`, `half`, `q8_0`, `u2` and `q2_k`.

**The arithmetic was right and, for a few hours, nothing in the repo would have
caught it if it were not.** `q2_k_matvec` was covered by `TestGeneratedKernelsCompile`,
which says only that the source is valid CUDA, and `TestQ2_KArithmeticReference`,
which compared two hand-typed Go copies of the same expression written fifteen lines
apart in the same test body — neither reading `lm/quant` (it is not on `main`) nor
executing the generated kernel. Measured on a throwaway worktree at `main`: editing
only the CUDA source string at `codegen.go:437` from `((group / 2) & 3) * 2` to
`(group & 3) * 2` — reintroducing precisely the element-map defect that had just been
fixed — left `go test -run 'Q2_K|Compile' ./x/egraph/` green. Looping the comparison
over all 256 elements scales a tautology; it does not make it an oracle
(`memory/format-check-is-not-a-gate.md`,
`memory/tautological-test-one-author-two-copies.md`).

**That is now closed.** Three commits later the package gates its quant kernels the
way it already gated `swiglu`, `rmsnorm` and `fp8` — build the `Expr`, `GenerateCUDA`,
NVRTC-compile, launch, compare elementwise against a float64 CPU reference in its own
function:

- `f1521f68` (3CE9C186) adds `q2k_cuda_test.go`: an independent `dequantizeQ2K` over
  the block bytes, `q2kMatvecReference` in float64, and `q2kTestBlocks` constructed so
  that dropping either scale nibble or reading elements sequentially changes every row
  rather than permuting the same values. Measured dissociation with `codegen.go` rolled
  back to `92b3cbd4`: the Go-to-Go test passes while the device test fails all four
  rows in both launch geometries.
- `3c61ed92` (9980FF84) adds `u2_cuda_test.go` for `u2_simd` and `u2_scalar`, whose
  `u2Reference` extracts each 2-bit field directly while the kernel packs four shifted
  fields into one word for `dp4a` — different arithmetic reaching the same number.
  Mutation-verified: `(rep >> 2)` → `(rep >> 3)` fails 60 of 64 rows on the simd kernel
  while the scalar kernel stays green.
- `f0eb6182` deletes `TestQ2_KArithmeticReference` and switches test and autotune
  compilation from `WithArch` (which emits `compute_XX`, PTX only) to
  `--gpu-architecture=sm_XX` with `WithOutput(OutputCUBIN)`, which is what made these
  launch at all given the dgx driver/toolkit skew
  (`memory/dgx-nvrtc-toolkit-pin.md`).

One defect introduced alongside those gates, reported and not yet resolved:
`f0eb6182` adds `if (cols % 256 != 0) return;` to the generated `q2_k_matvec`
(`codegen.go:424`), placed *before* `y[row] = row_sum`. A ragged width leaves the
output unwritten with no error, so a caller that uploaded a zeroed buffer reads an
all-zero logits vector — the fail-open shape
(`memory/gate-fail-open-pattern.md`). Both subtests use `cols = 512`, so the branch is
unreached; and a Q2_K row length is a multiple of 256 by format, so the guard cannot
fire on valid input and only silences the invalid case. Relevant to this plan because
Stage 2 would generate these kernels for ds4-min's real tensor shapes.

For this plan the consequence is narrower than it was: Stage 2 still cannot accept
a generated kernel on a compile test, but the oracle pattern and two of its
instances now exist on `main`, written from the format rather than from `lm/quant`,
so the branch merge is no longer a precondition for gating a generated quant kernel.
What Stage 2 adds is the remaining precisions and the requirement below.

## Why this increment stops at planning

The proposal's tier 2 is "lower the fused regions to CUDA C++ and JIT them."
The generator cannot express this region:

| Claim in the proposal | State at HEAD | Consequence |
| --- | --- | --- |
| `x/egraph` generates the fused megakernels | `GenerateCUDA` has kernels for exactly `fused_swiglu_gemv`, `fused_rmsnorm_gemv` and `add`, and rejects any precision but `float` (`nvidia/x/egraph/codegen.go:44`) | Every weight in this region is Q8_0, Q8_1, IQ2_XXS or Q2_K. No generatable kernel applies |
| `x/fusion` lowers the BlockOps | There is no `x/fusion`. `x/cudafuse` is the nearest package, and it is elementwise-only: its `rmsnorm` case is a per-element `rsqrtf` of the element's own square, `rope` is the identity, and `matmul` is an elementwise product (`nvidia/x/cudafuse/fuse.go:128`–`:132`) | Those three cases are placeholders, not implementations; routing this region through them would be silently wrong |
| An e-graph already drives a working inference path | `examples/nvidia/qwen-egraph-infer` saturates, extracts, then compiles a hand-written source constant and stores the plan only as `planSummary` (`egraph_opt.go:252`, `:278`) | The existing precedent is decorative. This plan declines to reproduce it without saying so |
| The e-graph's own rules feed the generator | Three of `BatchGEMMRules`' four outputs — `fused_batched_qkv`, `scaled_fp8_gemv`, `cublaslt_gemm` (`nvidia/x/egraph/batch_rules.go`) — are refused by `GenerateCUDA`; only `fused_swiglu_gemv` is emittable — while `FusedMegakernelCost` scores any `fused_*` at 0.2 and `cublaslt_gemm` at 0.5 against `gemv`/`matmul` at 2.0 (`extract.go:148`) | On a DAG where those rules fire, extraction actively *prefers* nodes the generator cannot emit, and the failure surfaces only at `GenerateCUDA`. Independently confirmed by iTerm 3CE9C186 |
| Autotuning selects among generated candidates | `benchmarkPlan` has launch plans for exactly `add`, `fused_rmsnorm_gemv` and `fused_swiglu_gemv` and errors on anything else, which discards the candidate (`nvidia/x/egraph/autotune.go:99`) | A new generated operator is silently dropped by the tuner until a launch plan is added alongside it |
| Apple Silicon portability from the same representation | `apple/x/metal/metalmegakernel`'s own doc says none of its kernels is a semantically complete transformer megakernel (`doc.go:6`) | Cross-backend portability is out of scope here; the flag is CUDA-only |

So tier 1 is the only tier that can land today without either a wrong kernel or
a fabricated claim. Reporting the plan next to the kernels that actually ran is
the honest shape of that increment, and it makes the later increments
falsifiable: once a generated region exists, its plan must be the plan this mode
already prints.

## Later increments (specified, not scheduled)

Each is a separate plan entry with its own gates. Listed so the flag's surface
is designed for them now.

- **Stage 2, pre-flight.** `-egraph` additionally compiles a candidate region
  with `nvrtcx` (already in the ds4-min CUDA path, `cuda_runtime_linux.go:25`
  concatenates the embedded `.cu` sources and the tests compile them at
  `cuda_attention_fused_linux_test.go:45`), inspects the CUBIN with
  `cudaoccupancy.InspectCubin` (`nvidia/x/cudaoccupancy/nvinfo.go:37`), and
  rejects candidates whose register or shared-memory demand drops occupancy or
  spills. Note the proposal names `EIATTR_SHARED_SIZE_SYSMEM`; this package does
  not read that attribute — shared memory comes from the `.nv.shared.<kernel>`
  section size, and registers from `EIATTR_REGCOUNT` in the global `.nv.info`.
  A generated quantized kernel is checkable on the CPU: `x/cudaquant` decodes
  Q4_0 and Q8_0 only, but `lm/quant` has `DequantizeQ2_K` and
  `DequantizeIQ2_XXS` (`lm/quant/kquant.go:49`, `:87`) against a pinned C oracle
  fixture (`lm/quant/testdata/ds4_golden.c`), and ds4-min's CUDA tests already
  import it (`examples/ds4-min/cuda_experts_q81_linux_test.go:11`). So all four
  of the region's formats have a reference; none of them has a generator. Both
  decoders and the fixtures are `codex/ds4-min`-only, per the provenance table:
  so an oracle built on *those* decoders needs the merge. That is no longer the only
  route: `main`'s own `q2k_cuda_test.go` writes `dequantizeQ2K` from the format in
  the test file, which needs no merge, and is the cheaper precedent to follow for
  the precisions that still lack a gate.
  Whatever the reference, **the test has to launch the kernel.** `main`'s
  `TestQ2_KArithmeticReference` is the counterexample: it compares two hand-typed
  Go copies of the same expression in one file, and the measurement above shows it
  stays green when the generated CUDA is mutated back to the defect it was written
  to catch. Stage 2's gate is the `swiglu_cuda_test.go` shape — build the `Expr`,
  `GenerateCUDA`, NVRTC-compile with `WithNoCache`, launch, compare against a
  float64 reference in its own function — and a mutation of the kernel string
  must turn it red before it counts as a gate. And a shape precondition the kernel
  cannot satisfy must fail loudly rather than return early without writing its
  output, per the `cols % 256` guard above.
  An opcode audit via `cudasass.AssertOpcodes`
  (`nvidia/x/cudasass/audit.go:71`) belongs here, with one correction: `require`
  lists for GB10 must not contain `WGMMA`. `ptxas` rejects `wgmma` on sm_121a
  (`memory/sm121-instruction-support-matrix.md`), so requiring it would fail
  every candidate on the host this work targets.
- **Stage 3, graph packaging.** `cudagraph` has both the in-place update the
  proposal wants (`Exec.Update`, `nvidia/x/cudagraph/graph.go:165`) and a
  `BucketManager` keyed on a changing dimension (`:229`). Its doc records that
  no production path in this repository uses it and that it promises no latency
  bound (`nvidia/x/cudagraph/doc.go:7`). The update path itself is now measured:
  `cudaGraphExecUpdate` at 0.23 us (1 node) and 0.33 us (20 nodes) against
  `cudaGraphInstantiate`+destroy at 3.54 us and 13.79 us, about 0.7 us per node,
  1000 iterations each on dgx / GB10 / sm_121a / CUDA 13.0.85 — measured by
  iTerm session 9980FF84, which holds the probe. That bounds the saving at
  3.3-13.5 us per rebuild; whether the ds4 decode loop rebuilds often enough for
  that to matter is still unmeasured here.
  `Exec.Update` is not a correctness check. It validates topology only: a
  measured acceptance matrix from iTerm 3CE9C186 (same host, driver 580.173.02)
  reports `Success` for a grid change with changed parameters, a different
  kernel in the same module, a kernel from a separate JIT-loaded module, an
  int-to-float parameter type change, 2 parameters becoming 3, and 1 KiB of
  added dynamic shared memory; only 1 node becoming 2 fails, as
  `TopologyChanged`. A snapshot-and-verify guard can therefore cover function
  identity, launch geometry and dynamic shared memory but *not* parameter count
  or types: `cudaGraphKernelNodeGetParams` returns a bare `void **kernelParams`
  with no arity field, no type descriptors and no terminator, and no API reports
  a kernel node's parameter count (measured by 3CE9C186). Session 9980FF84 has
  taken that guard and the doc invariant in `x/cudagraph`; this plan does not
  edit the package. For ds4-min the conclusion is unchanged: a bucketed decode
  path needs a logits digest per bucket, not a successful update call.
- **Stage 4, tuning.** `egraph.Tuner.SelectBest` and `BenchmarkCandidate`
  (`nvidia/x/egraph/autotune.go:239`, `:137`) benchmark candidates on the
  device. Any number they produce is subject to the instrument rules already
  paid for on this workload: device span rather than wall clock
  (`memory/device-span-not-wall-clock.md`), median and sign test on sub-millisecond
  kernels (`memory/median-and-sign-test-for-fast-kernels.md`), and `-count=1`
  so the test cache does not replay one draw
  (`memory/go-test-cache-replays-repeats.md`). Hardware stall data is also
  unavailable at the moment: on dgx the PC sampling unit answers
  `CUPTI_ERROR_HARDWARE_BUSY` on every `Start`, reproduced by 3CE9C186 via
  `nvidia/examples/ptx-sass-trace -pcsampling`, and `nvidia-smi --gpu-reset` is
  refused because GPU 0 drives the display. Any tuning keyed on stall reasons is
  blocked until the unit clears.

A prefill-throughput claim is explicitly not a deliverable of any stage above.
The measured prefill gap on this workload is arithmetic intensity — per-token
expert matvec against a matmul — not kernel count or launch overhead
The mechanism is measured on both
backends, not transferred by analogy: 44 to 2.1 command buffers per token bought
6% on Metal and left `gpu_ms/tok` unmoved
(`memory/ds4-prefill-gap-is-batching.md`), and the CUDA side measures the same
per-token expert matvec against matmul at a 30-60x ratio that varies by sample
and does not close with batch width
(`memory/ds4-cuda-prefill-gap-is-matvec-vs-matmul.md`). Fusing the gate/up pair
removes DRAM round-trips on 11% of per-token weight bytes at most
(`memory/ds4-per-token-weight-bytes.md`). Stage 1 claims nothing about speed.

## Files

| File | Change |
| --- | --- |
| `examples/ds4-min/egraph_plan.go` | New, no build tag: the DAG, the rules, extraction, the region assertion, the report. Pure Go so it is testable on darwin |
| `examples/ds4-min/egraph_plan_test.go` | New: the plan is deterministic, the two regions are selected, a renamed operator fails the assertion, the report has the documented prefix |
| `examples/ds4-min/main.go` | Register `-egraph`; add `egraph bool` to `config`; reject it with a non-cuda backend; call the report before `openSessionBackend`; document it in `usage` |
| `examples/ds4-min/main_test.go` | `parseConfig` accepts `-egraph` with cuda and rejects it otherwise |

`examples/go.mod` already requires `github.com/tmc/lib/nvidia`, so no module
change is needed. `nvidia/x/egraph`, `x/cudaoccupancy`, `x/cudafuse` and
`x/cudair` build and test on darwin/arm64 with `GOWORK=off` as of this date, so
Stage 1 is verifiable off the GPU host.

## Acceptance criteria for Stage 1

1. `ds4-min -egraph -backend=cpu` and, on darwin, `ds4-min -egraph` fail in
   `parseConfig` with the flag named in the message, before the model is opened.
2. On the GB10 host, one pinned prompt produces the same
   `-expect-final-logits` digest with and without `-egraph`, and the same
   accepted-token count.
3. The report names both fused regions; renaming any operator in the DAG makes
   the unit test fail rather than the report degrade.
4. `go test ./examples/ds4-min/` passes on darwin/arm64 with no GPU.
5. Criterion 2 counts only runs that executed. `go test` reports `t.Skipf` as
   `ok` with status 0 (`memory/skip-counts-as-pass.md`), so the GPU gate records
   the digest it compared, not that the test returned.

## Follow-ups outside this plan

- `x/cudafuse` should refuse `rmsnorm`, `rope` and `matmul` rather than emit the
  placeholders at `fuse.go:128`-`:132`. Refusal is the shape `egraph`'s codegen
  already chose for precisions it cannot emit, and a node that compiles, runs and
  returns finite wrong numbers is worse than an error. Agreed with 3CE9C186;
  owner unassigned.
- A test in `x/egraph` that saturates, extracts, and asserts the extracted root
  is generatable by `GenerateCUDA`. Today nothing connects the rewrite set to the
  generator's operator set, and the cost model tilts toward the ops it lacks.

## Considered and rejected

- **Replacing the hand-written kernels now.** Rejected: `GenerateCUDA` emits
  float32 only and this region is entirely quantized. The result would be a
  fused kernel that cannot be compiled for the pinned checkpoint's types.
- **Routing the region through `x/cudafuse` to get "something fused".**
  Rejected: its `rmsnorm`, `rope` and `matmul` cases are placeholders that
  compute the wrong values, and the ds4-min oracle gates would catch it only if
  they cover the affected layer — a wrong answer behind a green flag is worse
  than no flag.
- **Printing a plan with no assertion.** Rejected: the report would be
  indistinguishable from a pass in which no rewrite matched.
- **A `-egraph=stage` enum now.** Rejected: one boolean, extended when a later
  stage exists. An enum whose other values are unimplemented advertises
  capability the binary does not have.
- **Making the flag cross-platform with a Metal planning path.** Rejected: the
  Metal megakernel package does not claim complete regions, and the DAG above is
  named after CUDA encode steps. Cross-backend planning needs its own plan.
