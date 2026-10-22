# ds4-min CUDA parity campaign ledger

Every row ties a change to the capture that motivated it and the capture that
judged it. Captures are large binaries and are never committed; they live on
dgx under `~/tmp/richprof/` and are identified here by their cudaprof session
id, which is printed in the capture's status JSON and is unique per run.

Read this top to bottom for the narrative. Each entry states what was measured,
what was predicted, what was changed, and what the re-measurement said --
including when the prediction was wrong.

## The narrative, for a reader who arrives after the campaign

This section is the spine. Everything below it is chronological and much
longer; this is what the campaign was about and how to read the rest.

**What the campaign was asked to do.** Close the performance gap between
ds4-min's CUDA path and the C reference on a DGX Blackwell GB10, without
moving the output. The output constraints are in *Standing gates* below,
and they held throughout: the decode digest `665ac935` and the two-token
continuation `[15342, 201]` never moved.

**What actually happened, in one sentence.** Almost every performance
hypothesis this campaign generated was refuted by measuring it, and the
durable output is not a speedup but a set of instruments that can no
longer produce the errors that generated those hypotheses.

That is not a hedge. It is the finding. Reading the entries in order:

### Levers proposed, measured, and retired

Five plausible optimizations died on contact with a capture, each for a
different reason, and the reasons are the value. Four died to a measurement;
the second died to an estimate, and its row says so:

| lever | why it looked right | why it was wrong |
|---|---|---|
| CUDA graphs | 2904 ms predicted from unrecorded nodes | 33 ms measured; the profiler was hiding the work, not the work being slow |
| batching the 8 `attnOutA` GEMMs into 1 | C does it *in one arm of an `if`/`else`* (`ds4_cuda.cu:18863`, split at `:18911`), and which arm the run we measured against took is unproven; 344 launches -> 43 | **deprioritized by estimate, not refuted by measurement**: at 128 blocks the grid already fills 48 SMs and doubling blocks doubles time linearly, so there is no *wave tail* to recover -- but no batched-against-unbatched draw was ever taken, and that argument says nothing about dispatch or quantization cost |
| C's activation fold | C appeared to have a device op we lacked | the fold is a registry probe whose lookup is a **stub returning 0**; C is 5 device ops unconditionally against our 4 |
| `cublasSetAtomicsMode` | a determinism/perf knob we were not setting | handles already default to `AtomicsNotAllowed`, and it is scoped to symv/hemv |
| the LIB/HAND GEMM swap | two code paths, presumed different | the HAND path still calls `GemmEx` once per row |

The last three share a shape worth naming: **an arm that cannot fire is
not a control.** Two of them were proposed by me and caught by a peer or
a subagent before they burned GPU time; one was caught after.

### What the profiler cost us, and what fixed it

The single largest measurement error was not in an experiment. For a long
stretch the capture showed **12.6% of the work** and we ranked kernels as
if it were all of it, because graph replay records an envelope with zero
nodes. Turning graphs off for one capture moved the census from 12.6% to
whole for *that capture*, and the MoE repack appeared at 721.011 ms,
previously invisible. That is 58.0% of the wall-clock gap as an
accounting ratio -- kernel-time sum over elapsed -- and **not** a claim
that 58.0% of the gap is recoverable; E3 is the standing demonstration
that those differ. Every ranking published before that capture is
superseded.

### The corruption thread, which turned out to be two threads

An artifact-layer sweep produced divergent draws at a rate that rose with
artifact bytes: 1/30, 5/30, 9/30, trend p=0.0077. That series is real and
it is also **merged** -- in two ways, and the second was found late.

Decomposed, `non-finite` is monotone (0, 1, 7) and `wrong-digest` is
1/30, 4/30, 1/30 -- so the "loudest" cell is the *quietest* for the class
the experiment was pointed at. The two classes are disjoint in the data
(all nine non-finite rows across the full 120 carry no digest; all six
wrong-digest rows carry no marker).

The second merge is in the composite's own composition. 1 and 5 are
`wrong-digest + non-finite`; **9 is those two plus one `invalid UTF-8`
abort**, a third category that fires in exactly one cell. That is why the
two-class union at eight layers is 8 and the table beside it says 9 --
the endpoint mismatch corrected in *E8*. A fourth outcome, **graph-capture
abort**, is excluded from every numerator here while its draws stay in all
three denominators; it is a harness failure, not a numerical draw, and it
is not part of this series.

**They are not known to separate on any second axis.** An earlier version
of this paragraph said they did, citing a headroom effect significant for
one class and not the other. That is two p-values side by side, not a test
of a difference; run directly, the contrast has four assignments and
p=0.75. What *is* established, symmetrically and on the pre-registered
blocking, is the non-finite association itself: p=0.0031 blocked, 0.0062
D-only. Whether the two classes have different mechanisms is open, and
these draws cannot close it.

### The method failures, which are the actual deliverable

The campaign's recurring defect is not arithmetic. It is **claims whose
scope is wider than the thing that was checked.** The same shape, over
and over, in different clothes:

- an ablation sited in the cell with the fewest events, whose null printed
  as a statement about the hypothesis;
- a determinism claim that named `workspace` among the inputs it was
  deterministic in -- and workspace availability drifts, so the sentence
  contained its own refutation;
- a permutation test that assumed exchangeable draws over a covariate
  with runs z = -5.13, taking p=0.0002 to p=0.0935 -- a correction this
  ledger long described as "adding blocking" and as pre-registered, and
  which was neither: it changed four things at once (rank to mean,
  centered to raw, arm-specific to pooled, and the pool from modal digest
  to all outcomes), and the ten-draw block was fixed prospectively for
  E10, not for the run it was applied to here;
- a classifier that read column 7 instead of column 8 and silently
  re-merged the two classes, in a section that opens by invoking their
  separation; the tell ignored was *zero non-finite draws in a sweep whose
  entire subject is a non-finite class*;
- a diagnostic that reported the FIRST non-finite index, so a wholly
  poisoned vector and a single bad element at the front printed
  identically -- **an early-returning checker measures presence, never
  extent** -- and those two readings name different bugs, only one of
  which any arm in this campaign could reach;
- and then, fixing that: three sites converted out of **thirteen**, with
  a fourteenth found only because an unrelated darwin build error forced
  a second look. *A census that cannot state its denominator is a sample.*

Each of those is now a gate or an instrument rather than a lesson.
`TestNonFiniteChecksReportExtent` reads the directory rather than the
package -- `os.ReadDir` has no build tags -- states its scanned count,
fails below 100 files, and has been fired against a defect planted in a
darwin-only file while running under `GOOS=linux`.

### What is still open

- **QB**, M=32768 K=1024 — **sized, and its mechanism is now provably not
  separable by choosing a site.** Calling the whole 111.684 ms unexplained
  was reading a big number without dividing by the work: QB is 25.4% of the
  dense row's arithmetic against 30.3% of its time. The part that needs a
  mechanism is measured against `attnOutB`, which does **identical**
  arithmetic (68.719476736 G MACs, same `blocks x K`, both to the digit)
  and differs only in shape: **QB is 1.414x slower, 32.71 ms, 8.87% of the
  dense row.** That 1.414 is a **ratio of site means**, and it is the right
  statistic for the 32.71 ms, because total-time accounting needs the tails.
  It is not the only ratio the same data supports, and the three are not
  interchangeable -- see *A second capture, and three ratios that are not
  the same estimand* below. The earlier 21.7 ms figure was understated by its
  calibration. The two candidate mechanisms -- a fixed per-block epilogue
  and output store volume -- are **the same variable**: the output tile is
  128x128 fp32, so every block stores exactly 65536 bytes and store volume
  is block count times a constant at every shape this kernel accepts.
  No call site can break that tie; only varying `kDqNTile` can, which is a
  build and not a capture. See *E11* below.
- ~~**C's dense shape envelope**, and the A/B has not been run~~ —
  **run and refuted, `f209004a`.** Honouring it costs +3603 ms across the
  eleven sites it moves. More importantly the question was mis-framed here:
  C's envelope selects between *two tensor-core GEMMs*, ours between a GEMM
  and a CUDA-core matvec, so transplanting it is not matching C. See *E11*.
- **Whether C's envelope is right for C** — genuinely open, and unaskable
  in this tree until someone ports the raw-layout `ds4_mmq_q8_0_dense`.
  This is the question the transplant experiment does *not* answer.
- **The envelope at a short prefill.** At `n_tok=2048` the `n_tok >= 512`
  clause never binds, so it is a two-way test on shape. Below 512 it would
  reject **all fifteen** sites. Nobody has drawn from that regime.
- ~~The SwiGLU clamp is unchecked~~ — **closed, `ee60e647`**, and it
  corrected three things I had wrong. See *The clamp gate, and a gate I
  specified stricter than the oracle* below.
- ~~Decode SwiGLU lacks the non-finite guard our D2R kernel has~~ —
  **retired, not fixed.** C has the same asymmetry for the same reason;
  adding the guard would have been the divergence. See *The missing guard
  was correct parity* below.
- Whether the non-finite corruption is memory pressure or a **layout fault
  at the start of the buffer**. The instrument now reports extent rather
  than first index, and that reporting is now held by behavioural fixtures
  over the shapes themselves -- a scanner keyed to the old bug's spelling
  was the only thing guarding it before (`b54d3e92`). Two cautions the
  earlier wording elided: **extent is a symptom, not a cause.** A partial
  extent is *consistent with* a layout fault and does not demonstrate one,
  and nothing in the campaign varies layout, so no arm can currently
  produce the contrast. The draw that would read it has not been taken.
- **The cuBLAS workspace did not move the digest in 64 draws**, 32 per
  condition, ABBA inside one lock window, all on one digest. That bounds
  the effect on this host in this memory state; it does not exclude the
  workspace as a mechanism, and with 32 per arm the floor on a perfect
  zero is what it is. See *E10*.
- ~~**B5:** the non-finite `p=0.0010` is computed unblocked while the
  wrong-digest claim it is compared against is blocked.~~ **Settled.** The
  asymmetry was real; the conclusion survives it. Under symmetric treatment
  the non-finite association holds at p=0.0031 blocked, 0.0062 D-only. What
  does *not* survive is the **class-separation** claim -- "non-finite carries
  the entire trend" compares two p-values rather than testing a difference,
  and the direct contrast has four assignments and p=0.75. Also found: the
  earlier correction changed four things at once and was attributed to
  blocking alone. See *E8*, corrected in place.
- **Whether the two corruption classes have different mechanisms** --
  genuinely open, and this dataset cannot resolve it. Separating them needs
  draws designed for that contrast, not a re-reading of these.
- **B3, same audit: closed.** This ledger's account of C's batching combined
  opposite branches of an if/else (`ds4_cuda.cu:18882` against `:18924`;
  the split is at `:18911`). Corrected in place in *The dense row*. Which arm
  C took in the run we measured against remains unknown, and the capture
  cannot say.

### How to read the rest

Chronological, and it contains its own retractions in place. Where an
entry is superseded it says so and names what superseded it. Three
indexes, with different scopes: *Corrections to earlier entries*; the
*Change log* table, which covers only experiments carrying a capture and
stops at E3; and *The complete commit index*, which is all 69 commits
generated from `git log`, and the only one of the three that is complete.
**24 of those 69 are retractions of this ledger's own claims.** Treat any number published before the graphs-off capture
as covering 12.6% of the work unless the entry says otherwise.

## Standing gates

Any change below is only admissible if all of these hold after it:

- decode output digest `665ac935` (the bit gate; argmax absorbs low-bit float
  differences, so this is necessary and not sufficient -- see the per-kernel
  tests for bit-exactness claims). **It is a per-arm baseline, not an oracle
  anchor.** The package registers four different prefill logit digests across
  four arm tests (`f0e4094b`, `f97c718b`, `a92add81`, `3c190c73`), and
  `cuda_prefill_graph_linux_test.go:100` says so outright: "The digest is a
  deliberate baseline rather than a second invariant." An earlier entry in this
  ledger treated `f0e4094b` as the oracle anchor; that was wrong.
- greedy tokens for 2048 tokens remain `[15342, 201]` -- **this is the only
  oracle-anchored invariant here, and it is two tokens deep.** See the arm
  result below for what that fails to catch.
- `go test -tags cuda -count=1 ./` green in the ds4-min package
- every GPU run on dgx taken under `/home/tmc/ml-explore/bin/gpu-lock`, with
  the lock label recorded in the row

## Measurement conventions

- Comparisons are at matched `--ctx-alloc`. The oracle's prefill number moves
  from 472.60 t/s to 849.35 t/s on the same prompt between `--ctx-alloc 65665`
  and `4096`, so an unmatched comparison measures allocation, not code.
- C-vs-Go draws are interleaved C/Go/Go/C so warm-cache and position effects
  cannot masquerade as an implementation difference.
- `cudaprof gaps` prints total device idle, three attribution arms, and the
  overlay rows -- all ms-valued and similar in magnitude on an idle-dominated
  capture. Only total idle is the coverage denominator. Any per-gap or
  per-percent figure quoted below names its denominator row (errata 21.118;
  two readers re-derived it wrongly in one evening). It also names whether the
  row is an **attribution arm** or an **overlay**: an overlay says where idle
  sat, never what caused it, and reading one as a cause is what killed the
  graph hypothesis (21.140). And before building on any attribution row at all,
  check the instrument's adjacent output for a row that contradicts it: `gaps`
  prints its `Unmatched` count directly above the attribution block that
  silently absorbs it (21.143).

## Baseline: 2026-09-18, commit dc859655, label 4016301

Matched interleaved run, `--ctx-alloc 4096`, 2048-token prompt, 128 generated
tokens. Go digest `665ac935` on every draw.

| | prefill t/s | gen t/s | prefill ms/tok | decode ms/tok |
|---|---|---|---|---|
| C (oracle) | 845.37 | 18.59 | 1.183 | 53.18 |
| Go (ds4-min) | 559.93 | 17.41 | 1.786 | 57.16 |
| ratio | 0.662 | 0.936 | C 1.51x | C 1.07x |

Capture pair, cudaprof `b7f5230d`, agent `b93cd952`, both `outcome=collected`,
`run_completeness=complete`, `dropped_records=0`, `collector_finalized=true`:

| | session id | device active | kernels | graph_trace | copy_bytes |
|---|---|---|---|---|---|
| Go | `feb096df9f4690332f7093cb9e6d6762` | 2947.878 ms | 69454 | 10710 | 6.75 GB |
| C | `adedf19870d777a9a71ff0300888bb51` | 7511.133 ms | 94682 | 10458 | 93.38 GB |

The span-level diff between these two is **not usable**: C's span carries its
14.6 s aligned-artifact repack (one 2429 ms gap, 10906 ms host-behind idle)
that Go's span does not have. One axis survives that contamination, because no
startup phase produces it:

```
hardware queue idle    C 4023.463 ms  ->  Go 7875.243 ms   (+3851.780 ms)
```

That row is a difference of two measured idle sums. It needs no denominator and
no normalisation, and the campaign's ranking rests on it alone.

Supporting, with denominators named. Graph-replay overlay ms over the count of
gaps overlapping a replay, and overlay ms over **total device idle**:

| | overlay ms | gaps | ms/gap | coverage of total idle |
|---|---|---|---|---|
| Go | 7849.728 | 5676 | 1.3830 | 81.63% (of 9616.681) |
| C | 3838.595 | 5544 | 0.6924 | 22.68% (of 16922.210) |

Go's idle is almost entirely inside graph replays; C's almost entirely is not
(C's idle is 64.45% host-behind, which is its repack). The 3.6x coverage
asymmetry is the larger fact; the 1.997x per-gap ratio was standing in for it.
`5676`/`5544` are gap counts, not replay counts.

## Phase split, 2026-09-18, labels 4027530 (Go) and 4029xxx (C)

The 128-token run minus the 1-token run, per implementation, same binary and
flags and lock hold in each pair. `-tokens 0` is rejected by ds4-min, and the
capture of that rejection still reported `run_completeness: complete` with zero
activity records, so the low arm is 1 token (errata 21.119).

**Decode, per token:**

| | wall | device active | hw-queue idle | launches | replay idle x gaps |
|---|---|---|---|---|---|
| Go | 57.41 ms | 17.86 ms | 40.26 ms (70.1%) | 471.4 | 0.884 ms x 44.0 |
| C | 53.33 ms | 20.86 ms | 31.31 ms (58.7%) | 699.8 | 0.690 ms x 43.7 |

**Prefill arm:**

| | hw-queue idle | graph replays |
|---|---|---|
| Go | 2803.299 ms | 2942.9 ms over 88 gaps |
| C | **24.585 ms** | none -- C's prefill contains no graph replays at all |

> **Both cells verified at HEAD (`bb8de0cc`), correlated 100.0%, Unmatched
> zero.** An earlier revision of this ledger retracted them on stale-binary
> evidence; that retraction is withdrawn. But the row does not mean removable
> idle: the 2942.9 ms is the total duration of the `graph_trace` envelopes, and
> because zero of 9711 kernel records carry a `graph_id`, the graphs' execution
> is invisible to the timeline and lands in idle. See below.

## The plan changed here, and this is the important part

The original rank 1 was "batch the stream-13 micro-kernel flood". It was
supported by my own reading of the Go capture, by `cudaprof advise`, and
independently by the notebook, which also ranked it first. Three agreeing
sources -- all looking at the same single capture of one implementation.

The first measurement that could have come out the other way killed it.
**C launches 48% more kernels per decode token than Go (699.8 vs 471.4) and
idles less.** Go's *median* gap is shorter than C's (1.984 vs 2.464 us) while
its mean is 2.77x longer: this was always a tail, never a per-launch tax, and
no count-based remedy could have touched it. Go also does *less* device work
per decode token (17.86 vs 20.86 ms active) and is still slower.

A launch-count census of one implementation cannot rank a gap between two.
Only the control can. Errata 21.120's determinism result made that census cheap
and trustworthy, which is exactly what let three readers be confidently wrong
in the same direction.

## Revised target

**Prefill, not decode** -- this half stands, on the wall-clock ratio alone
(1.51x vs 1.07x). **The idle argument beneath it does not.** Go does carry
2803.299 ms of hardware-queue idle against C's 24.585 ms, verified at HEAD.
But ~2942.9 ms of Go's "idle" is the device working inside graph envelopes
whose node kernels emit no records, so the asymmetry tracks has-graphs, not
has-idle, and the graphs-off A/B showed the time is not removable. See below.

Go's prefill idle is 66.4% graph-replay overlay. Subtracting the one decode
token the low arm contains leaves roughly **2904 ms over ~44 gaps = ~66 ms of
device idle per gap, one per layer**, against a prefill layer budget of 85.6 ms
(3679 ms / 43 layers). About 77% of every prefill layer is a single device-idle
window inside a graph replay.

**What that window is has not been identified. Do not assume.** The "one per
layer" reading is itself provisional -- see the next section: prefill issues
two graph launches per layer, so the same idle may be two ~33 ms windows.
Next measurement: count graph replays directly in the capture, then determine
what the prefill graph contains and why the device drains inside it.

## Structure of the prefill path, read from source (2026-09-18)

Three read-only scouts mapped the paths the measurements point at. Source
reading, no measurement -- every claim here is a file:line, and the ones that
make a performance claim are marked as hypotheses until a control runs.

**Prefill replays exactly two graphs per layer.** There are four capture sites,
all the same shape (capture on first use, instantiate, cache the exec, launch
thereafter): batch QKV `cuda_session_linux.go:1149`, batch FFN `:1475`, decode
QKV `:1835`, decode FFN `:2064`. The batch pair covers all 43 layers, so a
prefill pass issues **86 graph launches**. The prefill arm's capture showed
graph-replay overlay spread across **88 gaps**.

That near-equality is suggestive and is *not yet a finding*: 88 is a count of
idle windows, 86 a count of launches, and this ledger has already recorded one
error from equating the two (21.117). If it holds, the honest restatement of
the revised target is **two ~33 ms windows per prefill layer, one at each graph
boundary**, not one ~66 ms window. The measurement that settles it is a direct
graph-replay count from the capture, not more arithmetic on gap counts.

**The compressor/attention span sits between the two graphs, un-captured.**
`evalLayerBatch` runs the QKV graph, then issues compress (pool / attention /
indexer / indexer_select) and `encodeMixedAttentionParallel` as loose launches,
then enters the FFN graph. Those kernels take `position` and `state.n` as kernel
arguments, which change per call -- the same reason decode layers 0-2 are
excluded from FFN graphing (comment at `cuda_session_linux.go:2051`). So every
prefill layer has a loose-launch span wedged between two replays, which is
exactly where a per-layer drain would sit.

**The 8192-launch `ds4_get_row_f16` run is one prefill loop with no batch
variant.** Kernel `cuda_probe.cu:572`, sole launch site
`cuda_runtime_linux.go:1050` (Grid 16 x Block 256). The loop is
`cuda_session_linux.go:703-710`: 2048 prefill tokens x 4 streams, where the
inner 4 replicate the *same* row into 4 adjacent slots. No `ds4_get_row_f16_batch`
exists anywhere in the registry. On the shipping HC-batch arm all 8192
destinations are one contiguous allocation (`:693-700`, `:844`) and the token
ids are already on the device at `:711`, so one launch is reachable there; the
scalar-HC arm is non-contiguous and needs a fallback.

**The compressor micro-kernels are decode, and the counts reconstruct exactly.**
21 even layers x 2 pools + 20 odd x 1 = 62 per token; 62 x 128 = **7936**,
matching the profile exactly, and 21x32 + 20x1 = **692** matches
`ds4_compressor_attention` exactly. Cross-token batching is impossible
(autoregressive) and cross-layer is impossible (sequential); the batched
siblings already exist and are already used, correctly, in prefill. This is an
independent confirmation of the refutation above, from the other direction.

**Latent, not on the measured workload: `compressGroupEligible`
(`cuda_session_linux.go:2099`) can only pass for the first 2048-token group.**
It requires `indexer.n + n/ratio <= 512`; after the first group `indexer.n` is
512 and every later group fails, falling back to 2048 x 62 = 126,976 single-block
launches per group. The captured workload had exactly one group, so this never
fired. Recorded because it is larger than anything in the decode path and is
invisible to every capture taken so far.

**D2R expert fusion is already ported, already ships on, and is structurally
excluded from decode.** `cuda_moe_d2r.cu` is a near-verbatim copy of the C
source (line numbers offset by a constant +9); both entry points are wrapped
(`:2810`, `:2827`) and registered (`cuda_runtime_linux.go:1346`), and
`cudaShipD2RExperts = true`. The only call sites are in `evalBatchExperts`
(`cuda_session_linux.go:1537`, `:1551`), reachable only through `evalBatch`,
which rejects `len(tokens) < 2`. Decode takes `evalLayerFFNPost` (`:1953`),
which has no D2R arm at all. So the unfused expert kernels in the profile are
not a missing port; they are a guard.

## The two shipping expert arms disagree, 2026-09-18, label 4034661

The gate question came first and it answered yes. ABBA x2, matched flags,
`DS4_CUDA_DISABLE_MOE_D2R` as the only variable (`cuda_session_linux.go:631`;
`on` = shipped D2R arm, `off` = gather arm):

| arm | digest | prefill t/s | generation t/s |
|---|---|---|---|
| on (D2R, shipped) | `665ac935` x4 | 562.79 / 554.61 / 554.03 / 552.60 | 17.43-17.45 |
| off (gather) | `7a44b494` x4 | 309.80 / 306.67 / 305.02 / 302.25 | 17.40-17.45 |

Both arms are perfectly reproducible within themselves and differ from each
other. This is not corruption: both outputs are fluent Italian continuing the
same passage, forking at byte 80 -- a greedy path that diverges a few tokens
in, not a broken kernel.

**What this means for the gates.** The divergence is not new information to the
codebase -- the four per-arm digests already record it, deliberately. What is
new is the consequence: `[15342, 201]`, the one oracle-anchored invariant, is
**two tokens deep**, and both arms pass it while producing different text by
roughly token twenty. So the semantic gate does not discriminate between two
arms that ship in the same binary. Every A/B in this campaign has been read
against a gate that cannot see a divergence this large.

That also retires the open question raised in the previous section: the answer
is no, they do not agree, and the door at `:631` -- documented as the way back
for an operator who hits the open corruption bug -- is not a behavior-
preserving fallback. Whoever takes it gets different logits.

**And a performance result, for free.** The gather arm is **1.82x slower at
prefill** (305 vs 554 t/s) at identical generation speed. The D2R expert fusion
is already worth 1.8x where it runs. Decode is exactly where it does not run
(`evalBatch` rejects `len(tokens) < 2`), which raises the value of the deferred
decode port -- and raises the cost of its digest problem in the same breath.

## The graph hypothesis is dead, 2026-09-18, label 4037664

The prediction was written before the run (errata 21.136): if the ~2904 ms of
prefill hardware-queue idle attributed to graph replay is real and removable,
`DS4_DISABLE_CUDA_GRAPH=1` must take prefill from 3.685 s to ~0.78 s, i.e.
~2623 t/s. ABBA x2, one lock, digest `665ac935` on all eight draws:

| arm | prefill t/s | generation t/s |
|---|---|---|
| graphs on (shipped) | 555.44 / 556.17 / 558.19 / 534.06 | 17.36-17.43 |
| graphs off | 560.68 / 558.47 / 563.97 / 561.00 | 16.39-16.54 |

**Measured: 33.1 ms. Predicted: 2904 ms. Off by 88x.**

The separation is clean -- every off draw beats every on draw, min-off 558.47
above max-on 558.19 -- so the direction is real: the prefill graph machinery is
a small net *cost*, about 0.9% or 33 ms, which is roughly 0.38 ms per graph for
86 captures and instantiations that buy 86 launches. That is a genuine finding
and it is two orders of magnitude away from the one I was chasing.

Decode moves the other way, -5.3% with graphs off, exactly as predicted: decode
replays the same graph objects thousands of times and those graphs earn their
keep. One flag, two directions.

### What I got wrong, and it is the same error a third time

`cudaprof gaps` prints "graph replay" as an **overlay row**, not an attribution
arm. It reports how much idle sat *inside a graph's span*. It never claimed
graphs caused it. With graphs covering most of prefill's kernels, "66.4% of
prefill idle is graph-replay overlay" is close to tautological.

I read an overlay as a cause. That is errata 21.118's lesson -- name the
denominator row -- wearing different clothes, and it is the third time this
campaign that a row of `gaps` output has been read as something it is not.
The measurement conventions section already said to name the denominator. It
now also says to name whether a row is an arm or an overlay.

### What survives

The asymmetry itself, as a reading. It is a direct instrument reading and it
reproduces at HEAD. But a reading of a row whose label overstates it: see the
next section for why 2803.299 ms is mostly graph execution rather than idle.

Also excluded, on the evidence gathered while chasing it:
- **Instrument artifact (the graph-body-collapse theory).** The agent forwards
  `graphId` and `graphNodeId` on every kernel record (`cupti_agent.c:85-89`,
  `:707`) and the advisor reads it (`advise.go:2586`), so graph-node kernels do
  emit activity records. The bodies are not collapsing into one gap each.
  Enabling no `GRAPH_TRACE` kind costs the aggregate record, not the nodes.
- **The MoE repack**, on cadence evidence: right cadence, right magnitude
  (~3.375 GiB/layer, 51 GB/s at 66 ms), wrong column -- these are kernels and
  count as active. The existing `DS4_CUDA_MOE_ARTIFACT_LAYERS=2` arm measured
  +1.8% where the mechanism predicts ~3.6%, and a six-draw rerun near +0.2%.
- **The per-layer blocking `cuMemcpyHtoD`** (`cuda_session_linux.go:1231`, 8 KB
  on the NULL stream against a blocking work stream). Real serialization, but
  it classes Host Behind, not hardware queue.

Next measurement should stop proposing mechanisms and instead ask the
instrument what the gaps actually border: for the largest prefill gaps, the
kernel that ended and the kernel that began. That is an observation, not a
hypothesis, and this campaign has now spent three rounds on hypotheses.

## The 114x is real as a reading, and the retraction of it was wrong

This section replaces an earlier one titled "The 114x was an instrument
artifact." That retraction was itself produced by a broken instrument and is
withdrawn. The sequence is worth keeping in full, because the failure mode is
the one this campaign keeps hitting.

### What happened

Chasing the prefill idle, I decoded the journals by hand and found that
`cudaprof gaps` was folding uncorrelated gaps into the hardware-queue arm, with
CUPTI overhead records manufacturing all of them. I reported it, retracted the
114x, and rewrote this ledger around "C has more hardware-queue idle than Go."

Both halves of that were stale. The overhead exclusion had landed a day before
my run (`86bad446`), and my `cudaprof` was a dirty build from 2026-09-17 --
**the same stale binary that had already cost me a withdrawn bug report.** I
read the source from my own worktree, which is on `codex/ds4-min`, not `main`,
so the line numbers I cited were never HEAD's either. The tool's owner caught
it before the correction did any damage.

Rebuilt from `main` at `bb8de0cc` and re-run on the same two captures:

| | span | active | duty | idle | host behind | hw queue | unattributed |
|---|---|---|---|---|---|---|---|
| Go prefill | 5183.460 | 753.775 | 14.54% | 4429.685 | 1580.528 | **2803.299** | 0.000 |
| C prefill | 17669.687 | 4840.984 | 27.40% | 12828.703 | 10955.391 (85.40%) | **24.585** | 0.000 |

Correlated 100.0% on both. Unmatched zero on both. **The original figures were
right to the millisecond** -- 2803.299 and 24.585, and C's 85.4% host-behind
too. The retraction is withdrawn in full.

### But the 114x still does not mean what it says

At HEAD `graph_trace` is excluded from the device timeline, on the stated
ground that the replay envelope "spans the kernel records of the nodes replayed
inside it" and counting both would double-count. That reasoning is right
whenever those node records exist. **On this capture they do not: zero of 9711
kernel records carry a nonzero `graph_id`.** CUPTI reported the envelope and
nothing inside it.

So the graphs' entire execution is invisible to the timeline and lands in idle:

```
reported active   753.775 ms  ->  duty 14.54%
graph_trace total 2942.915 ms  (reported as a 66.44% idle overlay)
753.775 + 2942.915 = 3696.690 ms  ->  duty 71.32%
```

Go's real prefill duty cycle is ~71%, not ~15%, and ~2942.9 ms of its "idle" is
the device working inside a graph. C shows 24.585 ms because C has no graphs at
all -- has-graphs and has-idle are 1:1, exactly the confound flagged and then
wrongly dismissed.

This is consistent with every other measurement, which is the point. The
graphs-off A/B moved prefill by 33 ms against a 2904 ms prediction: turning
graphs off does not remove work, it makes the same work *visible*, because the
kernels then emit their own records. Wall clock barely moves; the idle number
collapses. Both results were always telling the same story.

### The defect, stated correctly this time

Not "overhead pollutes the timeline" (fixed a day before I looked) and not
"Unmatched is folded into hardware queue" (real, since fixed, and irrelevant
here -- both captures show zero Unmatched at HEAD). The actual defect is
narrower: **excluding `graph_trace` assumes the node kernel records exist to
cover the envelope, and on a capture where CUPTI emits only the envelope, the
graph's whole execution is counted as device idle.** The tool cannot currently
tell "idle inside a graph" from "a graph whose contents were never recorded" --
and the capture already carries the discriminator, since a timeline with zero
graph-tagged kernels is exactly the second case.

### What this costs the campaign

Less than the retraction did. The prefill target is dead either way -- the
graphs-off A/B killed it on wall clock, independent of any attribution. What
comes back is the 114x as an honest instrument reading with a known cause, and
what stays dead is the idea that it points at removable device idle.

**Still standing, still untouched by any of this:** Go prefill ~553 t/s against
the oracle's 849.35 t/s at matched `--ctx-alloc`. Every profiler excursion in
this campaign has so far explained exactly none of it.

### The rule this time

Two rules, and the second is the expensive one.

1. **Rebuild the instrument before reporting a defect in it, and read the
   source from the branch the instrument ships on.** I did neither, twice,
   from the same stale binary. The first cost a withdrawn bug report; the
   second nearly cost a correct finding.
2. **A correction is a measurement and takes the same controls as the thing it
   corrects.** I treated retraction as automatically safe -- as if undoing a
   claim could not itself be wrong. It ran on one binary, with no control, and
   inverted a correct result. The audit that catches a bad finding must also be
   run against the finding that replaces it.

## The fix landed, and it moved the target, 2026-09-19, cudaprof 7645c217

F76E shipped `c40b1884` (count opaque graph replays on the device timeline) and
`7645c217` (advise duty cycle counts only device work), and asked for the check
I would have wanted from them: *two numbers from one capture that do not match
is the check I want on this.* Both halves are now run, on the same two captures,
with the same binary.

### gaps agrees with the prediction, to the millisecond

Re-running `gaps` on `~/tmp/claude-tmp/caps/go_prefill.pprof` -- **with no flags
at all**, see the control below:

```
activities 12614  api records 124834  correlated 100.0%
graph replays are opaque: the capture holds replay envelopes and no device record
naming a graph id, so each replay is counted as activity from its envelope alone.

timeline span      5183.460 ms
device active      3696.690 ms  (71.32% of span)
device idle        1486.770 ms  (28.68% of span)
idle attribution
  host behind        1394.805 ms  (93.81% of idle)
  driver submission    64.577 ms  (4.34% of idle)
  hardware queue       27.387 ms  (1.84% of idle)
  unattributed          0.000 ms  (0.00% of idle)
```

`753.775 + 2942.915 = 3696.690`, duty 71.32% -- exactly the arithmetic written
down before the fix existed.

**This is the entry that closes the 114x.** The share did not just move, the
millisecond figure collapsed:

| | Go, before | Go, after | C |
|---|---|---|---|
| hardware queue | 2803.299 ms | **27.387 ms** | 24.585 ms |
| host behind | 1580.528 ms (35.7%) | 1394.805 ms (**93.81%**) | 10955.391 ms (85.40%) |

Go and C were never different. The 114x was the graph replay envelopes landing
in idle and being attributed to the queue, and it is gone for the right reason
this time rather than the wrong one. Go's prefill idle now has the same shape as
C's: almost entirely host-behind, with a queue that is a rounding error.

### advise does not agree with 71.32, and does not sit at 14.54 either

It reports no duty cycle, and says the data required to compute one is absent:

```
advise item 15, evidence_level "unmeasured":
  "... but this capture carries no activity start/end timestamps, so neither
   device idle time nor host submission was measured"
  recommendation: "Re-capture with activity timestamps and host launch API
   records so duty cycle and submission latency can be measured ..."

-top, same file:
  "coverage: unknown -- the capture records no measured span for the 464.685ms
   of device time to be a fraction of"
```

**The control, because a correction is a measurement.** My earlier `gaps` runs
carried `--max-records=33554432 --journal-bytes=1073741824`, which is the obvious
confound: maybe `advise` just sees a truncated file. Re-ran `gaps` at stock
defaults (1048576 / 536870912) -- byte-identical output. The journal needs no
coaxing. `advise` and `-top` have no journal flags because they never open it.

So it is not two numbers that disagree. It is two readers announcing the absence
of data the third decodes by default, and one of them turning that into a
recommendation to go re-capture work already done correctly -- which on this
project costs a run under GPU lock. Reported as errata 21.151 with three ranked
fix requests, the first being: when a journal is attached, do not recommend a
re-capture.

A second reading falls out of the same numbers. These cannot both describe one
record set:

```
advise / -top    464.685 ms    sum of pprof kernel samples
gaps             753.775 ms    union of kernel+memcpy intervals
                3696.690 ms    union including replay envelopes
```

A union over N intervals is at most the sum over the same N. `753.775 > 464.685`,
so the journal holds activity the sample side does not. Same shape on C: `-top`
3322.414 ms against gaps' active 4840.984 ms.

### Where this leaves the campaign

Every attribution route out of prefill is now closed, and closed cleanly:

- graphs are not the cause (pre-registered prediction, 33.1 ms measured against
  2904 ms predicted);
- hardware-queue idle is not the cause (27.387 vs C's 24.585 -- no difference);
- device idle is not the cause (71.32% duty, and the remaining 28.68% is
  host-behind on both sides).

**What survives is the wall clock and nothing else:** Go prefill ~553 t/s against
the oracle's 849.35 t/s at matched `--ctx-alloc`. The gap is in the work, not in
the waiting. Every subsequent excursion should be aimed at device-active time --
the 464.685 ms of kernel samples, of which three kernels are 91.5%
(`ds4_attention_tokentile_hmma` 42.65%, `ds4_repack_q8_0_aligned` 25.01%,
`ds4_attention_mixed_prepare_batch` 23.80%) -- and not at idle.

## The profile was showing 12.6% of the work, 2026-09-19, label 4082203

Every kernel ranking in this campaign -- mine, `advise`'s, three scouts' -- was
computed over `go_prefill.pprof`'s 464.685 ms of kernel samples. That census is
real and it is not prefill. Decoding the journal by activity subtype:

```
kernel        n= 9711   sum  464.685 ms    12.6% of device-active
memcpy        n= 2734   sum  289.090 ms     7.8%
graph_trace   n=  169   sum 2942.915 ms    79.6%
kernels carrying a nonzero graph_id: 0 of 9711
```

**169 replay envelopes, each naming a distinct graph id, and not one kernel
record inside any of them.** The MoE and FFN path is captured into CUDA graphs,
so it was absent from every ranking. `ds4_repack_q8_0_aligned` reading "25.01% of
GPU time" was 25% of the visible eighth.

### The instrument fix was already paid for

`DS4_DISABLE_CUDA_GRAPH=1` is a near-unbiased window into the same work: the
graphs-on/off ABBA (label 4037664) priced graphs at ~33 ms of prefill wall clock,
0.6%, with prefill marginally *faster* off. Predictions were written into the
script before the run.

| | predicted | measured |
|---|---|---|
| graph_trace records | 169 -> 0 | 169 -> **0** |
| kernel records | materially up | 9711 -> **14168** (+4457) |
| visible kernel time | 464.685 -> 3200..3400 ms | 467.809 -> **3364.901 ms** |
| device active | within a few % | 3845.843 -> **3738.641 ms** (-2.8%) |
| prefill | ~560 t/s | **560.44** off, **559.88** on |

Both captures `complete`, 0 dropped. **27 kernel names, 2640.544 ms, 78.5% of
prefill device work, had never been visible in this campaign.**

As a side effect this validates F76E's `c40b1884` from the outside: promoting an
opaque envelope to busy gives duty 72.24%, and resolving the same work into real
kernel records gives 70.83%. The envelope arithmetic is right to 1.4 points
against an arm that does not need it.

### The real prefill census

Window cut at the load-time Q8_0 artifact build, boundary taken from the data
(last build repack ends 809.618 ms before the first attention kernel), not
assumed. Span 3349.350 ms, 3250.362 ms of kernel time, 13520 launches:

```
ds4_gateup_iq2_swiglu_q8_d2r        43    676.008 ms  20.80%
ds4_repack_iq2_xxs_aligned          86    436.171 ms  13.42%   <-- per group
ds4_dense_q8_d2r                   645    368.674 ms  11.34%
ds4_repack_q2_k_aligned             43    284.840 ms   8.76%   <-- per group
ds4_down_q2k_d2r                    43    259.411 ms   7.98%
ds4_rms                            520    222.425 ms   6.84%
ds4_attention_tokentile_hmma        43    198.455 ms   6.11%
```

Attention, which two scouts were sent to attack as 66% of prefill, is **6.11%**.

### The finding

**The MoE repack runs once per prefill group in Go and once per process in C.**

```
ds4_repack_iq2_xxs_aligned + ds4_repack_q2_k_aligned = 721.011 ms
  = 22.18% of prefill kernel time
  = 58.0% of the entire parity gap
```

Go prefill 3654.3 ms, oracle 2411.3 ms, gap 1243.0 ms. C builds the same
artifacts inside `ds4_engine_open_internal` -> `cuda_build_derived_artifacts`
(`ds4_cuda.cu:4595-4602`, `ds4_repack_build_q2k_aligned` and
`ds4_repack_build_iq2_aligned`) and never again. Go rebuilds them per group
because `cudaMoEArtifactDefaultLayers = 0` (`cuda_session_linux.go:2354`), so
`a.built()` is false at every layer and `repackLayerMoE` runs.

Removing it projects **698.2 t/s** against the oracle's 849.35 -- not parity, but
58% of the gap, and the first item this campaign has found that is both large and
structural rather than a scheduling artifact.

This was predicted from source by scout-repack, from the shipping default and a
traffic calculation, *before* any of it was visible in a capture. The arithmetic
was ~1.37 s against 721 ms measured -- a factor of two high, right target. Its
independent confirmation was the code's own recorded A/B at
`cuda_session_linux.go:2332-2338`: 2 artifact layers worth +1.8%, against 2/43 of
the predicted share = 1.7%.

### The known blocker, and why this is not a patch yet

`cuda_session_linux.go:2311-2316` and `:2340-2348` record that 43 layers of MoE
artifacts is 72.56 GiB and returns non-finite logits, with corruption at 8 layers
in 9 of 30 draws. The gate is memory residency, not the kernel: Go appears to
hold both the artifacts and the raw mapping. `buildMoEArtifact` does call
`releaseMapped` (`cuda_moe_d2r_linux.go:437`), the same as the dense side, so the
release exists; whether it drops the pages is the open question, and
`releasedMapping()` (`cuda_runtime_linux.go:538`) is the instrument that answers
it. **Nothing has been run against that yet.** The dense side is the existence
proof that the pattern works -- `cudaDenseArtifactLayers()` defaults to all 43 and
is worth +7.2% (`:2417-2420`).

### The rule

**A census is only as wide as the records the capture resolves.** Zero of 9711
kernels carried a graph id, and that was on screen in errata 21.145 a full day
before I understood it ranked nothing. I read it as a defect in the profiler's
attribution -- which it was -- and not as a statement that seven eighths of the
workload was missing from every ranking I had built since.

## E3: the fix makes it slower, 2026-09-19, label 4083518

Moving the MoE repack to load time should remove 721.011 ms of per-group kernel
work. `DS4_CUDA_MOE_ARTIFACT_LAYERS` x `DS4_CUDA_RELEASE_MAPPED_ARTIFACTS`,
2x3x2 interleaved, digest `665ac935` on **all twelve cells**, zero corruption,
zero non-finite:

| moe layers | release | prefill t/s (4 draws) | median | generation |
|---|---|---|---|---|
| 0 | off/on | 561.97 559.16 558.20 551.45 | **558.7** | 17.3-17.5 |
| 2 | off/on | 561.75 556.80 551.92 549.33 | **554.4** | see below |
| 8 | off/on | 502.60 512.65 535.22 521.05 | **516.9** | see below |

Monotone downward, and the 8-layer arm separates cleanly from the 0-layer arm
across all eight draws (max 535.22 < min 551.45). **Removing 134 ms of repack
work from a 3654 ms prefill made it 7.5% slower.** Both registered predictions
(+3.7% mine, +6.9% scout-repack's) are refuted, and refuted in sign.

### The second effect is cleaner than the first

Splitting generation by the release gate, which the prefill column hides:

| moe layers | release=0 | release=1 |
|---|---|---|
| 0 | 17.53, 17.27 | 17.39, 17.38 |
| 2 | 17.51, 17.45 | **15.19, 15.11** |
| 8 | 17.58, 17.56 | **12.28, 12.23** |

A double dissociation, both draws agreeing to two decimals in every cell.
Artifact layers cost generation **only** when the mapping is released. Release=0
is flat at 17.5 regardless of layers; release=1 falls 17.4 -> 15.1 -> 12.2. The
candidate mechanism is that `MADV_DONTNEED` drops page-table entries and decode
then re-faults the mapping per token, but that is a hypothesis with no
measurement behind it yet.

### What this does not yet say

`cuda_runtime_linux.go:405-406` records that in this regime the same
configuration drew 173.53 and 496.83 t/s, "2.9x apart and both correct, so the
throughput column is not a rate anywhere in it", and `:412-415` prescribes
repeated alternating draws, a median, and the digest as acceptance -- warning
that "one draw per cell produced a confident finding and its exact reversal
within the hour". This sweep is n=4 per layer count. The prefill separation at 8
layers holds across all eight draws and the generation dissociation is exact in
four pairs, so neither is a single draw -- but neither is 30 either. **Banked as
directional, not as a rate.**

### Corrections this forced, from scout-repack's source audit

- **`releaseMapped` is a no-op by default.** `cudaReleaseMappedArtifacts()`
  (`cuda_runtime_linux.go:432-440`) returns 0 unless the env var is "1" or "2",
  and `releaseMapped` returns immediately at `:516`. Confirmed in the sweep logs:
  `released=0/0.00GiB` in every release=0 cell.
- **The 9/30 corruption record's conditions are not in the source.**
  `cuda_session_linux.go:2340-2348` says "with the flags on" and never defines
  the phrase or names the release gate. The comment at
  `cuda_runtime_linux.go:381-383` -- "no failing run had done" the release --
  predates that sweep (`49a7fe50` 10:53 vs `e861706f` 14:11, and the former is an
  ancestor), so it cannot be a statement about it. Default-based inference only.
- **8 layers with release on was under-tested, not untested.** `:385-397`
  records three clean draws. Against a 30% rate, three clean draws have
  probability 0.34 -- consistent with "release fixes it" and "release does
  nothing" about equally.
- **The +1.8% for 2 layers was one end of a range.** The same paragraph records a
  six-draw rerun putting it "near +0.2%". My measurement-implied +0.92% sits
  inside +0.2%..+1.8%. scout-repack's projection agreeing with 1.8% to 0.1 point
  was agreement with a selected endpoint, not corroboration.
- **The MoE artifact digest is a witness, not a gate.** Off by default
  (`DS4_CUDA_MOE_ARTIFACT_VERIFY`), compares against no expected value, covers
  0.043% of a layer, and runs once rather than twice, so unlike the dense pair
  (`:2496-2498`) it cannot separate built-wrong from overwritten-after. It
  detects non-determinism, not incorrectness: a wrong constant yields a stable
  wrong digest that passes.

### Why it is slower, measured (label 4089442)

One graphs-off capture at `moe=8`, predictions registered first. Both launch
counts hit exactly, so the artifacts **are** being consumed and the framing is
right:

```
                                     moe=0              moe=8
ds4_repack_iq2_xxs_aligned    86x  436.171 ms    70x  382.828 ms   predicted 70  HIT
ds4_repack_q2_k_aligned       43x  284.840 ms    35x  263.907 ms   predicted 35  HIT
```

But the repack fell only 74.3 ms, not the predicted 134. The rest of the census
says why -- **identical launch counts, uniformly more expensive**:

```
ds4_dense_q8_d2r            645x -> 645x   +60.108 ms  (+16.3%)
ds4_quantize_mmq_q8_1       602x -> 602x   +38.559 ms  (+32.7%)
ds4_gateup_iq2_swiglu_q8_d2r 43x ->  43x   +30.920 ms   (+4.6%)
ds4_down_q2k_d2r             43x ->  43x   +17.018 ms   (+6.6%)
```

Per-launch, the surviving repacks slowed too: iq2 5.072 -> 5.469 ms, q2_k 6.624
-> 7.540 ms. Nothing changed path; everything got slower. Net: **-74 ms of
repack removed, +188 ms of uniformly slower everything, +114 ms kernel time,
+221 ms wall.**

That signature -- same launches, more time, across unrelated kernels -- is
memory pressure, not a code change. `availGiB` falls 107.4 -> 99.1 holding
13.50 GiB of artifacts, and on GB10 the device and host share one physical
pool.

### What the campaign actually has

The finding survives and so does its blocker, and they are now separate things:

- **The repack is real work C does not do.** 721.011 ms, 22.18% of prefill
  kernel time, 58.0% of the parity gap. Confirmed by launch count, grid
  dimension, and removal.
- **Precomputing it does not pay under release modes 0 and 1,** because holding
  the artifacts costs more than the repack saves. ~~This is not a tuning
  constant; it is the unified memory pool.~~ **Superseded by E4 below:** under
  release mode 2 it does pay. The cost was the page cache, and modes 0 and 1
  never reclaimed it.
- **So the prize needs residency reduction, not work relocation.** The source
  already says so at `cuda_runtime_linux.go:381-383`: repacking everything is
  "close to memory-neutral -- but only once the mapping is released". Release
  mode 1 (`MADV_DONTNEED`) is now measured and it is the wrong instrument -- it
  leaves the page cache and wrecks decode (17.4 -> 12.2 t/s). **Mode 2, which
  adds `posix_fadvise` to drop the cache, has never been run.** That is the
  open door and it is one env var.

### The rule

**A prediction that is refuted in sign is worth more than one refuted in size.**
Both of ours were sized from bytes and bandwidth and both said "faster". The
measurement says slower, monotone, at n=8. Whatever is happening is not in the
kernel accounting either of us did, which is the only reason it is now worth
chasing instead of shipping.

## E4: the reclaim works, and it was the page cache, 2026-09-19, label 4098372

> **Corrected by E4b below.** The prefill gain here is real but it is measured
> over a window that excludes the load-time cost the change adds. Read E4b
> before quoting the "does pay" conclusion.

The section above says precomputing the MoE repack does not pay on this part.
That is wrong, and this sweep is the correction. It does pay -- but only once
the reclaim is done with the call that actually reclaims.

Four arms, two interleaved draws each, order reversed in round two, digest
`665ac935` on all eight:

| arm | prefill | median | generation | availGiB | released |
|---|---|---|---|---|---|
| moe=0 | 561.42 552.90 | 557.2 | 17.4 | -- | -- |
| moe=8 release=0 | 534.96 537.71 | 536.3 | 17.5 | 99.1 | 0/0.00 GiB |
| moe=8 release=1 | 538.79 534.97 | 536.9 | 12.2 | 99.1 | 24/13.50 GiB |
| moe=8 release=2 | 576.75 567.51 | **572.1** | **3.4** | 98.8 | 24/13.50 GiB |

### Why mode 1 was never a test of the hypothesis

The model is mapped `PROT_READ MAP_SHARED`. `MADV_DONTNEED` on such a mapping
drops *this process's page-table entries*; the pages stay in page cache. Mode 1
did exactly what it claims -- `released=24/13.50GiB`, the gate is plumbed and
fired -- and freed no physical memory, which is why prefill did not move
(536.9 against 536.3). Mode 2 adds the `posix_fadvise(DONTNEED)` that reclaims,
and prefill jumps 35.8 t/s.

So "release does not help" was never a result. It was mode 1 being the wrong
instrument, and `releasedMapping()` is what let us tell that apart from an
unplumbed gate -- the exact use its comment at `cuda_runtime_linux.go:538-540`
was written for.

### availGiB is blind to the pressure

Prediction P2 said `availGiB` would recover toward 107 under mode 2. It did
not: 98.8 against 99.1, no movement at all, while prefill moved 6.7%. The
number I had been using as the residency readout does not see the page cache,
so it could not have shown this either way. **Every residency claim in this
ledger that cites availGiB is citing an instrument blind to the thing that
turned out to matter.** The E3 entry above leans on `availGiB 107.4 -> 99.1` to
argue pressure; the direction of that argument survives, its evidence does not.

### The double dissociation

One knob, three settings, prefill and generation moving in opposite directions:

```
release=0   prefill 536.3   generation 17.5     nothing released
release=1   prefill 536.9   generation 12.2     PTEs dropped, pages kept
release=2   prefill 572.1   generation  3.4     pages reclaimed
```

Prefill wants the raw bytes gone. Decode wants them resident, because decode
still reads experts through the raw mapping and now re-faults them from storage
every token. That is not a tuning conflict, it is two consumers of one mapping
with opposite needs, and it names the next piece of work precisely: **make
decode read the artifacts instead of the raw mapping.** That item is listed
under Deferred below as "D2R expert fusion at decode" with the note that it
would move the digest. It is no longer deferred for lack of a reason.

### What is claimed and what is not

- Claimed: at 8 of 43 layers, mode 2 gives prefill 572.1 against a 557.2
  baseline, +2.7%, on two interleaved draws with a clean digest.
- Not claimed: that this scales to 43 layers. The release=0 sweep fell
  monotonically in layer count; whether mode 2 inverts that slope is untested
  and is the next experiment, not an extrapolation.
- Not shippable: generation at 3.4 t/s is a 5x regression. This is a result
  about where the cost lives, not a configuration to turn on.

## E4b: nothing was removed, the window moved, 2026-09-19

F76E shipped the launch-count-split diff I asked for (64bb3a8b) and I ran it on
my own pair. It reported **no kernel changed launch count at all**, where I had
counted the repacks falling 86 to 70 and 43 to 35. One of us was wrong about the
central fact of E3 and E4. Neither was:

| | whole capture | prefill window |
|---|---|---|
| moe=0 iq2 | 86 / 436.171 ms | 86 / 436.171 ms |
| moe=8 iq2 | **86** / 492.415 ms | 70 / 382.828 ms |
| moe=0 q2_k | 43 / 284.840 ms | 43 / 284.840 ms |
| moe=8 q2_k | **43** / 318.530 ms | 35 / 263.907 ms |

Over the run the counts are identical. The artifact build **relocates** 16 iq2
and 8 q2_k launches out of the prefill window into load time. It removes
nothing, and the run's total repack time *rises* 89.934 ms doing it.

### What this does to the E4 claim

E4 is titled "the MoE repack does pay". On the metric I used, it does. That
metric is the problem: `prefill t/s` is measured over a window that begins
after the load, so work moved into load leaves the numerator and the
denominator both. The arithmetic end to end:

```
prefill 2048 tok @ 557.2 t/s (moe=0)      = 3.676 s
prefill 2048 tok @ 572.1 t/s (moe=8 rel=2) = 3.580 s
  saved per prefill                         0.096 s
  artifact build, paid once at load         1.96  s
  break-even                                ~20 prefills
```

**For a single-shot run the change is a net loss and my ledger said it was a
win.** For a server that loads once and prefills many times it pays after about
twenty requests. That is a real and defensible claim, and it is not the claim I
made.

### The rule, for the third time in this campaign

A census is only as wide as the records the capture resolves -- and a *rate* is
only as honest as the window it is divided by. Graph replay hid seven eighths
of the kernels (E1). The release-mode readout could not see the page cache
(E4). Here the window itself excluded the cost, so relocating work into it read
as removing work. Same error, three instruments.

**Before believing an improvement in a rate, ask what the rate's window
excludes, and whether the change moved anything across that boundary.** The
launch-count-split diff caught this in one run because it counts over the whole
capture and cannot be fooled by a window I chose. That is the value of an
instrument somebody else built.

### What survives

- The repack is still 721.011 ms of prefill-window work that C does not do
  (`ds4_cuda.cu:4595-4602` builds its artifacts once at load). That census is
  unaffected: it was a single-arm measurement, not a diff.
- Precomputing it is still the right idea, and mode 2 is still the reclaim that
  makes it viable rather than harmful.
- The win is a **load-time amortization**, not a speedup -- and E5 below
  withdraws even that as unproven, and puts a corruption ceiling between 8 and
  20 layers on how far the idea can go at all.

## E5: the corruption ceiling, and the effect was inside the noise, 2026-09-19

A layer sweep at release=2, two draws each, plus a release=0 control at 20 layers.

| layers | release | draw a | draw b |
|---|---|---|---|
| 0 | 2 | 554.20 OK | 549.11 OK |
| 8 | 2 | 576.13 OK | **422.83 OK** |
| 20 | 2 | 611.51 **MISMATCH 08cf0217** | 584.88 OK |
| 43 | 2 | 679.09 **MISMATCH f76777ee** | 440.41 **MISMATCH bdf304ff** |
| 20 | 0 | **rc=1** non-finite logit 0 | **rc=1** non-finite logit 0 |

### The ceiling is real and it is not stochastic in the way the source says

`cuda_session_linux.go:2340-2348` records corruption at 1/30, 5/30, 9/30 and
never states the conditions. This sweep localizes it: clean at 0 and 8 layers,
intermittent at 20, never clean at 43. Three distinct wrong digests appeared,
so when it corrupts the *value* is non-deterministic even though the
*occurrence* tracks layer count.

At release=0 and 20 layers it does not corrupt quietly -- it fails loudly,
both draws, `cuda runtime: non-finite logit 0`. That is the better failure and
it is worth knowing the release mode changes which one you get.

**So the artifact path has a hard ceiling somewhere between 8 and 20 layers**,
and every number above it is inadmissible. Note the shape: prefill *rises*
monotonically into the corrupt region, 554 to 576 to 611 to 679. The fastest
number this campaign has produced, 679.09 t/s at 43 layers, is a broken run.
A faster consumer that computes the wrong thing is the oldest trap there is,
and here it is dressed as a scaling curve.

### And the effect I claimed does not survive its own noise

The two 8-layer mode-2 draws here are **576.13 and 422.83 t/s** -- same config,
same binary, both digests clean, 36% apart. E4 claimed +2.7% from n=2 that
happened to land twice on the high mode.

`cuda_runtime_linux.go:405-406` says in as many words that the throughput
column is not a rate in this regime, and records one config drawing 173.53 and
496.83. It was in this ledger's own measurement conventions. I read it, wrote
"interleaved ABBA draws are mandatory" at the top of this file, and then
published a 2.7% effect on two draws per arm.

**E4's prefill win is withdrawn as unproven.** Not refuted -- unmeasured. A
paired n=8 alternating sweep at the two digest-clean arms is running as E6 to
settle it. What E4 established that stands: mode 1 releases page-table entries
and not pages, mode 2 reclaims, `availGiB` cannot see either, and the
generation collapse is real and large enough to read through any noise
(17.4 to 3.4 is not a 2.7% effect).

### The rule

**Two draws cannot measure a 2.7% effect in a distribution with a 36% spread,
and the tell is not in the numbers you got -- it is in whether you ever drew
the distribution.** Both E4 draws were high. Nothing in them said "bimodal".
The only thing that would have said it is more draws, which is why the
convention is written down in advance rather than inferred from the result.

## Corrections to earlier entries in this ledger

- The decode attention chain is **three** kernels, not four.
  `ds4_attention_mixed_prepare_batch` (0.11 s) is the *prefill* path -- with
  HMMA on by default and n>=4, prefill runs prepare_batch + tokentile_* +
  commit_raw. The decode chain is prepare + scores_tile + finalize = 0.34 s,
  not 0.45 s. Two chains sharing a name prefix; I summed across them (21.122).
- Both decode attention boundaries are **real RAW barriers**, so collapsing the
  chain is not free. `out` is one allocation partitioned by range, which makes a
  buffer-granular view see a false prepare<->finalize conflict that does not
  exist.
- A two-kernel attention path already exists and is **structurally
  unreachable**: `ds4_attention_mixed_fused` (cuda_compressed.cu:679) sits
  behind the same predicate as the arm that shadows it, so it can only run if a
  module lookup fails (21.123).
- Neither attention grid size consults any device property, so advise's
  "36 of 48 SMs idle" names a cause that does not exist in the source (21.124).

## E6: the withdrawn claim was true, and my retraction over-corrected

Paired, strictly alternating, n=8 per arm, at the two digest-clean
configurations. Every one of the sixteen draws gated clean at `665ac935`.

| arm | n | median | min | max | spread | bad digests |
|---|---|---|---|---|---|---|
| moe=0 | 8 | 553.12 | 549.52 | 559.76 | **1.9%** | 0 |
| moe=8 release=2 | 8 | **570.44** | 203.82 | 581.25 | 185.2% | 0 |

```
per-pair delta: -355.9  +27.6  +13.3  +22.9  +22.1  +12.3  +19.4  +18.4
median delta:   +17.32 t/s  (+3.13%)
sign test:      8 layers faster in 7 of 8 paired draws
```

### All three predictions refuted

- P1, both arms heavy-tailed: **refuted.** The control is tight and unimodal,
  1.9% spread across eight draws. Only the artifact arm has the outlier.
- P2, medians differ by less than the 2.7% claimed: **refuted.** They differ by
  3.13%, more.
- P3, the sign test does not reach significance: **refuted.** 7 of 8, one
  direction, consistent magnitude.

### What that means about E5's withdrawal

E4 claimed +2.7% from n=2. E5 drew the same config at 422.83 and I withdrew the
claim as unproven. The withdrawal was right as procedure and wrong in its
reasoning: I read one low draw as evidence that **both** arms were bimodal and
that the effect was noise. The control arm was never bimodal. What I actually
had was a tight control and a treatment arm with a rare collapse, which is a
different distribution and supports a different conclusion.

**The claim is restored, at +3.13% with n=8 and a 7-of-8 sign test.** It took
the same measurement to restore it that would have established it in the first
place, which is the whole argument for taking the measurement rather than
arguing from two draws in either direction.

[[correction-is-a-measurement]] cuts both ways and I only applied one edge of
it. A retraction is a claim about a distribution, and mine was made from the
same n=2-plus-one-outlier that could not support the original.

### The collapse mode is real and is not noise

One draw in eight at 203.82 against a 570.44 median is a 64% collapse with a
clean digest. It appears only in the artifact arm. It is consistent with a
page-cache re-fault storm -- mode 2 has reclaimed 13.50 GiB that prefill may
still touch, and the re-fault is transparent but not free
(`cuda_runtime_linux.go:425-431` promises correctness, never speed). **Not
measured. Needs its own experiment**, and until it is understood this
configuration is not shippable regardless of its median.

### What still stands from E4b

The +3.13% is a **prefill-window** gain and the window excludes the ~1.96 s
artifact build. End to end that is still amortization, not speedup: about 0.1 s
saved per prefill against a one-time 1.96 s, breaking even near twenty
prefills. E6 raises the per-prefill gain; it does not move the boundary that
E4b identified.

## E7: release is not the fault, and my gate was six bytes wide

Run under label 18295 to separate "the artifact build is wrong at high layer
counts" from "the build is right and decode misreads it". The discriminator: the
first generated token is the argmax of prefill's final logits, so a `-tokens 1`
run exercises prefill only.

| layers | release | tokens | digest | prefill t/s | released |
|---|---|---|---|---|---|
| 0 | 0 | 1 | `5ab98410` | 552.68 / 547.85 | -- |
| 43 | 2 | 1 | `5ab98410` | 648.85 / 415.33 | 129/72.56 GiB |
| 43 | 0 | 1 | `5ab98410` | **137.07** | 0/0.00 GiB |
| 8 | 2 | 1 | `5ab98410` | 419.76 | 24/13.50 GiB |
| 43 | 0 | 128 | **`82d9a8d8`** | 141.61 | 0/0.00 GiB |
| 43 | 2 | 128 | **`51164e28`** | 602.50 | 129/72.56 GiB |

### What this establishes: release is not the fault

**Both release modes corrupt at 43 layers, with different wrong digests.** The
release=0 row is the one that matters: `released=0/0.00GiB`, nothing dropped,
the mapping fully intact, and decode is still wrong. E5 has the other half --
20 layers at release=0 hard-failed with `cuda runtime: non-finite logit 0` on
both draws, also with nothing released.

That retires the campaign's leading theory. "`releaseMapped` drops pages decode
still reads raw" cannot explain a failure in a run where no page was dropped.
scout-attn-parity proposed it, I flagged it as suspicious on the grounds that
the same predicate holds at 8 layers where the digest is clean, and they
accepted the refutation when the measurement landed.

Release does buy a great deal of speed: 602.50 against 141.61 t/s at 43 layers,
and 648.85 against 137.07 on the prefill-only rows. A 4.3-4.7x prefill
difference from reclaim alone. It is the largest single number in this campaign.

### What this does NOT establish, and I claimed that it did

I wrote that the 43-layer prefill output was "byte-identical" to the reference
and concluded the artifact build is numerically correct. **The output file is
six bytes.** It is ` alla\n` -- one detokenized token, the argmax of prefill's
final logits over a ~129k vocabulary.

Argmax absorbs low-bit float differences. A numerically wrong prefill whose
top-1 ranking survives passes that comparison exactly as a correct one does. So
E7 established that the **argmax is stable** at 43 layers, not that the build is
right, and "byte-identical" did rhetorical work that six bytes cannot support.

F76EF7EB caught it by asking what the digest was taken **over** -- the same
question that produced E4's residency error and E4b's window error. Third
instance. This time the scope I had not checked was my own gate's strength.

**The instrument existed the whole time.** `ds4-min -h`:

```
-expect-prefill-logits string
      require this SHA-256 of little-endian float32 prefill logits
-expect-final-logits string
      require this final logits SHA-256 and exactly -tokens accepted tokens
```

A bit gate over every logit. I built a six-byte token comparison beside it.
[[wait-loop-matched-itself]] failed to ask what a pattern matched; this failed
to ask what the binary already offered.

### Two further corrections, from the scouts

**E7's release-safety half was a control that could not fire.** At 43 layers
`cuda_session_linux.go:1520-1523` binds the artifacts, so
`l.expertGate/Up/Down` are never passed to a kernel and a prefill-only run
reads **zero** of the 72.56 GiB that release dropped. The same digest would
appear if `releaseMapped` were `rm -rf`. The conclusion survives only because
the tokens=128 rows above replaced the evidence for it.

**An argmax gate has strong positives and weak negatives.** A mismatch or a
non-finite logit is a real difference argmax cannot manufacture; an OK means
only that top-1 survived. So every *clean* row in every table here is weak
evidence -- including the 8-layer cleanliness I used to argue against the
release predicate.

### The corruption is a rate, and it predates expert artifacts

The comment on `cudaMoEArtifactDefaultLayers`
(`cuda_session_linux.go:2340-2348`) already records, at **thirty draws per
arm**:

> corruption runs 1/30 at zero layers, 5/30 at two, and 9/30 at eight --
> monotone in bytes held, trend p=0.0077

Two things follow, and both invalidate how this ledger has been reading its own
tables. The corruption is **stochastic**, so every single-draw comparison
between layer counts -- E5's rows, and several sentences written here as facts
about a layer count -- is one Bernoulli draw from an arm with p near 0.3. And
the rate is monotone in bytes held, which was measured before this campaign
began.

**Correction, from scout-attn-parity: the 1/30 arm is not a zero-byte
control.** I first read "1/30 at zero layers" as showing the corruption
predates the artifacts. It does not. A zero-MoE-layer run still builds dense
artifacts -- `cuda_session_linux.go:276` calls
`buildDenseArtifacts(cudaDenseArtifactLayers())`, holding 5.62 GiB. So 1/30 is
a 5.62-GiB baseline, another point on the same monotone curve, and I read it as
an exception to the curve. That inverts the inference I drew from it.

**The series has therefore never been anchored at zero.**
`DS4_CUDA_DENSE_ARTIFACT_LAYERS=0` with no ballast and no MoE layers is a
genuine zero-byte cell and nobody has run it. If it corrupts at all, bytes are
not the mechanism and this entire framing is wrong. It is the cheapest
falsifier on the board and it remains open -- see the E8 entry in the capture
index, banked and not run.

F76EF7EB read the single draws as non-monotonic in layer count (20 at release=0
fails harder than 43) and proposed bisecting by layer index to find a
quant-type-specific bug. The shape is n=1 per cell and the 30-draw data says
monotone; the bisect would have answered confidently either way. That is the E5
mistake with a new independent variable, and it is the third time in this
campaign that a 30-draw answer already existed in the tree while someone argued
from two draws.

### Prefill is the artifact's only consumer

Worth recording because the inverted version fit every result we had.
F76EF7EB's sharper worry was that prefill might not read the artifact at all,
which would make a prefill logits gate blind to a build error. It resolves the
other way: `cuda_session_linux.go:1520-1528` is the **sole** artifact-
substituting branch in the tree and it is the batch/prefill path. Decode reads
raw unconditionally (`:1972`, `:1984`, and the gather variants at `:1562`,
`:1573`). So prefill is exactly where the artifact is read, and
`-expect-prefill-logits` does look at the thing under test.

### The ledger conflates two different failures

Also from scout-attn-parity, reading `logits_check.go:32-33`: a non-finite
value is caught **before** the digest is formed, and reported as `non-finite
value at N`. So the instrument classifies three ways -- clean, wrong digest,
non-finite -- and this ledger has been folding the last two into one bucket
called corruption. E5's `non-finite logit 0` and E7's `82d9a8d8` are not the
same event, and every rate quoted here mixes them.

### What is still open

Whether the build is numerically right. E7 cannot say. E7b re-runs the
discriminator on `-expect-prefill-logits`, behind three validations, since a
gate that has only ever passed is not known to be a gate: V1 asserts an all-zero
SHA and requires refusal; V2 runs a half-length prompt and requires the SHA to
**move**, which is the only one of the three that can catch a SHA computed over
an empty or pre-copy buffer; V3 establishes that the 0-layer reference is a real
forward pass. Registered prediction: the 43-layer SHAs differ. Standing caveat,
in F76EF7EB's words -- if they differ and the gate has not been shown able to
fail, the first candidate is the gate.

## The expert kernels are not the gap: C swaps where we stack

Source reading by two subagents, cross-checked against each other, every claim
cited. **Nothing in this section is measured by me** -- it is what the two trees
say, and it is labelled that way deliberately, because this campaign has already
lost a lead to a derived number quoted as a measured one.

### Our largest kernel is C's kernel

`ds4_gateup_iq2_swiglu_q8_d2r` is 676.008 ms, 20.80% of prefill kernel time and
the biggest named row in the census. It is not an approximation of what C does.
It is the same kernel:

- C: `cuda/mmq/ds4_mmq_d2r.cu:2524` defines `gateup_iq2_swiglu_q8_d2r_kernel`,
  launched at `:3019` as `<32, kFusedNTile>`. Ours instantiates `<32, 32>` at
  `cuda_moe_d2r.cu:2822`.
- Constants match: C `kMTile = 128` (`:43`), `kNTile = 64` (`:44`),
  `kFusedNTile = 32` (`:62`) against our `cuda_moe_d2r.cu:81,84,100`.
- The SwiGLU epilogue is line-for-line identical (C `:2665-2666`).
- Neither materializes a dequantized matrix; both dequantize inline through the
  256-entry `iq2xxs_grid` LUT.

C's prefill genuinely reaches it and cannot fall back: `ds4_mmq.cu:2121` sets
`direct_gateup_q8 = true`, taking the branch at `:1365` under an NVTX scope
named `ds4/prefill/moe/iq2_gate_up_swiglu_q8_d2r` (`:1611`); every guard
(`:1379-1383`) returns an error (-94/-95/-96) rather than degrading.

`ds4_down_q2k_d2r` is the same port at `TileN=64` where gateup is bounded to 32
by a 48 KiB shared-memory `static_assert` (`cuda_moe_d2r.cu:668`, `:2555`). That
predicts a traffic ratio of 3.15 against the measured time ratio
676.008/259.411 = 2.61 -- consistent, and identical in C.

**A lever was retracted here.** An earlier report offered gateup's ~2x expert
weight re-read as a Go-side parity lever. It is not one: C re-reads exactly as
many times at the same `TileN=32` and still runs 849 t/s. It stays on the list
as an absolute optimisation and is struck from the gap analysis.

### The asymmetry is residency policy, not kernel code

C builds every aligned artifact once at load, from the file, for every candidate
tensor, with no budget cap (`ds4_cuda.cu:4530-4672`, allocations `:4617-4628`).
Then it **releases the other side**: when artifacts cover a span it never
registers the model mmap at all, short-circuiting `cudaHostRegister` with
"CUDA aligned artifacts replace expert residency" (`:3856-3860`, `:4008-4012`),
and `ds4_gpu_cache_model_range` returns early for any span
`cuda_span_fully_replaced` covers (`:4709`, predicate `:571-599`), gated on
`g_derived_replaces_complete` (`:4630-4648`).

**C swaps. Our E3 stacked.** That is a better explanation of E3's inverted sign
than the page-cache framing in E4: at 8 of 43 layers you pay residency for 8
while the other 35 still stream raw expert weights through the mapping, so
nothing is released and everything is contended. 8/43 is not a point on a
monotone curve -- it is the trough.

It also sharpens what E4 found. `DS4_CUDA_RELEASE_MAPPED_ARTIFACTS=2` *is* the
swap, and it does reclaim. So "Go stacks" is true of the shipping default and
not of what E4 and E5 ran. **The swap is available, and at 43 layers it still
corrupts** (E5). Which relocates the question: C has a completeness predicate
before it releases, and we release per-layer regardless. That predicate, not the
residency, may be the finding.

### The 15 in 645

`ds4_dense_q8_d2r` runs 645 = 43 x 15 launches. The 15 is seven distinct weight
matrices, one launched eight times: `attnQA`, `attnKV`, `attnQB`, `gate`, `up`,
`attnOutA` x8 block-diagonal (`cuda_session_linux.go:1647`), `attnOutB`, `down`.
One chunk per layer, because `cudaPrefillQ81Chunk = cudaPrefillGroupWidth`
(`:177`, `:181`).

Eleven of the fifteen grids are under 550 blocks on 48 SMs; `attnKV` is 68.
**Needs measurement** before it is worth anything: a per-call-site split of the
645 launches' 368.674 ms. Fusing the eight `attnOutA` launches is worth doing
only if that row is a meaningful slice of the 368.674, and nobody has measured
which of the fifteen sites carries the time.

## The dense row, attributed, and a lead that died on arrival

`ds4_dense_q8_d2r` is 368.674 ms over 645 launches (43 x 15). Two subagents
independently flagged that C issues the eight block-diagonal `attnOutA` blocks
as ONE batched GEMM (`ds4_cuda.cu:18901`, `cublasGemmStridedBatchedEx` with
`batchCount = n_groups`) and quantizes ONCE (`:18919`, `x_rows = n_tokens *
n_groups`), where we issue 8 GEMMs plus 8 quantizes per layer. That is a real
structural difference and it looked like the next experiment.

> **Correction (B3, audit session C8332C52; reference lines re-read by me at
> unchanged SHA `c420f92f…0fc0b5`).** Those two citations are the two arms of
> one `if`/`else`, not two stages of one path. `if (out_a_f16)` opens at
> `:18863` and runs to `:18910`; `} else {` is `:18911`. The batched GEMM is
> the *then* arm and the single wide quantize is the *else* arm, so **C never
> does both.** `out_a_f16` is non-NULL only when quality mode is off, cuBLAS
> is ready, `n_tokens >= 2` (tunable by
> `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN`), and
> `DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A` is unset. Two further limits the
> original framing elided: **which arm the captured C run actually took is
> unproven** -- nothing in the capture records `out_a_f16`, and the else arm
> has its own grouped-MMA sub-branch at `n_tokens >= 8` -- and an API call
> count is not a device launch count, so "one batched GEMM" is one cuBLAS
> call, not necessarily one kernel. The structural difference is still real;
> it is one difference with two possible shapes, and we do not know which one
> we measured against.

It is measured now, from the capture already on disk -- no GPU needed. Grid
dimensions separate all fifteen sites (`gridX = numM * 16`), and within the
`gridX=128` group launch order separates QA from the eight `attnOutA`:

| call site | launches | ms | % of row |
|---|---|---|---|
| QB (M=32768, K=1024) | 43 | 111.684 | 30.3% |
| attnOutB (M=4096, K=8192) | 43 | 78.975 | 21.4% |
| **attnOutA (8 per layer)** | **344** | **90.008** | **24.4%** |
| gate + up (M=2048, K=4096) | 86 | 44.260 | 12.0% |
| shared `down` (M=4096, K=2048) | 43 | 24.295 | 6.6% |
| QA (M=1024, K=4096) | 43 | 11.713 | 3.2% |
| KV (M=512, K=4096) | 43 | 7.738 | 2.1% |

The split validates itself: `attnOutA` and QA have identical shape and come out
at 0.262 and 0.272 ms per launch.

> **Updated.** `attnOutB` and `down` were one merged `86 | 103.270 | 28.0%`
> row here until the retained journal was decoded; this ledger had said that
> separating them needed a capture it already had. See the correction under
> *The dense table merges a failing class with a passing one*.

### Why batching them would not pay

The case for fusing 344 launches into 43 is wave quantization -- a 128-block
launch on 48 SMs wastes a partial wave, eight times per layer. The same capture
tests that directly, because `gate`/`up` is the same kernel at exactly twice the
grid:

```
attnOutA   128 blocks   0.262 ms/launch
gate + up  256 blocks   0.515 ms/launch      2.00x blocks -> 1.97x time
```

**Linear.** At 128 blocks the grid already fills the device, so there is no tail
for fusion to recover. Fusing would save 301 launches of dispatch overhead,
which at 0.262 ms each is not where the 90 ms lives.

The one site that genuinely underfills is KV at 64 blocks, and it is 7.738 ms --
2.1% of the row. Not worth an experiment.

**So the launch-count difference from C is real and is not a performance
finding.** C batches because its cuBLAS path wants a batch, not because eight
launches cost it a wave. This is the second lever this campaign has retired by
measuring it instead of reasoning about it.

### Marked, not claimed

The per-unit-work comparison above holds K constant between the two rows being
compared. Across rows with different K (QB at K=1024 looks *less* efficient per
unit of `gridX * K` than the small launches) arithmetic intensity differs too,
so that comparison is suggestive and is **not** offered as a finding. QB is the
largest single site at 111.684 ms and nobody has explained it. **Needs
measurement.**

C's shape envelope is also still open and is the better question: C routes a
dense Q8_0 GEMM here only when `n_tok >= 512 && out_dim >= 2048 && in_dim <=
4096` (`ds4_cuda.cu:14862-14866`), which 11 of our 15 sites fail, and our gate
is a plain on/off with no envelope (`cuda_session_linux.go:654-662`). Whether
those 11 shapes belong on this kernel at all is an A/B nobody has run.

## The aligned decode family: two entry points and one branch

The live parity lead, specified from source by scout-attn-parity and verified
independently here against both trees. No measurement in this section.

### Why this lever is different from every other one tried

Every artifact lever in this campaign spends against the working-set budget --
20-24 GiB dormant, near 12 GiB touched (`cuda_session_linux.go:2304-2309`).
Routing decode onto the artifacts is the only one that spends the other way,
because **the raw expert stack decode currently scatter-reads is byte-neutral
with the repack**: 256 experts x 2048 x 16 x 66 = 553,648,128, exactly
`cudaIQ2XXSRepackedBytes` (`cuda_moe_d2r_linux.go:19`). It does not add a
working set, it *moves* one -- off the mapping the budget is measured against.

The reference author placed the branch for this reason and wrote it down.
`ds4_cuda.cu:24450-24453`, verbatim:

> The aligned artifacts replace the raw expert tensors on integrated CUDA
> systems. Route both prefill and decode before resolving a raw pointer,
> otherwise the fallback cache would duplicate tens of GiB.

### Our artifacts are already valid inputs

This is the load-bearing claim and it is what makes this small. C's documented
layouts (`ds4_mmq.h:503-520`, `:535-549`) are the same three-plane split with
the same 64-byte rounding as our offsets (`cuda_moe_d2r_linux.go:160-179`,
`:181-204`), and the kernels that fill ours are C's own, carried with the
header comment "Repack kernels from ds4_repack.cu" at `cuda_moe_d2r.cu:2740`.

Derived from the cited constants, at our shapes (256 experts, gate/up M=2048
K=4096, down M=4096 K=2048):

| | computation | total | our constant |
|---|---|---|---|
| IQ2_XXS | dq 16,777,216 + qs 536,870,912 | **553,648,128** | `cudaIQ2XXSRepackedBytes` |
| Q2_K | dm2 33,554,432 + sc4 134,217,728 + qs2 536,870,912 | **704,643,072** | `cudaQ2KRepackedBytes` |

Both land on C's `ds4_mmq_iq2_xxs_aligned_bytes` / `ds4_mmq_q2_k_aligned_bytes`
contract with zero slack. **No new repack, no new artifact, no layout work.**

### The dispatcher shape already exists in our tree, on the dense side

`decodeQ80Quantized` (`cuda_session_linux.go:2229-2233`) takes
`(t cudaModelTensor, aligned cudamem.Slice[byte], ...)` and routes to the
aligned entry when `aligned.Len() > 0`, else the raw one. It is called at
`:1968` as `decodeQ80Quantized(l.sharedDown, a.down, q, s.sharedOut)` --
**four lines above** the raw MoE reads. The MoE decode side is the same shape
with no implementation behind it.

### The two missing kernels

**K1** `ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec` (`ds4_mmq.h:640-653`,
called `ds4_cuda.cu:24519-24527`). Fused further than ours: it takes float32 X,
the router `weights` and a `clamp`, and emits `mid_f32` which goes **straight
into K2 with no intervening quantize** (`:24528-24537`) -- contrast the owned
branch at `:24499`, which does quantize between them. So K1 absorbs gate GEMM +
up GEMM + SwiGLU + router weighting. Read off the call sequence, which is
stronger than the header.

**K2** `ds4_mmq_q2_K_aligned_moe_vec` (`ds4_mmq.h:551-561`, called `:24530`).
Each (token, slot) assignment is its own token: `n_tokens = n_tokens*n_expert`
(6 at decode), `n_expert_used = 1`.

C gates K1 on `n_tokens <= 8 && ds4_gpu_device_is_spark() &&
!g_q8_dequant_gemm_enabled && expert_in_dim % 1024 == 0` -- an envelope, where
our gates are plain on/off, which is the same asymmetry found on the dense row.

### Which call sites each retires

| site | current | replaced by |
|---|---|---|
| `:1972` | `encodeMatvecIQ2SelectedQ81(l.expertGate, l.expertUp, ...)` | K1 |
| `:1976` | scalar variant | K1, or kept as raw fallback |
| `:1980` | `encodeSwiGLUQuantQ81(...)` | **absorbed by K1** |
| `:1984` | `encodeMatvecQ2KSelectedQ81(l.expertDown, ...)` | K2 |
| `:1988` | scalar variant | K2, or kept as raw fallback |

Three launches down to two per layer per token, on top of the residency effect.
The batch fallback sites (`:1562`, `:1573`) are unreached when D2R is on.

**Once those are routed, no kernel reads `l.expertGate/Up/Down` on a layer that
has an artifact.** That is the condition C's release is licensed by, and the
condition our `releaseMapped` currently assumes without holding.

### What could not be settled from source

Flagged rather than guessed, and each must be closed before any bit-identity
claim:

- **`clamp`.** C passes one into K1; our SwiGLU path has a clamp of its own
  (`metal_runtime_darwin.go:809`). Not traced to be the same one.
- **Whether `s.q81Input` has a second consumer** in the same layer. If it does,
  retiring the quantize at `:1980` is not free.
- **The kernel bodies.** Declarations, call sites and layout contracts were
  read; `ds4_mmq.cu` implementations were not. A signature is not a capability.
- **Stream.** C uses a dedicated `cuda_decode_stream()` at `n_tokens == 1`
  (`ds4_cuda.cu:24478`). No equivalent checked on our side.

One refinement to the spec as delivered: `s.expertMid` is allocated at
`cuda_session_linux.go:335` and is *not* unused -- the Metal path reads it at
`metal_runtime_darwin.go:830,842`. It is unused on the **CUDA decode** path
specifically, so it is available there but is not dead storage.

## Capture index

Every measurement in this ledger, its lock label, where the artifact lives on
`dgx`, and the script that produced it. Scripts are on the Mac under
`~/tmp/claude-tmp/` and uploaded to `~/tmp/` on `dgx`. All GPU runs held
`/home/tmc/ml-explore/bin/gpu-lock` (which takes no `--label`; it assigns one).

| label | script | artifact on dgx | what it established |
|---|---|---|---|
| 4016301 | -- | -- | baseline at commit `dc859655` |
| 4027530 / 4029xxx | -- | -- | Go/C phase split |
| 4034661 | -- | -- | the two shipping expert arms disagree |
| 4037664 | `grapharm-C6A7.sh` | -- | the graph hypothesis is dead; 8 draws, digest clean |
| 4082203 | `graphvis-C6A7.sh` | `~/tmp/graphvis/go_nograph_1.pprof`, `go_graph_1.pprof` | the 12.6% census; graphs-off resolves 27 kernel names and 2640.544 ms that had never been visible. **The baseline capture for everything after.** |
| 4083518 | `moesweep-C6A7.sh` | `~/tmp/moesweep/` (12 cells) | E3: artifact layers make prefill monotonically slower; both predictions refuted in sign |
| 4089442 | `moecap-C6A7.sh` | `~/tmp/moecap/moe8.pprof` | E3 mechanism: same launch counts, +146.6 ms across untouched kernels |
| 4098372 | `rel2-C6A7.sh` | `~/tmp/rel2/` (8 cells) | E4: mode 1 releases PTEs not pages; mode 2 reclaims; availGiB blind to both |
| 4109833 | `lay2-C6A7.sh` | `~/tmp/lay2/` (10 cells) | E5: corruption ceiling between 8 and 20 layers; the 36% spread that withdrew E4 |
| 4159561 | `n8-C6A7.sh` | `~/tmp/n8/` | E6: paired n=8 at the digest-clean arms |
| 18295 | `e7-C6A7.sh` | `~/tmp/e7/` | E7: release is not the fault; the six-byte gate |
| 40150 | `e7b-C6A7.sh` | `~/tmp/e7b/` | E7b: reported the INPUT hash; re-derived from the logs it shows prefill is not reproducible (1/8 at zero layers) |
| -- | `e8-scout.sh` | **not run** | E8: the zero-byte anchor (cell Z) and the ballast rate. Designed by scout-attn-parity with a pre-registered stopping rule, an honest power statement, and a contamination check. Banked because the capacity question is a constraint on the fix, not the gap. |

### Analysis scripts

The captures carry a per-record journal alongside the pprof; the pprof alone is
an aggregate with no timestamps. Decoding is done against
`*.journal.gz`, **not** the `.pprof` -- a scan for `{"kind":` with a
brace-depth parser that respects string escapes.

The decoder lives in the repository, at `capture_journal.py`. Every dense-row
number in this ledger is reproducible from it against the capture journal named
in the table above:

    python3 capture_journal.py <capture>.journal.gz \
      --sha256 9305cf641709811c73f478e109a9e63cdeabfec5900f16bed074ba61438344f9 \
      --pattern 128,64,4096,128,128,128,128,128,128,128,128,512,256,256,512 \
      --names 'QA,KV,QB,attnOutA*8,attnOutB,gate,up,down'

which prints `records 138686  malformed 0  truncated 0`, 645 launches, 43
complete layer patterns, the seven-site census, and the 368.673745 ms total.

> **This section previously cited four scripts that were not in the repository.**
> `e0.py`, `cmp.py`, `m8.py` and `win.py` existed only in a scratch directory on
> one laptop, so a table naming them read as a reproduction path and was not
> one. E12's design constraint 2 registered "the journal decoder" as its
> instrument while that instrument was unversioned. `capture_journal.py` is that
> decoder, written to be cited.
>
> Porting it surfaced a defect in `e0.py` worth recording, because it is this
> campaign's own class. `e0.py` wrapped `json.loads` in a bare
> `except Exception: pass`. A record that failed to parse was dropped silently,
> and the record count it printed -- the number every later script trusted as a
> census -- could not distinguish "the journal holds 138686 records" from "the
> journal holds more and the decoder lost some." It was a
> fail-open instrument sitting in the one place every measurement passed
> through. The
> replacement counts malformed and truncated records separately and prints a
> warning above any census taken from a lossy decode. On this capture both
> counts are zero, so the published numbers were not affected -- but that is a
> fact I can now state, and previously could not.
>
> The three checks this decoder makes -- the hash, the layer pattern, the
> malformed/truncated counters -- were fired against mutated input before this
> paragraph was written: a wrong `--sha256` exits 1, a single wrong grid entry
> in the pattern exits 1 naming the layer, flipping one byte inside one record
> moves the census to `records 138685 malformed 1`, and truncating the file
> yields `truncated 1`. That makes it the first gate in this campaign whose
> control fired before it was cited rather than after an audit found it
> vacuous. Three earlier ones are documented under *Three gates I built against
> this campaign's own defect, all passing vacuously*; the difference is only
> that this time the mutation came first.
>
> `cmp.py`, `m8.py` and `win.py` are not ported. They answered questions that
> are settled (E4b, the moe=8 scoring) and their inputs are the captures listed
> above; anyone re-deriving those results should do it from
> `capture_journal.py` rather than trust a citation to a file that is gone.

### A second capture, and three ratios that are not the same estimand

`capture_journal.py` made it cheap to run the paired comparison against a
second retained capture, `ex8` (journal sha256 `46923f6f...`, 139197 records,
0 malformed, 0 truncated, 645 launches, 43 complete layer patterns, same
provenance hash as `exn`). It had never been tested against.

| estimand | exn | ex8 |
|---|---:|---:|
| ratio of site **means** | 1.4142 | 1.9969 |
| ratio of site **medians** | 1.5097 | 1.4950 |
| **median of the 43 layer-paired ratios** | 1.4319 | 1.4535 |

> I first reported the middle row as "the paired finding replicates to one
> percent." Both halves of that sentence were wrong in a way worth keeping.
> **A ratio of site medians is not a median paired slowdown** -- it summarises
> two separate distributions, where the bottom row summarises 43 within-layer
> comparisons. Naming the wrong estimand is this ledger's own recurring defect
> (a claim whose scope is wider than the thing computed) applied to a summary
> statistic. The correction is `C8332C52`'s, from an independent raw-journal
> recomputation, and it arrived before publication rather than after.

What the second capture does support: **same-host, same-build repeatability in
two retained captures**, exploratory and post-hoc. All eight sites agree within
5.2% on medians. It is **not** independent replication across builds or hosts,
not 86 independent samples, and not an optimization ceiling.

The means disagree by 44% (QB 2.597 -> 3.742 ms/launch) because **two of ex8's
43 launches ran at 29.024 and 22.096 ms** against a 2.65 ms median; `exn` has
zero launches above twice its median. Those two are unexplained. Recorded
memcpy does not reach them: 6.687 of 6.689 GB (exn) and 21.183 of 21.184 GB
(ex8) of copy traffic completes before the first dense launch, and overlap with
every dense launch is 0.000 ms in both captures.

> **The within-phase figure here was first published as "47 copies totalling
> 1.5 MB" and that is wrong.** The filter was `start >= first dense start`,
> which has no upper bound and so swept in the post-dense tail. Bounded
> properly -- `start >= first dense start` **and** `end <= last dense end` --
> the dense phase holds **43 copies and 352256 bytes** in both captures; the
> four copies I wrongly included carry 1165312 bytes, so 77% of the bytes I
> attributed to the dense phase happened after it. A filter that states only
> its lower bound is a denominator with no ceiling. Caught by `C8332C52` on an
> independent recomputation.

That **excludes recorded memcpy overlap, not all contention** -- and an exact zero in all sixteen cells is also the
shape a dead instrument makes, so it is stated here as a measurement that
wants independent confirmation.

### A caution for anyone re-reading these

The prefill window used throughout is "records starting at or after the last
`ds4_repack_q8_0_aligned` end", which excludes model load. That boundary is the
right one for a per-prefill census and the wrong one for a before/after diff:
see E4b, where a change that moved work across it read as removing the work.
`cudaprof gaps --diff_base` (64bb3a8b and later) spans the whole capture and
splits its per-kernel census by whether launch count moved, which is the view
that catches this.

## Deferred, with reasons

- **Micro-kernel batching.** Refuted as a parity argument (above). The 32814
  micro-kernels are real and are 47% of Go's launches, but the oracle has more
  of them and is faster. May still be worth doing on its own merits; it is not
  a parity fix.
- **D2R expert fusion at decode.** Deferred with a *reason* now, not just a
  priority. The port exists and ships; extending it to decode would move the
  `665ac935` digest, and not by FMA re-association. The two paths compute
  different functions: the decode IQ2 kernel applies `sum * ls / 8` as **integer
  truncation per 32-element sub-block** (`cuda_experts_q81.cu:44`, stated as the
  intended contract at `cuda_experts_q81_linux.go:19`) where D2R uses
  `d * ls * 0.125f` in float over a whole MMA tile; and the decode Q2_K hoists
  `d`/`dm` outside the accumulation where D2R folds them into every fmaf step.
  The SwiGLU expression is the one place the two agree exactly. All accumulators
  are fp32 on both sides, so this is not accumulator width and no reordering
  argument reaches it. The precedent for holding the digest across an arm flip
  (`cuda_session_linux.go:1775`) does not apply: that flip was bit-exact by
  construction (`:1788`). Either the gate is re-baselined against the D2R arm or
  decode keeps the unfused kernels.
- **Open question this raises: ANSWERED, see the arm section above.** They do
  not agree. `665ac935` is the D2R arm; the gather arm is `7a44b494`.
- **PC-sample `ds4_f16_pair_ordered_chunks`.** BLOCKED on errata 21.115(b):
  `DS4_CUDA_PC_PROFILE_PHASE=decode` profiles an un-graphed decode, so a stall
  histogram taken under it describes a different program than the one that ships.
- **Residency / `cudaHostRegister` pinning.** Rejected on evidence: the notebook
  proposed it citing Go's 6580.9 ms of **Host Ahead** idle, which is the opposite
  of host-behind -- the host had already submitted, so pinning cannot shorten an
  idle window the host is not in (21.116).

## Change log: the experiments that carry a capture

**Scope, which this table did not state.** It indexes only experiments with
a before/after capture, and it **stops at E3**. Everything after it --
E4 through E12, every instrument fix, and every retraction -- is absent.
For the complete list see *The complete commit index* immediately below.

| # | date | commit | change | capture before | capture after | verdict |
|---|---|---|---|---|---|---|
| - | 2026-09-18 | dc859655 | baseline, no change | - | `feb096df` / `adedf198` | see above |
| - | 2026-09-18 | 778db619 | ledger only | - | - | no code change |
| - | 2026-09-18 | 034b5029 | plan revised on evidence | phase-split x4 | - | item 1 refuted, target moved to prefill |
| - | 2026-09-18 | 455029ed | prefill path mapped from source | - | - | no code change; three scouts, all read-only |
| - | 2026-09-18 | cc46a2f5 | expert-arm digest A/B, label 4034661 | - | ABBA x2 | arms disagree; gather 1.82x slower at prefill |
| - | 2026-09-18 | 54b221dd | graphs on/off A/B, label 4037664 | - | ABBA x2 | graph hypothesis refuted: 33 ms measured vs 2904 predicted |
| - | 2026-09-19 | 2b75c5f8 | gap-neighbour observation | `go_prefill` / `c_prefill` | - | **superseded**: ran on a stale cudaprof; its retraction of the 114x was wrong |
| - | 2026-09-19 | 1668425d | re-run at cudaprof `bb8de0cc` | same two captures | - | 114x confirmed to the ms; cause is unrecorded graph nodes, duty 71% not 15% |
| - | 2026-09-19 | 06a4acd9 | cross-check at cudaprof `7645c217` | same two captures | - | fix confirmed: hw queue 2803.299 -> 27.387 ms, idle 93.81% host-behind; advise still denies the journal (21.151) |
| - | 2026-09-19 | 98039717 | graphs-off capture, label 4082203 | `go_graph_1` | `go_nograph_1` | **78.5% of prefill work was invisible**; MoE repack is 721.011 ms, 58.0% of the parity gap |
| - | 2026-09-19 | (this) | MoE artifact sweep, label 4083518 | - | 12 cells, all `665ac935` | **refuted in sign**: 8 layers is 7.5% SLOWER; generation collapses 17.4->12.2 only under release=1 |

## The complete commit index

Every commit of this campaign on `codex/ds4-min`, oldest first, generated
from `git log` rather than transcribed. **84 commits, current as of
`7c46926c`.** A hand-maintained index drifts from the tree the moment it is
written, so regenerate rather than append:

```
git log --reverse --format='%h|%ad|%s' --date=short 778db619~1..HEAD -- ':(top)examples/ds4-min'
```

> The pathspec was `examples/ds4-min` until `3d7453e6`, and that form is
> **cwd-sensitive and fails silently**. Run from the repository root it
> returns all 84 commits; run from `examples/ds4-min` -- the directory this
> ledger lives in, and the first place a reader would try it -- it matches
> nothing and exits 0. An empty index and no diagnostic is the signature of a
> resolution failure, and it is indistinguishable from a campaign with no
> commits. `:(top)` anchors the pathspec to the repository root and returns
> the whole set from either directory. The regeneration instruction had the same defect
> as the index it was written to repair: it read as complete and was scoped
> without saying so.


The *Change log*
table above is not this: it indexes only the experiments that carry a
before/after capture, it stops at E3, and it never claimed otherwise
clearly enough -- the spine calls both "indexes" and only one of them was
complete. Roles:

| role | n | what it means |
|---|---:|---|
| **M** | 24 | a measurement, experiment or A/B -- the thing that produces evidence |
| **R** | 37 | a retraction or correction **of this ledger's own earlier claim** |
| **I** | 17 | an instrument, gate or test -- including the ones that were fixed after passing vacuously |
| **S** | 5 | a source reading of the C reference or our tree, no run |
| **D** | 12 | structure: the ledger, its spine, its indexes |

Roles are assigned once each and a few commits do two things: `3d7453e6`
both builds an instrument and retracts this section's claim that four named
scripts were available, and is counted only as **I**. Reading it as **R**
instead would move the ratio below to 38 of 95.

**38 of 97 commits are retractions of this ledger's own claims** -- 39%. The
index rows are current through `4e15e567`; `9d36702b` onward are counted in
this ratio but their rows are written when the index next runs. The newest
retraction is `0e9a96b9`, which withdrew a sentence from `2f9001b2` an hour
earlier -- the input that entry called unmeasured had been measured in the
same report.
The ratio has risen, not fallen, as the campaign matured, and the most recent
retraction withdraws a correction rather than a claim: an advisory issued
*about* an earlier correction, which would have discarded the campaign's
sharpest evidence. Layers of this regress keep looking like the last one from
inside.
That ratio is the campaign's most honest summary statistic. It is not a
sign of sloppiness being cleaned up after the fact; it is what happens when
every claim gets read adversarially by someone with no stake in it, and it
is the reason the instruments rather than the speedup are the deliverable.

| # | commit | date | role | what it did |
|---:|---|---|:---:|---|
| 1 | `778db619` | 2026-09-18 | D | add the parity campaign ledger |
| 2 | `034b5029` | 2026-09-18 | **R** | refute the micro-kernel ranking, move the target to prefill |
| 3 | `455029ed` | 2026-09-18 | S | map the prefill path from source |
| 4 | `cc46a2f5` | 2026-09-18 | M | measure the two expert arms, correct the gate framing |
| 5 | `54b221dd` | 2026-09-18 | M | refute the graph hypothesis with its own prediction |
| 6 | `2b75c5f8` | 2026-09-19 | M | retract the 114x, it was an instrument artifact |
| 7 | `1668425d` | 2026-09-19 | **R** | withdraw the retraction, the instrument was stale |
| 8 | `06a4acd9` | 2026-09-19 | M | close the idle route, the gap is in the work |
| 9 | `98039717` | 2026-09-19 | M | the census was showing an eighth of the work |
| 10 | `c2f10350` | 2026-09-19 | M | the MoE artifact fix is refuted in sign |
| 11 | `c2a1d4f2` | 2026-09-19 | M | the MoE repack does pay, with the right reclaim |
| 12 | `3b6e604f` | 2026-09-19 | **R** | withdraw the E4 win, twice over |
| 13 | `559691b6` | 2026-09-19 | D | index the captures behind the ledger |
| 14 | `50bb8020` | 2026-09-19 | M | the expert kernels are not the gap |
| 15 | `5ce9732a` | 2026-09-19 | M | attribute the dense row, and retire the batching lead |
| 16 | `57d5647b` | 2026-09-19 | **R** | the withdrawn win was real, at +3.13% |
| 17 | `4c392e08` | 2026-09-19 | **R** | release is not the fault, and the gate was six bytes |
| 18 | `0abe2cc3` | 2026-09-19 | **R** | the 1/30 baseline is not a zero-byte control |
| 19 | `5248cef1` | 2026-09-19 | S | spec the aligned decode family |
| 20 | `7805dce7` | 2026-09-19 | **R** | split the corruption bucket into its two classes |
| 21 | `7a5ea568` | 2026-09-19 | **R** | retract the decode fusion's launch-count claim |
| 22 | `8378434f` | 2026-09-19 | M | dense and expert bytes pull opposite ways |
| 23 | `4ddec73d` | 2026-09-19 | S | reconcile two independent decode-family reads |
| 24 | `05319ce4` | 2026-09-19 | **R** | the hash oracle is invalid independent of the cause |
| 25 | `bdc71d60` | 2026-09-19 | M | site the cuBLAS ablation in the loud arm |
| 26 | `e91e5e1a` | 2026-09-19 | M | the workspace is the input that drifts |
| 27 | `854c0e16` | 2026-09-19 | S | C's activation fold never fires |
| 28 | `649e25b9` | 2026-09-19 | M | divergent draws start with less headroom |
| 29 | `7ee24c1a` | 2026-09-19 | **R** | arm A was never the default |
| 30 | `8b4efa21` | 2026-09-19 | **R** | E9's covariate is also a clock |
| 31 | `29521ed3` | 2026-09-19 | I | a hash of NaNs is a stable digest |
| 32 | `981a2fdc` | 2026-09-19 | M | pre-register the blocking at ten draws |
| 33 | `7cc706ec` | 2026-09-19 | **R** | the NaN concern is prospective, not a correction |
| 34 | `87c184ee` | 2026-09-19 | I | report how far the poisoning extends |
| 35 | `0eb9a5d7` | 2026-09-19 | I | one non-finite check, not thirteen |
| 36 | `935f51b7` | 2026-09-19 | I | assert the non-finite checks report extent |
| 37 | `d3f2f6fb` | 2026-09-19 | D | give the ledger a spine |
| 38 | `d67bb2c7` | 2026-09-19 | I | a GEMM stability harness that states its own limits |
| 39 | `ee60e647` | 2026-09-19 | I | check the SwiGLU clamp against the model |
| 40 | `bdd23728` | 2026-09-19 | **R** | record the clamp gate and the gate I over-specified |
| 41 | `a4a4408e` | 2026-09-19 | S | the missing SwiGLU guard was correct parity |
| 42 | `7530ab02` | 2026-09-19 | M | pre-register E10 before running it |
| 43 | `658e29fd` | 2026-09-19 | I | judge a clamp literal by the call it belongs to |
| 44 | `b54d3e92` | 2026-09-19 | I | test that the extent walk reports extent |
| 45 | `d8cd4fb9` | 2026-09-19 | **R** | record the envelope result and two vacuous gates |
| 46 | `8e9f3296` | 2026-09-19 | I | free the pinned workspace before closing the device |
| 47 | `e5aa4432` | 2026-09-19 | I | count dense Q8 D2R launches |
| 48 | `f209004a` | 2026-09-19 | M | the reference shape envelope, measured and refuted |
| 49 | `a2171460` | 2026-09-19 | **R** | the table said nine, the p-value said eight |
| 50 | `907befd0` | 2026-09-19 | D | bring the spine's open list up to date |
| 51 | `f96fa94d` | 2026-09-19 | **R** | two p-values side by side are not a difference |
| 52 | `0be757a8` | 2026-09-19 | **R** | the spine still asserted what the body had retracted |
| 53 | `22768f89` | 2026-09-19 | **R** | say which instrument carries the extent guarantee |
| 54 | `091b6f32` | 2026-09-19 | **R** | C does not batch and prequantize, it does one or the other |
| 55 | `158826f8` | 2026-09-19 | **R** | the spine still said C batches, full stop |
| 56 | `ad5b7b99` | 2026-09-19 | **R** | say what the 1/5/9 series is a count of |
| 57 | `654d6ac0` | 2026-09-19 | **R** | the journal I said was missing was on local disk |
| 58 | `d965820e` | 2026-09-19 | I | run the runs test on the series it is quoted about |
| 59 | `f67a6e70` | 2026-09-19 | I | make the envelope test's outcome classes refuse |
| 60 | `6187fbf2` | 2026-09-19 | **R** | three E11 claims wider than what was measured |
| 61 | `8613b9c2` | 2026-09-19 | I | report the median the label names |
| 62 | `fe9165ba` | 2026-09-19 | **R** | a third gate, found after the section about the first two |
| 63 | `4da3d8e7` | 2026-09-19 | M | QB and attnOutB do identical arithmetic, 1.41x apart |
| 64 | `791fcb1e` | 2026-09-19 | D | bring the spine's QB item up to the paired measurement |
| 65 | `d0cbd1c4` | 2026-09-19 | M | pre-register E12, and retire the tile sweep by arithmetic |
| 66 | `cc692aac` | 2026-09-19 | I | register the envelope acceptance budget |
| 67 | `99dbc299` | 2026-09-19 | D | index all 69 commits, not the 11 with captures |
| 68 | `536d83b8` | 2026-09-19 | D | date the commit index and say how to regenerate it |
| 69 | `3d7453e6` | 2026-09-19 | I | version the journal decoder the ledger cites, and count what it drops |
| 70 | `7cc94dc7` | 2026-09-19 | **R** | the index's regeneration command returned nothing |
| 71 | `096c45b0` | 2026-09-19 | **R** | name the estimand behind the 1.414 |
| 72 | `f17176c5` | 2026-09-19 | **R** | the dense-phase copy filter had no upper bound |
| 73 | `e52c6362` | 2026-09-19 | **R** | E12 assumed its store width, now verified |
| 74 | `ca3f2d91` | 2026-09-19 | **R** | withdraw the oracle-failure shape argument |
| 75 | `124699eb` | 2026-09-19 | D | bring the commit index to 74 rows |
| 76 | `5f077277` | 2026-09-19 | **R** | step 5 is the discriminator, not an artifact |
| 77 | `656d87a5` | 2026-09-19 | D | index to 76 rows |
| 78 | `a54278c2` | 2026-09-19 | M | KVnorm differs by one ULP, unattributed |
| 79 | `6128cc78` | 2026-09-19 | D | index to 78 rows |
| 80 | `a3b377e3` | 2026-09-19 | **R** | scope the KVnorm ULP reading and the null test |
| 81 | `686fe815` | 2026-09-19 | D | index to 80 rows |
| 82 | `295853f4` | 2026-09-19 | **R** | name where my own errors concentrate |
| 83 | `efd220ec` | 2026-09-19 | M | first tolerance failure, at kqv_back |
| 84 | `7c46926c` | 2026-09-19 | M | the kqv_back failure changes partition with position |
| 85 | `29ae4f44` | 2026-09-19 | I | record the partition arms and their controls |
| 86 | `2c211655` | 2026-09-19 | R | commit index listed one commit twice |
| 87 | `73558be1` | 2026-09-19 | M | two writers of raw KV do not store the same values |
| 88 | `53f92985` | 2026-09-19 | R | the fp8 omission hypothesis is falsified |
| 89 | `858f7676` | 2026-09-19 | M | replay closes position 14 and not position 0 |
| 90 | `0a9818f1` | 2026-09-19 | R | withdraw the position-0 replay attribution |
| 91 | `f17fb03e` | 2026-09-19 | D | extend the commit index through 0a9818f1 |
| 92 | `3babe813` | 2026-09-19 | I | the tool-improvement inventory |
| 93 | `da8e76d1` | 2026-09-19 | R | inventory listed a fixed defect as open |
| 94 | `25537a20` | 2026-09-19 | M | both positions close under matched inputs |
| 95 | `4e15e567` | 2026-09-19 | R | the mutant's gating did not change |

## E8: the corruption is two failures, and only one of them tracks bytes

This section corrects the central statistic of the campaign. It does not change
the number; it changes what the number is about.

### The re-tabulation

Every rate this ledger has quoted for "corruption" came from one bucket that
merges two outcomes. `logits_check.go:32-33` catches a non-finite value *before*
the digest is formed, so the outcome space is three-way -- clean, wrong digest,
non-finite -- and the ledger has been collapsing the last two.

The 120-draw characterization (`e861706f`, 30 draws per arm, ABCD/DCBA
alternating) can answer this with no new GPU time: `~/tmp/corrupt-char/`
already records a `digest` column parsed from `final logits sha256=` on stderr
and a `crash` column holding the non-finite message. The reference is the
per-arm **modal** digest, decided in post, which is what `ds4-draw.sh` says in
its own header comment. Arms by bytes held -- C = 0 MoE artifact layers,
A = 2 layers set explicitly, D = 8; all three also build 5.62 GiB of dense
artifacts, so
none is a zero-byte control (`0abe2cc3`):

| arm | non-finite | wrong digest | merged |
|---|---|---|---|
| C (0 layers) | 0/30 | 1/30 | 1/30 |
| A (2 layers) | 1/30 | 4/30 | 5/30 |
| D (8 layers) | 7/30 | 1/30 | 9/30 |

Fisher one-sided, C against D: non-finite `p=0.0053`, wrong digest `p=0.7542`,
merged **`p=0.0061`**. The merged column reproduces the 1/30, 5/30, 9/30
recorded in `cudaMoEArtifactDefaultLayers` (`cuda_session_linux.go:2340-2348`)
exactly, so this is a decomposition of that comment and not a disagreement with
it.

> **Corrected.** This line read `p=0.0128` until an outside audit
> (C8332C52) traced the retained `fisher.py` and found it computes 1 vs **8**,
> not 1 vs 9. The table beside it prints 9. So the displayed endpoint and the
> tested endpoint were different quantities, and this section had already
> explained the gap two paragraphs down -- the `invalid UTF-8` row "is what
> makes D 9/30 rather than 8/30" -- without noticing the p did not include it.
> Recomputed independently: 1/30 vs 9/30 gives 0.006091, 1/30 vs 8/30 gives
> 0.012845. The merged column is the three-category composite (non-finite OR
> wrong digest OR invalid UTF-8), so **0.0061** is its p; the two-class union
> is 8/30 at 0.0128. Note the direction: the error quoted a p **larger** than
> its own endpoint warranted, so it understated rather than overclaimed. That
> is luck, not method. The same mismatch pointed the other way is an
> overclaim, and nothing in the pipeline would have caught it either.

**Non-finite rises 0 -> 1 -> 7 and carries the entire trend. Wrong digest goes
1 -> 4 -> 1: not monotone, and no pairwise comparison reaches significance
(best `p=0.1766`).** A flat low rate independent of bytes held fits the
wrong-digest series as well as anything does.

What this costs is the reading, not the figure. The capacity framing -- the
working set at 20-24 GiB dormant and near 12 GiB touched -- was derived over the
merged bucket, and is now supported only for the non-finite class. Wherever this
ledger says corruption is monotone in bytes held, it has to say which corruption.

> **Corrected, and the correction is about the paragraph above.** "Non-finite
> carries the entire trend" and "supported only for the non-finite class" are
> claims about the *difference between two classes*, and they are supported
> here by putting a significant p beside a non-significant one. That is not a
> test of a difference. The audit (C8332C52) ran the direct
> non-finite-against-wrong-digest contrast under the same arm and block
> conditioning: **two informative strata, four assignments, p=0.75.** This
> dataset cannot resolve the question the paragraph answers. I reproduced its
> analysis myself; every figure below is from that run.
>
> Under symmetric treatment -- same endpoint, same modal-digest pool, same arm
> and ten-draw block conditioning for both classes -- the **non-finite
> association survives**: exact p=1/324=0.0031, and D-only p=1/162=0.0062. So
> the demotion this section worried about does not happen, and the unblocked
> historical figure reproduces exactly at 1170/1184040=0.00099. The non-finite
> result is the sturdiest thing in this campaign.
>
> The wrong-digest class is a different story and not the one told above. At
> the pre-registered block of 10 it is p=0.083 by the mean statistic and
> p=0.208 by rank. At block 20 it is **p=0.000625**, and unblocked it is
> 1.3e-05. Its "flat, independent of bytes held" reading is an artifact of the
> blocking choice as much as of the data, and the pre-registered block is the
> only one entitled to be quoted -- which is why it is, above, and why no
> stronger claim about that class belongs here.
>
> **A separate defect in the correction that produced these numbers.** The
> earlier pass is described throughout this ledger as adding blocking. It
> changed four things at once: rank statistic to mean, centered to raw,
> arm-specific to pooled, and the comparison pool from the modal digest to all
> outcomes. Attributing the resulting move to blocking alone is an
> uncontrolled attribution inside a section whose whole subject is uncontrolled
> attribution. The script output shows the statistic alone moves the
> wrong-digest figure from 0.083 to 0.208 at fixed blocking.

### Two things the re-tabulation found that nobody asked for

**Every wrong digest is a singleton.** Six wrong-digest draws across all four
arms carry six distinct values, no repeats. Corruption has no deterministic
wrong state; each occurrence is a fresh random result. That is what reading
bytes that are not there looks like. A wrong kernel would repeat, and the
ballast result (`cuda_session_linux.go:2518-2526`) already said the same thing
from the other direction.

**The "no digest" bucket was two unrelated bugs.** Fifteen of the 120 rows print
no digest; nine are the non-finite aborts. Of the remaining six, five are

    cuda graph: capture batch ffn layer N: cudagraph: end capture:
    cuda: CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD

at layers 6, 7, 17, 19 and 33, in arms B and C, aborting at 8-55 s before any
logits exist. That is a thread-affinity defect in graph capture -- the goroutine
ending the capture is not always the one that began it -- at roughly 4% of runs,
unrelated to bytes held, and structurally invisible to any gate that reads a
digest, because there is no digest to read. It is new; nothing had logged it.
The sixth is `invalid UTF-8 output` at accepted token 32, which is real
corruption wearing a detokenize failure, and it is what makes D 9/30 rather
than 8/30. Partitioning the residual moved counts in both directions.

### The byte-matched pair that was sitting in the data

Arms A and B hold **byte-identical** artifacts: `moe layers=2 bytes=3.38GiB`,
`dense layers=43 bytes=5.62GiB`, `availGiB=104.8`, the same to the digit. They
differ only in kernel selection -- A sets `DS4_CUDA_MOE_D2R` / `DENSE_D2R` /
`ATTENTION_HMMA`, B sets nothing, and the readers at
`cuda_session_linux.go:634/644/661` are `!r.disableX || env == "1"`, so A
force-enables what B leaves to a runtime field. Their canonical digests differ
(`6e9c1d56` against `2a6edfaf`), which proves the two arms really are running
different kernels.

Same bytes, different kernels: **5/30 against 1/30**, `p=0.1766`.

Not significant at n=30, and not dismissible either. It is the only byte-matched
pair in the sweep and it points the opposite way from the bytes-held account.
The sweep contained its own control for the confound and nobody had read it.
Settling it needs a powered byte-matched A/B, not another layer-count sweep --
a layer sweep varies bytes and kernels together, which is how this went unnoticed.

### E7b, corrected: the table reported the input hash

E7b was built to answer whether the artifacts are numerically correct, with a
gate validated against the V1/V2/V3 ladder. Its extraction was

    sha=$(grep -oE '[0-9a-f]{64}' "$log" | grep -v "^$ZERO$" | head -1)

and the first 64-hex value in the log is not the answer. It is
`token_ids_sha256` on the startup banner -- the **input** digest, constant by
construction across every cell sharing a prompt. Every SHA in the E7b table was
that. The V1 rung was worse than wrong: every cell passes
`-expect-prefill-logits $ZERO`, so every cell returns `rc=1` by construction and
the exit status carries no information at all.

Re-derived from the same logs against `prefill logits sha256=`, which was
present the whole time:

| cell | prefill logits sha256 |
|---|---|
| 0 layers, rel 0, `ref-a` | `f0e4094b28390eab` |
| 0 layers, rel 0, `ref-b` | `f0e4094b28390eab` |
| 0 layers, rel 0, `v1-zero` | `992eeee5e3cfe7a2` |
| 0 layers, rel 0, `v2-half` (1024-token prompt) | `ee89d7f50fa5d2c4` |
| 8 layers, rel 2 | `f0e4094b28390eab` |
| 20 layers, rel 0 / rel 2 | `f0e4094b28390eab` |
| 43 layers, rel 0 / rel 2 | `f0e4094b28390eab` |

Two things follow, and they pull in opposite directions.

The gate is real: `v2-half` moves the digest on a different prompt, so the
value tracks the input through the computation. That is a genuine V2, unlike
the one I recorded, which was satisfied by an input hash tracking the input.

But `v1-zero` and `ref-a` are **the same configuration and the same prompt** --
both are `probe 0 0` with the default token file -- and they disagree.
**Prefill logits are not reproducible at zero MoE artifact layers: 1 of 8 draws
on the full prompt diverged.** So the bit-identity across 8, 20 and 43 layers is
six draws agreeing against a per-draw divergence rate the same experiment
measured as nonzero. The conclusion I was about to write, that the artifact
build is numerically correct, survives as the most likely reading and loses its
proof. It is six clean draws, not a demonstration.

This is the two-draw trap named exactly: a reference learned from `ref-a` and
`ref-b` agreeing, with the third draw of that same configuration sitting in the
same directory disagreeing, under a different label because I had assigned it a
different job.

V4 -- perturb an artifact and confirm the digest moves -- remains **unmet**.

## The decode-family spec, closed out: two of its four open items refute it

The four things `5248cef1` flagged as unsettled have been read out of the two
trees. Two close cleanly; two refute claims the spec made. All read-only; the
`antirez/ds4` tree was not written to.

### The launch-count benefit does not exist (refutes "three launches to two")

The spec counted named entry points. Entry points are not launches, and counted
as device operations the fusion **costs** a launch rather than saving one.

Ours, per layer per token: `encodeMatvecIQ2SelectedQ81` at `:1972` already
takes gate and up together in a single `cudalaunch.Kernel(r.iq2SelectedQ81Fn)`
(`cuda_experts_q81_linux.go:24,48`), so `:1972`/`:1976` are one arm and
`:1984`/`:1988` are another -- three launches, not the five the table implied.
With the shared quantize at `:1891` the expert block is four device ops.

C's: K1 is `iq2_aligned_quantize_xn` (`ds4_mmq.cu:4715`, a
`quantize_row_q8_1_cuda` launch) plus the gate/up/mid kernel (`:4751`) -- two,
or one with the folded sidecar. The `n_tokens >= 2 && n_tokens <= 8` dedup
variant (`:4733`) is not taken at decode. K2 is a `quantize_row_q8_1_cuda` on
the f32 mid (`:4498`), a `cudaMemsetAsync` (`:4512`), and the vec kernel
(`:4522`) -- three. **C total 5, or 4 with the fold. Ours 4.**

The mechanism the spec missed is ours, not C's: `ds4_swiglu_quant_q8_1`
(`cuda_q81.cu:188-210`) does SwiGLU, router weighting and Q8_1 quantize in one
kernel, where C splits that across the K1/K2 boundary -- the epilogue writes f32
`mid` in K1 (`ds4_mmq.cu:2630-2644`) because K2's contract takes
`const float *X_f32`, so the quantize must be its own launch. The claim is
retracted.

What C actually has over us here is not launch count. It is the aligned layout
(two wide `uint2` loads per block against our narrow reads of raw GGUF) and, at
`n_tokens` 2-8 only, the expert-overlap dedup kernel.

### `s.q81Input` has three other consumers, and K1 cannot take it anyway

The spec assumed none. It is written once per layer at `:1891` and read at
`:1943` (`sharedGate`), `:1946` (`sharedUp`) and then `:1972`; the batched path
is the same shape over `d.q81FFN` (`:1391` written, `:1419`/`:1422`/`:1563`
read), with `q81Input` a view of it (`:1022`).

The harder half: K1 takes `const float *X_f32` and quantizes internally
(`ds4_mmq.cu:4715`). It has no parameter for pre-quantized codes. So K1 is not a
drop-in over our quantized activation -- the `:1891` quantize would still be
needed for the two shared-expert projections, and K1 would quantize the same
`ffnNorm` a second time. C hit this too and built the hook: `ds4_mmq_folded_q81`
(`ds4_mmq.cu:4599`) exists to hand the producer's codes in and skip the prelude.
Either that sidecar is ported, or K1's signature changes and it stops being C's
kernel -- which is the whole argument for using it. The spec must say which.

### The clamp matches; two small independent gaps found

C zeroes non-finite gate and up, then clamps gate above only and up two-sided,
at `DS4_SWIGLU_CLAMP_EXP` default 10.0f (`ds4_mmq.cu:2630-2644`, `ds4.c:408`).
Ours is the same asymmetry and the same constant (`cuda_q81.cu:196-198`), and
prefill agrees (`:1546`). The fusion introduces no clamping difference.

Two pre-existing gaps, neither caused by the fusion:

- **We do not zero non-finite gate/up before clamping.** Under CUDA `fminf`/
  `fmaxf`, which return the non-NaN operand, a NaN gate becomes `10.0f` for us
  and `0.0f` for C; a `-inf` gate stays `-inf` for us. **This is a live
  candidate for the non-finite class in the section above**: where C absorbs a
  transient non-finite intermediate, we propagate it to the logits and
  `logits_check.go:32-33` aborts. It does not explain the byte-dependence, but
  it would set the conversion rate from "a transient occurred" to "the run died".
- **Our `10.0f` is a literal with nothing checking it against the model.** C
  refuses to run a model whose `deepseek4.swiglu_clamp_exp` array disagrees with
  the shape constant, per layer (`ds4.c:6474-6497`). A model with a different
  clamp would silently give us wrong numbers. Cheap gate, independent of all of
  this.

### The stream is not an ordering hazard

`cuda_decode_stream()` (`ds4_cuda.cu:988-991`) is three-way: the V4.1 shared
overlap stream, the decode-graph capture stream, or the legacy NULL stream; the
aligned MoE picks it only at `n_tokens == 1` (`:24478`). We have one work stream
(`cuda_runtime_linux.go:267-272,350`) and every decode encode rides it. A single
in-order stream is strictly stronger than an event-fenced pair, so every
dependency C enforces is already implied by our program order.

What we do not have is C's concurrency: `ds4_gpu_dsv41_shared_start` runs the
entire shared expert on a side stream alongside the routed experts
(`ds4_deepseek41_cuda.cuh:74,95-104`), fenced by `ready`/`done` events. Ours is
inline at `:1943`/`:1946`. That is an unexploited overlap, not a hazard -- and C
gates it to Spark and disables it under graph capture, so it is not free there.

### Where this leaves the decode lead

The residency argument survives intact and is now the whole of it: the byte-
neutrality result (`256*2048*16*66 = 553,648,128 = cudaIQ2XXSRepackedBytes`)
still says routing decode onto artifacts moves a working set rather than adding
one, which given the section above is the only lever pointing the right way.
The launch-count argument is withdrawn. The drop-in argument is withdrawn
pending a decision on the folded-Q8_1 sidecar.

## "Bytes held" was never one quantity

The section above ranks arms by bytes held and finds one class monotone in it.
That framing has a defect that reading `cudaDenseArtifactLayers`'s doc comment
(`cuda_session_linux.go:2412-2446`) exposes, and it applies to every rate this
ledger has quoted.

**Dense artifacts are protective.** From that comment, measured rather than
argued: behind a process that had committed 72.56 GiB -- the position where the
artifact was supposed to be dangerous -- `dense=43` failed **0 of 8** where
`dense=0` failed **4 of 8**, and the artifact's clean draws were the faster
ones. The mechanism it gives is direct: 5.62 GiB of repacked weights resident on
the device is 5.62 GiB the kernel does not read back through the model's HMM
mapping, which is what a large predecessor makes the driver reclaim. In a
separate clean sequence both arms failed zero in thirty draws, so the effect
appears only under predecessor pressure.

So the two kinds of byte pull in opposite directions. MoE artifact bytes raise
the failure rate; dense artifact bytes lower it. Ranking arms by a single
"bytes held" axis sums two opposing effects and reports the residue as one
curve. Arms C, A and D all hold the same 5.62 GiB of dense artifacts, so the
comparison between them is sound as far as it goes -- it varies only MoE bytes
-- but the axis label is wrong and does not generalise.

Two consequences.

**Cell Z is not a baseline, and was banked on a misreading.** `dense=0, moe=0`
is simultaneously the minimum-byte cell and the minimum-protection cell, and
the comment above predicts it fails *more*, up to 4/8. A floor rate is not
constructible by removing bytes, because removing dense bytes adds a different
failure mechanism. If what is wanted is the floor of the wrong-digest class, the
cell is maximum protection with minimum MoE artifacts -- `dense=43, moe=0` --
which is arm C, already at n=30 and already reading 1/30. The baseline may
already be in hand.

**The A/B discrepancy is not explained by this.** Arms A and B hold identical
dense *and* identical MoE bytes, so neither direction of the byte effect can
account for 5/30 against 1/30. That one still points at the kernels.

### Two instruments that were in the tree the whole time

Both named in the same doc comment, and both answer questions this campaign
spent experiments on:

- **`DS4_CUDA_DENSE_ARTIFACT_VERIFY`** prints a strided digest of the dense
  artifacts after the build and again at `Close` (`cuda_session_linux.go:2496-2515`).
  Equal digests across a failing run say the bytes were intact. That is the V4
  rung -- does the digest track the artifact -- for the dense side, and the
  comment already records its verdict: the build "is byte-identical across
  processes and across a failing run's own start and end." E7b was built to
  answer a question this flag had already answered.
- **`DS4_CUDA_BALLAST_GIB`**, whose result the comment states plainly: bytes no
  kernel reads run at full speed to 20 GiB. The limit is a working set, not a
  footprint.

This is the fifth and sixth time in this campaign that the answer to an
experiment I was designing was recorded in the doc comment of the flag the
experiment was setting. The pattern is stable enough to be a rule: in this
codebase the comment holds the sample size, the control arm and the verdict,
and the identifier holds none of it.

## Two scouts, one source, four disagreements

`scout-attn-parity` and `gateup-parity` closed the same four open items
independently and without seeing each other's work. They agree on the substance
of all four. They disagree on three points of fact and one of method, and the
disagreements are more informative than either report.

**Method: what counts as a launch.** `scout-attn-parity` wrote "the pair is two,
each of which does its own quantize internally" -- two entry points -- while in
the same report correcting its own spec to note that K1 emits f32 `mid` and K2
quantizes internally. It held the fact and did not cash it: an internal
quantize is a launch. `gateup-parity` counted launcher bodies and got C at 5
(4 with the fold) against our 4. **The device-op count is the right one**, and
the retraction in the section above stands. What makes this worth recording is
that the two reports are not "one right, one wrong" -- the same source line
(`ds4_mmq.cu:4496-4502`) appears in both, doing different work in each.

**Fact: the decode stream.** `scout-attn-parity` reported the aligned pair "does
NOT use" `cuda_decode_stream()`, because the call sites pass `aligned_stream`.
Checked at the source: `ds4_cuda.cu:24478` is

    const cudaStream_t aligned_stream =
        n_tokens == 1u ? cuda_decode_stream() : (cudaStream_t)0;

and that value is passed to both K1 (`:24527`) and K2 (`:24537`). At decode
`aligned_stream` **is** `cuda_decode_stream()`. The name at the call site was
read without tracing it four lines up. `gateup-parity` is right on the binding;
both are right that it is not an ordering hazard for us, since one in-order work
stream is strictly stronger than an event-fenced pair.

**Fact: `s.q81Input`'s consumers.** Two (`:1943`, `:1946`) against three
(those plus the batched twins at `:1419`/`:1422` and the gather at `:1563`).
Both reach the same conclusion by the same route -- K1 takes
`const float *X_f32` and binds `s.ffnNorm`, so the question is moot -- and the
count differs only in whether the batched path was included.

**Fact: the non-finite guard.** Both found that C zeroes non-finite gate and up
before clamping and our decode does not. `scout-attn-parity` adds the half that
matters: **our prefill D2R kernel does have the guard**
(`cuda_moe_d2r.cu:2667-2675`), and our decode SwiGLU (`cuda_q81.cu:194-198`)
does not. So this is not a gap against C; it is an inconsistency inside our own
tree between two paths that are supposed to compute the same expression.

### What `scout-attn-parity` established that nothing contradicts

Read out of the kernel bodies rather than the headers, and each checked against
our dimensions (`cudaWidth = 4096`, expert mid 2048, 256 experts, 6 used):

- **K1's guards pass.** `n_tokens` in 1..16 and `K % 1024 == 0`; our K is 4096.
  At `n_tokens == 1` it takes the per-slot kernel, not the 2..8 dedup variant.
- **K2's guards pass.** `n_expert_used == 1`, `M % 2 == 0`, `K % 256 == 0`; C
  satisfies the first by flattening each (token, slot) into its own token
  (`:24530-24537`). Its output is **per-assignment and unsummed**, `s2_dst = M`,
  which is the shape `encodeMatvecQ2KSelectedQ81Scalar` already asserts
  (`cuda_experts_q81_linux.go:112`). Nothing downstream changes.
- **The offsets were verified against C's own pointer arithmetic**
  (`ds4_mmq.cu:4516-4520`, `:4718-4721`), not only the header comment, and land
  on 553,648,128 and 704,643,072 at these exact dimensions.
- **There is a standalone C test binding both entry points**:
  `cuda/mmq/test/test_iq2_aligned_entry.cu`, 21 KB, K2 at `:54` and K1 at
  `:223`, `:229`, `:247`, `:332`, `:349`, including a 4-token shape. Verified
  present. It is a working reference for argument order and evidence these
  entries are exercised outside the model.
- **The batch sites are nearly free.** C's predicate is
  `n_tokens == 1u || (n_tokens <= 8u && ds4_gpu_device_is_spark() &&
  !g_q8_dequant_gemm_enabled && expert_in_dim % 1024u == 0)`
  (`:24516-24518`), verified verbatim. Our `:1562`/`:1573` batch sites are the
  same two entry points at a wider `n_tokens`, so they cost one predicate rather
  than a second implementation.

### One datum both scouts should stop citing

`scout-attn-parity` reads E7b's clean digests at 8 and 20 layers as evidence the
aligned pair is safe to land at those counts, carefully flagging that it is
consistent with rather than proof of it. The flag was right and the datum is
gone: E7b's table was `token_ids_sha256`, the input hash. Re-derived, six cells
agree and a seventh run of an identical configuration disagrees. Nothing about
release-safety at any layer count can be cashed from it.

## The oracle conclusion does not wait on the mechanism

Whether the wrong-digest class is cuBLAS or corruption is still open. The
consequence for the instrument is not.

**Selection is not the suspect.** `GemmDefault` picks by a deterministic
function of shape, dtype, arch, library version and available workspace. It does
not draw per process. So "cuBLAS re-picks per run" was the wrong form of the
hypothesis, and the sixty consecutive identical digests recorded in
`cudaDenseArtifactLayers` are what selection *predicts* rather than evidence
against the library. Of the two inputs that could move a pick, shape is fixed by
a fixed prompt and workspace depends on free memory -- and a workspace-driven
fallback would be byte-dependent, which fights the flat 1/4/1 the wrong-digest
class actually shows. Selection is squeezed from both ends by our own data.

**What is not squeezed is nondeterminism inside the chosen kernel.** A split-k
or stream-k GEMM can choose its split count at launch from occupancy and reduce
partials through atomics. That is per-run nondeterministic with the algorithm
held fixed, invisible to every selection argument, and consistent with both
numbers at once -- 0/60 where the chosen kernel does not split, nonzero where it
does.

**Our six-distinct-singletons datum discriminates between the two, and we had
been cashing it for the wrong claim.** A per-process algorithm choice has a
small finite outcome space and is constant within a process, so two runs falling
to the same alternate algorithm should yield the *same* wrong digest. Six for
six distinct is what a perturbation finer-grained than one startup choice looks
like. It points at the kernel, not the pick.

### The instrument verdict, which holds either way

This tree already declares the path not bit-reproducible. Its own test logs
`"cuBLAS order not certified"` and compares with atol/rtol rather than bitwise
(`cuda_f16_default_linux_test.go:19,95`), and the GEMM arm's doc comment says
the two implementations differ **only** in summation order
(`cuda_f16_default_linux.go:108-111`).

A SHA-256 over the output of a path whose written contract is "summation order
not certified" is a gate that can fail without a defect. **Sixty green runs do
not make it a bit-exact contract; they make it a contract that has not been
renegotiated yet** -- and renegotiation happens on a cuBLAS or driver upgrade,
on someone else's schedule. Every corruption rate in this ledger was measured
with that gate, so every one of them carries an unknown baseline of
library-legal variation.

The fix does not depend on resolving the mechanism: **a tolerance gate on the
logits, with a justified threshold, and an exact gate on the token ids** if one
that is allowed to be exact is wanted. The token ids are integers and survive
any summation order; that is the quantity for which equality is the right
question.

### The experiment, redesigned

Swapping cuBLAS for the hand-written arm varies precision policy, kernel and
summation order together -- the layer-sweep mistake one level down. Pinning the
handle holds the kernel fixed and varies only the suspect. In priority order:

1. **TF32 off.** The handle runs `SetMathMode(TF32TensorOpMath)`
   (`cuda_hc_linux.go:61`) -- ten mantissa bits. The proposed mechanism is a
   discrete amplifier at a top-k routing boundary, and near-ties are far likelier
   at ten bits than at twenty-four. A rate that moves a lot supports the
   amplifier; a rate that does not move badly hurts it. One flag, sharp either way.
2. **`CUBLAS_ATOMICS_NOT_ALLOWED`**, the direct test of the split-k story.
3. **`CUBLAS_PEDANTIC_MATH`**, pinning away tensor-op substitution generally.

Recorded as designed and not yet run.

## The ablation belongs in the loud cell, not the clean one

The cuBLAS experiment was designed to run in arm C — zero MoE artifact
layers, the cell with the fewest bytes held and the cleanest recorded
behaviour. That was the wrong cell, and the reason is arithmetic rather
than taste.

C's wrong-digest rate is 1/30. If turning TF32 off drove the rate to
*literally zero*, 0/30 against 1/30 is p=0.5 one-sided. The experiment
cannot produce a result. Worse than that, it cannot produce a *legible*
null: a flat reading in C would be written down as "the amplifier story
is hurt" when the only thing it licenses is "this cell was too quiet to
speak." A lock window spent there buys a sentence that sounds like
evidence and is not.

Arm A is the highest wrong-digest cell at 4/30. 4/30 against 0/30 is
p=0.056 one-sided — at least a result, and a large true effect is
visible. Power lives where the base rate is, and the instinct to run an
ablation in the cleanest available configuration inverts that: the clean
cell is clean *because* the phenomenon is rare there.

Two constraints came with the relocation:

  - **Hold arm A's kernel flags fixed.** A force-enables MOE_D2R,
    DENSE_D2R and ATTENTION_HMMA; B leaves them to the runtime field
    (`cuda_session_linux.go:634/644/661`). If the handle change lands on
    a build where those differ, two variables moved and the A/B is no
    longer the control that made this design better than the arm-swap it
    replaced. The harness sets the flag set explicitly rather than
    inheriting it.

  - **A null does not exonerate cuBLAS.** `GemmDefault` selection is
    deterministic in shape, dtype, arch, version and workspace, and
    split-k is shape-dependent: a shape that does not split returns a
    clean zero for a reason unrelated to whether the path is bit-defined.
    The oracle verdict stands on the written contract — the tree's own
    test logs "cuBLAS order not certified" and compares atol/rtol — not
    on any observed rate. This experiment asks about the *mechanism* of
    the corruption. It cannot re-open the question of the *instrument*.

## The divergence was recorded once already, and attributed to the workspace

`05319ce4` closed the oracle question with a sentence about mechanism that
I had no business writing:

> `GemmDefault` selection is a deterministic function of shape, dtype,
> arch, version and workspace, so 0/60 is what selection predicts.

Five inputs are listed and one of them is not a constant. Workspace
availability moves with the host's memory state from run to run. A
deterministic function of a drifting input drifts, so selection can vary
between processes with no randomness anywhere — no split-k, no
per-launch perturbation, no per-process dice. I wrote the refutation of
my own claim into the same sentence as the claim and did not see it.

The tree had already seen this happen. `ensureBatch`,
`cuda_session_linux.go:884-894`, on the per-token-slot cuBLAS handles:

> One cuBLAS handle per token slot, each with its own workspace [...]
> Past roughly 1700 slots it failed outright with
> CUBLAS_STATUS_ALLOC_FAILED, which was the harmless symptom; below that
> it was worse. At 512 slots the same pressure changed the logits, and
> changed them differently from run to run — two digests from one build
> — while costing a factor of three in throughput. Both looked at first
> like evidence that a wider group reassociates the reductions, and
> neither was: widths 256 through 2048 are bit-identical once these are
> not allocated.

Three things fall out of that paragraph.

**Run-to-run digest divergence from one build is not new.** It was
observed, reproduced, and cured — by removing an allocation, not by
changing an algorithm.

**The reassociation hypothesis has already been raised here once, looked
right, and was wrong.** That is the same hypothesis this campaign has
been pursuing under a different name. The comment is the record of it
failing.

**Workspace pressure fits the six-distinct-singletons datum at least as
well as launch-time split-k does.** A per-process algorithm pick from a
small menu should repeat across processes; that is why six distinct
digests argued against it. But workspace availability is near-continuous
rather than a small fixed set, so selection driven by it can land in
many cells while staying constant within any one process. The datum
that discriminated per-launch from per-process does not discriminate
this third thing, which is constant within a process and varying across
them.

So the experiment's decisive arm is not the math mode. It is
`Handle.SetWorkspace` — `cublas/doc.gen.go:390`, present in the bindings
— pinning a fixed, test-owned device buffer. `MathDisallowReducedPrecisionReduction`
(`cublas/enums.gen.go:285`) constrains how split-k partials are reduced
and so tests the kernel; it cannot separate "selection drifted because
its input drifted" from "the chosen kernel is nondeterministic." Pinning
the workspace removes the drifting input, and the two outcomes point at
different remedies: a surviving divergence indicts the kernel, a
vanishing one indicts selection and the fix is a pinned workspace in the
shipping path rather than a math-mode change.

Two corrections to the arm list that came with this, both from reading
the bindings rather than the API's reputation:

  - **`cublasSetAtomicsMode` is very likely a no-op here.**
    `cublas/handle.gen.go:7485` states that a default-initialised handle
    is already `AtomicsNotAllowed`, and our code never calls the setter.
    The documented scope is the symv/hemv alternates, not GEMM split-k.
    The arm stays, because an arm that measures `GetAtomicsMode()` on
    the live handle and prints that it set nothing is worth more than a
    quietly stable row that reads as a fifth exclusion.

  - **The LIB/HAND swap is not a cuBLAS-versus-no-cuBLAS comparison.**
    Under `disableLibraryGEMM`, `cuda_hc_linux.go:302` still issues a
    `GemmEx(..., GemmDefault)` — once per row at N=1 instead of once per
    group. That 2x2 discriminates wide-N heuristic selection from N=1.
    It was demoted to the secondary test for that reason.

The oracle verdict is untouched by all of this. It rests on the written
contract, and a mechanism that makes the divergence *more* explicable
makes the exact-identity gate *less* defensible, not more.

## The fold we were comparing against is a stub

The decode-family accounting carried a conditional: C is five device
ops, or four when `ds4_mmq_folded_q81` hits and K1 skips its quantize
prelude. The conditional is dead. `ds4_mmq_folded_q81`
(`ds4_mmq.cu:176-187`) computes nothing — it is a ten-line probe into a
registry keyed on the activation's device pointer and K, asking whether
a producer already emitted canonical `block_q8_1` codes for that exact
buffer. The registry lookup it probes, `ds4_cuda_q8_fold_take_q81`, is
in this tree:

    extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim,
                                             const void **q81) {
        (void)src; (void)in_dim; (void)q81;
        return 0;
    }

with the comment four lines above it (`ds4_cuda.cu:1173-1176`) saying
so outright: "The flat-pool producers are not ported, so nothing is
ever registered: always miss, and the entries run their own activation
quantize." All six call sites take the miss branch. The producer lives
in the upstream fork, deliberately left behind along with "the
aligned-SoA / D2R / producer-quantized tiers."

**C is at five device ops unconditionally, against our four.** The
qualifier that made our fusion advantage conditional was never a live C
behaviour.

It also dissolves the port question. There is no implementation to
bring over: porting the fold means writing a pointer-keyed registry and
the per-layer-reset, one-shot-pop freshness protocol its comment
describes, in order to bridge a producer and a consumer that sit four
lines apart in our code. Changing K1's launcher to take
`const block_q8_1 *` instead of `const float *X_f32` deletes one call
(`ds4_mmq.cu:4715`) and touches nothing else — the kernel body never
sees a float input.

### The layouts do match, field for field

`block_q8_1` (`cuda/mmq/ggml-common.h:248-259`) and our
`ds4_q8_1_activation` (`cuda_q81.cu:1`) are both 36 bytes with `d` at 0,
`s` at 2, `qs` at 4, both static-asserted. The scale is `amax/127` with
the same zero-block special case on both sides; the reductions span the
same 32 lanes; block ordering is linear in both at `n_tokens=1`; and at
K = `cudaWidth` = 4096, `ne10_padded == K`, so there are no pad blocks
and both sides compute 4608 bytes. Neither consumer reads the `s` half
— K1 takes `.qs` and `__low2float(.ds)` (`ds4_mmq.cu:2585-2586`), K2 the
same (`:4389-4390`).

One bit is unverified rather than matched: the half encoding of `d`. C
goes through `make_half2`; we go through our own RNE in
`ds4_q81_half_bits`. The `qs` bytes cannot differ — integers from an
identical formula — and `s` is unread, so the exposure is one
half-precision scale per 32-element block. That is a measurement, not an
argument, and C ships the measurement: the `DS4_Q8_FOLD_SELFTEST` block
at `ds4_mmq.cu:4604-4637` memcmps exactly these bytes. It is dead code
there for the same reason the fold is.

This is the third correction in this campaign where a name promised a
capability the file did not implement, and the second where the
implementing comment sat directly above the stub.

## E9: the covariate was already in the logs, and it is not flat

The workspace account needs workspace to actually drift. `SetWorkspace`'s
own doc (`cublas/handle.gen.go:9170`) says the recommended size "is based
on the cuBLAS default workspace pool size which is GPU architecture
dependent" — architecture-dependent, not free-memory-dependent. If the
default pool is a fixed per-arch claim, then on one device there is no
drifting input, selection is constant after all, and a pinned-workspace
arm pins something that was never moving. It would return "divergence
survives" and be read as "the kernel is nondeterministic" — a false
exclusion from a knob with no reach, which is errata 21.175 one more turn
around.

That is testable for free. The 120-draw sweep's `.err` logs record
`availGiB` at each artifact build, so device free memory is a per-run
covariate that has been sitting in the capture since 2026-09-18. No lock
window needed.

**It is not flat. It ranges 32.5 GiB across the sweep, and the divergent
draws sit at the bottom of it.** Classifying as in E8 and comparing
`availGiB` at the dense-artifact build, one-sided permutation rank test,
exact by enumeration except where marked:

| arm | wrong-digest draws | median availGiB | clean draws | median availGiB | p |
|---|---|---|---|---|---|
| A (2 layers) | 4 | 95.1 | 25 | 110.4 | 0.0002 |
| C (0 layers) | 1 | 91.3 | 28 | 108.2 | 0.0690 |
| D (8 layers) | 1 | 98.1 | 21 | 100.2 | 0.0455 |
| pooled, each arm centred on its own clean median | 6 | −15.3 | 99 | — | <0.0001 (MC) |

A wrong-digest draw starts with about fifteen fewer GiB of headroom than
a clean draw in the same arm. Centring per arm matters and is what makes
this a within-arm result: every draw in an arm holds identical artifacts,
so the surviving variation is host-side and not a restatement of bytes
held.

### This refines E8 rather than contradicting it

E8 established that the wrong-digest class does **not** track bytes held
*between* arms — 1/30, 4/30, 1/30, non-monotone, best pairwise p=0.1766 —
while the non-finite class does. That stands. What this adds is that the
wrong-digest class does track free memory *within* an arm, and strongly.
The two are not the same axis: artifact bytes are a configured constant
per arm, host headroom is a per-draw accident. A class can be flat in the
first and steep in the second, and this one is.

So a memory-pressure account of the wrong-digest class is back on the
table, having been set aside on the between-arm evidence alone.

### The internal control that makes it credible

Arm B's five no-digest draws sit at median 110.1 against a clean median of
110.4 — essentially no deficit, where every other divergent bucket shows
one of 14 to 21 GiB. Arm B's no-digest bucket is the one this ledger had
already identified, independently and for unrelated reasons, as a
*different bug*: `CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD` graph-capture
aborts, about 4% of runs. The one bucket known on other grounds not to be
memory corruption is the one bucket that does not show the memory deficit.
That is a control this analysis did not ask for and could not have
arranged.

### What it does not show

It does not establish the workspace mechanism. `cudaDenseArtifactLayers`
already says headroom is the currency — "a draw that starts with less of
it is the draw that corrupts" — for the reason that the non-expert weights
are addressed through host page tables. This measurement is equally
consistent with that account and with cuBLAS workspace-driven selection
drift, and it does not separate them. What it does is remove the
possibility that killed the workspace arm's premise: the input the arm
would pin is demonstrably not constant across runs.

Two cautions on the instrument. `availGiB` is scoped to one pool
(`free-memory-readout-scoped-to-one-pool`), so it is a proxy for host
state rather than a whole-device figure — a flat reading would not have
exonerated anything, which is why the non-flat reading is the useful
direction of inference. And a pool allocation can fail or fall back under
pressure without moving the free-memory readout much. Closing the loop
needs the workspace the handle actually receives at creation, recorded per
run; if the bindings expose no getter for that, the harness should say so
in its doc comment rather than let the absence read as agreement.

### Two figures corrected, and one arm was never "the default"

`cudaMoEArtifactDefaultLayers` is **0**, not 2 (`cuda_session_linux.go:2354`),
and its comment says the zero is deliberate: two layers is the throughput
optimum, declined because "the expert artifacts are bought back for safety,
not given up for speed." So arm A is two layers **set explicitly**, and
every place this ledger called it "the default" named arm C instead — the
quiet cell, which is the one the power argument two sections up exists to
argue against. Corrected in the E8 table above.

The other correction is arithmetic. `0/30` against `1/30` is **p=0.5**
one-sided, not the 0.75 recorded: the tail is C(30,1)/C(60,1) = 30/60. The
power argument is unaffected and marginally stronger.

Both found by a subagent reading the constant rather than the campaign's
name for it. The pattern is the one this ledger keeps recording from the
other side — a name that has been used consistently for long enough stops
being checked against the thing it names, and "default" is an especially
bad offender because it sounds like a property of the code rather than a
claim about a value.

A related trap this nearly walked into: the same subagent proposed moving
the harness to the eight-layer cell, 9/30 being louder than 5/30. Those are
**merged** rates. Decomposed (E8), the eight-layer cell is 7 non-finite,
1 wrong digest, 1 invalid UTF-8 — its wrong-digest rate is 1/30, identical
to the quietest cell. The mechanism under test produces wrong digests, not
non-finites, so that move would have taken the experiment to the quietest
available cell for the class under test while every number on screen said
it was the loudest. A merged bucket that is monotone because one class is
monotone does not just blur the rate; it inverts the cell ranking.

## E9 corrected: the covariate is a clock, and it was read off the wrong column

Two independent defects in the section above. Both were found within the
hour, one by a peer and one by the audit that followed, and together they
take the headline figure from `p=0.0002` to `p=0.09`.

### Defect one: the classifier read column 7, and the marker is in column 8

`summary.tsv` column 7 is `none` for all 120 draws. The non-finite marker
is in **column 8** (`non-finite_logit_0`, 9 draws). The E9 pass keyed on
column 7, so every non-finite draw fell through into the no-digest bucket
and the two classes E8 exists to separate were silently re-merged — in the
section that opens by invoking E8.

Corrected partition, which now reproduces E8 exactly:

| arm | clean | wrong digest | non-finite | no digest |
|---|---|---|---|---|
| A (2 layers) | 25 | 4 | 1 | 0 |
| B (2 layers, runtime flags) | 25 | 0 | 1 | 4 (all graph-capture) |
| C (0 layers) | 28 | 1 | 0 | 1 (graph-capture) |
| D (8 layers) | 21 | 1 | 7 | 1 (invalid UTF-8) |

### Defect two: `availGiB` is also a clock

Thirty-two GiB of range across a sweep does not come from nowhere. It
comes from a neighbour starting, page cache growing, a previous arm not
yet released — all of which drift with wall time. So `availGiB` is partly
a proxy for *when* a draw ran, and a permutation test that assumes draws
are exchangeable will overstate its significance by an amount it cannot
report.

They are not exchangeable, and not marginally so. Wald-Wolfowitz runs on
`availGiB` about the sweep median: **33 runs observed against 61.0
expected, sd 5.5, z = −5.13**. Lag-1 agreement is 87 of 119 against a
chance ~60. The covariate moves in blocks.

Re-testing by permuting labels only *within* contiguous blocks of run
order, which respects that structure:

| permutation | p (wrong-digest, pooled) |
|---|---|
| global (as published in `649e25b9`) | 0.0002 |
| within blocks of 30 | 0.0130 |
| within blocks of 20 | 0.0286 |
| within blocks of 10 | **0.0935** |

The most conservative blocking is the honest one — larger blocks approach
the global test and inherit its assumption. **The wrong-digest deficit
does not survive at conventional significance once run order is
respected.** It is a suggestive 15 GiB gap at p≈0.09, not the p=0.0002
result the previous section claims.

### And it was never replicated across three arms

Arms C and D contribute one divergent draw each. With n=1 the p-value is
just that draw's rank, so the smallest value either arm could ever produce
is 0.0690 and 0.0455 respectively — and those are exactly the values
reported. **Both arms returned their floor**, which is what a single draw
looks like when it is anywhere at all below the median. The pooled figure
was carried almost entirely by arm A's n=4. The correct sentence is one
arm at n=4 with two single draws pointing the same way.

### What survives, and it is the part that was buried

The dissociation stands and is now the strongest evidence in the section.
Arm B's four no-digest draws are `CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD`
graph-capture aborts — a bucket identified on entirely independent grounds
as not memory corruption. They show **no deficit**: median 110.1 against a
clean 110.4, p=0.1188 against a floor of 0.0385. It could have reached
0.0385 and did not. Meanwhile arm D's seven non-finite draws sit 14.6 GiB
low at p=0.0010.

That is a double dissociation nobody arranged and nobody could have tuned,
and unlike the three p-values it is a comparison whose outcome was
predicted by a mechanism fixed in advance. Its limit, stated: n=5 with no
deficit rules out that bucket sharing the mechanism; it does not establish
that bucket has no sensitivity at all.

So the honest summary of E9 is narrower than the section above it and
points the same way for a different class. **Non-finite draws start with
materially less headroom (arm D, n=7, p=0.0010), which is what
`cudaDenseArtifactLayers` already says. Wrong-digest draws show a 15 GiB
gap that does not reach significance once the covariate's autocorrelation
is respected.** E8's finding is untouched: wrong-digest is still the class
that does not track bytes held.

The pinned-workspace run should wait on this. A time confound that inflates
a covariate's apparent effect will also drift the pin arm's own baseline
between its two processes, which are by construction run at different times.

### Two digest functions in one tree disagree about whether a poisoned draw has a digest

`checkLogits` (`logits_check.go:31`) returns on the first non-finite value
**before** `h.Sum` is reached, so a poisoned draw emits no
`logits sha256=` line at all — which is why the sweep records it as a
separate outcome and why E8's three-way partition is well founded.

`prefillLogitDigest` (`cuda_prefill_graph_linux_test.go:190`) does the
opposite: it hashes every logit including the non-finite ones and returns
the offending index alongside. A poisoned draw therefore *does* have a
digest on that path.

The trap is that a hash of NaN bit patterns is perfectly deterministic. A
poisoned draw entering a digest tabulation appears as a second stable
digest — indistinguishable from the wrong-digest class it is not, and
capable in principle of becoming an arm's modal digest if the corruption
repeats its bit pattern. The gemm-stability harness had exactly this,
misfiling one draw in thirty at two artifact layers, worst in the cell
whose rate most needs to be trusted. It now carries `nonFinite` per draw,
excludes poisoned draws from both the digest tabulation and the engagement
comparison, and files the two classes as separate findings.

This is the same defect as the column-7 classifier above, arriving through
a different door: one merged the classes by reading the wrong field, the
other by using a digest function that does not distinguish them. Any
comparison that spans both functions is comparing different definitions of
"has a digest."

### The retraction is a second axis, not a loss

E8 asserted that the two corruption classes are different phenomena on
one axis: wrong-digest does not track artifact bytes held, non-finite
does. The corrected E9 supports the same partition on a *second,
independent* axis, and by a mechanism rather than only by a rate:

  - non-finite draws start with materially less headroom — arm D, n=7,
    median 14.6 GiB below clean, `p=0.0010` — which is what
    `cudaDenseArtifactLayers` already says headroom is for;
  - wrong-digest draws show a 15 GiB gap that does not reach
    significance under honest blocking.

A retraction that converts a spurious p-value into a second line of
support for the partition it was published under is not a net loss. The
headline number was wrong; the structure it was decorating got stronger.

### The blocking is pre-registered at ten draws

Choosing a block size after seeing three p-values is a knob, and it is
precisely the knob to reach for if a result comes out wrong. So it is
fixed here, before the pinned-workspace run, on a stated basis rather
than on its answer.

The 120-draw sweep gives 33 runs about the median, so the mean run length
of `availGiB` is about 3.6 draws. **A block of ten draws is about three
correlation lengths** — long enough that shuffling inside one is not
shuffling within a single excursion, short enough that the permutation
still respects the drift. That is the figure quoted for the workspace
result, whatever it turns out to be, and it is the figure the harness doc
comment carries.

Reporting the other block sizes as a sensitivity is fine. Quoting one of
them as the result is not.

#### Cross-tabulated: the two classes are disjoint in this data

A NaN's bits are only deterministic if its payload is. IEEE 754 fixes the
exponent and a nonzero significand and leaves the payload and sign to the
implementation, which splits the concern in two with opposite signatures:

  - a NaN **generated** by an operation — overflow to infinity, then
    `inf - inf` or `0 * inf` — is the default quiet NaN on real hardware,
    so its bits are stable and it hashes repeatably;
  - a NaN that arrives by **reading bytes that are not there** carries
    whatever those bytes were, which is arbitrary per occurrence.

The second case predicts exactly the signature this ledger has been
attributing to something else: six wrong-digest draws, six distinct
digests, no value repeating. A poisoned draw entering through a digest
function that hashes NaNs would look precisely like that.

It did not happen here. Cross-tabulating the non-finite marker against the
digest field over all 120 draws:

  - all **9** rows carrying `non-finite_logit_0` have **no digest at all**;
  - all **6** wrong-digest rows carry **no marker**, in either column;
  - the two sets do not intersect.

Which also confirms empirically what was previously an argument from
source: the sweep's digest comes from `checkLogits`, the function that
returns before `h.Sum`, so a poisoned draw cannot enter the digest
tabulation by that route.

So the six-distinct-singletons datum is uncontaminated, E8's partition
stands as measured, and the NaN-hashing concern is **prospective** — real,
worth the fix already made in the gemm-stability harness, and not a
correction to anything published. It becomes live the moment any analysis
sources digests from `prefillLogitDigest` instead.

Two details the cross-tab surfaced without being asked. Every one of the
nine markers reads `logit_0` — the first index, every time, which is what
a wholly poisoned vector looks like rather than a single bad element. And
arms A, C and D share one modal digest (`6e9c1d56`) while B's differs
(`2a6edfaf`), consistent with B being the arm that leaves the kernel flags
to the runtime field; no wrong digest equals any arm's mode.

#### Retracted: "wholly poisoned vector" was not measurable from that marker

The cross-tab's second free observation does not hold. Nine markers all
reading `logit_0` is not evidence that the whole vector was poisoned. The
guard returns at the **first** non-finite value, so it reports index 0
whenever index 0 is bad and cannot see past it. A vector where every
element is NaN and a vector where only element 0 is NaN produce the
identical marker. **An early-returning checker measures presence, never
extent**, and nine-for-nine at index 0 is what you get under either
reading.

This one is worth resolving rather than filing, because the two readings
name different bugs:

  - **every element bad** is consistent with the read-bytes-that-are-not-
    there account this ledger already favours, and with the headroom
    result;
  - **element 0 specifically, nine times of nine**, is the signature of
    something wrong at the *start* of the buffer — an offset, a mapping
    boundary, a first-tile launch. That is a layout fault, and it would be
    invisible to every experiment in this campaign, because they all vary
    pressure and none varies layout.

The instrument is now fixed in three places rather than argued about.
`cuda_session_linux.go` had two identical early-returning loops (the
decode and batch logit guards); both now call `checkFinite`, which walks
the whole slice and reports `non-finite logit %d (%d of %d, through %d)`
— first index, count, length, last index. `checkLogits`
(`logits_check.go`) decides finiteness over the whole slice *before* any
hashing, so a poisoned draw still produces no digest — the property E8's
partition depends on — while the error now carries the extent.

The next non-finite draw settles it. If the count is the vocab size the
poisoning is total and nothing changes; if it is one, or a short run at
the front, this campaign has been measuring the wrong axis.

#### Three was not the number: the same defect was written thirteen times

F76E asked the question the fix could not answer from inside itself: *what
makes three the number?* It was not three. Fixing three sites and
declaring the instrument sound is the same error as the instrument itself
— an answer that reports what it found and is silent about what it did not
look for.

A census of every non-finite check over a logits vector in this package
found **thirteen**, all written independently, all with the same defect:

| site | shape |
|---|---|
| `cuda_session_linux.go` decode, batch | inline loop, first index |
| `checkLogits` | fused scan-and-hash, first index |
| `prefillLogitDigest` | walks the whole buffer to hash it, then **records only the first index** |
| `oracleWorkloadHash`, `oracleWorkloadToken` | first index / no index |
| `cuda_prefill_{d2r,gather,hmma×2,dense_d2r}`, `prefill_gate`, `cuda_raw_attention_group`, `cuda_production_hash`, `cuda_observe`, `cuda_kernels`, `metal_observe` | fused scan-and-hash, first index |
| `cuda_single_token_repeat_linux_test.go:191` | **a verbatim copy of the production decode guard**, same message string |

Two of these matter beyond tidiness.

`prefillLogitDigest` is the second reader F76E pointed me to as the way to
settle the question cheaply — and it is *itself* extent-blind. It already
walks all 129 280 elements to form the hash, and throws the extent away
with a `nonFinite < 0 &&` guard. The instrument nominated as the
independent check had the defect under test.

`cuda_single_token_repeat_linux_test.go:191` reimplemented the production
decode guard, loop for loop and string for string. Any draw through that
path would have printed the old first-index-only marker *after* the
production fix landed, and the log would have read as a short poisoning at
the front — the layout signature, the interesting answer. A fix applied to
one of N copies leaves N−1 live, and the tell is the same as for probes:
**you cannot tell from the fixed site how many sites there were.**

All thirteen now route through one `nonFiniteExtent` in `logits_check.go`,
which is portable rather than `_linux`-tagged — the first draft put the
type in a `_linux_test.go` file and broke the darwin build, which is the
only reason the cross-GOOS half of the census got taken at all. The format
is the discriminator: `first-last/count/length`, and `grep` for the old
`non-finite logit %d` spellings now returns zero. Assert that zero before
reading any extent number out of a log.

#### The census was thirteen by method and one by luck, so the zero is now a test

F76E's follow-up refused the fix on its own terms: *what establishes
fourteen?* An unplanned darwin vet rejection, triggered by the shared type
landing in a `_linux_test.go` file. So the census is thirteen found by a
method plus one found by accident — and the method is the same one that
confidently reported three. A number arrived at that way is a sample
wearing a census's clothes.

`TestNonFiniteChecksReportExtent` converts it. There is now exactly one
`nonFiniteExtent` and one message format, so the old spellings are a
finite set of strings that must appear nowhere, and the test asserts that
as a property of the package rather than as a grep someone ran once.

Two details carry the lesson rather than decorate it.

**It reads the directory, not the package.** `os.ReadDir` has no build
tags, so all 199 `.go` files are in the denominator whatever GOOS the test
is compiled for. A scan that sees only what the current GOOS compiles
reports the package clean and says nothing about the other half — which is
precisely how the fourteenth copy survived. The test states its scanned
count and fails below 100, because a scan that matched nothing *because it
read nothing* is the exact failure it exists to prevent.

**It has a positive control.** The defect was reintroduced into
`metal_attention_darwin_test.go` — a darwin-only file — and the test run
under `GOOS=linux`. It failed. A gate never run against the fault it
guards is an unproven instrument, and this one guards a fault that by
construction lives where the build cannot see it.

The fifteenth copy now fails at the moment it is introduced rather than at
the moment it misleads.

**The general rule, which is larger than its instance:** a grep's
denominator is the build configuration it ran under, and *a census that
cannot state its denominator is a sample*. It is the same rule this
campaign has applied to every arm — [green must state its denominator] —
turned around and applied to the search itself.

F76E's own note on the other half: `prefillLogitDigest` was nominated as
the independent second reader *because* it walks the whole buffer, and
walking the whole buffer is not the same property as reporting what it
saw. It had the extent and discarded it one line later.

## The clamp gate, and a gate I specified stricter than the oracle

C treats the SwiGLU clamp as model metadata and validates it on load:
`validate_swiglu_clamp_metadata` (`ds4.c:6474`) reads the per-layer array
`deepseek4.swiglu_clamp_exp` and dies if any of the first `n_layer`
entries disagrees with the compiled shape. We carried the value as a bare
literal and never looked at the file, so a disagreeing model would have
been miscomputed rather than rejected. That is a real gap and it is now
closed.

The value is not a constant in the reference. It is a per-shape field
(`ds4.c:956`), defaulting to 10.0f (`:408`), and `DS4_SHAPE_GLM52`
(`:712`) sets it to **0.0f — clamping disabled**. A shipped configuration,
not a hypothetical, which is the argument for the gate.

**The gate passes vacuously on our checkpoint.** 43 entries, all exactly
10.0. The doc comment says so; its value is entirely prospective. Recorded
here because a gate that cannot currently fail is not evidence that
anything is correct, and this campaign has twice mistaken one for the
other.

### Three corrections to my own framing, none of them cosmetic

I briefed a subagent on this and got three things wrong. It read the
reference instead of my summary of it, which is the only reason one of
them did not ship.

1. **"Hardcoded in 9 places" was a LINE count.** `grep -rno` gives 18
   kernel-side occurrences plus one Go literal: 19. I reported a line
   count as an occurrence count *in the same stretch of work whose whole
   subject was an undercounted census.* Third consecutive erratum about a
   number whose denominator I did not state.

2. **"Every entry equal to our constant" — C does not test equality.**
   `config_expect_f32` (`ds4.c:6513`) accepts
   `|got − expected| <= max(|expected|, 1) × 1e-6`. Implemented literally,
   my instruction would have made a model that stores 10.0 as float64 and
   narrows — or differs by one ulp — a **hard load failure where the
   reference loads fine**, and I would have written the divergence up as a
   parity finding.

   That deserves a name, because it is the inverse of the failure this
   ledger keeps recording. **A gate whose predicate is tighter than the
   reference it claims parity with fires on conforming inputs, and every
   firing reads as a defect found.** A fail-open gate is silent; this kind
   manufactures findings. Of the two, this is worse.

3. **The 0.0f preset is `DS4_SHAPE_GLM52` specifically**, and naming it is
   the point — see above.

Also missed: C validates only the first `n_layer` entries and ignores a
longer tail.

### What the gate does and does not cover

The comparison mirrors `config_expect_f32` rather than `==`. The literal
scan reads the directory rather than the build set, so a `.cu` file is
checked from either GOOS, and it states **both** denominators as fatal
conditions: below 200 scanned files, or below 18 found literals, it fails
rather than passing quietly.

What it does not cover, stated rather than implied: a future bare numeric
literal passed as the clamp argument of a Go call. Its rule keys on lines
that both call a min/max/clamp and name `gate` or `up`; the one Go call
site spans lines and names neither, and is covered today only by there
being exactly one and it now using the constant.

Controls fired, and re-fired independently by the lead rather than taken
from the subagent's report: the validator rejects a missing key, a
non-array, a short array, a non-float element, a single disagreeing layer,
and the GLM 5.2 zero; the scan fails on `cuda_probe.cu:1072` mutated
10.0f→9.0f **run from darwin against a linux-only file**, and passes again
on revert.

## The missing guard was correct parity, and the gap was in the reading

This ledger carried an open hardening item: *our decode SwiGLU lacks the
non-finite guard our D2R kernel has.* Two of our kernels disagreed, so one
of them was missing something. Reading the oracle inverts it.

C has **two** SwiGLU kernels and they disagree in exactly the same way:

| C kernel | guard |
|---|---|
| `swiglu_kernel` (`ds4_cuda.cu:12432`) — the plain/shared path | **none.** Clamp, silu, multiply |
| `moe_mmq_swiglu_weighted_clamp_kernel` (`:1240`) — the MoE mmq path | `if (!isfinite(g)) g = 0.0f;` and the same for `u` (`:1259-1260`) |

And the reason is visible three lines above the guard: the mmq kernel
walks per-expert slots and `if (w == 0.0f) { mid_out[gid] = 0.0f; return; }`
for unowned assignments — *"Never read those slots."* The accumulator in
that path can hold whatever was left there. The plain kernel has no such
slots, so it has no such guard.

Ours line up with C's, kernel for kernel:

| ours | guard | C counterpart |
|---|---|---|
| `cuda_moe_d2r.cu:2668-2669` | yes | mmq kernel ✓ |
| `cuda_q81.cu:194-198` (decode) | none | `swiglu_kernel` ✓ |

**So the item was never a gap.** Adding the guard to decode would have
moved us *away* from the reference, and the change would have passed every
gate in this package — the digest is unaffected on finite input, which is
all our workload produces. It would have been an invisible divergence
installed as a hardening improvement.

The shape is the campaign's usual one wearing new clothes: **an internal
inconsistency read as a defect without asking which side the oracle is
on.** Two of our kernels differed, and the inference "one is missing
something" skipped the step where you check whether the reference differs
too. It does, deliberately, for a documented reason.

### One real divergence, found on the way, and already covered

`cuda_q81.cu:194-198` hardcodes the clamp at `10.0f`. C's `swiglu_kernel`
takes `clamp` as a parameter and applies it only under
`if (clamp > 1.0e-6f)` — so on the GLM 5.2 shape, where the clamp is 0.0
and clamping is *disabled*, C would pass values through and we would clamp
them at 10. That is a genuine behavioural difference.

It is unreachable on our checkpoint, and it is now caught at load rather
than at runtime: the clamp gate committed in `ee60e647` reads
`deepseek4.swiglu_clamp_exp` and refuses a model whose metadata disagrees
with the compiled constant, with the GLM 5.2 zero as an explicit test
case. A gate written for a prospective reason turns out to guard a
concrete one, which is the argument for writing it before you can
demonstrate it firing.

## E10: the workspace contrast, designed before it was run

Written before any result was seen, which is the only thing that makes
the word pre-registered mean anything here. If the numbers below the fold
disagree with what this section predicts, this section stays as written.

### What is being asked

Whether pinning a test-owned cuBLAS workspace, in place of the
library-managed pool, changes the set of prefill digests a fixed arm
produces. This is the arm that E9's retraction left standing as the one
input that plausibly drifts and that we can actually hold fixed.

### The design, and why it is shaped this way

**The factor is process-scoped.** cuBLAS exposes no call that returns the
library-managed pool once a buffer has been handed over, so `default` and
`pinned` cannot be varied inside one process. That is a property of the
library, not a convenience, and it forces every other choice below.

**Eight single processes, ABBA × 2, inside ONE `gpu-lock` window:**
`default, pinned, pinned, default, default, pinned, pinned, default`,
four rounds each, 8 draws per process, 64 draws total, 32 per condition.

- *ABBA and not alternation.* A strict alternation makes arm and period-2
  position the same column, so an order effect would be
  indistinguishable from the factor. This campaign already retired an 8%
  effect that vanished under ABBA.
- *One lock window.* Two back-to-back blocks would confound the factor
  with everything that drifts between them — which is exactly the
  criticism that took E9 from p=0.0002 to p=0.0935.
- *Interleaved at process granularity*, which is the finest available
  given the paragraph above. Finer is better and this is the floor.

**The covariate is recorded per draw**, `availGiB` sampled *before* the
prefill, since what bears on a workspace allocation is the state the
allocation is made under. A Wald-Wolfowitz runs z on that series prints
beside every table. Assessment permutes labels **within contiguous blocks
of ten draws**, the constant fixed at `cudaGEMMPermutationBlock = 10` and
derived from a mean run length of ≈3.6 over the 120-draw sweep.

### What a wrong answer looks like

Stated in advance so it cannot be discovered afterwards:

- **The baseline run preceding this one is a single condition.** 60 draws,
  `workspace=default` throughout, because I set it up wrong. It measures
  within-process digest stability for the `tf32` arm and it does **not**
  measure the workspace factor. It is reported as what it is.
- **This experiment is between-process on the factor and within-window on
  the clock.** A reader comparing a workspace effect against one of the
  handle arms is comparing two designs, not two magnitudes.
- **n = 32 per condition.** The harness prints `floorP`, the smallest
  one-sided p that n can produce even at a perfect zero. If the floor
  exceeds 0.05 the rate is a description and not a test, and it will say
  so itself rather than waiting for a reader to notice.
- **A null here is the expected outcome.** The baseline is already 60 of
  60 clean on one digest. If both conditions come back clean the
  experiment has bounded the effect, not found one, and the honest
  statement is a bound with its n — not "the workspace does not matter".
- **`availGiB` has been flat at 95.1 across every baseline draw.** If it
  stays flat the runs test returns *not defined* rather than a z of zero,
  by deliberate design, so a degenerate series announces itself instead of
  reporting perfect exchangeability. A flat covariate would mean the
  blocking is precautionary here and the pre-registration still binds.

### The one thing this cannot resolve

Nothing in it varies **layout**. If a non-finite draw appears and its
extent is partial rather than the full vocabulary, that is a fault this
design cannot have caused and cannot bound, and the harness prints a
`LAYOUT:` line rather than folding it into a rate. That would be a stop
condition, not a result.

## E11: C's dense envelope, transplanted and refuted, and what the transplant is not

Run by the `dense-envelope` subagent; every oracle citation below I read
myself at `ds4_cuda.cu` before accepting the report.

### The correction that matters: the envelope is GEMM-vs-GEMM, not fast-vs-slow

This ledger framed C's dense shape envelope as a fast path with a slow
fallback, and asked why 11 of our 15 dense sites fall outside it. That
framing is wrong, and the error is load-bearing.

`ds4_cuda.cu:14862-14863`, verbatim:

```
    if (aligned && n_tok >= 512u && out_dim >= 2048u &&
        in_dim <= 4096u && cuda_use_mmq()) {
```

My earlier transcription cited `:14862-14866`, which runs past the predicate
into the `getenv` and the call, and it dropped two conjuncts: `aligned` --
which is not a boolean but a *derived weight pointer* (`:14848-14853`), NULL
when the repacked aligned Q8_0 weights do not exist or the feature is off --
and `cuda_use_mmq()`, a runtime enable. Neither discriminates between our
sites, because `encodeDenseQ8D2R` already requires the same alignment of
every caller (`cuda_dense_d2r_linux.go:56` errors on `m%128 != 0 ||
k%1024 != 0`). So the 11-of-15 count stands. The framing does not.

Fourteen lines down, `ds4_cuda.cu:14901-14906`:

```
    if (n_tok > 1 && (in_dim % 256u) == 0 &&
        !g_q8_dequant_gemm_enabled && cuda_use_mmq()) {
        int rc = ds4_mmq_q8_0_dense(wptr, ...);
```

`ds4_mmq_q8_0_dense` is a batched MMQ **tensor-core** GEMM on the raw Q8_0
layout. All eleven of our envelope-failing shapes satisfy `n_tok > 1 &&
in_dim % 256 == 0`.

> **Correction (E11 audit; reference re-read by me at the same SHA).** This
> used to say "so in C all eleven land there" and "cuBLAS at `:14911` is
> reached only when MMQ is disabled." Both are stronger than the source.
> The MMQ branch also requires `!g_q8_dequant_gemm_enabled`, and **taking
> the branch is not landing in it**: only `rc == 0` returns, and a nonzero
> rc prints `ds4_mmq_q8_0_dense returned %d ... falling back` and falls
> through. The cuBLAS branch below it is guarded by `g_cublas_ready &&
> n_tok > 1` and nothing else, so **it is reachable with MMQ enabled** --
> via a declined or failed MMQ call, or via the dequant-GEMM flag. Meeting
> the shape conjuncts is necessary, not sufficient, and nothing here
> demonstrates runtime engagement for any of the eleven. Read the paragraph
> as source topology. Its conclusion survives unchanged: the Go experiment
> does not implement C's raw-layout alternative, so it is not a like-for-like
> routing comparison.

**C's envelope selects between two tensor-core GEMMs. Ours selects between a
GEMM and a CUDA-core matvec** (`ds4_q8_0_matvec_q8_1_batch`,
`cuda_q81_linux.go:126-144`, grid `((rows+7)/8, ceil(batch/32))`, which at
N=2048 re-reads the weight 64 times). We have no raw-layout MMQ dense GEMM.
"Give our gate C's envelope" therefore does not mean "match C"; it means
"route 11 sites off tensor cores." That is the only thing our tree can
express, and it is what was measured.

### The measurement

One process, artifacts loaded once, arm flipped per prefill. Three arms
chosen so the contrasts are disjoint by construction: BASE (ship, all 15 on
D2R), ATTNOUT (envelope in the attention-output block only), FULL (both
blocks). Sequence BASE ATTNOUT FULL FULL ATTNOUT BASE, block size 6
pre-registered as the minimal position-balanced sequence over three arms --
each arm's within-block positions sum to 7, so drift linear in position
cancels inside a block. 10 blocks, 60 draws, 20 per arm, 2 warmups
discarded.

The control is a **host submission count** -- `denseQ8D2RLaunches`,
incremented in `cuda_dense_d2r_linux.go` after the launch API returns --
predicted 645 / 258 / 172 before the run. It came back exactly 645 / 258 /
172 on every draw, so the arms demonstrably fired; an arm whose count does
not move would have had no interpretable timing. 645 = 15 sites x 43 layers,
which reproduces the census launch count independently. It is not a
device-maintained counter and not a CUPTI census: this ledger called it
"the kernel's own launch count", which it is not, and the distinction
matters wherever graph capture is in play.

The medians below are the **11th order statistic of 20**, which is what the
harness printed under that label until `8613b9c2`; conventionally they are
3314.15, 6758.0 and 6920.20. The difference is small against the effect and
moves nothing, and the values are left as published with the convention
named.

| arm | n | median ms (11th of 20) | sd | logits digest |
|---|---|---|---|---|
| BASE | 20 | 3314.9 | 15.1 | `f0e4094b28390eab` |
| ATTNOUT | 20 | 6758.3 | 18.9 | `f605f737ee56b623` |
| FULL | 20 | 6921.1 | 17.2 | `20beec010f5cbd0e` |

One stable digest per arm, three distinct: a second, independent witness
that the branches fired. All 60 draws produced the correct continuation
token 15342.

Within-block paired differences, positive = envelope slower:

| contrast | sites moved | mean | sign test |
|---|---|---|---|
| BASE -> ATTNOUT | 8x attnOutA + attnOutB | **+3439.9 ms** | 10/10 |
| ATTNOUT -> FULL | attnQA + attnKV | **+163.1 ms** | 10/10 |
| BASE -> FULL | all 11 | **+3603.0 ms** | 10/10 |

Leaving D2R costs the 9 attnOut sites ~21x (~170 ms -> ~3610 ms) and
attnQA+attnKV ~9x (19.45 -> ~183 ms).

**The envelope hurts, in aggregate, by a margin that needs no statistics.
Do not add it.** The lever is retired.

> **Correction (E11 audit).** This used to read "hurts at every one of the
> eleven sites." The experiment has three arms, and it moves nine
> attention-output sites together and then adds QA and KV together. Two
> positive group contrasts do not establish that each of eleven sites is
> individually worse: one could improve while the rest more than offset it.
> The ~21x and ~9x figures above are likewise estimates combining the older
> captured kernel totals with this run's whole-prefill wall clock -- not
> measured per-site factors, as the *What this cannot resolve* list already
> concedes for the split it was asked about. Rejecting the transplant at
> N=2048 needs only the aggregate, which is what was measured.

Note the direction was predicted and the magnitude was not: the registered
prediction was +100..250 ms for BASE -> ATTNOUT and the result was +3440, a
factor of ~15 in the predicting party's own favour. A registered direction
that hits while its magnitude misses by 15x is a weak prediction that
happened to be pointed the right way, and it is recorded as such rather than
as a hit.

The `availGiB` covariate read 98.5 at the first draw and 98.5 at the last,
and the per-arm runs tests read BASE z=-0.22, ATTNOUT and FULL z=-1.17. The
pooled-series z of -2.49 is the three arm levels showing through a pooled
series, not drift, and is not evidence of anything.

> **Correction (E11 audit, session C8332C52; verified by me in the source).**
> This paragraph used to conclude from those numbers that "the
> pre-registered blocking was **unnecessary rather than merely satisfied** --
> this run had no drift to remove." Two defects.
>
> **The runs test never saw `availGiB`.** `runsZ` was fed `draws[i].ms` and
> per-arm `d.ms`; the harness printed only first and last for the covariate.
> Every z quoted above describes a *latency* series. Citing it as evidence
> about memory headroom is the same error as reading a `%.1f` display as the
> series (errata 21.191), one level worse: not a coarsened view of the right
> variable but the wrong variable. Fixed at `d965820e`, which runs the
> statistic on the covariate and prints its min and max -- and the ten-block
> log does contain a 98.6 among otherwise-98.5 readings, so the path between
> the equal endpoints is not flat.
>
> **A non-rejection is not a demonstration.** Even correctly labelled, a
> runs test that fails to reject on 20 samples does not establish
> exchangeability or the absence of drift. And the six-position sequence
> balances drift *linear in draw position*; it does not balance nonlinear
> drift, warm-state carryover, or drift linear in wall time when the arms
> differ in duration by a factor of two, which these do.
>
> What is supportable: the blocked design was kept, and the covariate was
> observed stable to the resolution recorded. Not that blocking was shown
> unnecessary. Also worth stating, since it was not: `availGiB` is sampled
> **after** each draw's synchronize, so it describes the state a draw left,
> not the state it ran under.

### What this cannot resolve

- It cannot say whether C's envelope is right *for C*. The question that
  would matter -- is aligned-D2R the right kernel for those 11 shapes
  *versus a raw-layout MMQ GEMM* -- is unaskable here until someone ports
  `ds4_mmq_q8_0_dense`. This answers the transplant question only.
- One prompt shape. At n_tok=2048 the `n_tok >= 512` clause never binds, so
  the envelope is a two-way test on `out_dim` and `in_dim`, not the
  three-way shape test this ledger presented. **At a short prefill
  (n_tok < 512) it would reject all fifteen sites** -- a different regime,
  and nobody has drawn from it.
- Wall clock per prefill, not per-kernel. No CUPTI capture, so the +3440 ms
  is not split between attnOutA and attnOutB, nor between the matvec itself
  and the extra quantize launches the fallback adds.

### Two more corrections to this ledger, from the same reading

**The wave-quantization block counts were wrong.** `kDqMTile = kDqNTile =
128` (`cuda_dense_d2r.cu:18-19`), so at N=2048 attnOutA is 8x16 = **128**
blocks and gate/up is 16x16 = **256**, not the 136 and 272 this ledger
printed. Corrected in place above. The ratio is still exactly 2.00x, so the
"linear, no tail for fusion to recover" conclusion is unharmed -- but the
absolute numbers were not the kernel's, and a reader checking the argument
against the source would have found two numbers that appear nowhere in it.

**The dense table merges a failing class with a passing one.** The row
`attnOutB + down | 86 | 103.270` straddles the exact boundary the section
then asks about: attnOutB fails the envelope on `in_dim=8192`, `down`
passes. That is design constraint 3 violated inside the table that states
the denominator. A `blocks x K` model splits it ~82.6 / ~20.7 ms and a
MAC-share split gives ~82.6 / ~20.6 independently.

> **Correction (E11 audit, session C8332C52; reproduced independently by me
> with my own decoder).** This paragraph used to end "both are derived and
> neither is measured -- splitting it needs a new capture with a journal
> attached." **The journal was on local disk the whole time.**
> `~/tmp/claude-tmp/exn/worker-process-4082226-513005984782608-capture.journal.gz`,
> sha256 `9305cf64…438344f9`, named with that hash in the adjacent
> `manifest.json`. What I actually observed was that the capture **pprof**
> inflates to a protobuf with zero `{"kind":` records, and I read that as the
> bundle having no journal. *Inspecting one member of a bundle says nothing
> about the others* -- and this ledger's own *Reading a capture* section
> already says the records live in `*.journal.gz` and **not** the `.pprof`.
> I had written the rule and then reasoned past it.
>
> Measured, not derived:
>
> | site | launches | total ms | ms/launch |
> |---|---:|---:|---:|
> | attnOutB | 43 | **78.975054** | 1.836629 |
> | shared `down` | 43 | **24.295178** | 0.565004 |
> | combined | 86 | 103.270232 | — |
>
> The derived ~82.6 / ~20.7 was wrong in both directions by ~4 ms. The
> attribution is still **order-based**: the journal carries no source or
> CUBIN, the two sites share `gridX=512`, and only launch order separates
> them -- the same method already used for QA against attnOutA, tightened
> by requiring the whole 15-launch grid pattern
> `128,64,4096,128x8,512,256,256,512` in **all 43 layers**, which holds with
> zero deviations. The assignment's direction is independently corroborated:
> swapping it would make `down` four times attnOutB, against the arithmetic.
> My own decode: 138686 journal records, 645 `ds4_dense_q8_d2r`, per-site
> totals summing to 368.673745 ms -- the published row, to six decimals.
> Reproduce with no GPU:
> `python3 ~/tmp/agent-collab/ds4-min/20260919-C8332C52-e11-split.py`.

### QB: most of it was never a mystery

This ledger recorded QB's 111.684 ms over 43 launches, 30.3% of the dense
row, as unexplained. Most of it explains itself: QB is 68.72 G of the
270.59 G MACs per layer, **25.40% of the row's arithmetic against 30.30% of
its time**, and every other group lines up similarly (attnOutB+down 31.75%
MAC / 28.01% time; attnOutA 25.40 / 24.41; gate+up 12.70 / 12.01; attnQA
3.175 / 3.18; attnKV 1.587 / 2.10). Calling the whole 111.684 unexplained
was reading a big number as an anomaly without dividing by the work.

The residual is real and much smaller. Against a `blocks x K` model
calibrated on attnOutA (0.26165 ms per 128 blocks x 4096):

| group | blocks | K | predicted | measured | delta |
|---|---|---|---|---|---|
| attnQA | 128 | 4096 | 0.2617 | 0.2724 | +4.1% |
| gate/up | 256 | 4096 | 0.5233 | 0.5147 | -1.7% |
| attnOutB | 512 | 8192 | 2.0932 | 1.8366 | -12.26% |
| shared `down` | 512 | 2048 | 0.5233 | 0.5650 | **+7.97%** |
| attnKV | 64 | 4096 | 0.1308 | 0.1800 | +37.6% |
| **attnQB** | **4096** | **1024** | **2.0932** | **2.5973** | **+24.08%** |

The attnOutB and `down` rows are split here for the first time; they were
one merged `2.6165 / 2.4016 / -8.2%` row until the journal was found.

attnKV's excess is the known 64-block underfill. QB's is 0.504 ms/launch =
**21.7 ms over 43 launches, 5.9% of the dense row.** That is the number
worth a name; 111.684 was not.

Two candidate mechanisms, collinear and not separated. QB is the only K=1024
site, so `k128_iters = 8` where every other site runs 16-64
(`cuda_dense_d2r.cu:240`); and it has by far the largest ratio of fp32 output
(M*N*4 = 268.4 MB/launch) to Q8_0 weight footprint, which at the measured
2.597 ms is 103 GB/s of store traffic on a ~273 GB/s part. **Derived from
published shapes and the census; no DRAM counter was read.**

> **Three corrections to the paragraph above (E11 audit, arithmetic
> re-derived by me).**
>
> *It is not the only site whose output exceeds its weight read.* With
> N=2048, the ratio is 8192/K before scales, so **every** site with K<8192
> clears one. Including the Q8_0 scale planes (34 bytes per 32 weights):
> QA and each attnOutA group 8.389/4.456 = 1.88; KV 4.194/2.228 = 1.88;
> gate and up 16.777/8.913 = 1.88; `down` 33.554/8.913 = 3.77; QB
> 268.435/35.652 = **7.53**. QB is the largest, not the only. The quoted
> 33.5 MB also omits the scales; with them it is 35.65 MB.
>
> *A footprint is not traffic.* M*K bytes is the matrix's size on device,
> not what the kernel reads: N-tiling reloads it logically, and what that
> costs in DRAM depends on cache reuse and scheduling. Likewise 103 GB/s is
> output bytes over elapsed duration -- an **effective store rate**, not a
> measured DRAM counter and not evidence of saturation.
>
> *"Both are the same statement" was too strong.* Mainloop length, epilogue
> cost, store volume, cache behaviour and the groupM schedule are distinct
> properties that happen to be collinear in K across these shapes. No
> intervention here equates them. And `down`, named above as the site that
> would break the tie, is now **measured** (+7.97%) and does not break it:
> it differs from QB in M, output size and grid as well as K, so its
> residual is consistent with either mechanism. 21.676 ms is the residual
> **from this specified model** -- not a measured optimization budget, and
> not a mechanism that has been explained.

#### The split buys a paired comparison the merged row was hiding

Once `attnOutB` is a row of its own, it turns out to sit at **exactly** QB's
arithmetic:

| | M | K | MACs/launch | blocks | blocks x K | output | measured |
|---|---:|---:|---:|---:|---:|---:|---:|
| QB | 32768 | 1024 | 68.719476736 G | 4096 | 4194304 | 268.435 MB | **2.597309 ms** |
| attnOutB | 4096 | 8192 | 68.719476736 G | 512 | 4194304 | 33.554 MB | **1.836629 ms** |

Not approximately: the same product to the digit, and the same `blocks x K`
to the digit. Every predictor in the model above is held exactly equal and
the two differ only in how the work is *shaped* -- 4096 blocks of K=1024
against 512 blocks of K=8192. QB is **1.414x** slower, **0.7607 ms per
launch, 32.71 ms over 43 launches, 8.87% of the dense row.**

That is a stronger statement than the +24.08% residual above, and a larger
one. The model's calibration point was `attnOutA`, and `attnOutB` comes in
12.26% *under* that model, so calibrating on it folded part of this gap into
the baseline. **The paired excess is 32.7 ms, not 21.7 ms.** Both are
derived from the same 645 records; the difference is which comparison is
made, and this one needs no calibration constant at all.

#### Why no choice of site can separate the two mechanisms

The two candidates were a fixed per-block epilogue and output store volume.
They are not merely collinear in this dataset -- **they are the same
variable by construction.** The kernel's output tile is
`kDqMTile x kDqNTile = 128 x 128` fp32 (`cuda_dense_d2r.cu:18-19`), so every
block stores exactly 128*128*4 = **65536 bytes**, and total store volume is
`blocks x 65536` for every shape this kernel can be called at. QB's 4096
blocks and its 268.4 MB are one fact, not two. Choosing a different call
site cannot break the tie, because no call site can change bytes-per-block.

Separating them needs the *tile geometry* varied -- a build with a different
`kDqNTile`, holding the shape fixed -- which is a code change and an
experiment nobody has run. The ledger previously said this needed "an
intervention, not another capture", which was right without saying why.

A two-point fit does hint at the answer without settling it. `down` and
`attnOutB` share M=4096 and therefore 512 blocks, differing only in K, so
`t = a + bK` at fixed block count is exactly determined: **a = 0.1411 ms
fixed per launch, 0.2756 us per block**, b = 2.0697e-4 ms per unit K.
Extrapolating per-block to other block counts predicts QB at 2.825 ms
against 2.597 measured (**+8.75%**) and attnOutA at 0.2472 against 0.2617
(**-5.52%**) -- much closer than the pure `blocks x K` model's +24.08%, so a
fixed per-block cost is doing real work in the data. **But two points fitted
and then extrapolated eightfold in block count is not a measurement**, the
fit reproduces its own two calibration points by construction, and it cannot
say whether that per-block constant is the epilogue, the stores, or both.
Recorded as a hint, not a result.

## E12, pre-registered: the only geometry that separates QB's two mechanisms

**Written before anything is built or run.** Nothing below has been
measured; when it is, the result goes under this heading with this text
left standing.

E11 established that QB's excess -- 32.71 ms against `attnOutB` at identical
arithmetic -- has two candidate mechanisms which are *the same variable*
under the shipped kernel: the output tile is `kDqMTile x kDqNTile = 128 x
128` fp32, so every block stores exactly 65536 bytes and store volume is
block count times a constant. The obvious next move is to vary the tile.

**It does not work, and the arithmetic says so before the build.** Per
launch at QB's shape (M=32768, N=2048, K=1024), counting each panel once per
tile that re-reads it:

| MTile x NTile | blocks | weight re-read | activation re-read | stores |
|---|---:|---:|---:|---:|
| **128 x 128 (shipped)** | **4096** | **570.4 MB** | **570.4 MB** | **268.4 MB** |
| 128 x 64 | 8192 | 1140.9 MB | 570.4 MB | 268.4 MB |
| 128 x 256 | 2048 | 285.2 MB | 570.4 MB | 268.4 MB |
| 64 x 128 | 8192 | 570.4 MB | 1140.9 MB | 268.4 MB |
| 256 x 128 | 2048 | 570.4 MB | 285.2 MB | 268.4 MB |

Stores are invariant -- they are `M*N*4` whatever the tiling -- and **every
tiling that doubles the block count also doubles one input stream.** The
confound is not small beside the effect: the per-block fit predicts +1.13 ms
per launch for the extra 4096 blocks at `128x64`, while the extra 570 MB of
weight re-read is worth several ms at any plausible effective bandwidth. A
tile sweep would measure the sum and attribute it to whichever term the
experimenter had in mind. **This is the fourth lever this campaign has
retired before running it, and the first retired by arithmetic alone.**

### What does separate them

Hold the geometry fixed -- same blocks, same weight traffic, same activation
traffic -- and change *only* the bytes each block stores, by writing the
output tile as fp16 instead of fp32:

| arm | blocks | weight | activation | stores |
|---|---:|---:|---:|---:|
| STORE32 (ship) | 4096 | 570.4 MB | 570.4 MB | 268.4 MB |
| STORE16 (probe) | 4096 | 570.4 MB | 570.4 MB | **134.2 MB** |

Only one term moves. **STORE16 produces numerically different output by
construction**: it is a mechanism probe, not a candidate change, the decode
digest does not apply to it, and that is declared here rather than
discovered later. It must never be a shipping path, and the arm has to be
unreachable unless explicitly selected.

### Registered predictions

- **P1.** If output store volume is the mechanism, STORE16 cuts QB's kernel
  time by roughly half of whatever fraction stores contribute. Concretely:
  if stores were the *whole* 0.7607 ms/launch excess over `attnOutB`, expect
  about **-0.38 ms/launch**, QB falling from 2.597 to about **2.22 ms**.
- **P2.** If a fixed per-block epilogue independent of store width is the
  mechanism, expect **|delta| < 0.05 ms/launch** -- the same 4096 blocks,
  the same epilogue count, only narrower stores.
- **P3.** The two are not exclusive and the likely answer is intermediate.
  A result between -0.05 and -0.38 ms **apportions** the excess rather than
  choosing a winner, and will be reported as an apportionment.
- **P4.** `attnOutB`, at 512 blocks and 33.55 MB of stores, must move
  **much less in absolute ms** than QB under the same arm. If it moves by a
  similar absolute amount, the effect is not stores and not per-block
  either, and both hypotheses are wrong.
- **P5 (control).** Weight and activation traffic are unchanged by
  construction, so `attnOutA` -- 128 blocks, 8.39 MB of stores -- should
  move by **under 0.02 ms/launch**. A larger move means the arm changed
  something it was not supposed to.

P4 is the one that can embarrass both hypotheses at once, and it is the
reason the probe reads every site rather than QB alone.

### Design constraints, registered

1. One process, artifacts loaded once, arm flipped per prefill, ABBA inside
   one `gpu-lock` window -- the E10/E11 shape.
2. Per-kernel attribution from the **journal**, not wall clock. E11's
   aggregate could not be split per site; this question is per site and the
   journal is the instrument. The decoder is `capture_journal.py` in this
   directory, which refuses to attribute any launch to a site unless the
   whole 15-launch grid pattern holds in every one of the 43 layers, and
   which reports the records it fails to parse rather than dropping them.
3. The digest gate is **suspended for STORE16 and only for STORE16**, and
   the STORE32 arm in the same window must still produce
   `f0e4094b28390eab`. An arm that cannot reproduce the shipped digest
   invalidates the window, not just itself.
4. Launch counts must be 645 on both arms. Different counts mean the arm
   changed routing, and the timings mean nothing.

### The probe's premise, retracted as an inference and then verified

Every prediction above assumes the arm changes **bytes per store** while
holding **stores per block** fixed. That assumption was written here as
though it were established. It was not: it was read off the source, where
the epilogue stores one `float` at a time, and never checked against the
machine code. If ptxas vectorized those stores -- `STG.E.128` over four
contiguous floats is exactly the shape it looks for -- then the fp16 arm
would halve the byte count *and* keep or change the instruction count in
ways the source does not show, and P1 and P2 would no longer be separable.
The agent that first reported the store width **withdrew the claim itself**,
on the grounds that no disassembly of this kernel existed anywhere in the
tree. The withdrawal is the reason the check exists.

Verified 2026-09-19 against artifacts, not source:

| fact | value |
|---|---|
| stores in `ds4_dense_q8_d2r` epilogue | **32 x `STG.E`** (32-bit scalar) |
| wider global stores, whole kernel | **0** `STG.E.64`, **0** `STG.E.128` |
| store form at `cuda_dense_d2r.cu:262` | `STG.E desc[UR4][R22.64], R3`, address via `LEA ... 0x2` |
| geometry cross-check | 32 x 512 threads x 4 B = 65536 B = 128 x 128 x 4 |
| fp16 arm, **compiled not predicted** | **32 x `STG.E.U16`** -- same count, half the width |
| control arm vs shipped kernel | 1088 instructions, encodings **byte-identical**, REG:128 both |

The last row is what makes this an A/B rather than two builds: adding the
`_store16` sibling entry point leaves the default kernel's machine code
bit-identical, so the STORE32 arm *is* the shipped kernel and the comparison
moves one thing. Artifacts and a `regen.sh` are at
`dgx:/home/tmc/tmp/claude-d2r-sass-20260919/`; the compile driver reproduces
`cudaMathFiles`/`cudaSourceChunk`/`cudaMathCompilationSource` and spells the
flags out (`--gpu-architecture=sm_121 --generate-line-info
--extra-device-vectorization --include-path=/usr/local/cuda/include`) rather
than inferring them, so the artifact states its own input. No `gpu-lock` was
taken and no compute ran: NVRTC and `nvdisasm` are offline.

**Constraint 5, added by this check.** The arm must be read in **DRAM write
bytes**, not L2 sector or transaction counts. A warp's store covers four
columns x eight contiguous rows -- one aligned 32 B sector per column at
fp32, half a sector at fp16 -- so a request-count metric can stay flat while
the bytes genuinely halve, and would report P2 (fixed per-block epilogue)
for a probe that worked exactly as designed.

### What this cannot resolve, registered in advance

It cannot say whether the epilogue's fixed cost is address arithmetic, the
scale broadcast, or the store instructions themselves -- only whether it
scales with the *bytes* stored. And it says nothing about whether QB's shape
should exist at all: a kernel that did not produce a 32768-row output would
not have this cost, and that is a model-structure question this campaign has
no standing to ask.

### E12 runtime result: STORE16 is materially faster, but the probe is not a full-model arm

The corrected standalone probe ran on `spark-31db` under `gpu-lock` in an
isolated scratch worktree, with fixed synthetic QB inputs (`M=32768, N=2048,
K=1024`), ten ABBA blocks and 40 timed draws. The STORE32 control retained one
digest on all 20 draws, both arms fired exactly 20 launches after warmups were
excluded, every draw rechecked identical input hashes, and the isolated STORE16
buffer passed host float32-to-float16-to-float32 validation against STORE32.

Device-event timings were:

| arm | n | mean ms | median ms | SD ms |
|---|---:|---:|---:|---:|
| STORE32 | 20 | 2.63275 | 2.552 | 0.29330 |
| STORE16 | 20 | 1.97395 | 1.994 | 0.11156 |

Aligned STORE32-minus-STORE16 differences were positive on all 20 draws, with
mean `0.65880 ms` and median `0.63100 ms`. This is **measured** evidence that
the isolated destination-width/epilogue path materially affects QB time. It is
not a DRAM-bandwidth measurement: physical DRAM write bytes are **UNREAD**, and
the half conversion itself is an additional instruction change. It is also not
a full-model speedup or a shipping candidate; downstream consumers were not
run with the half buffer. Raw log and scope are in
`20260921-C8332C52-e12-store16-runtime-report.md`.

## Three gates I built against this campaign's own defect, all passing vacuously

Found by adversarial audits from session C8332C52, which had no stake in
these claims being true. All three are gates this ledger presented as
protective. Each was verified by me and each control was fired before its
fix landed. The third arrived a day later, from the same session's E11
pass, after this section had already been written -- which is the part of
it worth noticing.

**The clamp scanner could not see a sign.** It stripped a leading minus
before comparing, so `fmaxf(up[i], -10.0f)` and `fmaxf(up[i], 10.0f)` were
the same literal to it. The second collapses the clamp interval [-10, 10] to
the single point 10. No evasion of the count or the spelling was needed --
the gate was blind to half of every bound it checked, including on the line
it was written to guard. Fixed at `658e29fd` by judging each literal against
the call it is an argument to. Control: sign-flipping `cuda_q81.cu:197` now
names the file, the line, which bound, and both values.

**The extent scanner is a spelling detector, and the thing it guards had no
behavioural test.** `TestNonFiniteChecksReportExtent` matches two old message
formats, so it catches a re-introduced copy of the old check and cannot catch
the shared walk being weakened underneath it. That much was a known limit.
What was not known: **no test exercised `nonFiniteExtent` behaviourally at
all.** All ten call sites ask only `bad()`, which needs presence and not
extent. Making the walk return at its first non-finite value broke nothing in
the suite.

So the property this ledger spent three sections establishing -- that a
wholly poisoned vector and a single bad element at the front must not report
the same thing -- was protected by a test that checks how the old bug was
spelled. Fixed at `b54d3e92`. Control: under the early-return mutant a front
element and a poisoned vector both print `0-0/1/64`, the new fixtures fail,
**and the spelling scanner stays green.** That last clause is the finding.

It is also the reason this section's ordering is misleading if read as a
ranking. The scanner is named first because it is where the defect was, not
because it protects anything: after `b54d3e92` the guarantee is carried
entirely by the six behavioural fixtures, which assert the extent struct,
its rendering, and `bad()` over named shapes. The scanner was left in place
because a regex over old message formats is cheap and catches one real
regression -- someone pasting the old check back -- but it should not be
counted a second time. One instrument here has teeth; the campaign wrote
about the other one for three sections.

**The envelope A/B reported bad outcomes without refusing them.** Its
`run()` classifies each draw as OK, WRONGTOKEN, NONFINITE, COUNT or ERROR;
the harness cross-tabulated the classes by arm, printed them, and excluded
non-OK draws from the timings they would have contaminated. **No
`t.Error`, `t.Fatal` or `t.Fail` was reached by any of them.** The only
failing paths were setup and cleanup. So a run in which the launch-count
control failed on *every draw* -- the control this ledger leans on to say
"the arms demonstrably fired" -- recorded COUNT sixty times, printed a
tidy cross-tab, and exited 0. This ledger cited that exit status as
protection. Reporting a bad outcome and refusing it are different acts and
the test did only the first. Fixed at `f67a6e70`.

Control, on device under the GPU lock, one block, both arms in one window:
unmutated **PASS rc=0**; BASE's expected count changed 645 -> 644, both BASE
draws become COUNT and the test **fails rc=1** on both new paths -- *2 of 6
draws are not OK, budget 0* and *arm BASE has no OK draws; its contrasts are
empty, not null*. The recorded 60-draw run is not withdrawn: all 60 of its
rows say OK, and that is a fact about the draws rather than about the exit
status. What is withdrawn is treating the exit status as evidence.

The same fix records a second thing the classes were hiding: they are
**priority labels, not independent observations.** COUNT outranked
NONFINITE, so a COUNT row also silently meant nobody had recorded whether
that draw's logits were finite. A joint census needs every observable, and
the ones an ERROR draw never reaches have to be explicitly *unobserved*
rather than absent. All three checks now run and all three answers are
logged per draw.

The shape is one this ledger has now hit at every level. A defect was found;
the fix had the defect; the census of the fix had no denominator; and the
test asserting the census had teeth only against the spelling of the
original. Each layer looked like the end of the regress from inside it.

### Carried open from the same audit, not yet verified by me

- ~~The merged corruption series mixes a third outcome selectively.~~
  **Settled, and against my framing of it.** The series is a *consistent*
  three-category composite (non-finite OR wrong digest OR invalid UTF-8,
  excluding graph-capture aborts from the numerator while keeping every
  attempt in the denominator), so the trend `p=0.0077` does hold for the
  series it is printed beside -- a reconstruction over scores 0/2/8 returns
  0.0077549. What was actually wrong was narrower and worse: the E8 Fisher
  figure was computed on the two-class union (8) and printed beside the
  three-category table (9). Corrected above to 0.0061. The two-class union's
  own trend, for the record, is 0.0194. Manifest of all 120 rows, cross-checked
  against stderr, at
  `~/tmp/agent-collab/ds4-min/20260919-C8332C52-b1-outcomes.tsv`.
- The non-finite p=0.0010 is computed **unblocked** in `e9.py` while the
  wrong-digest claim it is compared against is blocked. Demoting one and not
  the other on a shared covariate is not established either way.
- `ds4_cuda.cu:18882` (GEMM) and `:18924` (quantize) are opposite branches of
  an if/else, not two stages of one path, so this ledger's description of C's
  batching combines mutually exclusive branches. **Verified by me** at the
  same SHA: `if (out_a_f16)` at `:18863`, `} else {` at `:18911`. Corrected
  in place; see *The dense row*.
- `ds4.c:6513` is a blank line; `config_expect_f32` begins at `:6514`.

## The oracle trace fails, and my argument about why was wrong

Reported 2026-09-19 by session C8332C52: `TestCUDAOracleTrace` fails on
current source. Every full-logit state mismatches the retained C reference --
on the order of 129240 of 129280 entries per state, max absolute difference
2.81 to 7.18 -- while the teacher-forced token *choices* largely agree.

**My advice was that this shape indicts the comparison rather than the
kernel**, on the reasoning that real numerical drift concentrates in a few
outputs and that a uniform, total violation is what a wrong reference looks
like. I recommended treating provenance as the leading hypothesis rather
than as a caveat. C8332C52 asked me to withdraw not just the prediction but
the shape argument itself, and gave two counterexamples:

1. **Error propagates globally through this architecture by construction.**
   A perturbation anywhere upstream passes through dense projections and
   normalization, which mix every channel into every output. Whole-vocabulary
   violation is the *expected* footprint of a single upstream defect, not
   evidence against one.
2. **Argmax agreement survives arbitrarily large error.** Add a constant to
   every logit: error unbounded, argmax identical at every step. Rank-
   preserving error is a large class, and token agreement excludes almost
   nothing within it.

Both are correct and the stronger claim is withdrawn. What survives is only
the weak, ordering form: provenance is cheap to check and worth checking
first. It was never an argument that a kernel defect could not produce this.

**The failure mode in my own reasoning is one this ledger already names.**
I reasoned from the *shape* of a residual to its cause without asking what
shapes the mechanism I was excluding can actually produce. That is the same
error as reading an exact zero as a dead instrument -- a signature argument
stated as an impossibility, when the signature has more than one preimage.

### What the diagnostics then established

Each of the three stacked differences I had flagged was eliminated
separately, and none of them was the cause.

| difference | disposition | evidence |
|---|---|---|
| context 128 vs 1024 | **excluded** | ABCCBA, six draws, one lock. ctx128 and ctx1024 batched are *bit-identical* at all ten states; within-arm repeats bit-identical too. |
| model weight bytes | **excluded** | both 87 GB files hash `ca22ae2f...61c0`. Different inodes, same bytes. |
| retained-observer provenance | **excluded** | the pinned C observer, rerun on the original model path and prompt IDs, reproduced historical `state-00/01/02/04/08.json` **byte-for-byte**, plus `input.json`, `selected.jsonl`, `decoded.bin`. |

So the reference is pinned and reproducible, and the discrepancy is a
**repeatable current-Go-versus-pinned-C gap** rather than drift in a
historical artifact. That is a sharper statement than the campaign had
before, and it is explicitly *not* a proven regression: it does not identify
which contract differs, and it does not establish parity against today's C
revision, which is a different build.

Scalar prefill changes the outputs (prefill hash `a4d6fc6c...` against the
batched `6c052b3d...`, max pairwise logit difference 8.080355) and still
misses the reference at all five sparse states. Host launch counts are 645
per prefill batched and **0** scalar, exactly repeated -- routing, as
expected, with no envelope enabled.

### A token-agreement claim, overgeneralized and self-corrected

C8332C52's first report described all ten teacher-forced tokens as agreeing.
On inspection of the raw logs that holds **only for the batched arms**: every
**Go scalar** draw chooses **8233 at step 5**, with the oracle's 1694 at rank
2 and a gap of 0.114128. The failure was in every raw log from the start; the
summary generalized across arms. Corrected at the source with a log citation,
and corrected here: this ledger's earlier "the tokens agree" framing was
unscoped and inherited the same error.

> **And my reading of the correction was wrong in a way that inverted it.**
> I proposed that step 5 be *quarantined from the parity claim*, on the
> reasoning that the divergence appeared in "both prior C draws" and so was
> reproduced by the reference implementation under a scalar schedule --
> making it a property of the prefill schedule rather than a Go-versus-C
> gap. **Arm C of the ABCCBA design is Go ctx1024/scalar, not the C
> implementation.** I read an arm label as an implementation name, in a
> design where both namespaces are single letters. The pinned C observer's
> `input.json` specifies `prefill_chunk=1` -- it *is* scalar -- and it
> selects 1694. So C scalar gives 1694 and Go scalar gives 8233 at matched
> schedule, matched context and matched weight bytes. Step 5 is not a
> schedule artifact to be quarantined; it is the **sharpest matched
> discriminator the campaign has**, and my advice would have discarded it.

### The localization, four boundaries deep

All at layer 0, position 0, scalar prefill, ctx1024 -- and all **byte-identical**
between the pinned C observer and Go, with max error, mean error, RMSE and
violations all exactly zero:

| stage | float32 elements | SHA256, both implementations |
|---|---:|---|
| `hc_attn_pre` | 4096 | `23e61898...a2b2` |
| `attn_norm` | 4096 | `74c4d1c7...9044` |
| `q_lora` | 1024 | `51b57d07...defb` |
| `KVraw` | 512 | `033f1813...ac2049` |

The gates fire before each result is read: C dump-off and dump-on both
reproduce the five historical states byte-for-byte, Go dump-off/on/mutant
reproduce all ten scalar baseline states, self-comparison passes, truncated
and NaN dumps are refused, and a live host snapshot mutation (`q_lora[0] += 1`)
fails at exactly index 0 while `KVraw` and the logits stay unchanged.

The disclosure attached is the one that keeps this honest: the mutant
validates **extraction and comparator wiring, not device arithmetic fault
detection**, and C8332C52 has declined to inject a device-side mutant until
there is a concrete kernel hypothesis to aim it at rather than one inferred
from a host mutation. Four boundaries are excluded at one position of one
layer. Nothing is excluded at later positions, later layers, or other
operations, and no kernel attribution follows.

### First localization, with its gates fired

Layer 0, position 0: `hc_attn_pre` and `attn_norm` are **byte-identical**
between C and Go -- 0 violations of 4096, max and RMSE exactly 0. The
instrument was validated before the result was read: C dump OFF and ON both
reproduce all five historical states byte-for-byte, Go dump OFF/ON/mutant all
reproduce the ten scalar baseline states, the comparator's positive control
passes, truncated and NaN inputs are refused, and a live host-snapshot mutant
(+1 at index 0) fails at exactly index 0 while every other endpoint is
unchanged. The disclosure attached to it is the right one: this validates the
snapshot and comparator wiring, **not** the ability to detect a device
arithmetic mutation, and it is one position of one layer.

Artifacts: `dgx:/home/tmc/tmp/ds4-C8332C52-baseline-20260919/`, with
`context-run.log`, `stage-run.log`, the six `context-*` and five `stage-*`
directories, and `stage-compare.py`. No production or reference-tree writes.

### The first boundary that is not byte-identical, and why it is not yet a finding

Layer 0, position 0, `go-scalar-ctx1024` against `c-pinned-scalar`:

| boundary | elements | max abs error | tolerance violations |
|---|---:|---:|---:|
| `q_lora_norm` | 1024 | **0** | 0/1024 |
| `KVnorm` | 512 | **4.76837158203125e-7** | 0/512 |

`KVnorm` is the first stage in six boundaries to differ at all. Mean absolute
error 2.3798e-8, RMSE 4.2186e-8, largest difference at index 474. The
notebook's registered exact-equality prediction is **falsified** for KV and
holds for `q_lora_norm`.

**The magnitude is consistent with rounding, and that is all it is.**
4.76837158203125e-7 is exactly `2**-21`, and at index 474 the raw values are
C `4.197505474090576` against Go `4.197504997253418` -- one ULP at that
element. I described this as "the smallest representable disagreement at that
scale" and read it as the signature of a rounding or summation order
difference. **Corrected by C8332C52 from the raw values: 344 of 512 elements
differ, with a maximum bit-distance of 2 ULP.** It is a broad low-order
difference, not a single-element one, and my phrasing implied a minimality the
data does not show. Rounding-order remains *consistent* with it; the magnitude
is not proof of the same computation, and not proof that fusion is the cause.

**And that is exactly what the instrument could produce on its own.** The C
`KVnorm` observer bypasses a possible fused normalization/RoPE route, and the
same pre-RoPE intermediate has not been captured under both C fusion routes.
The difference is therefore **UNATTRIBUTED between reference routing and
implementation**, and C8332C52 recorded that disposition rather than naming
`encodeRMSPair`. Endpoint equality is necessary and not sufficient here: it
proves the unfused path reaches the same logits, not that the extracted
intermediate is the one the fused path computes.

This ledger had the hazard in writing before the window opened, which is the
only reason the result arrives with its caveat attached rather than as a
localization that a later control would have had to withdraw.

**A sufficiency test would bound it without resolving the ambiguity** --
inject the observed difference into Go's KVnorm and see whether the endpoint
moves by anything like the residual's 4.57 to 7.18 max abs, or flips step 5.
I wrote that a negative result would exclude `KVnorm` as a cause "however it
is attributed." **That overclaims a negative intervention, and C8332C52
scoped it:** a null result excludes only *that injected delta, at that
position*, as a sufficient explanation. It cannot exclude a normalization
difference that recurs across positions and layers and compounds, nor one
that only matters in interaction with another difference. The test is worth
less than I said, and its value is a bound rather than an exclusion. **We cannot run it:** the validated mutant is an
*observing* one -- it alters the extracted snapshot only, leaving KV and every
model endpoint equal, which is what makes it a wiring control. A sufficiency
test needs an *injecting* mutant that writes back into the live buffer before
consumption, and that is a new capability rather than another dump.

**Do not infer that a sub-tolerance error cannot amplify downstream.** It is
recorded here because it is the obvious wrong inference from a passing
tolerance, and the tolerance (abs 1e-4 + rel 1e-4) was never chosen to bound
propagation.

### A second withdrawal, of a premise inside the plan

The plan asserted that attention overwrites `kvCur`, which is what made an
immediate KV snapshot necessary. Withdrawn by C8332C52 against source:
`cuda_probe.cu:720` and `cuda_compressed.cu:501` take **const** KV and rotate
into a separate output buffer. The snapshot boundary remains valid and its
*necessity* was never established. The pre-registered plan was preserved
unchanged rather than edited to match, which is the right handling -- a plan
rewritten after its premise fails is no longer pre-registered.

Decode step 5 is produced by **position 14**, after five teacher-forced tokens
are accepted. The next boundary takes positions 0 and 14 together, before any
depth sweep, and avoids internal stages that force a routing fallback.

## Where my own errors concentrate, measured rather than felt

Worth stating plainly now that there are enough of them to see a shape. Across
this advisory phase the corrections landed on me five times: the estimand
behind the 1.414, the dense-phase copy filter, the oracle-failure shape
argument, the arm-label misreading, and the two overclaims about the KVnorm
ULP. Every one of them entered at the same place -- **the summary sentence**,
not the measurement.

None was an arithmetic mistake. In each case a correctly scoped result was
compressed into a stronger general statement while being written up: a ratio
of medians became "the paired slowdown"; a lower-bound filter became "inside
the phase"; "uniform error is what a wrong reference looks like" became "this
shape indicts the comparison"; a one-ULP maximum became "the smallest
representable disagreement"; a bounded null result became an exclusion
"however it is attributed."

That is a different failure from the ones this ledger catalogues in the
instruments. A vacuous gate is wrong when it runs. These were right when
computed and wrong when described, which means no rerun would have caught
them and no control would have fired. The only thing that caught them was a
reader recomputing from the raw artifact -- which is why every claim here
carries its artifact path, and why the peer's habit of re-deriving from raw
values rather than from my summary is the campaign's most load-bearing
practice.

The practical rule it yields: **a summary sentence is a claim and needs its
own check.** Before writing one, state the estimand, the bound, and the
population it ranges over, and confirm the sentence still holds with those
attached. Most of these would not have survived that.

## The first tolerance failure, and the boundary it sits on

Layer 0, scalar ctx1024, `c-pinned-scalar` against `go-scalar-ctx1024`, six
fresh processes in one lock window. Each boundary is 32768 float32.

| boundary | pos | byte equal | max abs error | RMSE | violations |
|---|---:|:---:|---:|---:|---:|
| `Qraw` | 0 | **yes** | 0 | 0 | 0/32768 |
| `Qcur` | 0 | no | 9.5367431640625e-7 | 4.0307e-8 | **0**/32768 |
| `kqv_back` | 0 | no | 0.001972675323486328 | 8.1115e-5 | **564**/32768 |
| `Qraw` | 14 | **yes** | 0 | 0 | 0/32768 |
| `Qcur` | 14 | no | 1.5735626220703125e-5 | 3.8915e-7 | **0**/32768 |
| `kqv_back` | 14 | no | 0.03794819116592407 | 5.8047e-4 | **6419**/32768 |

Three things are new here, and they are of different strengths.

**`Qraw` is byte-identical at both positions.** Query expansion is exact,
including at position 14 where fifteen positions of history have accumulated.
This is the strongest clean result the campaign has: not merely within
tolerance, but bit-for-bit, on 32768 elements, twice.

**`Qcur` differs bitwise and passes tolerance at both positions**, at
9.54e-7 and 1.57e-5. The pos0 figure is exactly `2**-20`.

**`kqv_back` is the first boundary in nine to fail tolerance**, and it fails
at both positions. From pos 0 to pos 14 the max absolute error grows 19.2x
and the violation count 11.4x. Growth with position is the first quantitative
signal of accumulation this campaign has produced.

### The failure's distribution flips between the two positions

Partitioned per head into the first 448 coordinates and the 64-element rotary
tail -- 28672 and 4096 elements respectively, and the two counts sum to the
totals above:

| pos | partition | violations | share of its own coords | max abs error |
|---:|---|---:|---:|---:|
| 0 | non-rotary (448/head) | 13 | 0.045% | 0.00019848346710205078 |
| 0 | **rotary tail (64/head)** | **551** | **13.45%** | 0.001972675323486328 |
| 14 | **non-rotary (448/head)** | **5560** | **19.39%** | 0.03794819116592407 |
| 14 | rotary tail (64/head) | 859 | 20.97% | 0.0046090781688690186 |

At position 0 the violations are **297x enriched per coordinate in the rotary
tail**, which holds 97.7% of them in 12.5% of the elements, and the tail also
carries the maximum. At position 14 the enrichment is **1.08x** -- essentially
uniform -- and the maximum has moved to the non-rotary bulk, where it is 8.2x
the rotary maximum. The concentration does not merely dilute with position; the
dominant partition changes.

C8332C52 records this as **descriptive, not a cause test**, and adds the
specific trap: rotary coordinates at position 0 can differ even though the
rotation angle is zero there, because the query and KV *representations* differ
going in. A partition that looks like it indicts RoPE at pos0 is equally
consistent with RoPE faithfully carrying an input difference.

### What this does not establish, recorded with it

**The KV and cache inputs to attention were not matched in this run.** The
difference can come from those inputs, from rounding, from score/softmax/
reduction arithmetic, from inverse RoPE, or from interactions among them.
This localizes the first observed tolerance failure to the attention-output
*boundary among this slice's observations*. It is not attribution to an
operation, and no full-model causal claim follows.

`kqv_back` also guards a TP head-split route in source, but the observer runs
TP=false, so that guard cannot switch this run's path -- checked rather than
assumed, as with every other routing question in this slice.

### My pre-registration was the wrong shape, and would have under-read this

Before the window I proposed reading the three boundaries against different
priors: that a bitwise difference at `kqv_back` would be *expected*, inherited
from the known `KVnorm` difference by propagation, and "would add nothing",
while a bitwise difference on the query path would be the informative one. I
marked the dependency as inferred from stage names and asked for it to be
checked. Two corrections came back:

1. **An upstream difference does not guarantee a downstream one.**
   Quantization can erase it. So "expected to differ" was never safe to
   pre-register as a prior in the first place.
2. **The query branch is not automatically independent** -- statistically or
   causally -- merely because it is a different branch.

And the outcome cuts against the framing from the other side: `kqv_back` did
not merely differ bitwise, it **failed tolerance and grew 19x with position**,
which is categorically more than propagation-neutral inheritance. A reader
primed by my prior would have been set up to discount the campaign's first
real localization signal. The half of the prediction that landed -- a
bitwise-clean query path being informative -- landed for reasons I had not
established.

This is the sixth item in one phase to enter through the summary sentence
rather than the measurement, and the first where the summary sentence was
written *before* the data rather than after. Pre-registration does not protect
a claim whose reasoning was never checked; it only fixes the time at which the
unchecked claim is made.

Raw evidence: `dgx:/home/tmc/tmp/ds4-C8332C52-baseline-20260919/attention-*`,
with per-stage SHA256 for both implementations at both positions. All controls
pass: three C arms reproduce all five historical states byte-for-byte, three Go
arms reproduce all ten scalar logit payloads, exact file/count/shape checks
hold, self-comparison passes, truncation/NaN/shape mismatch refuse, and the
host mutant is detected at exactly index 0 at both positions while leaving
`Qcur`, `kqv_back` and the logits unchanged. Device-arithmetic detection is
still not claimed.

## The raw KV cache is written by two paths that do not agree

Read by me in `ds4_cuda.cu`, read-only, while reviewing C8332C52's
matched-input replay plan. This is a source reading; no run.

> **Correction: not at the pinned SHA.** I wrote "at the pinned SHA" and the
> line numbers below are from the local reference checkout at **`11e8cad`**
> (2026-09-17, clean), not from the pinned observer build, where the same
> function sits at `:7072` and its caller at `:16711`. I cited a moving target
> as though it were the frozen one. The structural reading survives -- the
> pinned scalar path at `ds4.c:20851` calls quantize-plus-store or a fused
> quantize/store, so two writers with different non-rotary content do exist --
> but every line number here is revision-scoped and only `11e8cad` is what I
> actually read.

The replay reconstructs the fifteen history rows as `half(KVcur)`. Two of its
premises check out and one does not hold unconditionally.

**The buffer is right.** Attention reads `raw_kv` as `float *` -- `:7259`,
`:7305`, `:7725`, `:8159`, `:8235` among others -- not an fp8 cache. And both
writers of `raw` apply the same round trip: `:7196` stores
`__half2float(__float2half(xr[d]))`, and `store_raw_kv_batch_kernel` at
`:7252` stores `__half2float(__float2half(kv[...]))`. Half-rounding is the
correct reconstruction for the read path under either writer.

**But the two writers do not store the same values.** `fp8_kv_quantize_row`
at `:7146` **mutates `xr` in place** before the raw store, and only over the
first `n_nope = head_dim - n_rot` coordinates -- the **non-rotary** part. The
rotary tail is untouched. It is an e4m3fn quantize/dequantize round trip with
a per-64-block power-of-two scale (`:7164-7165`).

| writer | callers | non-rotary coords | rotary tail |
|---|---|---|---|
| `fp8_kv_quantize_store_rows_kernel` | `:16870`, `:16901` | `half(fp8_roundtrip(KV))` | `half(KV)` |
| `store_raw_kv_batch_kernel` | `:16909`, `:16917` | `half(KV)` | `half(KV)` |

So `half(KVcur)` is the right reconstruction only if every history row was
written by the same path as the captured `KVcur`. If any of the fifteen came
from the batch writer while `KVcur` is captured post-quantize, the
reconstruction is wrong **on the non-rotary coordinates specifically** -- and
a replay built on it would attribute its own reconstruction error to the
kernel. It is a source question and costs no GPU time to settle.

### And it predicts the partition we already measured

The fp8 round trip applies to non-rotary coordinates **only**. The `kqv_back`
failure at position 14 concentrated in exactly those coordinates: 5560
violations against the rotary tail's 859, with a maximum 8.2x the tail's.
Position 0, where far less history has been stored, was rotary-enriched
instead.

If the Go side omits that round trip, or applies it with a different scale or
clamp, the resulting error lands on precisely the coordinates that showed it.

**Falsified, on the same day, by reading the Go side I had not read.** Go's raw
prepare explicitly FP8-rounds the first 448 coordinates and then half-stores all
512 -- the same split, in the same order. **No omission exists.** The hypothesis
was cheap to falsify, which is the only thing it had going for it, and it was
falsified by the one party who checked both sides instead of one.

## Matched-input replay: position 14 closes, position 0 does not

C8332C52's replay feeds C-derived `Qcur`, a source-reconstructed cache and the
pinned sinks into Go's `raw_attention_eval`, so the inputs are matched rather
than merely similar. Three arms: baseline, a `query-f32` candidate that removes
only the query half-round, and a device `+1` mutant.

| position | arm | violations/32768 | max abs error |
|---:|---|---:|---:|
| 14 | in situ (prior run) | 6419 | 0.03794819 |
| 14 | replay baseline | **160** | 0.00063419 |
| 14 | replay `query-f32` | **0** | **6.65e-6** |
| 0 | in situ (prior run) | 564 | 0.001972675 |
| 0 | replay baseline | **564** | **0.001972675** |
| 0 | replay `query-f32` | 521 | 0.00176549 |

**At position 14 the gap closes completely.** Matching the inputs removes 97.5%
of the violations, and removing the query half-round removes the rest: zero
violations, max error 6.65e-6, three orders of magnitude below the in-situ
failure. Query rounding is a real contributor, and it is now measured rather
than argued.

**At position 0 nothing moves.** The replay baseline reproduces the in-situ
figures *exactly* -- 564 and 0.001972675 to every digit -- and the candidate
that zeroes position 14 barely dents it, 564 to 521. The same intervention that
closes one position leaves the other essentially untouched.

That is a **dissociation**, and it is worth more than either number alone. One
mechanism cannot produce both an effect that vanishes under `query-f32` and one
that survives it. Position 0 has a distinct path, which is what C8332C52 is
taking next, and the campaign's registered candidate is **incomplete rather
than wrong**.

### The pos-0 exact reproduction is an unplanned control

Feeding C's `Qcur` and a reconstructed cache into Go at position 0 returns the
in-situ numbers to the last digit. Two readings, and they are not the same:

- the replay harness is faithful -- it neither adds nor removes error, which is
  exactly what a substitution control should show; or
- at position 0, with no history and a one-token attention case, the replay is
  close to the identity and reproduction is nearly trivial.

Both are probably true, and the second limits how much credit the first
deserves. Recorded as a consistency check that *passed*, not as a validated
instrument: it was never registered as a control, and a gate found after the
fact is the pattern this ledger has retired three times.

The `+1` device mutant was verified against the **replay baseline** across all
28672 non-rotary coordinates per position, not against a failing C comparison.
That is the right shape: an already-failing baseline would have let a no-op
mutant validate itself vacuously, and this design forecloses it.

### The position-0 replay was not input-matched, so its null result attributes nothing

C8332C52 read the pinned `ds4.c:29864-29924` and found the premise under the
position-0 arm of the replay is false. The zero-prefix raw attention path reads
the contiguous batch `KVcur` **directly**; the ring cache is half-stored
separately, for later chunks and for decode. The replay reconstructed every
position's attention operand by IEEE half-rounding the captured `KVcur`. That
reconstruction is right for cached history and wrong for the position-0
operand, which C never half-rounds.

The source reading has an independent numerical witness. Across coordinates,
C's head/`KVcur` ratio is constant; head/half(`KVcur`) is not. A single
reconstruction error that applies uniformly would keep the ratio constant in
whichever operand C actually used, and it is constant in the un-rounded one.

Consequences, stated as withdrawals rather than as a new result:

- The pos-0 row of the replay table measures a comparison against an operand
  the reference did not use. **Withdraw the arithmetic attribution at pos 0.**
  "Position 0 does not move under query-f32" is not evidence about position 0's
  arithmetic; 564 -> 521 is a difference between two mismatched inputs.
- The **dissociation reading is withdrawn with it.** "One mechanism cannot both
  vanish under query-f32 and survive it" only holds if both arms were
  input-matched. One arm was not. The pos-14 result stands on its own terms;
  it no longer licenses a claim that pos 0 has a distinct arithmetic path.
- The pos-0 exact-reproduction consistency check did not catch this, and could
  not have: it compared the replay to itself, so a wrong operand reproduces
  exactly.

What survives: pos 14 (6419 -> 160 -> 0 violations, max 6.65e-6), where the
cached-history reconstruction is the one the source says C uses.

`prefill_chunk=1` does not collapse this. It does not make the first-prefill
and decode kernel routes identical -- the route is selected by whether a
zero-length prefix exists, not by chunk size. I had treated chunk size as
though it settled the route.

Next arm, per C8332C52: query-f32 plus an un-half-rounded current KV at pos 0,
keeping stored history half-rounded.

### Both positions close under matched inputs, and the retraction bought the result

The corrected replay finished at 13:09:28 UTC in one lock window with four
arms, and the operand contrast I asked for was inside it rather than inferred
across runs.

| arm | pos | max abs error | violations/32768 |
|---|---:|---:|---:|
| baseline | 0 | 0.00197267532349 | 564 |
| `query-f32`, half KV | 0 | 0.00176548957825 | 521 |
| `query-f32`, **first-prefill float KV** | 0 | **9.53674316406e-7** | **0** |
| baseline | 14 | 0.00063419342041 | 160 |
| `query-f32`, half history | 14 | 6.64778053761e-6 | **0** |
| new arm, same half history | 14 | 6.64778053761e-6 | **0** |

Two changes, each carrying one position, and the second was only visible after
the first was withdrawn:

- **Query rounding**, which closes pos 14 on its own (160 -> 0).
- **Operand form** at the zero-prefix position: C's first-prefill attention
  reads contiguous post-FP8 float KV, not the half-stored ring. Feeding the
  same query arithmetic the float form takes pos 0 from 521 violations to 0,
  max 9.54e-7 -- three orders below the in-situ failure.

Yesterday's reading of this was a dissociation between two positions. It was
an artifact of feeding one of them an operand the reference never used. The
real structure is simpler and was hidden by the error: **one arithmetic
difference and one operand difference, neither of which is a distinct
mechanism per position.**

Four things make this arm stronger than the one it replaces:

1. The half/unhalf contrast ran **in the same process and window**, so
   "operand choice matters here" is measured, not read off source. The outputs
   differ. This is the control I asked for before the window opened, and it
   fired.
2. All six old-arm binaries compare **byte-identically** (`cmp`) against the
   prior replay's outputs, so the new arm did not perturb the old ones.
3. The `+1` device mutant changes all 28672 non-rotary coordinates against the
   **replay baseline** at each position and is detected. It is a device
   mutant, not a host snapshot, so it tests device arithmetic.
4. Outputs start poisoned and are verified finite.

**Correction to point 3, before it propagated.** I first wrote that the mutant
became device-arithmetic detection once the baseline passed. C8332C52 corrected
it: the mutant has **always** been gated against its own unmutated Go replay
output, never against C, and the original replay already required the `+1` at
all 28672 coordinates independently of the failing C comparison. The current run
repeated that control unchanged. So the control acquired no new status when the
C comparison passed -- it had the status the whole time.

This is my error concentration exactly as the ledger records it: the design was
stated correctly in `858f7676`, which says the mutant was gated against the
replay baseline *rather than* against a failing C comparison, and then the
summary sentence one entry later turned a fact about the design into a change
in the design. No rerun catches this and no control fires on it.

What this is not. It is parity on **two frozen-input fixtures at layer 0**, not
full-model parity, not a performance result, and not a certification of other
attention paths or layers. The full-model disagreement is unchanged: Go scalar
still selects 8233 at step 5 where pinned C selects 1694, and no candidate has
run against the full model. A fixture that closes tells you what to implement;
it does not tell you that implementing it closes the model.

### The layer-2 compressed capture measures a symptom, not a boundary

Position 127, layer 2, both implementations dumping five buffers. The arms ran
in **separate** lock windows with the same prompt ids and forced token 0.

| buffer | elements | max abs | mean abs | violations |
|---|---:|---:|---:|---:|
| `Qcur` | 32768 | 12.1381855 | 0.0867263 | 31917 |
| `KVcur` | 512 | 1.9114060 | 0.0265013 | 496 |
| `KVcompress` | 512 | 4.5 | 0.3866392 | **512/512** |
| `kqv_back` | 32768 | 0.0163092 | 0.0002684 | 15519 |
| `attn_out` | 4096 | 0.0336562 | 0.0070112 | 4025 |

C8332C52 scoped this correctly and I am recording the scoping as the finding:
the two **inputs** to layer-2 compressed attention already differ materially,
so nothing downstream of them is attributable. This is a measurement of how far
the two runs have drifted by position 127, not a measurement of the compressed
evaluator. Three readings worth stating before someone quotes a row:

- **The small numbers are the dangerous ones.** `kqv_back` max 0.0163 is three
  orders *below* `Qcur` max 12.14. That is not the compressed path behaving
  well. Softmax and the value average contract a large query difference, so a
  small output delta downstream of a huge input delta is the expected shape.
  Read the other way -- "`kqv_back` is nearly fine, so the kernel is nearly
  right" -- it would be exactly backwards.
- **Near-total violation counts are consistent with upstream divergence**, and
  `KVcompress` at 512 of 512 is what a fully diverged input looks like after a
  projection. None of that indicts the projection.
- **This does not close the position-0 operand question, it avoids it.** That
  question was about the zero-prefix route at first prefill. A position-127
  capture takes a different route, so the operand hazard does not arise here;
  it is untested, not settled.

What the capture does establish, and it is not nothing: position 127 exercises
the C compressed dump path and the Go overlay reaches the same position. The
fixture works. It is the comparison that is uninterpretable.

The separate lock windows are fine for a bitwise comparison of deterministic
output, but they mean there is no in-window contrast here of the kind that made
the corrected replay a result rather than a source reading.

Next: matched-input replay, or a shallower boundary walk to find the first
upstream divergence. Not optimization against these deltas.

### The matched replay at position 127 is not executable, and that is a tooling fact

The feasibility check found a concrete missing input rather than a result. The
pinned C observer exposes only the **current** `KVcompress` row at the decode
position; its CUDA decode path emits neither the prior raw-cache rows nor the
compressed-cache history that `encodeMixedAttention` consumes. A position-0
prefill capture does expose `attn_comp_*_raw` and compressor scratch, but no
committed compressed row and no reusable decode cache, so it cannot supply the
position-127 inputs either.

C8332C52 is holding attribution rather than reading the layer-2 deltas as
evidence, which is the correct disposition: **a fixture that cannot be built is
not a negative result about the code it would have tested.** Nothing here bears
on the Go compressed evaluator in either direction.

Two constraints on the way forward, stated before a window opens rather than
after one closes:

1. **The reference is read-only.** Any observer extension lives in a scratch
   copy; `/home/tmc/go/src/github.com/antirez/ds4` is not written to.
2. **An extraction hook can change the route it observes.** This campaign has
   already paid for that once: a C dump that disables fused norm/RoPE yields an
   unfused intermediate compared against Go's fused one, and endpoint
   invariance does not catch it, because the endpoints agree while the
   extracted value is from a different computation. An observer extended to
   commit and dump compressed-cache history is exactly the shape where this
   recurs -- the act of retaining history is a change to the cache path. If the
   extension is taken, the check is not "do the endpoints still match" but
   "does the extended observer take the same route", verified rather than
   assumed.

The shallower boundary walk avoids both constraints and needs no new
extraction. It is the cheaper arm and, given (2), also the safer one.

### The fixture predicted a full-model effect, and the effect is not parity

Three results landed together. Taken in the order they constrain each other.

**1. The layer-0 walk at position 127.** With Go's dump semantics corrected to
rotated query plus stored raw cache row:

| boundary | max abs error | violations |
|---|---:|---:|
| `attn_norm` | 0 | byte-identical |
| `Qraw` | 0 | byte-identical |
| `Qnorm` | 9.54e-7 | 0 |
| `Qcur` | 1.9646e-4 | **1**/32768 |
| `KVcur` | 1.5316e-3 | **11**/512 |
| `kqv_back` | **5.2918** | **32762**/32768 |
| `attn_out` | 0.00964 | 3825/4096 |

This is the **inverse of the layer-2 shape** and that is what makes it worth
having. There, huge inputs produced a small output delta and the contraction
was the explanation. Here the compared inputs are nearly matched -- one
violation in 32768, eleven in 512 -- and the output is off by 5.29 across
essentially the whole tensor.

**The amplification reading is available and I am not taking it.** "Nearly
equal inputs, wildly different output, therefore the attention arithmetic is
wrong" requires that the compared inputs are *the* inputs. They are not. Raw
attention at position 127 also reads the **cache history for positions 0..126**,
and that history was not compared at all -- it is precisely what the observer
cannot emit. The dominant input to this boundary is the one input nobody has
measured. A large difference in an uncompared input and a fault in the kernel
produce the same table.

So the honest statement is the one C8332C52 made: this localizes the first
actionable boundary to **layer-0 raw attention or its history/cache contract**,
and it moves the target off compressed layer 2. It is not a kernel attribution.

**2. The full-model `query-f32` arm, ABCCBA, and it moves the discriminator.**

| arm | digest | step 5 | max logit error |
|---|---|---:|---:|
| baseline (repeats) | `a4d6fc6c` | 8233 (C: 1694) | 7.1834 |
| `query-f32` (repeats) | `97262fa4` | **1694** | 4.99837 |

Both arms are deterministic across repeats, and route counts are identical --
20 half-cache calls, no dense launches -- so the change did not move the
program onto a different path. That is the control that makes this an arm
rather than a coincidence, and it was run.

**This is the campaign's first full-model movement of the step-5 token**, and
the fixture is why: query rounding was identified in a frozen-input replay at
layer 0 and then predicted an effect in the whole model. That is the fixture
methodology working end to end, and it is the strongest procedural result here.

**3. And it is not parity, in a way the token flip actively disguises.**

- **About 129k of 129280 logits still violate tolerance.** Essentially the
  whole vocabulary. Max error fell 7.18 -> 5.00, roughly 30%, while the
  violation count barely moved.
- **The step-5 match is the weak direction of a weak gate.** This ledger has
  recorded it twice: argmax agreement has strong positives and weak negatives,
  and *this is the negative direction*. A mismatch would be real evidence; a
  match says only that the top-1 ordering survived. With a runner-up margin of
  **0.0893** against a max logit error of **5.0**, the selected token sits well
  inside the error bar. The flip is a threshold crossing on a still-badly-wrong
  vector, and a further correct change could flip it back.
- Which means the right reading is: `query-f32` is a **measured partial** that
  moves the headline discriminator and does not close it. Treating the token
  match as parity would be adopting the most broken configuration that happens
  to agree -- the failure mode this ledger already has a section for.

**4. The notebook's direction, and the part of it that is unverified.** The
proposal is a scratch observer extended for exact decode raw-cache history,
then a matched layer-0 replay with OFF/ON endpoint invariance, poison/readback
and the `+1` mutant. The controls are the right ones and match the hazard this
ledger registered before the branch was chosen. But its claim that a
wrapper-only rebuild suffices is **unverified** -- locally the current
wrapper/decode hooks emit no `raw_cache` -- and C8332C52 is correctly declining
to treat a received recommendation as a measured fact. Extraction-off/on
endpoint identity has to be proven first, not assumed.

### The uncompared input was then compared, and the hazard check held

The entry above says the cache history at position 127 is the dominant input
to `kqv_back` and that nobody has measured it. That was true of the walk. It
is no longer true, and the resolution arrived in the same report.

A scratch observer was built from a **copied** `ds4.c` with one decode hook
dumping the physical raw cache after the current-row store; the pinned
CUDA/reference objects were not modified. **Extraction OFF and ON at the same
128-token workload produced byte-identical `state-00/01/02/04/08/16.json`
hashes.** That is the control this ledger registered as the hazard before the
branch was chosen -- an extraction hook can change the route it observes, and
a C dump that disables fused norm/RoPE yields an unfused intermediate while
the endpoints still agree. Here the endpoint-identity check was run first and
passed, so the extension is admissible as an observer.

With all 256 raw-cache rows frozen, the matched replay ran at both raw layers:

| layer | arm | `kqv_back` max abs | violations/32768 |
|---:|---|---:|---:|
| 0 | baseline (query half-round) | 0.00103450 | 62 |
| 0 | `query-f32` | 1.25766e-5 | **0** |
| 0 | device `+1` mutant | 1.41222 | 32768 |
| 1 | baseline (query half-round) | 0.00176430 | 509 |
| 1 | `query-f32` | 1.30879e-5 | **0** |
| 1 | device `+1` mutant | 1.41220 | 32768 |

The mutant moves all 32768 outputs at both layers, so the device-arithmetic
control is live and not vacuous.

**What this changes in the walk's reading.** The alternative I refused to
discount -- a large difference in an uncompared history -- is now excluded for
this position: with the history frozen and matched, `query-f32` takes
`kqv_back` to zero violations. The 5.2918 in the walk was therefore not
attention-kernel arithmetic and not a history mismatch; it was the walk's own
inputs being the unmatched thing. Query half-rounding is sufficient to explain
the position-127 raw discrepancy at layers 0 and 1.

**What it does not change.** This is one position, two raw layers, frozen
inputs. It says nothing about compressed attention, about other positions, or
about full-model sufficiency -- and the full-model arm in the entry above is
precisely the measurement that a local pass does not settle it. A prior
full-model candidate that also changed the current-KV route had already
failed; that is why the fresh arm was restricted to `query-f32` alone.

**The sequence is the point, and it is worth the ledger saying so.** Hazard
registered before the window -> control run first -> observer admitted ->
matched replay -> full-model arm with route counters held fixed. Five of this
campaign's retractions came from skipping one of those steps.

### The layer-2 table was mis-sized, and it was the table I reasoned from

A stride correction on the Go side -- the scratch dump used the state width
1024 where the emitted cache head dimension is 512 -- invalidates the layer-2
capture this ledger recorded and reasoned from. Corrected Go-vs-C snapshots:

| buffer | max abs | violations | previously recorded |
|---|---:|---:|---|
| `comp_cache` | 0.125 | **993**/16384 | `KVcompress` 4.5, **512/512** |
| `Qcur` | 0.0455 | 31817/32768 | 12.1381855, 31917/32768 |
| `KVcur` | 0.0078125 | **52**/512 | 1.9114060, 496/512 |
| `kqv_back` (in situ) | 0.01631 | 15519/32768 | 0.0163092, 15519/32768 |

`kqv_back` is unchanged, which is the sign that the fault was in the dump's
indexing of the cache buffers and not in the capture as a whole.

**The withdrawal is mine and it had already propagated.** I read the
`KVcompress` row -- 512 of 512, every element -- as the worst buffer in the
layer, and I sent that to C8332C52 as the leading hazard on the layer-2 replay
plan: freeze C's compressed cache and you mask the largest measured
disagreement in the layer. The corrected figure is 993 of 16384, about 6%.
The compressor is not the largest measured disagreement and may not be a
disagreement worth chasing at all.

The scoping half of that hazard survives and is still worth stating in any
layer-2 report: a replay fed C's frozen compressed cache tests the compressed
**evaluator**, never the **compressor**. What is withdrawn is the argument that
this masking mattered *because the compressor was badly wrong*. It was one
number, read off a buffer indexed at twice its true width, and "512/512" is
exactly the shape a stride error produces -- every element wrong is what you
get when you compare misaligned memory, and I read it as severity instead of
as a diagnostic.

### The compressed evaluator arm is falsified or vacuous, and one control separates them

The matched layer-2 evaluator replay, under one lock with committed compressed
history (32x512) and raw cache frozen from a scratch C capture:

| arm | `kqv_back` max abs | violations/32768 |
|---|---:|---:|
| baseline mixed evaluator | 0.4039359 | 31915 |
| mixed `query-f32` | **0.4039359** | **31915** |
| device `+1` mutant | 1.4597204 | 32768 |

The candidate and the baseline agree to every printed digit. That is two
findings with opposite meanings, and the run as designed does not choose
between them:

- the compressed score path never half-rounded the query, the edit was a
  no-op on that path, and query rounding is a **raw-attention defect
  specifically** -- a sharper and more useful claim than "no second defect"; or
- the edit never reached the executed kernel, and the arm measured nothing.

**The `+1` mutant does not separate them.** It is a different edit at a
different site. It establishes that the replay reaches the device and that the
outputs are sensitive to device arithmetic -- it says nothing about whether
*this candidate's line* executed. A mutation the dispatcher or the compiler
absorbs reports all-clear in exactly the shape of a true negative, and this
ledger has recorded that failure three times already.

The separator is a positive control **at the same site**: perturb the query
where the half-round was removed -- bf16-round it, or add one ULP -- and
confirm the outputs move. Move-and-not-move establishes the first reading;
neither-moves establishes that the edited line is not on the executed path.
Until that runs, this entry records a *candidate* falsification, not a
falsification.

**The control ran, and it moved the question rather than settling it.** A
`+1e-3` perturbation inserted at the same site -- after each query half-round
-- moved the output to max 0.40302455 and 31921/32768, against the baseline's
0.4039359 and 31915. So the site is reached and the path is sensitive there:
the vacuous-arm branch is dead, and that is a real gain.

But the pair of numbers now forces a third reading that neither branch
anticipated. Half-rounding an fp32 query perturbs it by order 1e-3 *relative*
for values near 1 -- the same order as the control that visibly moved the
output. If the removed round had been doing anything to these values, it
should have moved the output by a comparable and clearly detectable amount.
It moved it by exactly zero, to every printed digit.

A live, sensitive site whose half-round is a bit-exact no-op has one dominant
explanation: **the values are already half-representable there.** fp16
rounding is idempotent, `round(round(x)) == round(x)`, so if the compressed
path receives a query that an upstream stage already rounded, the round that
was removed was the second one and removing it is free. That inverts the
conclusion:

- not "query rounding is absent from the compressed evaluator", but
- "query rounding in the compressed path happens **upstream** of the edited
  site, and the edited site was redundant"

Same measurement, opposite implication for where to look next -- and the
second reading keeps query rounding alive as a compressed-path candidate
rather than retiring it.

The separator needs no GPU window and no new capture: over the frozen C
inputs already on disk, test `q == float32(float16(q))` for every query
element arriving at the edited site. All-equal means the round was idempotent
and the branch **relocates** rather than falsifies. Any-differ means the round
was live and genuinely changed nothing downstream, and the falsification
stands as written.

Recorded here as a candidate falsification with its alternative named. This is
the third time in this campaign that a correctly-run control has been read as
answering a question one step broader than the one it addressed.

**The host check ran and falsified my hypothesis. I named the wrong
direction.** C's `Qcur` has **32764 of 32768** non-idempotent fp16
round-trips, max 0.00194454. The values arriving at the edited site are
emphatically *not* already half-representable, so idempotence cannot explain
the zero delta and the upstream reading is dead.

The replay also now logs output hashes, which upgrades the observation itself:
baseline `41507a99`; `query-f32` **exactly `41507a99`, 0 of 32768 floats
changed**; same-site `+1e-3` `9c299e32`; device mutant `cfe43836`. "Agrees to
every printed digit" is now "bit-identical", and that is what makes the
redundancy reading solid rather than suggestive.

The mechanism C8332C52 gives is that the executed path performs a subsequent
`__float2half_rn`, which makes the preceding `ds4_half_round` redundant. So
the round is removable **because a half conversion remains downstream** -- not
because one happened upstream, and not because the path never rounds. My
alternative had the right shape and the wrong location.

Their phrasing is the correct one and is adopted here: *the edited round is
redundant in the executed compressed path; a half conversion remains
downstream* -- not *query rounding is absent*. The distinction is load-bearing:
the second sentence would retire query rounding as a compressed-path
candidate, and a downstream conversion is still a live one.

**What is still inference.** The `__float2half_rn` mechanism is prefaced
"likely" and is a source-level reading, not a measurement. The gate is the
same-site control applied one step later: perturb or remove the *downstream*
conversion and confirm the output moves. If it does, the redundancy is both
confirmed and located, and the compressed path gets a named rounding site to
test next. If it does not, the explanation is wrong and the zero delta is
still unexplained. Cheap, and it reuses the harness already standing.

**The gate already existed inside the control that was run.** The same-site
`+1e-3` mutant perturbs the value *immediately before* the half conversion,
and it moves the output (`9c299e32` against baseline `41507a99`). That is a
downstream-conversion gate: the conversion's input is live and the path is
sensitive to it. The candidate removes only the inner explicit round while
retaining the conversion, and is bit-identical. Both halves are therefore
measured, and the inference is discharged without a new window.

Removing the outer conversion is **not representable at this site**: the
destination is half storage, so assignment performs a float-to-half conversion
by type. There is no arm in which it is absent. That is a structural fact
about the site, not a gap in the experiment, and it is the right reason to
stop asking for one.

Two measurements combine into something neither gives alone. The host check
found **32764 of 32768** query values non-idempotent under fp16 round-trip,
and the destination is type-required half storage. Together those say the
downstream conversion is **lossy on this data** -- not merely present. The
inner round was redundant because the outer conversion was already doing the
same discarding.

**So the candidate relocates from arithmetic to storage, and that is the
second time this campaign has landed there.** The remaining compressed-path
candidate is a change to the downstream half-*storage route*, not the deletion
of a type-required conversion. The echo is with position 0: that boundary also
closed not on an arithmetic change but on **operand form** -- C's contiguous
post-FP8 float KV against the half-stored ring. Two independent boundaries,
reached by different fixtures months apart in the ledger's ordering, have now
pointed at half **storage** rather than half **arithmetic**. Each was
individually scoped as a local result; the repetition across unrelated
boundaries is the first structural pattern this campaign has that neither
fixture was built to find. Recorded as a pattern, not yet a claim -- two
instances is a direction to look, and this ledger has a section on what
happens when it is read as more.

Scope either way: frozen C compressed cache, so the compressor is untested;
and C/Go cache inputs still differ materially, so this is an evaluator result
and not a layer-2 parity result. The full-model `query-f32` arm remains a
partial -- step 5 fixed, max error 4.998.

### The storage arm is negative, and I am the one who promoted it

One lock window, baseline-comp-f32-baseline. The candidate removed only the
attention compressor cache half-store (`cache[row*512+t] = ds4_half_round(x[t])`
becomes `x[t]`); query, routing and indexer unchanged.

| arm | digest | step 5 | max logit error |
|---|---|---:|---:|
| baseline | `a4d6fc6c` | 8233 (C: 1694) | 7.1834 |
| `comp-f32` | `e67c2085` | 8233 | **8.11987** |
| baseline repeat | `a4d6fc6c` | 8233 | 7.1834 |

The baseline returned its original digest and error, so the window is sound
and the arm is interpretable. **Reject storage unhalving as a full-model fix.**

**The direction is the finding, and it is a positive one inside a negative
arm.** Widening the store did not fail to help -- it made things *worse*,
7.1834 to 8.11987. Moving Go's storage from half to float moved Go **away**
from C. The natural reading is that C's compressed cache is half-stored too,
so the two already agreed at this site, and the candidate broke a match rather
than repairing a mismatch. A null result could not have said that; the sign
could. This site is now positively characterized as *format-matched*, which is
worth more than the rejected candidate cost.

**And it is a disconfirmation of the direction I promoted, one message
earlier.** I wrote that two boundaries pointing at half storage meant "a
storage-width arm now has two priors behind it and the arithmetic arms have
none left." The first storage-width arm run after that advice is negative.

The rescue is available and I am refusing it. I could observe that this tested
a *third* site -- the compressor cache -- rather than either of the two the
pattern was drawn from, and that the pattern therefore survives untouched.
That is true as far as it goes, and it is also exactly the move this ledger
has a section about: narrowing a generalization after its first
disconfirmation so that the disconfirmation lands outside it. The pattern was
offered as a direction for choosing the next arm. It chose one, and the arm
failed. The honest accounting is that the direction is now 0 for 1 as a
predictor, and that its two supporting instances remain what they always were
-- two local results, each correctly scoped, which I read as a trend one
message too early.

What the two original instances still support is narrower and does not license
site selection: at the two boundaries actually measured, operand and storage
*form* mattered where arithmetic did not. That claim never extended to a third
site, and I extended it anyway.

**Not worth a window:** `comp-f32` combined with `query-f32`. The direction
evidence here is site-local -- widening moves away from C because the formats
already agree -- and that does not flip under a query change elsewhere.

Also recorded: 6 of 10 leader margins sit below the max error on the candidate
arm, against a runner-up margin of 0.0220184 at step 5. That is the fragility
figure for this arm and it is worse than the baseline's, which is consistent
with an arm that increased error without changing the selection.

### The indexer arm is ungated, not negative

Baseline-`indexer-f32`-baseline under one lock. **All three arms returned the
identical digest `a4d6fc6c`**, step 5 at 8233, max error 7.1834, baseline
repeats matching.

Bit-identity between candidate and baseline is the same signature the
compressed query round produced, and it has the same two readings:

- the edit is live and genuinely does not affect this trace; or
- the edit never executed on the path this trace takes.

The reported conclusion -- "indexer unhalving is a no-op for this ctx1024
scalar trace (likely selection path does not consume a changed row here)" --
is the first reading, and the hedge inside it is doing real work. Nothing in
the window distinguishes them. **The ABA baseline control does not separate
them either: it establishes that the window is sound, not that the edit ran.**
A change that never executes reproduces the baseline digest perfectly, which
is precisely what was observed.

Recorded here as an **ungated arm**, not as a negative result. The separator
is the same-site positive control that already resolved this once: perturb at
the indexer store site and check the digest moves. Moves -- the site is live,
the no-op is real, and the conclusion is established rather than surmised.
Does not move -- the site is not on this trace's path, this arm measured
nothing, and the more useful fact is learned anyway: that the ctx1024 scalar
trace does not exercise the indexer store at all, which would retire a class
of indexer experiment on this fixture instead of leaving it ambiguous.

**The asymmetry is worth stating once, because it governs which arms need
which controls.** `comp-f32` yielded a direction because its output *moved*;
a moved output carries its own evidence that the edit took effect.
`indexer-f32` yielded bit-identity, and bit-identity is the one outcome that
cannot distinguish "no effect" from "no execution" without a separate control.
Every arm in this campaign that returned an unchanged digest has needed a
same-site control, and each time the control has been added after the
conclusion was first written rather than before.

### The indexer site is not on this fixture's path, and 21.83 said so first

The same-site control ran and did **not** move the output: baseline,
`indexer-site-mut` (`+1e-3` at `cache[row*128+t]`), baseline -- all three
`a4d6fc6c`, step 5 at 8233, max 7.1834.

So the second reading holds. The earlier `indexer-f32` arm measured nothing,
and C8332C52's recording is the correct one: **no output sensitivity /
coverage gap, not a rejected parity candidate.** An arm that cannot move the
output cannot reject anything.

**The scope of this is larger than the arm, and it is a fixture-level fact.**
If perturbing the indexer store cannot move the decode digest, then *every*
indexer arm judged on that digest on this fixture is vacuous -- past ones as
well as the two run today. That is a statement about what the fixture covers,
not about any candidate.

**And the campaign already recorded the symptom, from the other side.**
Errata 21.83: *"the indexer warp rewrite was not bit-exact, and the digest
gate could not see it."* A real 1-ULP numerical regression in the indexer was
invisible to the decode digest, caught only by a dedicated unit test. Today's
control is that same fact approached from the opposite direction -- then, a
real change the digest could not see; now, a deliberate perturbation the
digest cannot see. The symptom was documented; what is new is the mechanism,
and the pairing is what makes it a characterization rather than an anomaly.

**The mechanism remains an inference and should stay one.** At ctx1024,
layer 2 has roughly 256 compressed rows, below the 512-row indexer-selection
threshold, which would explain a cache that is written and never consumed.
C8332C52 declined to promote that to a measured site-execution claim without a
direct cache dump, and that restraint is right. Weak corroboration only: the
21.83 unit test exercised 513, 514, 1000, 1024 and 2049 rows -- all above the
threshold -- but that reflects the sizes the test chose, and says nothing
about behaviour below it.

**The discriminator is cheaper than a cache dump and reuses the harness.**
Repeat the same-site perturbation at a context that produces more than 512
compressed rows; the campaign already runs a 2048-token fixture for the
continuation-token invariant. If the output moves there and not here, the
threshold explanation is established by behaviour rather than by source
reading, and the coverage gap acquires a boundary instead of a hypothesis.

**What this costs the campaign, stated plainly.** The indexer is not
exonerated and is not implicated. It is *unmeasured on this fixture*, and it
has been unmeasured for longer than today -- 21.83 is the receipt. Any future
indexer work needs either a fixture above the threshold or the unit test, and
the decode digest is not an instrument for it.

## The campaign ends on an exhaustion decision, and the best result is a passing gate

Bounded diagnostics stopped with no parity claim and no performance claim.
This section is the narrative close.

### A boundary that passes tolerance was the cause

The single most useful result of the campaign is this one, and it inverts a
rule the campaign spent weeks learning.

At layer 0 position 0 under the scoped arm, `kqv_back` compared against C at
max **9.536743e-7** -- one ULP, mean 1.65e-8, **0 of 32768 tolerance
violations.** By every gate this campaign owns, that boundary passed.
Immediately downstream, `attn_low` failed at max 0.0018969178 with
**5484 of 8192** violations.

Three rival explanations were killed by measurement, each with the candidate
disabled and the output compared:

| control | result | verdict |
|---|---|---|
| block-diagonal `attnOutA` disabled, scalar per-group fallback forced | `attn_low` **bit-identical** to the block-diagonal arm | dispatch rejected |
| `DS4_CUDA_DENSE_ARTIFACT_MATVEC=0`, model-mapped weights forced | `attn_low` **bit-identical** | weight provenance rejected |
| `DS4_CUDA_NO_Q8_MMA=1` on the C side | C `kqv_back` and `attn_low` **bit-identical** to default C | optional C MMA dispatch rejected |

Then the causal test, which is the one that pays. Frozen C `kqv_back` floats
were injected into Go immediately before the `attnOutA` Q8 quantization, with
every later kernel unchanged. **`attn_low` became bit-identical to C: max 0,
mean 0, 0 of 8192 violations.**

So the one-ULP difference at a boundary that *passed its tolerance gate* is
**sufficient** to produce the failure downstream. The host-side reconstruction
says why: of 1024 Q8_0 blocks, 583 scales differ by at most 7.51e-9 and only
**3** int8 codes differ. A quantizer is a discontinuous function, and a
one-ULP input shift moves values across block boundaries.

**The rule this establishes, and it is the mirror of one already here.**
Earlier this ledger recorded *the small number is the trap*: downstream of a
huge input difference, softmax and the value average **contract** the error,
so a small output delta means nothing good. This is the same lesson from the
other side -- a quantization boundary **amplifies**, so a *passing* upstream
gate means nothing good either. Tolerance gates are calibrated for smooth
propagation; they are not evidence at any boundary that is discontinuous.
Neither direction can be read off a single number without knowing what the
next operator does to it.

And the injection bought a positive exoneration, not just a localization:
`attnOutA` weights, the quantizer and the matvec are all **unchanged Go code**
in the arm that matched C exactly. They are positively cleared on this
fixture. The target is exclusively the upstream `kqv_back` producer.

### The propagation is a chain, and the chain was bisected

Layer 1 `Qraw` diverged from C at 15775 of 32768. Injecting frozen C layer-0
`attn_out` **or** C layer-0 `hc_attn_post` each made layer-1 `Qraw`
**bit-identical** (0 of 32768). So the FFN/HC tail is not independently wrong:
the discrepancy is propagated, and it enters at or before layer-0 raw
attention/output. A separate injection of C layer-0 `hc_ffn_post` reproduced
the same result.

That is a genuine bisection -- two injection points, one conclusion, and the
alternative (an independent layer-1 QKV defect) excluded rather than argued
away.

### The cache history is a major measured contributor

At position 14, replacing Go's prior raw-cache rows with C's history and
keeping Go's own current Q/KV moved `kqv_back` from max 6.545609 with
**32763** of 32768 violations to max 0.0379656553 with **6426**. The mapping
was gated: the first 128 C rows and the post-injection Go cache are
byte-identical over all 262144 bytes, and a same-site `+1.0` to the first
cached float moved 32597 of 32768 outputs. Live channel, validated comparator.

Adding `query-f32` on top of matched history moved 6426 to **6215**. Query
rounding is live and real and **small** -- a measured contributor, never the
mechanism.

### What ended it: every current-row dump is stage-incompatible

Three separate attempts to inject C's current row failed the same way. C
`Qcur`, C `KVcur` and C `KVraw` each produced *regressions* when injected at
the corresponding Go point -- and in each case a same-site mutant fired,
proving the channel live. A live channel plus a regression on the unmutated
value is the signature of a **stage-mapping mismatch**, not of attribution.
C8332C52 read all three correctly and refused to name a cause from any of
them.

The supporting discovery is exact and worth keeping: **C's captured `KVcur`
values lie exactly on the binary16 grid** (0.375, 0.5625, ...) while Go's
pre-store `kvcur` is arbitrary float. The dumps are taken at different sides
of the half store. That single observation retroactively narrows the earlier
full-C-KV injection result, and C8332C52 narrowed it rather than keeping the
stronger reading.

So the residual is **unattributed, and honestly so.** Reopening needs a new
in-kernel C observer with a writeback ABI at the exact CUDA staging boundary,
plus route-invariance and live-injection controls. That is production-adjacent
work against a read-only reference, and it is correctly out of scope.

### The null family, which is a result in itself

| arm | outcome |
|---|---|
| raw-evaluator FMA forced (`__fmaf_rn`) | full-trace digest and both dump hashes **byte-identical**; NVCC already emits it |
| Go RMS-pair split into two launches | digest changed, `attn_low` **worsened** (4943 vs 5484 at higher max); rejected |
| fused mixed route disabled (`C833_DISABLE_MIXED_FUSED=1`) | every stage hash byte-identical; route not on this trace |
| C paired QKV projection disabled | C output **byte-identical** |
| C fused QKV RMS disabled | C output **byte-identical** |
| C score reduction forced to 4 and 8 lanes | C output **byte-identical** |
| Go current-KV FP8 round-trip removed | **failed its own cache control** (440/512 rows), and worsened everything; FP8 current storage is necessary |
| un-half-rounded current KV at *every* position | reproduces pos-0 exactly, worsens the full trace, 2/10 selections disagree; generalization rejected |

Four of those are C-side route controls returning byte-identical output. Taken
together they say something the campaign never had before: **C's own dispatch
choices do not affect C's output on this fixture.** Every route difference
that remains is Go's.

### The indexer discriminator ran, and it refuted my expectation

I proposed running the same-site indexer perturbation at a context above the
512-row threshold, arguing it would establish the threshold explanation *by
behaviour* and give the coverage gap a boundary. It ran: ctx4096, prompt 10 +
2040 accepts, position 2049. **Baseline and `indexer-site-mut` returned the
identical digest.** (The ctx1024 attempt correctly failed with "prompt exceeds
context" and was not interpreted -- the right handling of a fixture that
cannot host the arm.)

So the digest is blind to the indexer perturbation **above** the threshold
too. I described both outcomes in advance, but I framed this one as the path
to a boundary, and it delivered a broader blindness instead. The threshold
explanation is **not** behaviourally confirmed, and the honest statement is
unchanged from before the run: the indexer is neither exonerated nor
implicated, it is unmeasured by this instrument, and errata 21.83 said the
digest could not see indexer changes before any of this.

### Where it actually stands

- **Closed, with a causal control:** the layer-0 projection mismatch is caused
  by the one-ULP `kqv_back` form crossing the Q8 boundary. `attnOutA` weights,
  quantizer and matvec are exonerated.
- **Closed, with a bisection:** layer-1 divergence is propagated from layer 0,
  not independently generated.
- **Measured contributors, neither sufficient:** cache history (32763 -> 6426)
  and query rounding (6426 -> 6215).
- **Rejected with measurement:** storage unhalving, block-diagonal dispatch,
  weight provenance, C MMA, FMA, RMS-pair, fused mixed route, KV FP8 removal,
  all-position current-KV.
- **Unattributed:** the 6215 residual, blocked on an observer ABI that does
  not exist.
- **Unmeasured:** the indexer, by this instrument, at any context tried.
- **Never achieved:** full-model parity. Go scalar still selects 8233 at step
  5 where pinned C selects 1694, with roughly 129k of 129280 logits violating
  tolerance. The one arm that moved step 5 to 1694 -- `query-f32` -- did so
  with a runner-up margin of 0.0893 against a max logit error of 5.0, and this
  ledger never called it parity.

## Feature-parity implementation ledger (2026-09-27)

These rows track the host-side slices implemented from
`~/tmp/ds4-parity/GOAL-features.md`. They do not supersede the CUDA/runtime
receipts above.

### Oracle receipt index

The pinned renderer receipt uses the clean `11e8cad` oracle exported on the
DGX. The hashes below are copied from the receipt, not shortened from memory:

| item | value |
|---|---|
| oracle source | `11e8cad4473716a41baf4d62e92ed9764a00fce6` |
| oracle `ds4` | `15cda4e22f5c18ede1946403afef94cd57f7b58e80b373a3fd8f940fdf96da30` |
| oracle `ds4-bench` | `700c391d80b355ae6ffd08aa13acc3b6528941d5ee09de1efcfeaba175c6c8b6` |
| oracle `ds4-eval` | `deb7e53b65d34cdf2207c31d5e354a2446c12554732c7433e5ceddacf7f8497d` |
| gate model | `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |
| host/backend | DGX Spark / CUDA `sm_121` |
| context | **UNREAD**: the renderer receipt does not record a numeric context value; none is inferred |
| candidate receipt | `~/tmp/agent-collab/ds4-min/20260927-C8332C52-renderer-gate-2dcc13fb3.md` |

The current renderer result is the `2dcc13fb3` receipt: candidate binary
SHA-256 `d4209b7d9bac47a17430611d13b703c8e14264605e45f53de5b73926f5e41719`,
288/288 exact, zero mismatches and zero skipped. Earlier 264/288 receipts are
historical failures and must not be used as the current renderer verdict.
This receipt covers renderer/tokenization behavior only; it is not a
full-model CUDA numerical or performance qualification.

| gate | verdict | commit | evidence and limit |
|---|---|---|---|
| sampler and CLI sampling controls | PASS (host) | `294f40793`, `63f49a1a4`, `075595f3a`, `8d11e2f53` | independent float32 reference, 7x256 seeded replay grid, xorshift transition and no-draw mutation tests; no GPU seeded-decode oracle |
| chat thinking/system modes | PASS (host) | `eb3f2492a`, `989b0b9f6`, `2559ed2ea`, `c6eaa82fd` | renderer/parser tests, fail-closed mode-specific stop boundaries (`-nothink` stops on reasoning markers), and provenance-bound corpus validation (ordinal, matching case IDs, rendered-input and token hashes); upstream default decision B2 remains open |
| full renderer corpus against upstream | PASS | `11e8cad` (full hashes in the Oracle receipt index above) | Candidate `2dcc13fb3`, SHA `d4209b7d9bac47a17430611d13b703c8e14264605e45f53de5b73926f5e41719`, ran 288/288 exact, 0 mismatches, 0 skipped under one DGX `gpu-lock` lease. Punctuation mutation failed 60/96 and rendered-input mutation failed 12/12. Host/backend: DGX Spark / CUDA `sm_121`; numeric context is UNREAD. Receipt `~/tmp/agent-collab/ds4-min/20260927-C8332C52-renderer-gate-2dcc13fb3.md` |
| renderer corpus JSONL gate | PASS (host) | `401ff7324` | Versioned 24-case oracle/candidate schema with rendered-input and signed-int32 little-endian token hashes, bounded records/lines, and mutation tests; missing corpora return `UNMEASURED`, and the named upstream corpus comparison remains a separate runtime receipt |
| renderer corpus gate CLI | PASS (host) | `888b1f1ca` | `ds4-renderer-gate` emits the deterministic receipt and exits 0 only for PASS, 1 for FAIL/UNMEASURED, and 2 for usage or malformed input; it does not fabricate a missing oracle corpus |
| prefix files and raw token dump | PASS (host) | `c1faccc49`, `a60d9e7a1`, `0afb83883` | strict parser and byte-format tests; no local C token corpus |
| model-free evaluation tooling | PASS (host) | `7b8096cae`, `f2e20d43e`, `029b13be6`, `303ae8f1d`, `d68165488` | `dsv4eval` package provides 142-case validation, listing, fail-closed trace regrade, and strict hard-smoke cross-trace provenance/shape comparison; matching all-failure traces are rejected |
| perplexity and decode consistency | PASS (host) | `4ca9d5d87`, `517bbc0a9`, `442efb3d6`, `5f98cfffb`, `0634b8318` | float64 scoring, replay comparison, and incremental-live-versus-fresh decode consistency oracle with top-1/max-error/RMS reports; physical backend and cross-implementation gates remain unrun |
| benchmark schema/frontier artifacts | PASS (host) | `5779bd886`, `4781e491a`, `216870fc2`, `8221834ea`, `4e4b98f0c` | `dsv4bench` preserves the strict 14-column CSV and frontier JSON validators; summary output rejects empty, non-finite, negative, and unordered ranges |
| teacher-forced logit agreement | PASS (host) | `0b95777f9` | forced-token hash, matching step set, top-1/max-abs/KL metrics and mutation tests; no GPU C-vs-Go receipt |
| benchmark summary statistics | PASS (host) | `0b95777f9` | strict CSV reader with median/range JSON summary; no qualified runtime sample set |
| seeded sampler replay command | PASS (host) | `a1d0c022e` | 256-seed parameter-grid replay and RNG-movement gate; no upstream golden corpus yet |
| eval case selection filters | PASS (host) | `df087659b` | source/domain/case-id and bounded question selection; fail-closed matching tests |
| logprob dump schema | PASS (host) | `0abc3e638` | upstream token objects/bytes, top-k cap and no-RNG argmax behavior; no cross-implementation receipt |
| eval extractor self-test | PASS (host) | `3ca517a0a` | all answer kinds, aliases, last-answer and incomplete-output fixtures; no model execution |
| teacher trace completeness | PASS (host) | `b215f3293` | requires forced-token provenance and contiguous steps; sparse/fallback traces fail closed |
| bounded tool schema/DSML parsing | PARTIAL (host) | `b460e47f3`, `5403c298d`, `a462d9dd3`, `9e44ffa09`, `c49a5791e` | validates bounded OpenAI schemas; parses plain, short, and full `<｜DSML｜...>` markers, nested parameters, and malformed/size/depth mutations; adds mutex-protected bounded exact-DSML LRU memory with validated atomic Record and byte-exact Replay; adds an optional executor/replay seam that fails closed without an executor; live model tool execution remains unmeasured |
| nested dsv4 State contract | PARTIAL (host seam) | `c4fb160d1`, `77ec722ca`, `3e79e1cf2`, `f4471e153` | independent State/Snapshot API, lifecycle, and fail-closed CUDA/Metal adapters; source audit confirms both runtimes still share one mutable device state, arena and graph set, so F1 package/runtime gate remains open; HOLD receipt `~/tmp/agent-collab/ds4-min/20260927-device-state-f1-hold.md` |
| nested tokenizer package seam | PASS (host seam) | `d40357e6c` | GGUF tokenizer, rendering, prefix and mutation tests; CLI still uses compatibility implementation |
| bounded HTTP dsv4server seam | PARTIAL (host) | `5da4d7a86`, `3834bdad5`, `590ff02dc`, `8e84e7162`, `b460e47f3`, `49259874c`, `73fb94d4d`, `e844b8c4f`, `dd84bbc6b`, `92e0afd49`, `81b082eec`, `c0c653a87`, `7046c87bc`, `7f226dacf`, `a244737c6`, `030f2e97e`, `c847baa0e`, `855330099`, `76d607544`, `df2cfd2c4` | `dsv4server` package with deterministic non-streaming and bounded SSE for `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/models`, model aliases/resources, OPTIONS, bounded JSON/trailing-value rejection, structured context-limit errors, deterministic normalized transcript tests, and an opt-in FIFO independent-session pool; request model IDs are now validated against the configured engine or documented aliases; chat SSE `stream_options.include_usage` is gated and emits the usage-only frame before `[DONE]`; `/v1/completions` emits text-completion SSE with optional usage; `/v1/messages` emits bounded text-only Anthropic SSE events and `/v1/responses` emits sequenced text-only Responses events, all rejecting streamed tool calls; session leases retain common prefixes and reset on changed suffixes, and four-slot concurrent requests are guarded against state sharing; a bounded txtar loader/comparator now enforces request/response/token transcript completeness and masks only documented volatile fields; validated schemas can reach an explicit `ToolAwareModel`, while plain models and actual tool execution remain fail-closed; the upstream golden transcript corpus is absent, so cross-implementation receipt remains unmeasured; no device batching or CUDA runtime receipt |
| versioned KV/session persistence seam | PARTIAL (host) | `90015e0c9`, `4ab756faf`, `069177da8`, `93bbb4e23`, `34e583f32`, `0a61cb5d4`, `d0aaceb87`, `935f082f5`, `0df849d47` | bounded private Session Save/Load and DSKV Q2/Q4 envelope; host prefix Store adds atomic files, longest-prefix lookup, filters, touch, and eviction; session continuation, save/load/save identity, aligned 2048-token split invariance, trailing-data and geometry mutation gates pass; a synchronized upstream KVC sample is now available and its payload header is named correctly (`HeadDim`/`IndexerHeadDim`), but payload restoration still requires a backend codec and runtime qualification; CUDA payload integration remains unmeasured |
| GGUF config/support and inspect | PASS (host) | `ada211498`, `296f65a7d` | metadata-derived support and backend-free summary tests |
| think-budget controller | PASS (host) | `cd1617f1c` | hard/soft/rank/queued-close mutation tests |
| independent State/package split | PARTIAL (B3 resolved) | `c4fb160d1`, `77ec722ca`, `3e79e1cf2`, `ee8478b43`, `53288f751`, `482c2d3ec`, `6e7151e87` | nested package and lifecycle seam landed; CUDA/Metal request adapters own one request state and preserve batch evaluation; CUDA State bridge passes the registered F1 digest gate on DGX; Metal hardware F1 and multi-request runtime qualification remain open |
| server/KV persistence | PARTIAL (host) | `590ff02dc`, `90015e0c9`, `4ab756faf` | bounded server endpoints and private Session/DSKV persistence exist with mutation tests; full upstream API/tool semantics, CUDA payload integration, and upstream `.kv` interoperability remain unmeasured |
| speculative decode | PARTIAL (host contract) | `9eadf0ecd`, `e9cb67256`, `fde99122f` | backend-independent drafter/verifier contract, snapshot rollback, greedy acceptance, exact verifier-row shape gate with restore-on-mismatch, and a seeded point-mass exact-sampling/replacement distribution gate; GPU kernels and cross-implementation gates remain unmeasured |
| quant variants and cross-implementation gates | PARTIAL (host manifest) | `e4cf6093a`, `dd84bbc6b`, `fbc452cc6` | fail-closed registered ds4f-q2 profile, float64-reference Q8_1 activation validation, and strict 43-layer IQ2_XXS/Q2_K, Q4_K and MXFP4 layout validation, including required routed-expert dimensions; no CUDA/Metal dispatch or cross-implementation gate |
| head/first-token diagnostics | PARTIAL (host) | `aa82e9d26`, `576fe9071` | bounded `dsv4tools` vector stats formatter and `cmd/ds4-stats` emit min/max/RMS/top-8 with mutation gates; no full-model CPU backend exists, so upstream head/first-token parity remains unmeasured; receipt `~/tmp/agent-collab/ds4-min/20260927-C8332C52-diagnostics-HOLD.md` |

### State contract implementation update (2026-09-27)

The B3 package-location decision is now recorded as nested packages under
`examples/ds4-min`. Commit `c4fb160d1` defines the backend-independent
`dsv4.Backend`, `dsv4.State`, `dsv4.Snapshot`, and optional batch contracts;
backends that have not split device state must fail closed with
`ErrUnimplemented`. Commit `77ec722ca` adds `dsv4.Session`, which allocates
one independent State per request and keeps token-prefix reuse, batching,
reset, stop handling, and logits ownership behind that seam. Contract and
session tests pass. CUDA is wired through the State bridge and passes the
registered F1 digest gate. Metal has the corresponding adapter and
compile/test coverage, but no hardware F1 receipt yet.

### CUDA legacy-path F1 receipt (2026-09-28)

The exact registered digest workload passes on the current source tree's
legacy CLI path. Under `gpu-lock`, graphs disabled and `-temp 0 -nothink`, the isolated
binary `7688d038d04893e8d8bf6e731609bb72026dff1d96e812102b6ed3e5163c8b84`
from source `d1e08246e` produced the registered prefill digest
`6d2ad5c9a07fd2d27619e864fbdd6e24de9532b67125b1ea3052c75629d40ce5` and
final digest `77d04501c83be683f7c55289e45160a82e75d02f16584905f4ba516bdb0b7895`.
The model SHA is `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
Receipt: `~/tmp/agent-collab/ds4-min/20260928-C8332C52-state-f1-receipt.md`.

This does not qualify the `dsv4.Backend.NewState` adapter: `openSessionBackend`
still uses the legacy runtime and the adapter remains fail-closed. State-split
F1 is therefore still UNMEASURED.

The registered final digest is tied to the retained position-2049 workload
(`-tokens 2`). The goal text says `-tokens 16`; that configuration reproduces
the prefill digest but reaches a later final position and yields a different
final digest. This contract discrepancy remains explicit rather than silently
changing either the digest or the workload.

### Quantized non-expert format gate (2026-09-28)

`LayoutOf` now rejects unsupported non-expert tensor types before backend
dispatch. The gate covers F16 HC/compressor/indexer weights, F32 norms/scales,
I32 expert maps, declared dense Q4/Q8 formats, and F16/Q8 indexer projections;
mutation tests cover each class and accepted-type cases. Commit `93e9c8ad0`;
host receipt: `GOWORK=off go test ./ds4-min -run
TestLayoutOf(RejectsNonExpertTypeMutation|AcceptsDeclaredNonExpertTypes)$`.
This is a host contract gate, not a CUDA/Metal quantized runtime receipt.

### CUDA State bridge F1 receipt (2026-09-28)

`openSessionBackend` now constructs a request through `dsv4.Backend.NewState`
and preserves `dsv4.BatchEvaluator`. The exact registered workload passes under
`gpu-lock` with prefill `6d2ad5c9...` and final `77d04501...` bit-exact.
Clean binary `e1a8dbfc29691fa72aebaab63c90974b51c69d65b0c673793a8d2f1541db99d1`
reports `vcs.modified=false`. Receipt:
`~/tmp/agent-collab/ds4-min/20260928-C8332C52-state-bridge-f1-receipt.md`.
This qualifies the single-request CUDA digest path; concurrent request
isolation, snapshots, persistence, and Metal hardware F1 remain unmeasured.

Commit `d49fb1801` makes both adapters terminal after `Close`: subsequent
`NewState` and `Eval` calls return `dsv4.ErrClosed`, with focused Darwin tests
and a Linux cross-compile receipt. This is lifecycle hardening, not a new
runtime parity claim.

### Upstream default-generation mode (2026-09-28)

The resolved Decision 2 is now reflected in the CLI parser: the default is
HIGH thinking with sampling enabled (`temp=1`, `top-p=1`, `min-p=0.05`). The
existing deterministic gates remain explicit via `-nothink` and `-temp 0`.
Focused parser tests pass. This is a host/configuration change; it does not
qualify CUDA or Metal numerical parity.

### Checklist reconciliation update (2026-09-28)

The external feature checklist still describes the 02 renderer comparison as
unmeasured. That is stale: the named `11e8cad` oracle and candidate receipt
already establish a 288/288 renderer comparison with zero mismatches and zero
skips (the receipt index above). The result is a renderer/tokenization receipt,
not a CUDA numerical or performance gate, and the numeric context remains
UNREAD.

The 03 sampler and host evaluation rows are likewise implemented and covered
by host mutation tests, but their upstream golden replay and physical backend
receipts remain open. G7 cannot be promoted from host tooling to parity: it
depends on those cross-implementation inputs and on the unresolved evaluation
decisions. No synthetic corpus or host-only rerun can close that gate.

### Metal hardware F1 qualification hold (2026-09-28)

The focused Metal runtime gates all skipped because the pinned GGUF and oracle
state are available only on DGX, not on this Mac. No model execution occurred;
the zero test exit is therefore not a qualification. Receipt:
`~/tmp/agent-collab/ds4-min/20260928-metal-f1-qualify-hold.md`.

### Speculative Q8 decode kernel slice (2026-09-28)

Commit `7165c4c90` adds the small-batch (`n <= 8`) Q8_0 quantized matvec
kernel and dispatches speculative-width batches to it while retaining the
existing prefill kernel for larger batches. Host differential tests cover
batches 2--8 against the sequential oracle, and source/register guards fail
closed if the kernel symbol or width contract drifts. Linux cross-compilation
and `git diff --check` pass. This is an implementation slice only: no DGX
CUDA compilation or runtime qualification has been performed, and the
IQ2_XXS/Q2_K speculative kernels and cross-implementation gates remain open.

### Speculative selected-expert small-batch adapter (2026-09-28)

The bounded CUDA adapter in `cuda_speculative_experts_linux.go` accepts
token-major IQ2_XXS and Q2_K selected-expert inputs for `1 <= n <= 8` and
invokes the already-qualified one-token kernels once per row. F32 and Q8_1
activation entry points are covered. The contract rejects zero, oversized, or
non-positive-slot batches before device dispatch; it does not claim a fused
multi-token weight layout or a complete speculative verifier. The differential
test compares the adapter with sequential device calls and independent CPU
dequantization for n=1,2,8. On this host the CUDA test is UNMEASURED because
there is no runnable device; Linux cross-compilation and host contract tests
pass. Commit and receipt are recorded with this entry; a DGX runtime receipt,
speculative state/snapshot integration, and cross-implementation gate remain
open.

The bounded adapter now has a DGX runtime receipt. Source `cbf598097224af90ef6af9e5804782be78ce20eb`, test binary SHA
`83e30c80738742b38419d931c23e0273f06e27f22f3a9f78f1fbbb8fa732b20b`, and raw
log SHA `94409211c6194a9b1ea064f12b9860ce9d114a919f3104cb08a51a1fed043076`
were exercised on spark-31db / GB10 under gpu-lock. Batch widths 1, 2, and 8
passed sequential-kernel and host-reference comparisons, along with existing
Q8_0 device tests and invalid-shape gates. Receipt:
`~/tmp/agent-collab/ds4-min/20260928-C8332C52-speculative-qualification.md`.
This qualifies the bounded synthetic adapter, not production speculative
decode or cross-implementation parity.

### Fail-closed feature gate wiring (2026-09-28)

Commit `e2a6ac525` adds explicit `SpeculativeBackend` methods on CUDA and
Metal adapters that return `ErrUnimplemented` until rollback-capable device
snapshots exist, with lifecycle mutation tests. It also adds the machine-
readable `dsv4-eval -g7-smoke LEFT:RIGHT` receipt path and session-load
mutation coverage. These gates report missing inputs or unsupported device
state explicitly; they do not claim production speculative or G7 parity.

### G7 receipt provenance repair (2026-09-29)

Commit `d1197f202` gives the machine-readable smoke receipt distinct
`left_trace` and `right_trace` JSON fields. The previous struct tags serialized
both paths as `left_trace`, which made a valid comparison receipt lose the
identity of its right-hand input. `TestSmokeReceiptJSONKeepsTraceSidesDistinct`
is the regression gate. Focused tests, package tests, vet, Linux and Darwin
cross-compilation, and `git diff --check` pass. This repairs host tooling
provenance; it does not create the missing upstream G7 traces.

### KV cache boundary policy (2026-09-29)

Commit `c63251c02` adds the host-side `dsv4kvstore.Options` and `StoreLen`
policy from spec 05: prefixes shorter than 512 tokens remain whole, while
longer prefixes use the 32-token trim margin and 2048-token alignment. Custom
policy and mutation tests cover the boundary arithmetic. `go test
./dsv4kvstore` and `git diff --check` pass. This is a host policy seam only;
backend payload save/load, upstream KVC interoperability, and runtime cache
qualification remain unmeasured.

Commit `eafb4b4dd` adds `DefaultOptions` and the continued-checkpoint
`ContinuedStoreTarget` policy, including the source-grounded 10,000-token
interval rounded to 10,240, minimum-token and already-stored guards, and
mutation tests. `go test ./dsv4kvstore`, `go test ./...`, `go vet ./...`, and
`git diff --check` pass. This remains a host policy seam; no payload or
runtime persistence claim is made.

Commit `67000eb95` adds the source-grounded `ChatAnchor` helper. It stops at
the first assistant marker, selects the last preceding user marker, and
requires the configured minimum prefix length; invalid markers and short
prefixes return `-1`. Boundary and mutation tests pass. This is policy logic
only; tokenizer IDs, filesystem lookup, payload persistence, and runtime
interoperability remain outside the receipt.

Commit `0baf06b5e` adds runnable package examples for the four exported
host-side KV policy APIs. `go test ./dsv4kvstore`, `go vet ./dsv4kvstore`, and
`git diff --check` pass. This improves API verification only; it does not
expand runtime or persistence qualification.

### Persistence and decode-gate hardening (2026-09-30)

Commit `21e496b6c` makes speculative verification restore its saved frontier
when verifier evaluation returns an error after mutating state. The mutation-
and-error regression test passes. This closes an error-path invariant in the
host speculator; device-backed rollback remains unimplemented.

Commit `4eb7f1986` makes session serialization reject short or zero-progress
writes with `io.ErrShortWrite` instead of emitting a truncated snapshot. The
`dsv4` and `dsv4kvstore` tests and vet pass. This is persistence-envelope
hardening, not device-state persistence qualification.

Commit `f299ac870` makes output-agreement receipts explicit about their gate:
fixed gates require preregistered limits, while relative gates require a
nonzero C-vs-C control and factor. Teacher-forced receipts require a forced-
token hash and contiguous state numbering. Mutation tests pass. Existing
decode step-2 runtime evidence still exceeds the original fixed bound, so
decode numerical parity remains UNMEASURED under that contract.

Commit `5f24c83f5` extends the host server's `/v1/completions` request parser
to accept the upstream string form and the first string in an array, while
rejecting empty, null, and non-string forms. Handler tests pass. This closes
request-shape coverage only; the upstream server golden transcript gate and
device-backed session isolation remain UNMEASURED.

Commit `52dc5366c` extends the host server response contract: all four
supported APIs report `length` when the token limit is exhausted, using the
upstream `incomplete`/`max_tokens` finish reasons for OpenAI and Anthropic
shapes. Regression coverage passes. The server golden transcript and runtime
backend gates remain UNMEASURED.

Commit `9e1bdffc6` makes the sampler logprob path reject both NaN and infinity
inputs, with mutation coverage. This is a host numerical-safety gate; it does
not qualify backend logits against the upstream runtime.

Commit `11e8c50f6` adds explicit temperature, top-p, min-p, top-k, and seeded
sampling to chat, completions, responses, and messages, including streaming.
Invalid controls and zero seeds fail closed; omitted temperature preserves the
existing greedy behavior. Thinking controls remain unsupported because the
current Model interface has no request-scoped renderer mode. Host tests and
vet pass; upstream server golden qualification remains UNMEASURED.

Commit `913ccce4e` adds an optional `ThinkingModel` interface for request-
scoped NONE, HIGH, and MAX rendering. Chat, completions, Responses, Messages,
and streaming paths parse thinking/reasoning aliases and dispatch only when a
model implements the interface; legacy models fail closed with `ErrUnsupported`.
Host dispatch tests and vet pass. This qualifies request routing only; full
thinking output and upstream server golden parity remain UNMEASURED.

Commit `cb6058098` adds strict generated-thinking parsing. Non-streaming
OpenAI and completion responses expose `reasoning_content`, while Anthropic
and Responses expose ordered reasoning/text blocks. Embedded, nested,
unterminated, and closing-only markers fail closed; plain output is unchanged.
Unit and HTTP tests pass. Streaming still needs an incremental parser, and
full upstream server golden parity remains UNMEASURED.

Commit `f1454d831` extends thinking parsing to all three streaming APIs. The
stateful parser handles markers split across token chunks and emits reasoning
and text events in the existing OpenAI, Responses, and Anthropic SSE shapes.
Malformed, late, nested, and unterminated markers fail closed; plain streams
remain unchanged. Parser and HTTP tests pass. Upstream server golden parity
remains UNMEASURED.

Commit `d1a7a9a57` adds bounded stop-sequence handling. OpenAI string/array
forms and Anthropic stop sequences are validated and applied across
non-streaming and streaming output, including marker splits across chunks.
Malformed, invalid UTF-8, empty, oversized, and over-cardinality forms fail
closed. Handler tests pass; upstream server golden parity remains UNMEASURED.

Commit `d7127311d` closes a host-side server lifecycle gap. `Handler.Close` is
idempotent, rejects new requests with the upstream-shaped 503 shutdown error,
releases idle sessions, and safely closes sessions returned by in-flight leases.
Server tests, vet, and `git diff --check` pass. This is lifecycle hardening;
the upstream G1-G9 golden transcript gate remains UNMEASURED.

Commit `2493b6d69` adds bounded metadata-only inspection for upstream KVC
files, following `ds4_kvstore.h` and `ds4_kvstore.c`'s 48-byte header and
payload ABI 2. `InspectUpstream` validates magic, version, quantization,
token/context fields, declared text/payload sizes, truncation, and total file
limits without allocating or interpreting device state. The decoder reports
both upstream-format and explicit unsupported-payload errors. Mutation and
truncation tests, package tests, vet, and `git diff --check` pass. This is a
compatibility-inspection receipt only: upstream device payload loading,
State restoration, and runtime KVC interoperability remain UNMEASURED.

Commit `9de125a96` adds the source-grounded upstream KVC eviction score:
decayed hits, six-hour half-life, file-size normalization, anchor weighting,
and superseded-continued replacement scoring. Boundary and clock-clamp tests
pass. This is cache-policy logic only; filesystem eviction and backend payload
interoperability remain UNMEASURED.

Commit `05fb06a86` adds a runnable package example for the metadata inspector;
it exercises the fixed header and demonstrates that the opaque payload remains
a reported size rather than a restored state.

Commit `d3b8fea42` hardens the G7 hard-smoke receipt. The parser now retains
post-summary fields separately, requires passed/failed/incomplete/total
counters, and checks that they exactly match the twelve case records. Missing
or mismatched counters, duplicate summaries, and cases after the summary fail
closed. Focused and full Go tests, vet, and `git diff --check` pass. This is a
stronger host receipt gate; upstream runtime traces and cross-implementation
G7 execution remain UNMEASURED.

Commit `944fb4530` adds a fixed 200,000-draw host sampling distribution
self-oracle with deterministic RNG seeds and a documented six-sigma-plus-
constant tolerance. An always-argmax mutation is rejected by the same gate.
The test is scoped to sampler behavior only; upstream and device sampling
qualification remain UNMEASURED.

Commit `81603c39c` adds bounded host references for Q4_K and MXFP4 blocks,
including dequantization and whole-block dot products. Block sizes, nibble
order, scales, malformed inputs, and mutation cases are covered by tests.
These are serialized-data oracles only; physical CUDA/Metal quantized
dispatch and registered cross-implementation gates remain UNMEASURED.

The new `dsv4` State contract test also covers failed-load atomicity for the
host fixture: malformed, truncated, trailing, and oversized payloads must
leave the serialized state unchanged. This is a host mutation gate for the
contract's `Load` failure guarantee; device-backed State loading remains
UNMEASURED.

Commit `f0dd3a7f7` adds overflow-safe KVC budget accounting. `KVCSize` includes
the upstream fixed header, text, payload, and trailer bytes; `KVCBudgetRequired`
applies the upstream ceil-one-percent safety slack; and `KVCFitsBudget` keeps a
zero budget as unlimited while rejecting all uint64 overflow cases. Boundary,
trailer-omission, and overflow mutation tests pass. This is host budget
accounting only; it does not establish payload restoration or device-state
interoperability.

Commit `f3fb71663` completes the host-side KVC split-invariance self-oracle.
Encode and Decode are exercised through short-write and split-read patterns
from one byte through 63-byte chunks; the envelope remains byte-identical and
the decoded record remains identical. This closes the DSKV G3 host gate. It
does not qualify upstream KVC payload restoration or device-backed State.

Commit `568bfb84c` hardens the host speculative verifier contract. The bounded
constructor defaults to the qualified eight-token draft limit, validates draft
and verifier token IDs (with optional vocabulary bounds), rejects malformed
results before commit, and restores the frontier on invalid verifier output.
Mutation tests cover empty limits, oversized drafts, negative IDs, invalid
tops, and rollback. This is a host fail-closed gate; device snapshots and
production CUDA/Metal speculative verification remain UNMEASURED.

Commit `e6ab04d00` hardens the server golden transcript gate against duplicate
JSON object members. Archive loading and observed request/response comparison
now reject duplicate keys instead of accepting the last value supplied by
`encoding/json`; mutation tests cover both paths. This closes a host receipt
fail-open defect without claiming an upstream golden corpus or runtime parity.

The 2026-09-30 KVC restore audit (`~/tmp/agent-collab/ds4-min/20260930-
C8332C52-kvc-restore-blocker.md`) confirms that upstream's payload ABI is
specified, but ds4-min still lacks the synchronized device copies, logical
ring mapping, atomic preflight/restore, graph invalidation/rebuild, and DGX
continuation gate needed to implement it safely. `cudaRequestState` therefore
continues to fail closed with `ErrUnimplemented`; the metadata inspector must
not be promoted to payload loading.

Commit `0db85b0bf` centralizes routed-expert format validation in
`ExpertFormatForTypes`, so the IQ2_XXS/Q2_K, Q4_K, and MXFP4 pair rules are
shared by the GGUF layout walker and direct mutation tests. Gate/up mismatches
and unsupported mixed pairs fail closed. This is profile/shape validation;
physical quantized dispatch and cross-implementation gates remain UNMEASURED.

Commit `0a61cb5d4` adds a bounded versioned `DS4MSSN` host session envelope and
the opt-in `-kv-save`/`-kv-load` CLI flags. Token history, logits, and backend
payload are restored atomically; malformed input is rejected before mutating
the destination session. Host round-trip and mutation tests pass. CUDA and
Metal State adapters still return `ErrUnimplemented`, so this is not an
upstream KVC payload-interoperability or device persistence receipt.

Commit `7608bbab7` adds the bounded `-prefill-chunk` control (256..2048,
multiple of 256) and routes non-default values through CUDA batch evaluation.
Backends without the setter fail closed with `ErrUnimplemented`; parser and
unsupported-backend mutation tests pass. This exposes a route-control seam but
does not qualify numerical parity or performance for alternate chunk sizes.

Commit `d0aaceb87` adds the host-side private prefix store: atomic temporary
file installation, SHA-1 text keys, longest-compatible-prefix lookup, opaque
backend payloads, touch accounting, and bounded budget eviction. Store tests
cover prefix selection, model/context/quant filters, atomic failed writes,
private file modes, and foreign-entry rejection. This is not upstream KVC byte
compatibility and does not restore CUDA or Metal device state.

Commit `129d868f9` adds the opt-in `-sync-prefix` control, which evaluates a
prompt prefix and then resumes the full prompt through the normal session
prefix path. Invalid ranges and conflicts with `-kv-load` fail closed. This is
a split-invariance diagnostic; it does not claim device snapshot or upstream
KVC payload interoperability.

Commit `fb4d1ff2e` adds the corresponding host self-oracle: split-prefix and
direct session evaluation must produce identical token history and logits.
This validates the session seam only; CUDA/Metal device-state restoration is
still unimplemented.

Commit `29db7f0e0` extends the session persistence gate to reject trailing
bytes and context-mismatched envelopes without changing the destination
session. The new cases pass under the race detector; this remains a host
format gate, not device-state restoration.

The accompanying `PARITY-BLOCKERS.md` inventory records the remaining
UNMEASURED/PARTIAL requirements: device-backed State persistence, upstream
KVC payload interoperability, production speculative verification, physical
quantized dispatch and decode/prefill qualification, the upstream server
transcript corpus, and the unresolved CLI default decision. It is a scope
receipt, not a parity claim; each item names the missing authority or runtime
evidence needed before implementation can be promoted.

The Q8_0 CUDA dispatch slice is PARTIAL. On DGX GB10, the isolated
`TestCUDAQualificationQ80Quantization` and `TestCUDAQualificationQ80Matvec`
suite passed under `gpu-lock`: quantizer fixtures were byte-exact, and matvec
shapes 17x32, 17x96, 17x4096, and 17x8192 stayed within the registered 1e-4
tolerance with mutation/shape coverage. Receipt:
`~/tmp/ds4-q80-qualification-20260930.txt`, SHA-256
`24bc96bd95fa7d9df4ef113f2f1119b31e4eacb31e5a45e496fa4cc30f838a08`.
This does not qualify IQ2_XXS/Q2_K production paths, Metal, or full-model
cross-implementation parity.

The targeted IQ2/Q8_1 truncation control also passed on DGX GB10 under
`gpu-lock`: `TestCUDAQualificationIQ2Q81IntegerTruncation` observed device
results +/-5 and the independent mathematical reference +/-5.375. Receipt:
`~/tmp/ds4-iq2-q81-qualification-20260930.txt`, SHA-256
`15c0778a7a8839a39ff78830789237703906043d3d162188c177d800c3ee8533`.

The Q8_1 synthetic dispatch controls also passed under `gpu-lock` on DGX GB10:
`TestCUDAQualificationQ81Quantization` (all fixtures byte-exact),
`TestCUDAQualificationQ81BatchQuantize` (batch widths 2 through 32 plus
short-input/output/stride and zero-batch mutations), and
`TestCUDAQuantMMQBlocksByteEqual`. Receipt:
`~/tmp/ds4-q81-qualification-20260930.txt`, SHA-256
`c8f019a837d8706143e57220a6df4ac7c93941f4778169c7519a5477abb7e6fe`.
These are synthetic dispatch gates; production expert and full-model parity
remain open.

### 2026-10-04 upstream KVC payload integrity and ring mapping: HOST PASS

The KVC reader now validates the complete payload envelope before a backend
can attempt restoration. `ValidateUpstreamRecordLayout` checks the declared
payload length and SHA-256 fingerprint, then walks every bounded section using
the caller's version-qualified compression ratios. `MapUpstreamRawRows`
maps the upstream logical raw window into ds4-min's compact ring, including
wraparound, and `ReadUpstreamPayloadSection` copies only an already-validated
section into caller-owned storage. `ValidateUpstreamRestoreContract` requires
model/token provenance, payload ABI, dimensions, token counts, and target ring
geometry; it records that upstream's full raw capacity may differ from the
target ring capacity. Host tests cover fingerprint, section bounds, geometry,
and ring-wrap mutations. These are transport and mapping gates only: no CUDA
upload, state restoration, or cross-implementation logits claim is made.
Commits: `228150463`, `2eb3bcead`.

Follow-up host-contract corrections are in `ec3dbe274` and `9fa9110de`.
`TargetRawCapacity` is validated as the destination ring allocation: zero or
smaller-than-live capacities fail, while a capacity larger than the current
upstream live window is valid. `HashUpstreamPayloadTokenIDs` hashes the fixed
little-endian token section in bounded 64 KiB chunks, with truncation,
overflow, nil-reader, and mutation tests. Focused KVC tests and Linux vet pass.
These commits strengthen provenance and geometry checks only; they do not
restore device state or qualify cross-implementation logits.

### 2026-10-04 upstream server corpus loader: HOST PASS

`dsv4server.LoadCorpusCase` now reads the owner-labelled G1-G9 corpus case
shape (`metadata.json`, response/request JSON, token traces, and optional
terminal metadata) with bounded files and fail-closed provenance validation.
`ValidateCorpusProvenance` binds source revision, binary/model hashes, backend,
context, temperature, and thinking mode before a transcript comparison. The
loader is tested with missing-provenance, malformed-token, and provenance
mutation cases. It does not include generated corpus files or claim that the
upstream runtime and ds4-min responses match; the physical G1-G9 comparison
still requires the finalized external corpus and a ds4-min run.

`dsv4server.LoadCorpus` adds the finalization gate: it requires manifest and
determinism metadata, all nine G1-G9 IDs with PASS verdicts, bounded case
artifacts, and case-to-root build provenance. Tests cover a complete synthetic
G1-G9 manifest and a missing-case mutation. This remains a host gate until
F7's external corpus is finalized and compared against ds4-min.
Manifest case directories are constrained to remain beneath the corpus root;
path traversal and missing-case mutations fail closed.
Error-contract cases may omit thinking metadata; generation cases still retain
their explicit thinking mode for provenance comparison.
Malformed request bodies are preserved for non-2xx error cases while successful
cases remain canonical-JSON validated; the live G1 corpus cases were smoke-
loaded from the DGX recording.

### 2026-10-04 speculative identity margin validation: HOST PASS

Commit `cc2918f41` makes the speculative greedy identity gate validate every
margin, including entries after the first near-tie boundary. NaN, infinity,
negative, and later-case mutations now fail closed instead of being silently
ignored. The full `ds4-min/...` host suite and Linux vet pass. This strengthens
the G1 evidence gate only; no device speculative qualification is implied.

Commit `702c4d9bc` also makes teacher-forced evaluation reject missing,
malformed, or non-hex 256-bit forced-token and token-ID provenance hashes.
Mutation coverage passes; this hardens the host evidence gate and does not
qualify seeded numeric parity.

Commit `0ef444b7c` closes two corpus fail-open paths: root provenance now
requires and binds temperature and thinking mode, and each case must carry a
nonempty HTTP status. Mutation coverage rejects missing root sampling metadata
and missing case status. Error cases may still omit thinking mode by contract.
This strengthens transcript admission only; the external G1-G9 runtime corpus
is still required before server parity can be claimed.

Commit `1a9f18dca` further validates the status grammar: HTTP/1.0, HTTP/1.1,
or HTTP/2 with a three-digit 100-599 code and a nonempty reason phrase when
present. Arbitrary nonempty values such as `OK` now fail closed under mutation
coverage. This is still a transcript-admission gate, not a server runtime
parity result.

### 2026-10-04 decode F16 matvec: DEVICE/DECODE PASS

The default decode F16 half-input matvec now accumulates in float with pinned
FMA contraction instead of double (`a1e103abd`). Its Linux qualification bound
tracks the actual float reduction order and retains a non-vacuous column
mutation (`5a1258b65`). The registered 128-token final-logit digest was
updated to `735e5dde15cd8971d430e5397a3b950b42fd933d0ae471bf4cf8e614a7606175`;
the prefill digest is unchanged. The owner receipt reports a paired decode
median ratio of 1.0012 versus 0.9519 before the fix, with batch/single
bit-identity and determinism controls passing. This closes the previously
measured F16 decode performance gap for the registered workload only; full
cross-implementation seeded decode and other model contexts remain separate
gates. Receipt: `~/tmp/agent-collab/ds4-min/20261004-F7FA2607-f16-landing/receipt.md`.

The selected-expert Q8_1 gate was repaired by `e3ddbbdf6`: replacing fast-math
division with `__fdiv_rn` removes deterministic half-step activation errors.
The DGX rerun passed widths 256, 1024, and 4096 with source-reference deltas
within the registered tolerance. Receipt:
`~/tmp/ds4-experts-q81-fdiv-rn-20260930.txt`, SHA-256
`f4304004588a2731b859688efd05bc65ee6f745f6b992e4ed233df23954bc306`.

The selected-expert Q8_1 gate remains FAIL/UNMEASURED for production use. After
correcting its test helper to exercise the batched quantizer for multi-row
inputs (`a504e87b8`), DGX runs fail deterministically in the activation byte
check at widths 256, 1024, and 4096. Receipt:
`~/tmp/ds4-experts-q81-fixed-test-20260930.txt` (SHA-256
`02aaec556bbade40801de3e98767966ab9e1937c04e38fcc28182c96e799aacd`). This exposes a real device
rounding/layout mismatch and must be fixed before expert dispatch is promoted.

That failure was corrected by `e3ddbbdf6` and is superseded by a full-suite
DGX receipt. Quantization, batch quantization, Q8_1 matvec (32 through 8192
columns), selected-expert Q8_1 (256/1024/4096), and MMQ block byte identity all
pass under one lock window. Receipt:
`~/tmp/ds4-q81-full-fixed-20260930.txt`, SHA-256
`58430a598007375bcabef1e73d38bc5bf09e045b5adacbb56f6ff10de0baee71`.

The registered full-model CUDA oracle trace remains FAIL on DGX: the
teacher-forced run disagreed on 1 of 10 selected tokens, with four steps below
the leader-margin safety threshold and maximum logit error about 4.61. Receipt:
`~/tmp/ds4-oracle-trace-20260930.txt`, SHA-256
`ab6abb86a06171cb0e1966b0687c88a4ebc35e882769fa50509b808d67f05d41`.

The Q8_K expert route was repaired by `4bf86d253`, using `__fdiv_rn` and
`__fmul_rn` for the quantizer's scale and integer-code calculation. The DGX
rerun passed `TestCUDAQualificationExperts` for 256, 768, and 4096-column
inputs, including IQ2/Q2 selected projections, Q8_K byte identity, invalid-ID
poison checks, and output mutation controls. Receipt:
`~/tmp/ds4-experts-q8k-fdiv-rn-20260930.txt`, SHA-256
`64b979298929c7b8d777f798ad6ee37ab06e3897702460072fcec047a3fe264e`.

The combined expert-dispatch, gather-differential, and SIMD-vs-scalar receipt
also passes on the DGX under one lock window. It covers production and odd-tail
SIMD shapes, IQ2/Q2 selected routes, rotated-routing negative controls, and
Q8_K byte identity with invalid-ID checks. Receipt:
`~/tmp/ds4-expert-dispatch-fixed-20260930.txt`, SHA-256
`54750853383d29673de80eeead6764edfc7dfc265a8b13662890462f61464f81`.

The mapped-model control does not explain the full-model oracle failure. With
`DS4_CUDA_MAPPED_MODEL=1`, the same registered state bundle produced the same
step-0 maximum error (4.60992), violations (129225/129280), and step-9 choice
(Go 2052 versus oracle 1) as the resident/default run. Receipt:
`~/tmp/ds4-oracle-mapped-20260930.txt`, SHA-256
`3052f9a900cb3e68b8dd278efe5023026201699ca8bb9afb1ff8377ecb8b25e7`.
Mapped versus resident loading is therefore UNMEASURED as a causal lever but
not supported by this matched control; first-divergent-state localization
remains open.

The host State persistence contract now has a focused self-oracle. It verifies
snapshot restore, Save-Load-Save byte identity, truncated-load rejection, and
failed-load atomicity across independently created states. This is host-only;
CUDA and Metal persistence remain unqualified. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-state-contract-receipt.md`; commit
`e8fb91af5`.

The private DSKV store key now binds rendered text, model digest, and quant
variant. The previous text-only key could overwrite a Q2 entry with Q4 state
for the same prompt, or collide across model identities; the format contract
explicitly forbids that upstream flaw. `TestStoreKeySeparatesModelAndQuant`
proves all three entries coexist and resolve to their matching payloads. This
is a host-only policy gate; old text-only filenames remain readable as private
legacy entries, but are not rewritten or claimed byte-compatible with KVC.
Commit: `c76fcb97f`; package receipt:
`~/tmp/agent-collab/ds4-min/20260930-kv-key-gate-receipt.md`.

The G7 hard-smoke comparator now rejects terminal cases missing `picked`,
`expected`, or a non-empty `MODEL_OUTPUT` block, with mutation coverage for the
missing-output case. Focused tests and vet pass; this remains a host contract,
not a physical cross-implementation smoke receipt. Commit `5b0a9e795`.

The speculative host contract now has a fail-closed identity gate. It requires
accepted drafts, complete traces, valid margins, and a pre-registered near-tie
position before tolerating divergence; mutation tests cover missing acceptance,
incomplete traces, invalid margins, and absent near-ties. This is a host G1
gate only; device snapshot/restore and physical speculative qualification are
still open. Commit `549d319d6`.

The server now honors the upstream chat `tool_choice:"none"` policy by
validating supplied schemas but excluding them from prompt construction; other
unsupported policies fail closed. Focused server tests pass. Commit
`94b744fb4`.

The registered deep-context CUDA decode qualification passes on DGX. The
2048-token prefill reaches the expected ratio-4 state (`attn.n=indexer.n=512`),
16 decode positions cross the 512-row boundary to 516, every measured phase
retains positive unattributed residue, and the median full decode step is
71.29 ms (63.75--89.51 ms). Receipt:
`~/tmp/ds4-deep-decode-20260930.txt`, SHA-256
`21382db3ad2af3eb49ad892ad70dc371f78f552f3b01071272535f373cce2a47`.
This is a backend self-qualification, not C-vs-Go parity.

Source audit of the full-model failure found no safe existing layer-dump hook
for the short oracle trace's terminal position 9: the current Go diagnostic
hook only accepts positions 2047/2048 and fixed layer-0 probes. Mapped and
resident controls are already identical. The proposed next diagnostic is an
opt-in position-9 dump for `attn_out`, `hc_attn_post`, and `hc_ffn_post`, but
the CUDA session file is collision-owned, so no edit was made. Report:
`~/tmp/agent-collab/ds4-min/20260930-oracle-localization-report.md`.

The failed oracle trace was rerun with `DS4_ORACLE_STATE_OUT` enabled so the
Go-side states are reusable for a later matched localization. The run remains
the same FAIL: one of ten selected tokens disagreed, step 0 reached maximum
absolute error 4.60992 with 129225/129280 tolerance violations, and step 9
selected 2052 versus oracle token 1. The captured state files are retained as
`~/tmp/ds4-oracle-go-states-20260930`; their manifest SHA-256 is
`106be6117ab3dac6138adec0afcdb21dea19c1ea71d6cb1c0e16fb8783f4d31f`, and the
test receipt SHA-256 is
`bc30100fb3d725d1a6b374902285e4f460baae10b9d44d7864b21c95c836622c`.
This artifact is diagnostic only; it does not localize the first divergent
layer or establish C-vs-Go parity.

The CLI now derives its default temperature, top-p, and min-p from the shared
sampler defaults, and the parser test asserts that they remain synchronized.
This closes configuration drift for the upstream-compatible defaults but does
not resolve the separate user decision about whether those defaults should be
the public policy. Commit `31cb623fe`.

The server golden comparator now masks per-run nested identity fields (IDs,
signatures, timestamps, and sequence numbers) while retaining strict request,
content, usage, and token comparisons. Mutation coverage proves that identity
changes are ignored and content changes fail. The upstream G1-G9 runtime
corpus is still unavailable, so this remains a host contract rather than a
cross-implementation server qualification. Commit `607751bbe`; receipt:
`~/tmp/agent-collab/ds4-min/20260930-server-golden-receipt.md`.

The server now uses the upstream sampling defaults when request controls are
omitted (`sampler.Default`); an explicit `temperature: 0` remains greedy.
Deterministic HTTP fixtures now state that choice explicitly, and repeated
server tests pass. Commits `6f4a28b76` and `2ccfaa0eb`. This is host/API
parity; the upstream runtime transcript corpus is still unavailable.

Quantized expert admission now has a host-side dispatch plan. `dsv4tools.PlanExpertDispatch`
accepts the supported IQ2_XXS/Q2_K layout for CUDA and Metal before allocation,
and rejects Q4_K, MXFP4, mixed q2-q4, CPU, unknown backends, and malformed
layer layouts. Mutation tests cover each rejection. Commit `b4c316d10`.
This is an admission gate only; Q4/MXFP4 kernels and physical full-model
quantization parity remain unmeasured.

The model configuration path now validates the complete supported shape before
backend use. Invalid dimensions, non-finite clamps, malformed metadata, and
unknown compression ratios fail closed; an unknown ratio no longer silently
falls into the coarse route. Mutation tests cover invalid routes and NaN
clamps. Commit `2756c02d9`; receipt:
`~/tmp/agent-collab/ds4-min/20260930-foundation-slice2-config.md`.
Device-backed State snapshot/persistence remains open.

The G7 hard-smoke comparator now regrades every terminal output with the local
case table before comparing the two traces. It rejects unknown case identities,
missing answer lines, producer `picked` values that disagree with the output,
and statuses that disagree with the grader; matching fabricated status fields
can no longer produce a host PASS. Mutation coverage is in `dsv4eval` and the
full host test and vet suites pass. This is a fail-closed host receipt
improvement only: upstream hard-smoke traces, physical model execution, and
cross-implementation G7 qualification remain UNMEASURED.

The KV compatibility inspector now accepts a seekable upstream KVC file plus
its caller-verified size. `InspectUpstreamFile` validates the fixed metadata
and declared text/payload bounds without reading the opaque payload, computes
trailer bytes, and rejects truncation or oversized input before body access.
This is a host inventory gate, not payload restoration: the backend-owned
upstream state remains unread and unsupported. Commit `262163ab3`; focused
`go test ./dsv4kvstore`, full tests, vet, and diff checks pass.

`dsv4bench.Compare` now provides a strict paired analysis of two benchmark
tables. It rejects empty, malformed, or differently shaped workloads and
reports base, target, and target-minus-base ranges for prefill TPS, generation
TPS, first-token latency, and steady-state TPS. This makes the paired delta
explicit without promoting it to a performance verdict; the campaign still
requires its warmup, ordering, host-condition, and digest gates. Commit
`4f24d747d`; focused bench and full host tests pass.

The `ds4-bench` command now exposes the paired analysis as
`-compare-base FILE -compare-target FILE`, emitting the same JSON contract and
failing closed on missing or conflicting inputs. Commit `667f2b39a`; CLI and
bench package tests pass. This remains analysis tooling, not a runtime or
qualification receipt.

The exact-sampling self-oracle now runs the registered distribution check for
each possible draft token and includes an always-accept mutation that the gate
rejects. Commit `b107fec58`; `go test ./dsv4` passes. This is a host sampling
receipt only; no device verifier or speculative runtime qualification is
claimed.

`dsv4kvstore.InspectUpstreamPath` now opens and stats a KVC path, rejects
missing/non-regular or oversized files before body access, and delegates to
the bounded metadata inspector. Commit `9461d1f56`; focused KV tests pass.
The backend-owned upstream payload remains opaque and unsupported.
The remaining server G1-G9 runtime slice is HOLD: local golden comparison and
mutation coverage are complete, but the upstream `ds4srvgolden` corpus and
pinned runtime receipts are unavailable. No cross-implementation server
qualification is claimed. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-server-golden-next-hold.md`.

Speculative host coverage now exercises every accepted-prefix length, including
zero-accept restoration and rejected-draft retention. Commit
`8e344cd17b8cc174cd60f5fd756bce23f027b782`; receipt:
`~/tmp/agent-collab/ds4-min/20260930-speculative-prefix-host-gate.md`.
Device snapshot/restore and physical speculative qualification remain
UNMEASURED.

The host quant references now cover Q4_K packed high scale/min bits and MXFP4
E8M0 exponent and signed-zero encodings, with a mutation-sensitive decoder
gate. Commit `584d78bdb`; receipt:
`~/tmp/agent-collab/ds4-min/20260930-quant-reference-edge-gate.md`.
This does not qualify CUDA or Metal quantized variants.

The Phase 2 G1-G3 audit found the host persistence contract complete: round
trip, Save-Load-Save identity, split invariance, malformed/truncated input,
and failed-load atomicity all have tests. CUDA/Metal snapshot persistence is
HOLD pending device synchronization/copy/graph policy and physical receipts.
Receipt: `~/tmp/agent-collab/ds4-min/20260930-phase2-g1-g3-audit-hold.md`.

The CUDA and Metal State adapters were audited again and remain deliberately
fail-closed with `dsv4.ErrUnimplemented`; current tests cover refusal,
non-aliasing, close behavior, and host persistence. No safe non-collision-owned
implementation remains. Physical State persistence is HOLD pending an
authorized device payload/synchronization contract. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-device-state-adapter-audit-hold.md`.

Read-only upstream KVC compatibility now has bounded host loading through
`dsv4kvstore.ReadUpstream` and `ReadUpstreamPath`. The envelope is validated
before allocation and text, opaque payload, and trailer bytes are returned;
backend restoration and cross-implementation logits remain UNMEASURED.
Commit `0f3cf2a2e`.

The upstream KVC sample is retained outside the worktree at
`~/tmp/ds4-agent-kvc/20261004-sample`, with manifest SHA-256
`b80e3bcf5ac43dc428e5e79778dfa6945951d7a98bd0c73f8e331f965cf2de42` and
receipt `~/tmp/agent-collab/ds4-min/20261004-F7FA2607-kvc-sample-receipt.md`.
The cold file contains 2057 rendered-chat token IDs and a 52,292,016-byte
opaque DSV4 payload from clean upstream `de2a9491b`; it is an authority for
future restore testing, not a restore qualification. Commit `0df849d47`
corrects the host metadata names for payload words 9 and 10 to the upstream
shared head and indexer dimensions.

The follow-up G4 interoperability audit is explicitly **HOLD**, not a host
PASS. The pinned upstream file can be read into a bounded opaque envelope, but
the required save -> fresh process -> load -> continue gate cannot run until
CUDA/Metal State codecs define synchronization, device-copy, and graph
semantics. Receipt: `~/tmp/agent-collab/ds4-min/20260930-g4-upstream-kvc-hold.md`.

A bounded CUDA smoke passed on DGX under `gpu-lock` for the existing Q8_1
activation/batch quantization and matvec qualification tests plus the
byte-equal block quantizer control. The speculative test pattern matched no
tests in that older tree, so no speculative result is claimed. The run was on
the separate `codex/ds4-min` tree at
`0af7ff20f679e487476737627c4878bac9c890a9`, which had unrelated dirty CUDA
files; it used no model and no cross-implementation logits. This is evidence
for the bounded synthetic/device adapters only, not production speculative
snapshot/restore or full-model quantized parity. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-cuda-bounded-qualification-receipt.md`.

The current ds4-min source was then run in an isolated DGX worktree at commit
`3ad52de0c` under `gpu-lock`. Q8_1 scalar and batched quantization/matvec,
`TestCUDAQuantMMQBlocksByteEqual`, and the bounded speculative batch and
selected-small-batch kernels all passed on GB10 (`rc=0`, 10.804s). This still
does not qualify full-model quantized parity or production State snapshot/
restore; the exact test output and provenance are in
`~/tmp/agent-collab/ds4-min/20260930-current-branch-cuda-qualification-receipt.md`.

A follow-up audit of specs 01, 05, and 06 found no additional safe host-only
State, KV, or speculative implementation to land. The existing host contracts
and mutation gates are complete; CUDA/Metal State persistence still needs an
owner-defined device payload, synchronization, copy, and graph policy, and
upstream KVC restore needs a version-qualified codec. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-state-host-gap-audit-hold.md`.

The G7 hard-smoke comparator now exposes `ran` and `skipped` denominators and
refuses PASS when any of the twelve cases is incomplete. The new mutation test
proves an incomplete trace cannot pass, while terminal-case validation remains
strict. Commit `5257b4641`; `go test ./dsv4eval ./cmd/ds4-eval` is the focused
host gate. This hardens the host comparator but does not supply the absent
upstream runtime corpus or a physical G7 execution receipt.

The upstream KVC reader now has a version-qualified header inspection slice in
`dsv4kvstore.InspectUpstreamPayload`. It validates the fixed 52-byte metadata
header, context/raw-cache geometry, token bounds, and raw-live bounds without
allocating tensor state; mutation and truncation tests pass. Commit
`2420908fc`; receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-g4-payload-header-receipt.md`.
This narrows the G4 input contract but does not restore opaque backend payloads
or qualify fresh-process continuation.

The CUDA request State now has an owner-bound device-resident snapshot and
restore implementation. It synchronizes before copying mutable request
buffers, preserves compressed-cache row counts, request position, and host
logits, and rejects cross-runtime restores. The opt-in DGX GB10 gate passed at
commit `ccfe203ce` (`rc=0`, one `gpu-lock` lease): evaluate, snapshot, mutate,
restore, replay, and byte-compare logits, plus foreign-state rejection. The
first runtime attempt exposed zero-length optional-buffer allocation and was
fixed in `ccfe203ce`; host tests and vet remain green. Legacy serialized
Save/Load, Metal persistence, and graph interaction remain UNMEASURED.
Receipt: `~/tmp/agent-collab/ds4-min/20260930-C8332C52-state-runtime-receipt.md`.

The current branch's CUDA quantized/speculative synthetic qualification suite
passed on DGX GB10 under one `gpu-lock` lease: Q8_0/Q8_1/Q8_K quantizers and
matvecs, expert paths, Q8_1 batching, MMQ byte equality, QKV D2R, HC post, and
small-batch speculative kernels (`rc=0`, 16.447 s). This is a device receipt,
not full-model cross-implementation parity or a registered quant digest. The
separate raw-attention qualification still fails at capacity 128, positions
127-129, so that path remains unqualified. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-quant-physical-receipt.md`.

The CUDA State now exposes a rollback-capable scalar `dsv4.SpeculativeVerifier`.
Its DGX gate passed through the public `dsv4.NewSpeculator` path: a two-token
proposal accepted one token, rejected the second, restored the frontier, and
replayed the accepted prefix (`accepted=1 committed=1`, `rc=0`). Commit
`383750814`; receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-speculative-runtime-receipt.md`.
This qualifies correctness and rollback only; batched speculative performance
remains unmeasured. Serialized State persistence is recorded below.

The Metal request-state adapter now has a bounded versioned Save/Load format
covering mutable buffers and layer counters. Malformed payloads are validated
before mutation, with regression tests for capacity and rollback behavior.
Commit `94710e6bc`; host and Darwin-target tests pass, but Metal hardware
round-trip remains unmeasured. The seeded decode comparator is now also
fail-closed on matching model/context/temperature/seed provenance, contiguous
steps, valid digests, and early token divergence (`27a5daad9`). It is a
host-artifact gate only; no C-vs-Go GPU trace has been qualified.

CUDA State serialization is now runtime-qualified. Under one DGX GB10
`gpu-lock` lease, Save after two evaluations, mutate with a third evaluation,
Load, and replay produced byte-identical logits (`rc=0`, payload 54,595,444
bytes). Test commit `46b882e8d`, implementation `8e047eccf`; receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-state-saveload-receipt.md`.
Metal hardware persistence, graph interaction, and batched speculative
performance remain unmeasured.

The seeded-decode artifact gate is now a committed host tool. `ds4-eval
-compare-seeded LEFT:RIGHT` strictly reads JSON traces, rejects unknown or
trailing data and incompatible provenance, and fails before the first differing
logit digest when sampled tokens diverge. Commits `abcabd5ec` and
`587c393d8`; focused tests and vet pass. This does not qualify the GPU gate:
the pinned upstream binary does not emit a matching per-step logit trace and
no candidate binary is pinned to this branch. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-seeded-gpu-hold.md`.

Phase 4 cross-implementation quantization remains HOLD. Synthetic GB10 Q8,
expert, batching, MMQ and small speculative cases pass, but raw-attention
capacity-128 cases fail their registered tolerance and there is no matched C
oracle, per-profile logit corpus, or cross-implementation mutation receipt.
Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-quant-phase4-hold.md`.

The upstream server G1-G9 runtime corpus also remains HOLD. The pinned source
and model are hashable, but no owner-approved fixed request corpus, recorder,
or source-to-binary receipt exists; post-tokenizing rendered responses is not
provenance-safe. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-server-golden-probe-2-hold.md`.

The seeded decode artifact path is now exercised on the DGX with a pinned
candidate (`93a060e24`, numeric-gate commits `0f381a73b`/`7710256b0`) and the
11e8cad CUDA oracle. Both arms emitted 16 raw per-step logit vectors for the
same 2048-token stream under one GB10 `gpu-lock` lease. Top-1 tokens agree
through step 8 and diverge at step 9; the first step already has max absolute
logit error 4.05506802, far outside the registered F5 finite-logit tolerance.
The §5.4 attribution shape is exercised, but numeric F5 remains unqualified.
Disposition is **PARTIAL / numeric parity unqualified**, not a performance
result. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-seeded-gpu-receipt.md`.

The Metal state adapter now validates its serialized header sizing against the
configured attention and indexer counts rather than relying on the fixed
Flash shape. Commit `34dd290c2` adds the Darwin-target invariant test and
passes the host and cross-compiled Darwin checks. This is a host-layout gate;
Metal snapshot/restore and physical split-invariance remain UNMEASURED.
Receipt: `~/tmp/agent-collab/ds4-min/20260930-state-metal-header-receipt.md`.

The quant dispatch host gate now exercises every layer in the 43-layer
CUDA/Metal dispatch plan and rejects a Q4 substitution at each layer. Commit
`568f6a0ce`; `go test ./dsv4tools` and the full host suite pass. This improves
mutation coverage but does not qualify physical quantized or cross-
implementation parity. The server G1-G9 audit likewise found no safe runtime
implementation without the pinned upstream corpus; receipt:
`~/tmp/agent-collab/ds4-min/20260930-server-g1-g9-audit-hold.md`.

The current branch (`14bd24e04`) was rebuilt in an isolated DGX worktree
(`vcs.modified=false`, binary SHA-256
`65f66f492cb7bec375cf159fab4a67b4ebf31696921470599a6d522bc217bcad`). Under
separate `gpu-lock` leases, the focused CUDA State, speculative, and bounded
quantization suite passed. Including raw-attention tests reproduced only the
known tolerance failures at positions 127-129. This is a current-branch
runtime receipt, not full cross-implementation parity. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-current-branch-dgx-receipt.md`.

Metal request snapshots are now implemented as immutable serialized state
copies with backend-owner and closed-snapshot checks. Commit `4fedae775`
passes the Darwin-target compile and host tests for round-trip metadata,
foreign ownership, and closed snapshots. Metal hardware and graph-interaction
qualification remain open. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-metal-snapshot-receipt.md`.

Seeded decode numeric localization remains HOLD. The raw step-0 vectors have
correlation 0.99343, median absolute error 0.369, p90 0.913, and max absolute
error 4.055; 9,412 of 129,280 logits differ by more than one unit and 80 by
more than two. The error is broad rather than a single output-head element,
but endpoint-only artifacts do not identify the first device stage. Matched
stage dumps (normalized output, Q8_1 activation/scales, output-head partial
sums, then logits) are required before a CUDA change. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-seeded-numeric-localization-hold.md`.

A fresh F5/F6 audit confirms that no CUDA fix is justified from the endpoint
artifacts: step 0 already differs, later token divergence follows the first
logit difference, and the retained C dump cannot expose the normalized
output-head input, Q8_1 bytes/scales, or partial sums needed to localize it.
The next safe slice is a matched opt-in stage-dump contract with C-vs-C and
Go-vs-Go controls plus mutation tests. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-seeded-f5f6-audit.md`.

The server CLI audit remains HOLD: `dsv4server.New` intentionally requires a
caller-owned Model, while GGUF/backend construction is private to the main
package. Exporting an adapter would guess model lifetime and State ownership,
and the pinned upstream G1-G9 recorder/corpus is absent. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-server-cli-audit-hold.md`.

Upstream KVC interoperability also remains HOLD. The version-qualified header
inspection is tested, but restoring the opaque payload still needs a backend-
owned decoder, CUDA/Metal synchronization and graph policy, exact layout
checks, and a fresh-process oracle receipt. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-g4-kvc-audit-receipt.md`.

A second audit of the pinned upstream payload confirms the same boundary. The
13-field envelope and DSV4 payload layout are source-specified, but importing
it would still require an owner-approved backend decoder, model/quantization
checks, ring/count assignment, graph invalidation, stream synchronization, and
fresh-process restore/continue evidence. No decoder was guessed or added.
Receipt: `~/tmp/agent-collab/ds4-min/20261001-C8332C52-g4-kvc-decoder-audit.md`.

The F4-F8 cross-implementation audit remains PARTIAL/HOLD: F4 renderer
qualification is 288/288, F5 numeric seeded parity fails at step 0, F6 only
attributes later token divergence after the first logit difference, F7 lacks a
retained frontier sweep, and F8 has no completed noise-control pair. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-cross-impl-f4-f8-audit-hold.md`.

The physical G7 evaluator smoke is also HOLD. The pinned oracle lists the 12
hard-smoke cases and a lock-protected run started, but the candidate currently
has no equivalent case runner, licensed payload corpus, grader, or strict
trace writer. No comparison was made. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-eval-smoke-runtime-hold.md`.

Raw-attention capacity-128 remains a narrow oracle HOLD. On DGX GB10,
capacity 4 passes; positions 127-129 fail only chronological-score checks
(`2.12e-5`, `2.26e-5`, and `4.53e-5` against their tolerances), while shape,
cache-row, output, and fused-vs-scalar bit checks pass. A temporary
float32/theta-recurrence probe reproduced the same errors and was reverted;
production CUDA was not changed and tolerance was not widened. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-raw-attention-capacity128-hold.md`.

The physical G7 runner audit is HOLD: the repository has the strict trace
writer, parser, comparator, provenance checks, and mutations, but no
owner-approved hash-pinned 12-case prompt/answer corpus or supported model
runner that emits complete traces. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-g7-runner-hold.md`.

Seeded numeric stage probing is likewise HOLD. Existing Go dumps lack schema
sidecars and the retained C observer records only one raw-attention coordinate;
neither exposes normalized output, Q8_1 scales/codes, or output-head partial
sums. A versioned bounded C/Go stage-record ABI with positive and adjacent
negative controls is required before tooling can be added. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-numeric-stage-probe-hold.md`.

Speculative G1-G3 host contracts are complete and fail closed, but device
qualification remains HOLD. The missing evidence is DSpark/MTP support and
real-device runs for the eight-prompt greedy gate, per-buffer snapshot cases,
and exact-vs-nonspec sampling distributions. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332A52-spec06-g1-g3-audit-hold.md`.

The Darwin/arm64 Metal adapter now has a scalar rollback-capable speculative
verifier (`b1b81a701`). It uses the existing request snapshot/restore contract,
ordinary Eval calls, deterministic argmax, and accepted-prefix replay, and it
fails closed for foreign or incomplete state. Host tests and a Darwin-target
compile pass. This is an implementation increment, not a G1-G3 qualification:
the eight-prompt greedy gate, full snapshot cases, exact-sampling distribution,
Metal hardware run, and batched speculative performance remain open. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-metal-speculative-verifier-receipt.md`.

The CUDA ballast pressure control is now exposed as the validated `-ballast-gib`
CLI flag. It preserves `DS4_CUDA_BALLAST_GIB` when the flag is absent, rejects
non-positive explicit values, and passes the full host test and vet suites. This
is a reproducible host/CLI entry point for the existing DGX pressure control;
it is not a new parity or performance result. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-C8332C52-ballast-cli-receipt.md`.

The existing CUDA mmap warming path is now exposed as `-warm-weights` with an
explicit presence bit, so `-warm-weights=false` overrides
`DS4_CUDA_WARM_MODEL` while omission preserves the prior environment/default
behavior. Parser and Linux selector tests pass; this is a CLI/control
implementation, not a warm-vs-cold performance or upstream-default claim.
Receipt: `~/tmp/agent-collab/ds4-min/20260930-warm-weights-cli-receipt.md`.

A fresh audit of Amend 01 found no further source-safe patch. Backend/State
contracts, independent CUDA/Metal request allocation, server State slots, and
host persistence gates already exist. Full multi-request CUDA isolation, Metal
hardware/graph qualification, and the upstream server/KV corpus remain
runtime-level holds; changing ownership or promoting the adapters would guess
at device lifetime and graph semantics. Receipt:
`~/tmp/agent-collab/ds4-min/20260930-state-contract-amend01-audit.md`.

The current committed tree `c24cf7c4b` passed the registered CUDA State gates
on DGX GB10 under `gpu-lock`: speculative verification accepted one token and
committed one, snapshot/restore replayed bit-identical logits and rejected a
foreign State, and serialized Save/Load replayed 129,280 logits bit-identically
from a 54,595,444-byte payload. This strengthens the CUDA single-request
qualification but does not close Metal hardware/graph or full multi-request
isolation. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-current-branch-state-gates.md`.

The interleaved CUDA State gate now passes on DGX GB10 at commit `e863bdf59`:
two request states evaluated alternating token/position sequences and matched
serial reference states bit-for-bit across three positions. This strengthens
single-process multi-request isolation for CUDA, but is not a concurrent,
Metal, graph, or multi-process qualification. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-state-interleaved-receipt.md`.

The F7 tooling gap now has a strict JSONL frontier comparator (`afaf2a302`).
It validates workload/provenance alignment, top-1 agreement, absolute and
relative tolerances, and first failure, with mutation tests and explicit CLI
flags. This is a host evidence improvement; it does not create the missing
upstream frontier corpus or a physical F7 receipt.

A same-runtime batch-vs-serial split probe is explicitly HOLD. Its DGX run
found `max_abs=0.85561305` at logit 61784, so the result is not a rounding-
level split difference and cannot qualify the BatchEvaluator contract. The
probe was removed from the passing interleaving gate; route/position and
reduction-contract localization are required before changing tolerances or
implementation. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-state-split-hold.md`.

A fresh spec-06 audit found no safe implementation beyond the existing scalar
rollback verifier and bounded selected-expert adapters. A fused batch path
still needs pinned DSpark/MTP semantics, KV/compressor/indexer row mapping,
graph policy, and device mutation controls. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-D2546E0A-spec06-batch-audit-hold.md`.

The host quantization surface now classifies the four source-documented Flash
expert layouts (pinned IQ2_XXS/Q2_K, the final-six-layer Q2/Q4 splice, all-Q4_K,
and MXFP4) with strict 43-layer and dimension checks. Backend dispatch remains
limited to the pinned IQ2/Q2K pair; no unqualified kernel support is implied.
Focused `dsv4tools` tests and vet pass. Commit `683bbbad3`; physical Q4/MXFP4
kernel and cross-implementation gates remain open.

### 2026-10-01 server G1-G9 corpus re-audit: HOLD

The Phase 2 server surface was re-audited against the pinned upstream source
at `11e8cad4473716a41baf4d62e92ed9764a00fce6` and the available DGX tree.
The current `dsv4server` package still has the safe host-side seam and strict
txtar comparator (`dsv4server/golden.go`, with mutation tests in
`golden_test.go`). It does not have a cross-implementation transcript corpus.

The upstream checkout contains `ds4_server.c`, `docs/SERVER.md`, server
integration tests, and model-local top-k vectors, but no `ds4srvgolden` helper,
server request/response/token transcript corpus, or checked-in
`cmd/ds4-server/testdata` matching spec 04. The upstream `local-golden.vec`
files are model logit/top-k fixtures, not HTTP server transcripts and cannot
prove G1-G9. A DGX search likewise found no retained server-golden or
transcript artifact. The pinned oracle binary and model alone are insufficient:
recording still needs an owner-approved fixed request corpus and a reproducible
trace/response receipt.

No safe source-backed implementation slice was found. Adding synthetic
fixtures, a guessed recorder, or a backend/model adapter would either duplicate
the existing self-oracle or invent the missing provenance and ownership
contract. G1-G9 remains `UNMEASURED / HOLD`; no parity claim is promoted.
Receipt: `~/tmp/agent-collab/ds4-min/20261001-C8332C52-server-g1-g9-reaudit-hold.md`.

### 2026-10-01 seeded decode and G7 eval follow-up: HOLD

The Phase 03 seeded-decode follow-up remains unqualified. The existing
artifact comparator is fail-closed, but the retained cross-implementation
artifact already differs at step 0 (`max_abs=4.05506802`). The available
diagnostics do not expose matching normalized-output, Q8_1 code/scale, or
output-head partial-sum stages. A guessed decoder or a repeat of the endpoint
run would not localize the failure. Closing this gate needs an owner-approved
bounded C/Go stage-record contract with C-vs-C, Go-vs-Go, positive, and
negative controls. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-seeded-decode-slice-hold.md`.

The Phase 05 G7 eval smoke also remains HOLD. The host comparator validates
strict provenance, case counts, and complete traces, but the required
owner-approved hash-pinned 12-case prompt/answer corpus and a supported
candidate evaluator that emits complete traces are absent. Assigning
metadata-only cases or inventing a corpus would invalidate the gate. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-g7-eval-smoke-slice-hold.md`.

### 2026-10-01 State BatchEvaluator contract: source fix

The State adapter no longer routes `EvalBatch` through the unqualified
optimized CUDA batch schedule. It evaluates each token through the ordered
State path, restoring the public contract that batch evaluation is
observationally equivalent to ordered `Backend.Eval` calls. The optimized
legacy runtime path remains available but is not promoted without a separate
split-invariance proof. Commit `8e782d7dd` is signed; host test and vet pass.
Receipt: `~/tmp/agent-collab/ds4-min/20261001-D2546E0A-state-split-fix-receipt.md`.

The Amend 01 State host audit found no additional safe implementation slice.
The remaining contract is runtime-owned: device allocation/lifetime, stream
synchronization, graph invalidation, Metal qualification, and upstream KVC
restore/continue. A host lock, copy, or decoder patch would guess those
semantics. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-amend01-state-host-audit-v2.md`.

### 2026-10-02 resumed feature audit: HOLD

The checklist reconciliation confirms the resolved B2/B3/B4 decisions and the
existing State slices, but no newly unblocked implementation. Remaining gaps
are full multi-request/Metal/graph State qualification, F5-F8 numeric and
frontier evidence, seeded decode, KV/server runtime gates, speculative G1-G3,
physical Q4_K/MXFP4 admission, and physical G7 execution. Receipt:
`~/tmp/agent-collab/ds4-min/20261002-C8332C52-checklist-audit.md`.

The physical-gap audit found no safe performance edit without a new DGX
capture; the next evidence-bearing gate is a matched seeded-decode stage
observer at position 9. The retained endpoint failure is real but does not
localize normalized output, Q8_1 activation, output-head matvec, or another
device stage. Receipt:
`~/tmp/agent-collab/ds4-min/20261002-C8332C52-physical-gap-audit.md`.

The observer-owner audit found no newly authorized broad observer ABI. The
retained scalar/window observers leave delegated routes, cuBLAS, graph replay,
model/session plumbing, and full-model parity unread. Receipt:
`~/tmp/agent-collab/ds4-min/20261002-C8332C52-observer-owner-audit.md`.

The resumed KVC, server, and physical-quant searches independently confirmed
the same boundary: KVC has headers but no safe body decoder contract; no
owner-approved server transcript corpus exists; and Q4_K/MXFP4 lacks an
admitted production route, packing ABI, fixtures, and cross-implementation
oracle. Receipts:
`~/tmp/agent-collab/ds4-min/20261001-kvc-contract-recheck.md`,
`~/tmp/agent-collab/ds4-min/20261001-server-artifact-recheck.md`, and
`~/tmp/agent-collab/ds4-min/20261001-quant-admission-recheck.md`.

The speculative host sampler now rejects invalid exact-random draws
(`negative`, NaN, Inf, and values >= 1) instead of accepting them. Four
mutation cases guard the fail-closed behavior. Commit `c648a891b` is signed;
focused `dsv4` tests and diff checks pass. Device speculative batching and
snapshot/restore qualification remain open. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-D2546E0A-spec06-host-draw-gate.md`.

The quantized-device audit confirms that Q4_K/MXFP4 cannot be added safely by
copying isolated upstream kernels: ds4-min lacks production admission,
activation packing, artifact ownership, route selection, graph policy, model
fixtures, and cross-implementation oracles. The existing host codecs remain
arithmetic references only. Physical Q4/MXFP4 support is HOLD. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-quant-device-hold.md`.

The `dsv4server` default request-body limit now matches upstream's 64 MiB
bound rather than the previous 1 MiB local default. `TestHandlerDefaultBodyLimit`
guards the value. Commit `e825614cd` is signed; focused server tests and vet
pass. This is a host API compatibility fix, not a server transcript parity
receipt.

The backend-independent State package now has a mutation-backed host test for
the `BatchEvaluator` contract: a serially equivalent implementation passes,
while a batch implementation that changes token values fails. This protects
the contract restored by `8e782d7dd`; it is not a CUDA optimized-route or
cross-implementation qualification. Commit `267de9dd7` is signed.

The Darwin State adapter now has an ownership gate: closing a borrowed request
runtime must not close the backend-owned model file. The new test passes under
the Darwin-target host build; physical Metal F1, graph interaction, and
multi-request hardware qualification remain HOLD. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-metal-state-ownership-gate.md`.

The refreshed upstream-KVC audit confirms the bounded reader is the maximum
safe source-backed slice. A backend-specific decoder still needs an
owner-qualified CUDA/Metal payload contract, synchronization and graph policy,
model/layout/quant checks, and a two-way fresh-process restore/continue gate.
No opaque payload decoder was added. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-g4-kvc-audit-receipt.md`.

### 2026-10-01 Amend 01 independent State host audit: HOLD

The backend-independent `dsv4` State contract, request adapter, persistence
mutation tests, CUDA snapshot/Save-Load gates, and sequential interleaved
CUDA gate are already present. A source audit found no additional safe
host-only ownership patch: the remaining requirements are device allocation,
stream synchronization, graph lifetime/invalidation, Metal hardware
qualification, and upstream KVC restore/continue. Adding host locks, copying
opaque device buffers, or guessing the upstream payload decoder would change
the contract without an owner-approved runtime schema. Host tests and vet
pass. Receipt:
`~/tmp/agent-collab/ds4-min/20261001-C8332C52-amend01-state-host-audit-v2.md`.

### 2026-10-01 State interleaved CUDA qualification

The current State adapter passed the two-request interleaved isolation gate on
DGX GB10. Each request's logits matched its serial reference while the
requests were interleaved; the optimized legacy batch route remains outside
the State contract. The locked run used source `6114095187fce7f523a6140e03c4655c765ba7cc`,
State fix `8e782d7dd5b8378613016da44f1fccbc95df62de`, test binary
`5db29e222da513c52f0dfa6fbac41c8f879618947a0d12a02f2e7893a8e4ac50`, NVIDIA
GB10 driver 580.173.02, and model
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
Receipt:
`~/tmp/agent-collab/ds4-min/20261001-D2546E0A-state-contract-dgx-receipt.md`.

### 2026-10-01 remaining feature-slice audit: HOLD

An independent source audit found no additional safe host-only implementation
slice for the unchecked features. CUDA/Metal State payload and graph policy,
upstream KVC restore, the pinned server transcript corpus, seeded decode and
F4-F7/G7 runtime artifacts, physical quantization support, and production
speculative qualification all require external artifacts or backend/device
ownership. Adding a guessed parser, adapter, or synthetic corpus would
manufacture evidence rather than advance parity. Focused host tests and vet
pass; the goal remains active pending those runtime inputs.
Receipt:
`~/tmp/agent-collab/ds4-min/20261001-D2546E0A-goal-features-host-slice-hold.md`.

### 2026-10-02 seeded decode position-2049 stage gate: MEASURED, parity HOLD

The retained locked C2/Go G3 artifacts were recomputed with byte-identical
C-vs-C and Go-vs-Go controls across all 43 layers. The ordinary Go run first
amplifies its position-2047/2048 baseline at `hc_ffn_post` L07 (ratio
1.9855) and `attn_out` L08 (ratio 1.9962). A forced-route run is mutation
sensitive but is not a clean localization control. Seeded decode parity is
therefore not closed. Receipt:
`~/tmp/agent-collab/ds4-min/20261002-seeded-stage-gate.md`.

The speculative host contract now has a regression test for commit failure:
`Speculator.Verify` must restore the pre-verification frontier when
`CommitPrefix` fails. Commit `065f80145` is signed and the focused `dsv4`
tests pass. This hardens the host rollback contract; device snapshot/restore
and production speculative qualification remain HOLD.

The version-qualified upstream KVC header inspector now validates that
`raw_live == min(tokens, raw_window)` and rejects a mutated live-row count.
Commit `56f9280dc` is signed; focused KVC/server tests and vet pass. This is
an envelope-integrity gate only: opaque backend payload restore remains HOLD.
Receipt:
`~/tmp/agent-collab/ds4-min/20261002-kvc-server-slice-receipt.md`.

Host admission now applies the shared non-expert tensor-type contract before
backend binding, rejecting unsupported dense, normalization, indexer, and
control tensors instead of failing later. Commit `3a20efa2b` is signed and
adds mutation coverage for an unsupported HC tensor. Q4_K/MXFP4 production
profiles remain unregistered and fail closed; no device parity claim is made.

The CUDA and Metal State adapters now validate and honor a requested
per-request context instead of silently cloning the owner context. Commit
`05a6dbd46` is signed, with shared bounds tests and a Linux State smoke pass.
This advances the host State contract; independent device allocation,
synchronization, graph, and physical Metal qualification remain HOLD.

### 2026-10-02 CLI default-policy audit: HOLD

The CLI default implementation is synchronized with `sampler.Default`: the
parser selects HIGH thinking and sampling enabled, while explicit `-nothink`
and `-temp 0` retain deterministic compatibility controls. Parser tests check
the default mode and exact sampler values, and the server tests cover omitted
sampling controls plus explicit greedy temperature. No source-backed host gap
remains to repair. Whether this should be the public policy is a separate
owner decision, so changing the default without that decision would be an
unsupported product change. CUDA/Metal and upstream runtime parity are not
qualified by these host checks.

Receipt: `~/tmp/agent-collab/ds4-min/20261002-C8332C52-cli-default-audit-hold.md`.

### 2026-10-03 seeded decode follow-up: HOLD

The matched C2r/Go Gx1 audit places the first clear step-2 amplification at
`hc_ffn_post` L08 (ratio 2.93) and `attn_out` L08 (ratio 2.00), with expert-set
differences beginning around L07/L08. Existing dumps do not provide an
owner-qualified matched input-stage contract, so no source fix is justified.
Receipt SHA-256:
`06b3e678100a537d7a7f5f3e4230dcf1e82c76708b64b450c3a2831a8196675c`.

### 2026-10-03 CUDA State runtime receipts: PASS, scoped

Under `gpu-lock`, the existing CUDA State snapshot/restore gate passed in
119.025s, including rejected-evaluation restore and cross-state rejection.
The existing Save/Load gate passed in 105.925s; a 54,595,444-byte state and
129,280 logits round-tripped identically. This qualifies the existing State
behavior only; it does not add an ABI or replace the prior interleaved
qualification. Receipt SHA-256:
`099a26d02d4b273454e87b8151d0d416b032a5e179440e87eafd09995a6a0a77`.

The benchmark CLI now rejects mixing frontier-comparison mode with ordinary
input, frontier, or summary modes instead of silently selecting one. Commit
`1bb572d1a` is signed and adds mutation coverage for each conflicting flag
combination; benchmark package and CLI tests pass.

### 2026-10-03 physical quantized qualification repeat: PASS, scoped

The locked DGX synthetic suite passed Q8_0, Q8_1, Q8_K, IQ2/Q2 expert,
selected Q8_1, QKV D2R, HC post, MMQ byte-identity, aligned-bit, and bounded
speculative-kernel fixtures (`rc=0`, 16.702s) on GB10 driver 580.173.02,
source `6114095187fce7f523a6140e03c4655c765ba7cc`. This repeats existing
synthetic/device coverage and does not qualify full-model parity, Q4_K/MXFP4,
or production speculative decoding. Receipt SHA-256:
`cd4430c92a433371bf75dabd540a4b5e99938a97fa15a10fcb278b8d9ab6ae0e`.

### 2026-10-03 blocker summary refresh

The top-level blocker summary was reconciled with the latest receipts. CUDA
State snapshot/restore/save/load is now an owner-bound device implementation
with locked GB10 receipts; its remaining gaps are multi-request/graph scope
and physical Metal qualification. Synthetic CUDA quantization coverage now
includes the registered Q8/IQ2/Q2 routes, selected experts, QKV D2R, HC post,
and bounded speculative fixtures. Production quantized dispatch, Q4_K/MXFP4,
and full-model cross-implementation parity remain open. This documentation
change promotes no additional parity claim.

### 2026-10-03 current-head CUDA State interleaving: PASS, scoped

The current branch was synchronized into an isolated DGX checkout and the
four-state interleaving gate passed under `gpu-lock` at commit
`6bb56ff43f1b55b8e7e67bcd84244d95459b7ccc`. Two request states were evaluated
in alternating token/position order and two independent serial references
matched bit-for-bit across three positions. The run used the pinned model on
GB10; model residency, artifact-build timing, and available memory were
retained in the raw receipt. This qualifies single-process CUDA request
isolation only; concurrent scheduling, graph interaction, Metal hardware, and
cross-implementation parity remain open.
Receipt SHA-256: `6bbb9681fde9b264001885c117a977de84f51661c1512aa4048f91e27a09636c`.

### 2026-10-03 explicit CUDA stage-dump selector: PASS (host contract)

Commit `a890ea4a5` adds the opt-in `DS4_DUMP_POSITION` selector to the
existing CUDA diagnostics. Absent selectors preserve the registered rows;
explicit non-negative positions reach the layer, QKV, attention, HC, and raw
KV dump helpers needed for seeded first-difference localization. Malformed or
negative values fail closed. The Linux parser gate passed under `gpu-lock` on
the current synchronized DGX checkout. This is a diagnostic contract only;
it does not claim numerical parity or alter the default execution path.

### 2026-10-03 decode position-2049 layer-dump contract: PASS, diagnostic

Commit `6c8a57200` removes the decode dump hook's hard-coded position-2048
guard. With `DS4_DUMP_POSITION=2049`, the current DGX build emitted all 43
`hc_ffn_post` layer files for the two-token decode run under `gpu-lock`;
before the change, the selector emitted only the L00/L01 raw-attention files.
The run used binary SHA-256
`55a8d4ffce51cd6602b275410060628e1c7c2d5897dcefc2e66e52c8249e9cff` and
returned `rc=0`. The isolated DGX checkout reported
`vcs.revision=6bb56ff43f1b55b8e7e67bcd84244d95459b7ccc` and
`vcs.modified=true`; the source change is pinned to the signed local commit
above. This qualifies observability only. The C artifacts available for the
comparison were produced by a different pinned runner, so no per-layer
cross-implementation conclusion is drawn. Raw receipt:
`~/tmp/ds4-lib-workflow/20261003-step2-compare/receipt.txt`.
The same binary without `DS4_DUMP_LAYERS` produced identical prefill and
final logit hashes, so the new host-copy hook did not explain the divergence.
The no-dump control's prefill/final hashes were
`86e45c0dff2cf9c7b16921b8df1edee3af7bef2415f30ad334ecd14ba589bf01` and
`b76ee92c3d13d28926ad09b5d04c78674b23fea5dcfa6efedd7a0c66fd60b64c`;
they differ from the retained historical G3 hashes. Because the binaries and
runner revisions are not matched, this is recorded as an unresolved
provenance discrepancy, not as a numerical regression or parity result.

### 2026-10-03 CUDA scalar speculative verifier: PASS, scoped

`TestCUDAStateAdapterSpeculatorGate` passed under `gpu-lock` on the pinned
GB10 model (`rc=0`, 100.008s). The gate exercised a seed evaluation, a
two-token speculative proposal, snapshot/restore after one rejected suffix,
and replay of the accepted prefix: `accepted=1 committed=1 tops=[5 223]`.
This qualifies the scalar CUDA verifier's state transition only; batched
verification, cross-implementation speculative parity, and throughput remain
open. Receipt SHA-256:
`06c149aef43b7a64dd87654a9bcd66586e16b933328c54a0670898660e2037d1`.

### 2026-10-03 current-checkout CUDA State persistence: PASS, scoped

On the same isolated current DGX checkout, both locked State persistence gates
passed: snapshot/restore in 100.74s and save/load in 100.13s (`rc=0`, total
201.297s). Save/load serialized 54,595,444 bytes and restored 129,280 logits.
This pins CUDA snapshot/restore and save/load behavior to the current checkout;
Metal persistence, graph interaction, concurrent scheduling, and
cross-implementation parity remain open. Receipt SHA-256:
`8afeb05c7b6d5c8e505e8aa79a61752b1934247507e656c0110bd0fdc2b18166`.

### 2026-10-03 current-checkout CUDA State interleaving: PASS, scoped

`TestCUDAStateInterleavedIsolationGate` also passed on the same current DGX
checkout under `gpu-lock` (`rc=0`, 107.015s). Two request-local states were
evaluated in alternating token/position order and matched independent serial
references. This closes the single-process CUDA isolation receipt for the
current checkout; host concurrency, graph replay, Metal, and C-vs-Go parity
remain open. Receipt SHA-256:
`f763c3b7c8e3aea34c74368c1efc962f7c61e8596df21f7e85c9fe768e064c0d`.

### 2026-10-03 CUDA State graph replay: PASS, scoped

Commit `a5dcbde42` adds the opt-in `TestCUDAStateGraphReplayGate`. On the
current DGX checkout it captured the legacy runtime's graphs, reset the
backing state, replayed the same token, and compared logits byte-for-byte;
the gate passed under `gpu-lock` (`rc=0`, 101.745s). This qualifies graph
replay across reset for one CUDA runtime. Concurrent request-state graph
replay, Metal graphs, and C-vs-Go parity remain open. Receipt SHA-256:
`9ba10db86a6d33c0840b521abaa254579c5b1d6c0d00f369c04201d4121d8993`.

### 2026-10-03 synthetic CUDA qualification regression: PASS, scoped

The current checkout passed the selected synthetic/device CUDA qualification
set under `gpu-lock` (`rc=0`, 16.941s). The set covers the Q80/Q81 aligned
fixtures, Q8K and expert routes, speculative batch contracts, attention QKV
D2R, HC post, MMQ block-byte equality, and Q81 quantization/matvec checks.
This is a synthetic fixture receipt only: it does not establish full-model
numerical parity, production quantized dispatch parity, Q4_K/MXFP4 parity,
C-vs-Go parity, or performance parity. The exact command, lock label,
checkout revision, and scope are recorded in the retained receipt:
`~/tmp/ds4-lib-workflow/20261003-synthetic-regression/receipt.txt`.
Receipt SHA-256:
`bcf6e66dff640e17a928ada482b551f0f7e4990682a7ea8526e069eeaca6f7e8`.

### 2026-10-03 teacher-state token provenance: PASS, host-scoped

Teacher-state comparison now requires a nonempty, consistent
`token_ids_sha256` alongside `forced_tokens_sha256`, compares both hashes
across state directories, and preserves both in `TeacherReport`. Missing or
mutated token-stream provenance is rejected by tests. This strengthens host
transcript validation only; it does not qualify CUDA, Metal, the upstream
corpus, or cross-implementation runtime behavior. Receipt:
`~/tmp/ds4-lib-workflow/20261003-teacher-provenance/receipt.txt`.
Receipt SHA-256:
`0f27eeb851ac24236acb8e8f9e7f80f7bb6783c7a9eba1b9fc8e2c1094576ebe`.

### 2026-10-03 CUDA State interleaving rerun: PASS, scoped

The current-checkout CUDA isolation gate was rerun on the DGX under
`gpu-lock` with `DS4_STATE_MULTI_REQUEST_GATE=1` and the pinned model. Two
request-local states were evaluated in alternating order and matched their
independent serial references (`rc=0`, 99.445s). This is a current-checkout
isolation receipt only; host concurrency, concurrent graph replay, Metal, and
C-vs-Go parity remain open. Receipt:
`~/tmp/ds4-lib-workflow/20261003-state-interleaved-current-rerun/receipt.txt`.
Receipt SHA-256:
`6b64f2f2447615a7d86449f58abd695e257d747d100b36f7dafc1c16b01d11c6`.

### 2026-10-03 e-graph Q8_1 lowering: PASS, host-scoped

Commit `57e2a5c5e` adds a `q8_1_quantize` CUDA source lowering, terminal
extraction entry, autotune launch plan, and an independent float64-backed
Q8_1 value oracle with mutation coverage. The `nvidia/x/egraph` package tests
pass. This does not enable the deferred ds4-min e-graph planner, qualify the
generated kernel on a device, or establish production/full-model parity; the
remaining selected-expert, slot-aware, and fused routed operators still need
independent device oracles. Receipt:
`~/tmp/ds4-lib-workflow/20261003-egraph-q81/receipt.txt`.
Receipt SHA-256:
`1513a886137737029366956e42028f7984aa2632713b6fa2c3f8917780cc2b70`.

### 2026-10-03 e-graph Q8_1 device gate: PASS, isolated

The generated `q8_1_quantize` kernel passed on the DGX GB10 against the
independent Q8_1 value oracle. A source mutation replacing normalized division
with raw rounding failed the oracle, as did the CPU mutation (`rc=0`, 1.050s
under `gpu-lock`). This qualifies only the isolated generated quantizer
fixture; ds4-min production dispatch, selected-expert routing, the deferred
e-graph planner, and full-model parity remain open. Receipt:
`~/tmp/ds4-lib-workflow/20261003-egraph-q81-device/receipt.txt`.
Receipt SHA-256:
`ccd6fa2dc96942bbc27a16d012155e2a037768fbe308fdec96f11ba15c27bb46`.

### 2026-10-03 raw-attention capacity-128 oracle correction: PASS, scoped

The raw-attention qualification fixture previously compared scores computed
from a host `math.Sincos` query against a device query rotated with CUDA
`cosf`/`sinf` and recurrent float32 theta. At capacity 128 this produced
false score failures at positions 127--129 while cache and output state were
otherwise valid. Commit `dccfa0a7d` keeps the analytic query comparison as an
independent check, then derives downstream score/output expectations from the
verified device query bits. The locked GB10 run passed capacity 4 and 128 at
positions 127--129; fused, unfused, and score-parallel attention tests passed.
This repairs the qualification oracle only and makes no full-model parity
claim. The device raw-attention implementation was not changed.
Receipt SHA-256: `71bb90d44f823aa89a52069b60fa6986341f355613902a9c605eed75f11ecce0`

### 2026-10-03 e-graph selected Q2_K x Q8_1 device gate: PASS, isolated

Commit `485091836` adds a slot-indexed selected-expert Q2_K x Q8_1 CUDA
lowering, extraction and autotune coverage, plus an independent CPU oracle and
device mutation gate. The generated kernel matched the oracle on the DGX GB10
under `gpu-lock`; mutating expert selection from `ids[slot]` to `ids[row]`
failed the oracle. The receipt also records and corrects the initial oracle
defect that accumulated each four-subwindow reduction. This qualifies only
the generated isolated operator and does not establish production ds4 dispatch,
full-model parity, or performance parity. Receipt:
`~/tmp/ds4-lib-workflow/20261003-egraph-q81-selected-device/receipt.txt`.
Receipt SHA-256:
`2dd2e241b908516271fb88afe5496ed6d835c4d078e38a83e470dc68378dcab6`.

### 2026-10-03 e-graph selected IQ2_XXS x Q8_1 device gate: PASS, isolated

Commit `f96870eb9` adds a slot-indexed selected-expert IQ2_XXS x Q8_1 CUDA
lowering, terminal extraction and autotune coverage, with a fixture oracle that
uses independently fixed IQ2 codebook values and a device slot-index mutation.
The generated kernel and mutation gate passed on the DGX GB10 under
`gpu-lock`. This qualifies only the generated isolated operator; production
dispatch, full-model parity, and performance remain open. Receipt:
`~/tmp/ds4-lib-workflow/20261003-egraph-iq2-selected-device/receipt.txt`.
Receipt SHA-256:
`06ab1c4f3afc99853bce4d484b2a3c0fb4e2b4177fb9d10f4bc8520f162e11bb`.

### 2026-10-03 e-graph Q4_K matvec device gate: PASS, isolated

Commit `18929ba34` adds a packed Q4_K matvec lowering, terminal extraction and
autotune coverage, with an independent packed-layout reference and a device
low-nibble mutation gate. The generated kernel matched the reference on the
DGX GB10 under `gpu-lock`, and the mutation failed as required. This qualifies
only the generated isolated operator; production quantized dispatch,
cross-implementation behavior, full-model parity, and performance remain
open. Receipt:
`~/tmp/ds4-lib-workflow/20261003-egraph-q4k-device/receipt.txt`.
Receipt SHA-256:
`a6275a83c368cd4d964170dd5411f2fbeac904cd813c03427b3cdd28c0979cfa`.

### 2026-10-03 e-graph MXFP4 matvec device gate: PASS, isolated

Commit `088770f2a` adds an MXFP4 matvec lowering with E8M0 block scaling,
terminal extraction and autotune coverage, plus an independent CPU value oracle
and a device mutation gate. The generated kernel matched the oracle on the DGX
GB10 under `gpu-lock`; mutating a packed nibble failed the oracle. This
qualifies only the generated isolated operator; production dispatch, full-model
parity, and performance remain open. The follow-up oracle indexing correction
is `44303e4b1`. Receipt:
`~/tmp/ds4-lib-workflow/20261003-egraph-mxfp4-device/receipt.txt`.
Receipt SHA-256:
`4deb7a6d575a72f49feb75d7d4421932e678f44db41fd2bd9df5f1bcfff533a8`.

### 2026-10-03 CUDA contract preflight guards: PASS, host/build scoped

Commit `bdc37749f` makes CUDA speculative batch validation fail closed before
snapshot or allocation for negative seeds, positions, and draft token IDs;
`76292bf6b` validates restored CUDA attention/indexer layer counts against
request-owned capacities before state mutation. Linux cross-build tests pass
with `CGO_ENABLED=0`; physical State runtime and batched speculative parity
remain open. The DGX CUDA State test reached initialization but timed out before
the server hello, so no device qualification is claimed.
Host receipt: `~/tmp/ds4-lib-workflow/20261003-contract-guards/receipt.txt`.
Receipt SHA-256:
`12bb13d72a9fb98e1e446db20c437bcc0dbaea54568bd6b08d10513727f4c0e8`.

Commit `2f47a7ae9` extends the speculative CUDA preflight to reject seed and
draft token IDs outside the model vocabulary before device work. Linux cross-
build tests pass; this remains a contract guard and does not qualify batched
speculative runtime behavior.

Commit `a696339ce` adds `PayloadSHA256` to the bounded upstream KVC reader and
hashes opaque payload bytes without interpreting or restoring backend state.
The focused `dsv4kvstore` tests pass. This strengthens provenance only; an
upstream payload sample, backend restoration, and cross-implementation logits
remain unmeasured.

### 2026-10-04 request-state graph snapshot gate: PASS, scoped

`TestCUDAStateGraphSnapshotRestoreGate` now checks the request-state contract
that was missing from the legacy graph-reset test: after a request captures
its QKV/FFN graphs, snapshot/restore must preserve the captured graph set and
replay the rejected token with byte-identical logits. The gate is opt-in via
`DS4_STATE_GRAPH_SNAPSHOT_GATE=1` and requires the pinned CUDA model plus
`gpu-lock`; the Linux cross-build passes with `CGO_ENABLED=0`. The gate passed
on the DGX under `gpu-lock` (`rc=0`, 111.616s) with the pinned model. This
qualifies one request-local CUDA graph path only; concurrent request graph
replay, Metal graphs, and C-vs-Go parity remain open. Receipt:
`~/tmp/ds4-lib-workflow/20261004-state-graph-device/receipt.txt`.
Receipt SHA-256:
`6b059be6ad26584d8c0ce34501db0a896015ef082017ff7fc4ad272dee6db743`.

### 2026-10-04 concurrent request graph replay: host/build gate added

Commit `5ae8d8343` adds an opt-in Linux gate that captures two request-local
graph sets independently, replays both request streams concurrently, compares
each result with its serial reference, and verifies graph counts remain stable.
The gate passed on DGX under `gpu-lock` (`rc=0`, 102.511s) with the pinned
model. This qualifies two independent request streams only; it does not claim
same-request concurrency, Metal graph safety, or C-vs-Go parity. Receipt:
`~/tmp/ds4-lib-workflow/20261004-concurrent-graph-device/receipt.txt`.
Receipt SHA-256:
`1f23677e66702a56df035be330be296175030eb24deab5915f8d2bb432cdde1e`.

### 2026-10-04 Q8_1 decode fusion lowerings: host/generated-source PASS

Commit `b20d70a3e` adds correctness-first CUDA lowerings for the mandatory
`rms_quant_q8_1` and `fused_swiglu_q8_1` e-graph forms. The RMS path uses a
serial reduction and block-max Q8_1 quantization; the SwiGLU path is slot-aware,
handles a short final block, and keeps all lanes convergent around its barrier.
The e-graph package tests pass, including generation tests for both forms.
This is generated-source coverage only; production ds4 dispatch and full-model
decode qualification remain open.

Commit `7e102e2f8` also accepts the explicit selected-matvec spellings used by
the decode FFN graph (`q2_k_selected_matvec_q8_1` and
`iq2_xxs_selected_matvec_q8_1`) by routing them through the existing
slot-indexed lowerings. Registry tests cover both the historical and explicit
spellings; no production dispatcher or device receipt is implied.
### 2026-10-04 host gate inventory and Metal fail-closed regression: PASS

Commit `b3523b45d` keeps the oracle-gate inventory synchronized with the
diagnostic and CUDA State graph gates, and makes an incomplete Darwin Metal
runtime return `ErrUnimplemented` before context validation. The focused host
suite passes, including the full `examples/ds4-min` package test. This is a
host-contract correction only; it does not qualify Metal persistence or any
cross-implementation runtime behavior.

### 2026-10-04 Q8_1 activation matvec lowerings: host/generated-source PASS

Commit `5953c249b` adds generated CUDA forms for Q8_1 activations paired with
the existing Q8_0, Q2_K, and IQ2_XXS matvec kernels. The float activation paths
remain unchanged; generation and lowerability tests cover all three new names
and verify the Q8_1 block load marker. `GOWORK=off go test -mod=mod
./nvidia/x/egraph -count=1` passes. This is an e-graph lowering and host gate
only: production dispatch, CUDA compilation on the target toolkit, device
oracle results, and ds4 decode parity remain unqualified.

An isolated DGX compatibility receipt exercised the Q8_1 fusion tests against
the remote CUDA API (`/Users/tmc/tmp/ds4-lib-workflow/20261004-q81-device/receipt.txt`).
The focused RMS/SwiGLU/selected-matvec suites passed under `gpu-lock`, but the
remote source was `6f598fee7` rather than this branch's `5953c249b`; the run
therefore demonstrates API/device compatibility only, not branch-matched
qualification. Receipt SHA-256: `94e5c39f8bd1e5037e692c8726c885e6976de710ba7df889d0fc538573f5a560`.

Commit `7e8c82ee1` adds a regression assertion that the legacy float
activation matvec still emits a direct float load rather than the Q8_1 block
helper. The focused generation tests pass; this is a source-generation guard,
not a device or full-model parity result.

Commit `4c0452f10` adds an NVRTC compile gate for the Q8_1 activation matvec
family and its selected-expert forms. The gate covers generated-source
compilation only; it does not qualify dispatch, device numerics, or full-model
decode parity.

Commit `12dfec8dc` registers the Q8_1 decode terminals with extraction and adds
matching autotune buffer/launch plans. The e-graph package tests pass. This
connects the lowering to host extraction and tuning only; production dispatch,
device qualification, and full-model parity remain open.

### 2026-10-04 selected Q8_1 terminal aliases: extraction/autotune coverage

The explicit `q2_k_selected_matvec_q8_1` and
`iq2_xxs_selected_matvec_q8_1` spellings were lowerable by the CUDA generator
but were absent from `TerminalOps`, so extraction could leave a selected
decode graph at a non-terminal alias. The aliases now participate in terminal
extraction and use the same bounded autotune plans as their historical
spellings. Host tests cover terminal registration and all four selected Q8_1
plans. This is an e-graph integration fix only; production dispatch, device
qualification, and full-model decode parity remain open.

The branch-matched Q8_1 device suite was then run in an isolated DGX worktree
at source `7342746933748ef29817e051c066ac89376e61f5` under `gpu-lock`.
Generated quantization, RMS/SwiGLU fusion, selected Q2_K/IQ2_XXS, activation
matvec compilation, aliases, and mutation tests all passed on GB10. Receipt:
`~/tmp/ds4-lib-workflow/20261004-q81-branch-matched/receipt.txt`.
Receipt SHA-256:
`4de5ec9305670922276465eabf39dd3f5271228e245082dab161084cf5ac2c12`.
This qualifies the synthetic Q8_1 suite for this source revision only; full
model dispatch and C-vs-Go parity remain open.

Commit `6bf178e38` adds independent float64 host references for Q8_1
activation matvecs across Q8_0, Q2_K, IQ2_XXS, and selected-expert layouts,
with invalid-slot and weight-mutation checks. The focused reference gate passes;
it strengthens the synthetic oracle but does not replace CUDA or full-model
qualification.

Host receipt: `~/tmp/ds4-lib-workflow/20261004-q81-reference-verify/receipt.txt`
(`7b5ae1d524d010e8b6076dd9f451c7599549b769929aa018a28dca2de3863abc`).

### 2026-10-04 decode FFN e-graph S0 gate: PASS, host/generated-source scoped

Commit `0796a9585` adds the missing full decode FFN gate. It exercises the
mandatory Q2_K scale/weight hoist, saturation, extraction, and CUDA generation,
then checks the Q8_1 RMS, SwiGLU, ordinary and selected-expert terminals. The
selected forms are checked for slot-indexed routing markers, and a non-selected
mutation is rejected. The optional NVRTC compile checks remain behind the
existing device-test capability gate. `GOWORK=off go test -mod=mod
./nvidia/x/egraph -count=1` passes. This qualifies the decode graph lowering
contract only; production dispatch, device execution, and full-model parity
remain open.

### 2026-10-04 finalized upstream server corpus: host loader PASS

Commit `3e6a03473` updates `dsv4server` to load the finalized owner-recorded
G1-G9 corpus. Root build identity remains strict; heterogeneous sampling is
bound through the root `think_mode_by_step` map, while SSE responses, missing
GET request bodies, and intentionally malformed non-2xx requests are handled
according to the corpus receipt. The loader and full package tests, Linux vet,
and a checker against the 159-file corpus pass. Corpus receipts are external:
`/Users/tmc/tmp/ds4-agent-kvc/20261004-server-corpus/corpus/RECEIPT.md`, with
`SHA256SUMS` `daed4c876e6a9770bfb59a82acd8cdaef14a6d60250eba3f5853e3601fa5a10e`
`manifest.json` `be64076eacc72b6bb3d2e86edb1b8a635c57d6604e9c49e1ca11b2baf5c40017`,
and `provenance.json`
`ba08cdb271417483d40ee1272f0cc41f0364396c67b21016ba3f92fd1bbbc3a8`.
The unchanged `determinism.json` hash is
`de6852e436d227ba34f798a9565e6b835d6ac510c1eb7078f2e07849e79ba05d`.
This is a host corpus-ingestion gate only; it does not establish ds4-min
runtime transcript, numerical, KVC restore, or performance parity.

Commit `d0a67926d` adds `dsv4server.CompareCorpusCase`, a fail-closed host
comparator for finalized cases. It compares status, canonical request JSON,
masked response JSON or SSE events, and exact token traces; mutation tests cover
each mismatch and missing-response cases. The race test, Linux vet, and diff
check pass. This is a transcript-comparison utility only and does not execute
the model or establish cross-implementation runtime parity.

Commit `2eb4b5ba5` extends the comparator to the complete corpus set. It
requires exactly one observation for every stable case/step key, rejecting
missing, extra, or duplicate observations before delegating to the per-case
comparison. Mutation tests cover the set cardinality gates; race tests and
Linux vet pass. Runtime server execution remains unqualified.

The G7 hard-smoke reader also rejects duplicate source/case identities before
grading terminal output (`6ec520148`). This closes a host denominator gap but
does not substitute for the physical cross-implementation G7 run.

Commit `8c357926a` binds the hard-smoke suite to the exact upstream 12-case
source/ID set rather than table position. Selection and validation reject
missing or mutated identities, and the CLI lists 12 cases plus its header.
This is a host denominator gate; no physical evaluator run was performed.

Commit `3a30b23af` makes raw seeded-decode conversion validate build, model,
context, temperature, seed, and MTP metadata before reading logits. Mutation
tests cover each invalid field. This strengthens artifact provenance only; the
existing cross-implementation seeded numeric gate still fails on DGX.

Commit `40b9fb399` adds direct regression coverage for heterogeneous corpus
sampling provenance. The test exercises a `per-case` root with both `high` and
`none` steps and rejects missing or mismatched step entries. The targeted race
tests pass; this verifies the host validation contract only and does not change
the runtime parity status.

Commit `99c35bcdd` preserves the root `think_mode_default` in the loaded
provenance and includes it in strict root comparisons, with a regression test.
This prevents the default policy from being silently discarded while retaining
the per-step exception for deliberately disabled-thinking cases.

Commit `aab1eb8f6` refreshes the blocker inventory so it no longer describes
the owner-bound CUDA State persistence as unimplemented. The remaining State
limits are Metal hardware persistence, same-request graph concurrency, and
cross-implementation runtime qualification.

The blocker wording is now narrowed by the State contract: a State is owned by
one request and is not safe for concurrent calls, so same-request concurrent
graph use is unsupported by design rather than an unimplemented parity mode.
The qualified scope remains request-local replay and independent request
streams; Metal hardware and cross-implementation runtime qualification remain
open.

### 2026-10-05 CUDA-backed server entrypoint: HOST PASS, RUNTIME UNMEASURED

Commit `1bcf1a9a4` adds a Linux-only `ds4-min -server` mode that binds the
existing CUDA runtime and tokenizer to `dsv4server.Handler`. It uses the
owner-bound `dsv4.Backend` to create independent request-local `dsv4.Session`
states, keeps the model/backend alive for the process, and provides an
explicit unsupported error on non-Linux builds. Focused Linux tests, Darwin
compile tests, Linux vet, and diff checks pass. No GPU workload was run for
this slice. The finalized upstream G1-G9 corpus can now be exercised by a
branch-matched server process, but numerical, KVC, transcript, and performance
parity remain unmeasured until that DGX receipt is produced.

Commit `6e0bcbb62` narrows the server prompt adapter to the upstream transcript
shape: omitted thinking uses the configured default, explicit system messages
must precede turns, assistant history is preserved with completion markers, and
the server no longer injects an implicit system prompt unless `-system` is
provided. Focused Linux tests and vet pass. This remains a host rendering fix;
the DGX G1-G9 runtime comparison is still pending.

The 2026-10-05 KVC restore audit found no safe host-only implementation to add.
The upstream payload is fully bounded and mapped, but restoring it requires a
version-qualified CUDA codec with explicit model/token provenance, synchronized
uploads, graph invalidation, and atomic rollback; the existing private
DS4CSTAT loader remains separate. Receipt:
`/Users/tmc/tmp/ds4-lib-workflow/20261005-kvc-restore/REPORT.md`.

The 2026-10-05 speculative audit found the host `dsv4.Speculator` contract
complete and fail-closed, with CUDA scalar rollback already bounded-tested. No
safe host implementation remains for G1-G3; production drafter loading,
batched verifier behavior, cross-implementation token identity, and device
gates remain unmeasured. Receipt:
`/Users/tmc/tmp/ds4-lib-workflow/20261005-speculative-audit/RECEIPT.md`.

Commit `c42c49461` closes a separate host failure path: if verification fails
and compensating snapshot restore also fails, `Verify` now returns
`ErrDesynchronized` while preserving both causes. The regression test checks
the joined errors. This strengthens fail-closed behavior but is not device or
cross-implementation speculative qualification.

Commit `1692d4924` fixes the server qualification crash at its source. Every
Linux NVRTC compile now snapshots and restores the complete SIGINT, SIGTERM,
and SIGXFSZ `rt_sigaction` records under a process-global mutex, preserving
Go's signal-handler flags. The synthetic restoration test and host package
checks pass. Commit `e2100a99e` restores graceful server shutdown on top of the
guard. This is a source-level fix, not a runtime qualification receipt; the
G1-G9 server corpus must be rerun from this exact tip under `gpu-lock`.

The 2026-10-05 DGX replay of the finalized upstream G1-G9 corpus is measured but
not qualified. It used `cb37e93c5`, binary SHA-256
`1ebbc5f5b6a3be9a85aa24702dae2a317ced76490a841d1259d4dd488c188b40`, and 32
normal process exits with zero SIGSEGVs. Receipt and comparator:
`/home/tmc/tmp/ds4-agent-kvc/20261005-server-runtime/rcb37/`. Exact matches are
limited to G1 models and malformed JSON. Remaining diffs cover generated
thinking/content placement, completion/SSE shape, usage/cache fields, disabled
thinking, incomplete output, tool-call history (500), assistant/tool history
(400), and concurrent G9 CUDA graph desynchronization. Serial A/B repeatability
is present, but token identity is UNREAD; no server parity claim follows.

Commit `ffe3c6e78` extends the host chat contract to accept assistant history
messages and adds handler coverage. This addresses the observed G8 extension
request-shape rejection without inventing a model-independent tool-result
format. It is a host contract change only; a DGX replay is still required to
measure its effect on the upstream corpus.

The targeted DGX G8 replay of `ffe3c6e78` used a clean binary
(`vcs.modified=false`, SHA-256
`e65f377c9f3f02dfc4317a1b6c1eb1be3da208b2eff135c840bba257d0ae1387`). Both
base and assistant-history extension requests returned HTTP 200, matched the
upstream prompt-token counts (12 and 35), and were A/B repeatable. The former
extension rejection is therefore fixed. The two cases still differ in three
response fields: ds4-min puts the generated `1.` in `content` rather than
leaving content empty, exposes it as `reasoning_content` instead of upstream
reasoning placement, and omits `prompt_tokens_details`. This receipt is scoped
to G8 only; the broader targeted replay of the default-thinking and usage
changes remains pending. Receipt: `/home/tmc/tmp/ds4-agent-kvc/20261005-server-runtime/rffe3-g8/`
(`compare.json` SHA-256
`81c179d7ceca1b45c589ebe9af8e694f45908df97f03afbee476f0598728dd91`).

Commit `acf5bbf96` adds a host-side G5 tool-history correction: when a tool
request's generated response begins with the empty `</think>` boundary, the
handler treats it as ordinary content rather than rejecting it as malformed
thinking output. Ordinary and explicit thinking paths retain strict marker
validation. The regression is host-only; a DGX replay is required before
claiming that the G5 transcript now matches.

The three-pass expert grouping optimization from historical commit `6d18b17a2`
is already represented on this branch by `31b5d02df`, with later dispatch
changes in `e15db0509`. The historical receipt cannot be transferred to the
current tip: a fresh `gpu-lock` qualification is required before reusing its
performance or digest claims.

Commit `942c0e7b0` adds a model-declared default-thinking capability to the
server adapter. Unmarked generated text is classified as reasoning only for a
model that declares a non-empty default thinking mode; explicit disable requests
and plain models remain content. Host tests pass; the corpus response-field
effect remains pending a DGX replay.

Commit `2ed499048` adds the cache and reasoning-token detail fields used by the
upstream chat, completion, response, and messages usage objects. The values are
derived from the bounded prompt and generated counts; transcript and handler
tests cover the wire shape. Runtime corpus revalidation remains required.

### 2026-10-07: independent request graph capture gate

Signed commits `82440ba1b` and `51e00195d` add explicit capture modes and use
thread-local capture on nonblocking request streams. Ordinary runtime capture
remains global. No request-wide serialization was added. Existing request
arenas, graph executables, streams and cuBLAS handles remain independently
owned. Thread-local capture does not permit unsafe APIs on its own capturing
thread; it removes the global prohibition on unrelated threads. Nonblocking
streams remove implicit ordering with synchronous legacy-stream copies.
The relevant contract is the NVIDIA CUDA Driver API stream-capture contract.

Host mode-forwarding, invalid-mode, context-ownership and example tests pass,
and the Linux arm64 test binary compiles. The NVIDIA tests used a scoped
temporary module file rather than modifying tracked module dependencies.

On DGX GB10, `TestCUDAIndependentGraphConcurrencyGate` creates four fresh
request states and compares concurrent first capture and next-token replay
against serial controls. The exact `51e00195d` archive test binary passed in
117.72 seconds, rc 0. SHA-256:
`c19f5b91ec6e9033800a70abe4300551d938694c2d6b6d602ab72ba478e0bc7c`.
A second binary adds only the capture-witness test hunk from old
`2ed5244e3` (re-recorded tree-identically as `41e22f7cc`), not other later
implementation commits. It is an explicitly identified test overlay, not a
clean build of `41e22f7cc`. Its SHA-256 is
`87561b726443ac1f76f236b3b23c206d8470ee287956002661d755a34a192063`.
It passed in 94.17 seconds, rc 0: all four states retained 43 QKV and 40 FFN
graphs and matched their serial-control logits byte-for-byte.

The negative build restores only GLOBAL capture in that witness source;
nonblocking streams remain. SHA-256:
`d3faf69b4c3eac17f6269908c451388bfc4d59454da3639d59d24ec6dd643ab1`.
It failed in 91.81 seconds, rc 1, at concurrent layer-0 QKV graph
instantiation with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. This tests the
mode choice at the executed capture site rather than an unrelated mutant.
All three runs held gpu-lock, and each release logged its workload return code.

Hardware UUID `GPU-0b2649cd-f5c3-ee67-5b8e-b18d60595e9d`, driver 580.173.02,
Go go1.27.1 linux/arm64. The fixture uses the registered chat-v2 IQ2XXS model,
context 128 and two evaluations per state. Raw logs and the mutant patch are
retained on both hosts beneath `~/tmp/ds4-lib-workflow/20261007-g9-graph`
(DGX) and `~/tmp/ds4-min/20261007-g9-graph` (Mac). Raw-log SHA-256:

- base: `0a3c11f1ce96f909325f1f65833b0ad7199bd6e70fe6e5639498342789256b2d`
- witness: `2c6f694ff4a4329c21d06b00989062fd1d50244acbeeba21b6705625f224b76d`
- mutant: `a655ccbe0ef1eeb5ffbe1df0910afa7dbecf64f30b166d481ea89f33a63bd49f`

This qualifies bounded independent graph capture/replay, not the HTTP G9
golden transcript or arbitrary concurrent server workloads. F7's golden G9
replay remains required before server concurrency parity can be closed.

### 2026-10-07: opt-in grouped speculative verifier

Signed compute commit `3e6f89bfb` adds an actual multi-row transformer route
with every row's vocabulary projection and finite readout. The frontier already
contains the seed, so top zero is the current frontier prediction and top i
comes from draft i-1. `f13af28da` separately rejects nonfinite frontier logits.
`DS4_CUDA_SPECULATIVE_GROUPED=1` admits this route; the default stays scalar.
Accepted-prefix commit restores the frontier and replays only accepted tokens.

On DGX GB10, the clean `04d9cd7378dc97c16703e1c6bb3193549c209d92`
test binary passed `TestCUDASpeculativeGroupedGate` under gpu-lock, rc 0,
34.12 seconds. Batches of 2 and 8 covered all accepted prefix lengths 0..N:
12 cases ran, zero skipped. Every predicted token matched ordered decode;
all 129280 logits after each prefix replay matched exactly. The fixture starts
with token zero at position zero and verifies drafts beginning at position one;
it does not cover arbitrary prompts or the ratio-128 emission boundary.

Binary SHA-256:
`d57502e007310af714f9e4064f34fba0f7ea2995303a2a53950397b168b3a0ed`.
Model SHA-256:
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
Go 1.27.1, linux/arm64. Test binaries omit VCS build settings; the clean clone
HEAD, source-bundle hash and binary hash are retained in build-receipt.txt.

An isolated readout-index mutant changes only top zero to `(first+1)%vocab`.
Binary SHA-256:
`fe32201f8467c5cd5dfbad67122268edee29a53d8aecfef9028a37aa86f2f1e2`.
It failed this same gate under gpu-lock at batch 2, row 0 (top 6, ordered 5),
rc 1, 98.61 seconds. This is a comparator/readout-wiring control, not a
device-arithmetic fault control. Both lock releases retain their workload rc.
No performance, seeded distribution or cross-implementation claim is made.

Artifacts are retained on the DGX beneath
`~/tmp/ds4-lib-workflow/20261007-speculative-grouped/` and locally beneath
`~/tmp/ds4-lib-workflow/20261007-speculative-batch/`. Raw-log SHA-256:

- baseline: `4028fde554da1dae8397c449c0d38b8a2bd57ebac1d6d8d64df3ebbf8dbb58d2`
- mutant: `b0d63d3364a528604613fd2e15bbf1fb525d7f1abff981addaa25836610a43a6`

### 2026-10-07 focused server response-contract replay: PASS, G9 open

F7FA2607 ran clean `2021a664d` over `dgx-ts`, binary SHA-256
`8591f6bf5602405d3f43809789782991866e0bee0df6008ff68afa0be2b829d2`,
with `vcs.modified=false`. G3 streaming, G4 completions, both G6 messages
modes, and G7 responses each MATCH the retained upstream corpus, with
identical normalized A/B results. All 10 fresh processes exited 0; no SIGSEGV.
Denominator: five logical cases ran, five matched; the other 15 comparator
entries are explicitly UNMEASURED in this focused replay.

The parent independently recomputed the remote comparison SHA-256:
`626664f5b9f323ce8b4b45c99209aafad4f1b108d2ac642cad8497d839c83d65`.
Raw SSE hashes differ in masked id/created fields; this is normalized-wire
equality, not byte identity of raw HTTP. Each G3 stream has five data lines.
Artifacts: DGX `~/tmp/ds4-agent-kvc/20261005-server-runtime/r2021/`;
local comparison/log mirror `~/tmp/ds4-min/20261007-server-r2021/`.

G9 serial/concurrent replay at `2b8e4caa9` remains a separate running gate.
This receipt does not qualify full G1-G9 at one branch head, device token IDs,
Metal serving, or all stream variants. Corpus model/oracle identities remain
those recorded in the finalized 92069b8b server-corpus receipt.

### 2026-10-07 CUDA cleanup and native CPU foundations: PARTIAL

Signed `2ba4ade80a131544e733479a21c94fd42e6d6a18` retains acquired
foundation resources through constructor failures, distinguishes a completed
release from a later context-restoration error, and rejects new work after
teardown begins. The coherent Linux ARM64 compile and host gate passed:
14 top-level tests, 65 leaves, zero skipped. Nine compiled production-source
mutations were rejected. These are ownership/admission controls, not model or
device numerical qualification. Child runtime, batch and snapshot cleanup is
a separate pending slice. Receipt:
`~/tmp/ds4-min/20261007-cuda-constructor-audit/FOUNDATION-LANDING-RECEIPT.md`.

Signed `ea826727f9a778351bf631bb80d932d7f51c4f61` adds native CPU tensor
decoding, scalar matrix multiplication and hyperconnection operations. The
integrated readonly race gate passed 16 tests/examples with zero skips,
including literal C controls from `11e8cad4473716a41baf4d62e92ed9764a00fce6`.
The source is pinned by SHA-256
`62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7`.
The control uses scalar association, not ARM SIMD equivalence. The locked log
is `~/tmp/ds4-min/20261007-root-review/cpu-foundation-integrated.log`.

This does not admit a CPU model backend. Attention, cache, native activation
quantization, routing, compressors, complete forward execution and State gates
remain separate integration work. Neither foundation closes the full feature
parity goal.

### 2026-10-08 completion reconciliation

This was audited at `4873c493d`, not a new runtime qualification. The newer
CPU indexer commit `362c88fcf5e5dd8f11debad90ce934eb65cb9555` is recorded below.
The full scope remains A, then B. Earlier artifact-unavailable notes are
historical where a newer retained corpus or codec exists.

| Slice | Disposition | Source and retained evidence | Remaining requirement |
|---|---|---|---|
| CPU dense/router, KV/RoPE and compressor | PARTIAL | `4f3577533`, `eedd28e70`, `4873c493d`; integrated race logs contain 46, 53 and 61 top-level tests/examples respectively, zero skips | These runs repeat previous foundations; do not add their counts. Bind and execute the complete model forward and request State. |
| CPU indexer activation | PARTIAL | `362c88`; retained `indexquant-integrated-race.log` has 65 top-level tests/examples passing, zero skips | This newer run repeats earlier foundations; do not sum 46, 53, 61 and 65. Complete model-bound forward remains required. |
| CUDA child cleanup | HOST PASS | `188d0d464`; eight tests, 35 terminal leaves, zero skips in the clean child receipt | Native free/context-restoration fault behavior remains unqualified. |
| CUDA verifier cleanup | IMPLEMENTED; physical UNREAD | `22ae9ffb5` retains failed Snapshot cleanup owners | No new physical speculation claim is made by this audit. |
| V4.1 scalar forward | IMPLEMENTED; model execution UNREAD | `645c97115`: model binding and the 40-layer scalar request path; integrated race gate has 42 tests/examples passing, zero skips | Full real-model execution, portable persistence and native admission remain required. The host gate does not execute a complete model. |
| Shape-derived CUDA cache | HOST PASS; device UNREAD | `53d1f33cb`: dynamic Flash/Pro cache planning and ownership with checked tensor extents | Shape recognition and allocation controls do not qualify Pro forward execution. |
| Vision language attention bounds | HOST PASS; image inference UNREAD | `ba4f3ab92`: 250 literal pinned-C bounds/chunk cases; integrated package race run has 24 tests/examples passing and three existing opt-in skips | Complete encoder and image-conditioned language dispatch remain required. This does not expose an image CLI. |
| Metal native Load compensation | BOUNDED PHYSICAL PASS | Clean `373e43acb` plus test SHA `1bb20efc0a30cda803196d19d51d66d14fae2002cd73e164bb7998f7da79ed6d`; two baseline branches passed, one skipped-rollback mutation failed as intended | Actual tiny shared buffers only. Real-model continuation, independent States and queued work remain required. |
| Metal teardown and scalar prefix gates | IMPLEMENTED; full-model UNREAD | `01210a044` and `bc19c9893` | Tests are opt-in; their source presence is not evidence that they executed. |
| Focused server response replay | FOCUSED PASS | `2021a664d` versus retained `92069b8` corpus; five logical cases and ten A/B processes | G9 and the full integrated-head/specification matrix remain pending. |
| Upstream KVC continuation | FAIL RETAINED | Cold 2057-token to continuation 2058-token oracle pair; KL 0.0909, maximum absolute error 4.51786 | Exact upload/readback does not replace the strict continuation gate. |
| Hard-smoke runtime | RUNNING / UNREAD | Coordinator reports an active gate; no terminal receipt reviewed in this audit | Report the twelve-case runtime/mutation denominator after completion. |
| KVC and vision device jobs | QUEUED / UNREAD | Coordinator status only | Queue admission is not a parity result; retain terminal raw logs and exact source/build identity. |

Retained receipts:

- [CPU indexer integrated race log](/Users/tmc/tmp/ds4-min/20261007-cpu-foundation/indexquant-integrated-race.log).
- [CPU integrated logs and C controls](/Users/tmc/tmp/ds4-min/20261007-cpu-foundation/).
- [Clean CUDA child ownership](/Users/tmc/tmp/ds4-min/20261007-cuda-constructor-audit/CHILD-LANDING-RECEIPT.md).
- [Tiny native Metal compensation](/Users/tmc/tmp/ds4-min/20261007-metal-state-qualification/SMALL-TERMINAL.md).
- [Focused server comparison](/Users/tmc/tmp/ds4-min/20261007-server-r2021/).
- [V4.1 scalar forward integrated race log](/Users/tmc/tmp/ds4-min/20261007-dsv41-runtime/forward-integrated-race.log).
- [Vision attention integrated race log](/Users/tmc/tmp/ds4-min/20261007-dsv41-runtime/vision-integrated-race.log).
- [Full A/B requirement audit and next ownership](/Users/tmc/tmp/ds4-min/20261008-completion-audit/MATRIX.md).

No implementation or completion requirement is removed by this reconciliation.
