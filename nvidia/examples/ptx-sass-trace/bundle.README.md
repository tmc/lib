# Go → CUDA → PTX → SASS

This is a real GPU capture of a DeepSeek-V4 (ds4) inference pipeline:
RMSNorm, Q8_0 quantized matrix-vector multiplication with warp reductions,
SwiGLU activation, and sparse MoE expert dispatch with indirect indexing,
along with sub-byte IQ2 SIMD vs scalar dot products and shared-memory staging.
NVRTC compiled the code, nvdisasm decoded the CUBIN, and the program executed
that same CUBIN on NVIDIA hardware. Every output was checked against a CPU
reference after every stage.

## Open the demo

Share just `capture.pprof`. It embeds the source, PTX, SASS, CUBIN, Go trace,
and capture manifest in standard pprof metadata. With the NVIDIA-aware viewer:

```sh
cudaprof -http=127.0.0.1:8087 capture.pprof
```

No CUDA installation or GPU is needed to view this recording. In the optional
share bundle with prebuilt viewers, `./view` opens the same pprof interface.

- **Flame Graph:** compare `vector_add` with `compute`.
- **Source:** search for `compute` to see CUDA, PTX, and SASS. Click a source
  row to expand its assembly instructions.
- **Disassemble:** search for `compute.PTX` or `compute.SASS` to choose a
  compiler stage. Search for `compute` to see both.
- **Details:** capture parameters and timing interpretation.

Command-line views: `cudaprof -top capture.pprof`,
`cudaprof -list=compute capture.pprof`, and `cudaprof -disasm=compute capture.pprof`.
Stock `go tool pprof` reads the samples and preserves the embedded files, but
needs the NVIDIA-aware viewer to interpret them as source and disassembly.

The recording includes a Go execution trace. Run
`cudaprof -extract recording capture.pprof` to extract it, then use the Go
version in `recording/manifest.json` to open `recording/go.trace` with
`go tool trace`. In the optional share bundle, `./trace` does this directly.
Open **User-defined tasks → CUDA pipeline**. PTX and SASS appear as
numbered task events; timeline regions show host launch-and-wait intervals.

## What the numbers mean

`gpu_time` is the CUDA event interval around an entire kernel launch, and it is
the default view. `samples` counts launches, one per recorded launch. Profile
cost is attached to each kernel's SASS entry; PTX arithmetic rows provide source
navigation. It is not a cost for that individual instruction.

Whether this capture holds sampled hardware PCs depends on how it was recorded.
Without `-pcsampling`, the Go/CUDA/PTX/SASS stack expresses the launch and
compilation relationship rather than a sampled hardware stack, and the capture
carries no sampled PCs or stall counters; the sample types it advertises are
only the two it can fill. With `-pcsampling`, the manifest's `pc_sampling`
object records the period, how many samples were collected, how many resolved
to a PTX statement, and what was dropped -- read those counters before trusting
the attribution, because a high resolved fraction over a batch that dropped
most of its samples is not a working measurement.

Under `-pcsampling` the hardware counts live in their own types. `pc_samples`
counts samples taken at an address, `inst_executed` the subset where the
scheduler issued, and the `stall_*` types the subset where it could not, broken
down by reason: `stall_mem`, `stall_wait`, `stall_scoreboard` and `stall_math`
for the back end, `stall_fetch`, `stall_branch`, `stall_drain` and
`stall_dispatch` for the front end, and `stall_not_selected` for warps that were
ready while the scheduler issued a different one -- issue contention rather than
latency, which is why it is not folded in with the rest.

They reconcile against the manifest exactly: the `stall_*` totals plus
`inst_executed` plus the manifest's `unmapped_stalls` equal
`pc_sampling.total_samples`. A reason with no type of its own appears only in
`unmapped_stalls`, where no pprof view reaches it, so read that map before
reading the breakdown -- on a GB10 capture it held 172 of 2765 samples across
five reasons that now have types.

A non-empty `unmapped_stalls` is a statement about the workload, not about the
classifier. It says the kernel hit a reason no sample type models -- `misc` and
`sleeping` on a GB10 -- and the identity above still closes with those samples
on the `unmapped_stalls` side. Read it as coverage to extend, not as a
regression, and read an empty one as evidence the workload stayed inside the
modelled set rather than as evidence about the routing.

The identity sums every stall type, so it stays closed when a reason is counted
in the wrong one: a fetch stall filed under `stall_mem` balances exactly and
sends a reader to the memory system for a front-end problem. `reason_routing` in
the manifest is the check the identity cannot be. It names every reason the
hardware inventory reports -- sampled on this run or not -- and where each one
would be counted: a sample type, `companion` for a `_not_issued` reason whose
count is discarded as a subset of its base, or `unmapped`. `reason_samples`
records how much each reason actually contributed, so the routing is checkable
against the same file rather than taken as a declaration.

Two type names read as answers to questions they do not answer, and both are
worth stating here because the numbers under them are correct.

`stall_mem` counts memory-pipe *throttling* -- on a GB10, `lg_throttle`,
`mio_throttle` and `tex_throttle`. A warp waiting for global memory data is
`long_scoreboard`, which is a scoreboard stall and counted under
`stall_scoreboard`. So `stall_mem` near zero does not mean the kernel is not
memory bound, and the distinction is measurable rather than argued: a shuffled
pointer chase, which is pure DRAM latency and nothing else, reports
`long_scoreboard` at 99.71% of 1.69M samples with `lg_throttle`, `mio_throttle`
and `tex_throttle` all exactly zero, while a kernel issuing wide independent
loads from the same data puts `lg_throttle` at 89.94%. Queue pressure is what
`stall_mem` sees; latency is what it cannot.

A production capture shows the same inversion from the other side, where it
costs something: a DeepSeek-V4-Flash IQ2XXS decode arm on a GB10 (4 tokens at
context 1024, captured 2026-09-13), 1313345 samples,
reports `stall_mem` at 11.75% -- entirely the throttle family, `lg_throttle`
106412 + `mio_throttle` 47631 + `tex_throttle` 271 -- while `long_scoreboard`
alone is 82.97% and `short_scoreboard` 1.68%. Sizing that workload's memory
problem from `stall_mem` gives 11.75% and points at bandwidth, when the 82.97%
is dependency latency. The split of `stall_scoreboard` into its two reasons is
what decides that, and it exists only in `reason_samples`.

One reason in `stall_mem` is observed where nothing explains it, and a reader
should know before sizing anything from that bucket: on a GB10 at driver
580.173.02, `tex_throttle` reaches 9.4% on fp64 arithmetic kernels whose SASS
contains no `TEX` instruction at all, with a threshold at eight independent
accumulators (0 / 0 / 0 / 4.89% at 1 / 2 / 4 / 8, work-matched). It is exactly 0
in every memory kernel measured on the same part, including a read-only `__ldg`
path, whose `LDG.E.CONSTANT` is not the texture unit. Either the driver binds
this counter to the wrong name -- it cannot be a mismap on our side, since
routing is keyed by the name CUPTI reports and never by an index we hold -- or
the fp64 units traverse a dispatch queue the counter names after texture. The
experiment that would separate those is a real texture object with `tex1Dfetch`:
a mislabel predicts 0 there, a shared queue predicts it fires. It has not been
run. Until it is, a small `stall_mem` on an fp64-heavy kernel may be counting no
memory instruction at all, and fp32 at the same geometry is only a control for
block shape and accumulator count -- both readings predict 0 there.

`stall_wait` is the other one. It counts the `wait` reason, which is where
NVIDIA puts a dependency on a *fixed-latency* result: the compiler knows when an
FMA retires, so it inserts stall cycles rather than allocating a scoreboard
slot, and scoreboards are left for results whose latency is not known at compile
time. A dependent chain of arithmetic therefore appears as `stall_wait`, not as
`stall_scoreboard` -- measured, a serial FMA chain reports `wait` at 71.88% and
`short_scoreboard` at 0.06%. Reading `stall_scoreboard` as "dependency stalls"
and `stall_wait` as "synchronization" inverts both: `stall_wait` also carries
`barrier` and `membar`, so it mixes the two, and `reason_samples` is the only
place they separate.

These descriptions are of the PC sampling reasons. The classifier also maps the
activity API's legacy stall names, which are routed so that a bucket means the
same thing whichever API produced the capture: `memory_dependency` joins
`long_scoreboard` in `stall_scoreboard`, and `exec_dependency` joins `wait` in
`stall_wait`. The second is a union of `wait` and `short_scoreboard` with no
split available on the parts that reported it, so it is routed to its dominant
member -- 71.88% against 0.06% on the measurement above. No capture from that
path has been taken here.

Two things make that identity hold and are worth knowing before reading a reason
breakdown. CUPTI reports every reason twice, as `<reason>` and
`<reason>_not_issued`, the second counting a subset of the
cycles the first already counted; only the base reason is added, and both are
listed in `unmapped_stalls` when the reason has no type of its own. And
`samples` still counts launches only, so it stays comparable with `gpu_time`
across both capture modes.

`pass`, `layer`, `moe.expert` and `moe.slot` are numeric labels, so pprof's
range filters apply: `-tagfocus='layer=0:3'` selects the first four layers.
Those filters reach launch rows only. A PC sample aggregates every launch of its
kernel, so a label that differed between those launches describes none of it and
is left off; only labels identical across all of them are attached. The profile
names the ones it dropped in a `label-omissions` comment, and `phase`,
`kernel.impl` and `kernel` are what remain to filter stall data by.

SASS addresses in Disassemble are CUBIN file offsets: each function's .text
section offset plus its instruction offset. PTX addresses are byte offsets in
the recorded text, assigned at capture time against a second logical image, so
any pprof consumer sees them and not only `cudaprof`. Profile addresses add
separate logical image bases.
These are reproducible file coordinates, not live device addresses.
The viewer reads saved NVIDIA assembly; it does not substitute CPU opcodes.

## Reproduce and inspect

`manifest.json` records the device, compiler, Go version, parameters, per-launch
times, validation result, and SHA-256 checksums. The viewer checks the bundled
files before displaying them. Original CUDA source, PTX, CUBIN, complete SASS,
Go source, and compiler/disassembler identity are included.

The `src` directory in the share bundle contains the buildable source and
vendored dependencies. See `BUILD.md` there for capture and rebuild commands.
View code and recordings locally; nothing is uploaded by the demo.
