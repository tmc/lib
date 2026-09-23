# A candidate-kernel harness

This example takes one CUDA C++ source file, compiles it with NVRTC, checks it
against a CPU reference the candidate never sees, measures its device time, and
prints a classified report. It is the piece a generate-and-refine loop sits on
top of: the loop writes a kernel, this runs it, and the report says what
happened in enough detail to change what the loop writes next.

It is modelled on the agent workspace released with
[CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent)
(`agent_workdir/utils/{compile,verification,profiling}.py`), and departs from it
in four places, each noted below under **Departure**.

## Run

```sh
go run ./examples/kernelharness -list
go run ./examples/kernelharness -task axpby -kernel my_kernel.cu
go run ./examples/kernelharness -task axpby -kernel my_kernel.cu -json
```

Every candidate must define exactly this entry point:

```
extern "C" __global__ void ENTRY(float *out, const float *a, const float *b, float alpha, int n)
```

where `ENTRY` is the task's entry name. The signature is fixed so the harness
packs launch arguments without parsing the candidate. A candidate that does not
match it is rejected as `missing_entry` or `launch_rejected`, which are
different signals from a wrong answer.

### With no GPU

The development host here is darwin/arm64 with no CUDA. The harness builds and
its tests run there:

```
$ go test ./examples/kernelharness
ok      github.com/tmc/lib/nvidia/examples/kernelharness

$ go test -v ./examples/kernelharness
--- SKIP: TestFixtureOutcomes (0.15s)
--- SKIP: TestCorrectCandidateIsMeasured (0.04s)
--- SKIP: TestBaselinesPassTheirOwnOracle (0.09s)
```

Every CUDA-touching test goes through `x/cudatest.Require`, which skips with a
reason naming the missing driver. The skip is visible under `-v`; a green `ok`
with no `-v` proves only that the pure-Go tests ran.

The CLI on a host with no device prints outcome `no_device` and **exits 3**.
Zero is success and one is a failed candidate, so a loop that reads only the
exit status still cannot mistake an absent GPU for a correct kernel.

## What the report says, and why

### The oracle is not the producer

Each task owns a CPU reference in `tasks.go`, written from the task's
definition, accumulating in `float64` and narrowing once. It is compiled into
the harness. A candidate cannot supply it, influence it, or read it.

The task's *baseline* kernel — the performance comparison arm — is not an
oracle. It is judged by the same CPU reference the candidate is
(`TestBaselinesPassTheirOwnOracle`).

Inputs are regenerated from a fresh seed for every correctness trial, so passing
means being right on several draws rather than on one memorized one.

The output buffer is prefilled with **NaN** before every launch. An element the
kernel never writes is then counted as non-finite rather than read back as a
plausible zero. `testdata/axpby_partial.cu` is the fixture that proves this
fires: it writes only the first grid's worth of elements and is caught.

**Mirrors CUDA-Agent**: an independent reference (`model.py`'s eager PyTorch
`Model`, which `SKILL.md:26` forbids the agent to modify), five trials on fresh
inputs (`utils/verification.py:90-95`).

**Departure**: CUDA-Agent's verification prefills nothing — `torch::empty_like`
in the binding leaves the output buffer uninitialized, so an unwritten element
reads whatever the allocator last held. Its tolerance is `atol=1e-2, rtol=1e-2`
(`utils/verification.py:35`), loose enough that a materially wrong kernel on
small-magnitude data can pass; the tolerances here are per task and each is
checked against a deliberately wrong fixture
(`TestTolerancesRejectTheWrongFixtures`).

### Time is device time

Kernel timing comes from `x/cuda.EventTimer`: CUDA events bracketing the
submitted work on the timed stream. The report names the clock in
`timing.source` as `cuda_event_device_span`.

- Warmup launches run and are discarded (`timing.warmup_launches_discarded`).
- Every sample is reported (`timing.sample_device_nanos`), not just a summary,
  so the spread is visible.
- The headline figure is the **median** of samples divided by the launches per
  sample. The median, not the mean: on a kernel this short one scheduling
  hiccup moves a mean further than the effect being measured.
- If any sample's device interval came back zero, no rate is derived at all and
  `timing.resolved` is false. A zero-nanosecond span is not a fast kernel.
- Wall clock appears twice and is named both times: `compile.compile_wall_nanos`
  (compilation is host work, so no device timer applies) and
  `timing.wall_nanos` (the host span over the timed section, for contrast).

The candidate and baseline arms are interleaved **ABBA**, not run one after the
other, so a clock or thermal drift over the measurement window falls on both
arms rather than on whichever ran first.

**Mirrors CUDA-Agent**: it also measures device time, via
`torch.profiler` with `ProfilerActivity.CUDA` (`utils/profiling.py:21-26, 71`).
That part is right and is worth copying.

**Departure**: three things.

1. CUDA-Agent sums `device_time` over all CUDA events and divides by the
   iteration count (`utils/profiling.py:71-73`). That is a sum of per-kernel
   busy time, not an elapsed device span: kernels overlapping on separate
   streams are added rather than spanned, and no per-iteration distribution
   survives, so there is no median, no spread, and one outlier iteration is
   invisible. Here each sample is its own event interval and all of them are
   reported.
2. Its arms run in a fixed order — candidate, then eager, then compiled
   (`utils/profiling.py:112-116`) — once each, with no interleaving and no
   repetition. Here the arms are interleaved and repeated.
3. Its `SKILL.md` config-tuning recipe (`SKILL.md:198-231`) uses
   `time.perf_counter()`, wall clock, contradicting its own profiler. There is
   no wall-clock tuning path here.

### A failure is classified

`outcome` is one of:

| outcome | meaning | context usable after |
| --- | --- | --- |
| `ok` | compiled, loaded, ran, matched on every trial | yes |
| `compile_error` | NVRTC rejected the source; the log is the evidence | yes |
| `load_error` | the driver rejected the image: PTX JIT failure, unsupported PTX version, no binary for this device | yes |
| `missing_entry` | the module loaded and does not define the required symbol | yes |
| `launch_rejected` | the driver refused the launch before running it | yes |
| `device_fault` | the kernel ran and faulted: illegal address, misaligned access, device assert | **no** |
| `timeout` | the kernel did not finish inside the deadline | **no** |
| `wrong_answer` | ran to completion, output outside tolerance | yes |
| `no_device` | no CUDA device; not a verdict on the candidate | — |
| `harness_error` | the harness failed, or met a driver code it does not classify | — |

An unrecognized driver code becomes `harness_error` with
`error_recognized: false`. It is never absorbed as `ok`
(`TestClassifyRefusesToGuess`).

The CUDA driver documents the fault codes as leaving the process in an
inconsistent state: every later CUDA call returns the same error. When that
happens the report sets `context_poisoned` and the harness must be re-executed
before the next candidate — a loop that reuses the process would see the first
fault reported for every candidate after it. `testdata/axpby_fault.cu` is the
fixture, and it is deliberately excluded from the in-process fixture set:

```sh
KERNELHARNESS_FAULT=1 go test ./examples/kernelharness -run TestDeviceFaultFixture
```

`TestFixtureOutcomes` is the test that matters here. It runs five fixtures and
requires that they do not collapse onto one another. A taxonomy is only worth
having if the fixtures prove it discriminates.

**Departure**: CUDA-Agent does not classify at all. `utils/compile.py:51-54`
catches a compile failure and returns 1; `utils/verification.py` uses bare
`assert` and `torch.testing.assert_close`, so a shape mismatch, a numerical
drift, an import failure of the `.so`, and a `cudaErrorIllegalAddress` surfacing
as a `RuntimeError` at the next sync all arrive as the same thing — a Python
traceback and a non-zero exit. Nothing distinguishes them; there is no timeout
anywhere in that repository, so a hanging kernel hangs the harness; and there is
no machine-readable output, so the only consumer of the distinction is whatever
model happens to read the traceback text.

### What was not measured is named

Every report carries an `unmeasured` array. It names occupancy, achieved memory
throughput, cache state across samples, SM clock and power, CUBIN and SASS
digests, per-thread register and shared-memory use, behaviour outside the
generated input range, and concurrent device contention. None of those is
reported, and none is inferred. An absent field is visible rather than read as
a zero.

`speedup` is absent, with a line added to `unmeasured`, whenever a baseline arm
did not resolve. It is never estimated.

**Departure**: CUDA-Agent's `SKILL.md:352` sets a hard gate — "at least 5%
faster than torch.compile" — that no code checks. `utils/profiling.py:76-82`
prints three microsecond figures and computes no ratio at all; the comparison
is left to the model reading the line. `utils/compile.sh:3-10` reports
`[TIME] Compilation took Xs`, which is the wall time of the whole Python
process including interpreter startup and the `torch` import, labelled as
compilation.

## Artifact identity

`compile.source_sha256` and `compile.ptx_sha256` are recorded by the producer,
in the same call that produced the artifact. They bind the PTX this report names
to the source this report names.

They bind nothing else. No CUBIN or SASS digest is recorded, and nothing
downstream consumes these — in particular `x/cudacapture.LoadCode` does not
read them. `x/nvrtcx/QUALIFICATION.md` names the missing digest chain as the
blocking gate for that package and records that **no producer in the tree
records one**. This example is a producer that records one for two of the tiers;
it does not close that gap, because closing it needs the consumer side as well.

## Documented gaps

These are things the design wants and this implementation does not do.

1. **No source-line attribution on a failure.** The stack has the pieces —
   `x/cudaptx` parses `.loc`, `x/cudasass` reads nvdisasm output,
   `x/cudacapture.LoadCode` fuses the tiers, `x/cudalowering.DefinitionLine`
   maps a mangled symbol to its definition — so a wrong answer could in
   principle be reported against the candidate's own source line. It is not
   wired up here. `QUALIFICATION.md` gap 9 records that PTX carries absolute
   build-host paths and that the cross-tier join discards the directory, so two
   files sharing a base name join as one; and gap 1 records that artifact
   identity across tiers is unmeasured beyond the line tables. Attribution
   built on that would be reported at a confidence the chain has not earned.
2. **No counter collection.** `x/cuptibench` collects hardware counters through
   CUPTI, and `x/examples/cuptibench` shows the lifecycle. Adding it here would
   let the report carry achieved occupancy and memory throughput instead of
   listing them as unmeasured. Counters come from separate replay executions,
   not the timed iteration, so this is a second measurement to design, not a
   flag to set.
3. **No timeout that actually stops a kernel.** The correctness deadline
   (`-timeout`) bounds the *wait*, not the kernel. Nothing in CUDA cancels a
   running kernel; the harness reports `timeout`, marks the context poisoned,
   and the process must be abandoned. A harness that had to survive adversarial
   candidates would run each one in a subprocess. This one does not, and says so
   rather than implying the deadline is a kill.
4. **The timed input set is uploaded once and reused.** Regenerating it per
   sample would put a host-to-device copy inside the measured interval. The
   consequence — a memory-bound kernel reading from warm cache — is recorded in
   `timing.input_policy` rather than controlled. Flushing the cache between
   samples is not implemented.
5. **One shape per task.** Each task runs at a single element count. A candidate
   that is correct at 2^20 elements and wrong at a non-multiple of the block
   size would pass. Sweeping shapes is the obvious next thing.
6. **Three elementwise tasks.** The fixed ABI covers elementwise problems over
   two input vectors and one scalar. It does not reach reductions, GEMM, or
   anything with a second output. That is a real ceiling on what this harness
   can pose, not an oversight to be read around.

## Files

| file | contents |
| --- | --- |
| `doc.go` | package documentation |
| `outcome.go` | the failure taxonomy and the driver-code classifier |
| `compare.go` | tolerance comparison and worst-element selection |
| `tasks.go` | the tasks, their CPU references, and their baseline kernels |
| `run.go` | compile, check, measure |
| `report.go` | the report types and the derived summaries |
| `main.go` | the CLI |
| `testdata/*.cu` | one correct candidate and five deliberately broken ones |
