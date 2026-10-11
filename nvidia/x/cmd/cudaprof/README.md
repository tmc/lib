# cudaprof

Cudaprof records a CUDA program's device timeline with CUPTI and reads
the result: kernel totals and source through upstream pprof, idle gaps,
per-op duration comparisons, the
verdict of an ABBA timing campaign, and a Perfetto trace.

```sh
cd nvidia
go install ./x/cmd/cudaprof
go generate ./x/cmd/cudaprof   # Linux with CUDA only: builds cupti_agent.so into GOBIN or GOPATH/bin
```

| Command | Needs |
| --- | --- |
| `capture`, `ab`, `campaign` | Linux, an NVIDIA driver, and `cupti_agent.so` built against the CUDA toolkit's CUPTI (CUDA 13.4 validated; CUDA 12 build limits are recorded below; the agent refuses a runtime of another major version). Tested on an NVIDIA GB10 (arm64, SM 12.1) with CUPTI 13.0. Follow-up native controls used CUPTI 13.4 and driver 580.173.02; see [validation limits](evidence-validation.md). |
| `gaps`, `compare`, `shape-compare`, `comparison-summary`, `counter-receipt`, `correlate`, `overview`, `dependency`, `memory`, `iterations`, `qualify`, `export`, `export-compare`, `view`, `-top`, `-http`, `version`, `help` | Any OS. No GPU, no CUDA. |

`probe` additionally needs native CUDA/CUPTI for its controlled workload. Run
`probe` and `campaign` inside an existing shared GPU lock.

A capture is one `.pprof` file: a pprof profile of device time that also
embeds `capture-status.json`, the per-record journals, the module images
and `capture-provenance.json`. `cudaprof -extract newdir run.pprof` writes
them out. A module image is kept up to `--capture-file-bytes` (64 MB by
default); when the total budget runs out, module images are dropped
before any journal.

## Capture

```sh
cudaprof capture -o run.pprof -- ./app -n 100
```

The agent is found beside the cudaprof binary or in GOBIN; otherwise pass
`--inject path/to/cupti_agent.so` or set `CUDAPROF_INJECT`. Exit status is
the target's own when it fails, 128 when a signal killed it, and 2 when the
target succeeded but the capture is incomplete. An existing `-o` file is
kept and the capture refused unless it holds no activity records.

### Capturing part of a run

By default collection runs from the first CUDA call to exit. To collect a
region, bracket it in the target with the profiler API:

```sh
cudaprof capture --capture-range=cudaProfilerApi -o region.pprof -- ./app
```

Collection opens when `cudaProfilerStart` (or `cuProfilerStart`) returns
and closes when `cudaProfilerStop` (or `cuProfilerStop`) is entered.
Synchronize before the stop: the range closes at the call, not when queued
work finishes. Start and stop may repeat; each pair becomes one phase.

`--finalize=profiler-stop` publishes the capture at the first stop instead
of at exit. A Go program leaves through `exit_group`, which runs neither the
agent's atexit handler nor its destructor, so a Go target needs it:

```sh
cudaprof capture --capture-range=cudaProfilerApi --finalize=profiler-stop \
	-o step.pprof -- ./gomodel -steps 8
```

```go
import "github.com/tmc/lib/nvidia/driver"

warmup()
if err := driver.ProfilerStart(); err != nil {
	return err
}
step()
if err := driver.CtxSynchronize(); err != nil {
	return err
}
if err := driver.ProfilerStop(); err != nil {
	return err
}
```

`--finalize=profiler-stop` ends the capture at the first stop, so do not
use it with a target that stops and starts again.

On linux, `--finalize=exit-group` finalizes a Go target without changing
it: a seccomp filter holds the target's `exit_group` until the capture is
published, then lets the process exit with its own status. The filter sets
`no_new_privs` and is inherited by children, and a child that outlives the
target has its `exit_group` fail with ENOSYS.

A target that dies of a fault ends the capture before any finalization.
`--crash-trace=N` has the agent keep the target's last CUDA API calls and,
on SIGSEGV, SIGBUS, SIGILL, SIGFPE or SIGABRT, write the last N of them to
the target's standard error before the process dies with the same signal:

	cudaprof: signal 11 at 0x8 on thread 4121; last 3 of 9 CUDA API callbacks, oldest first:
	cudaprof:   15us before, tid 4121 driver cbid 307 enter correlation 4 cuLaunchKernel

Every API call then passes through an agent callback, and a Go target that
faults in C code dies without its own traceback. The capability record's
`crash_trace` says whether the handler was armed.

From C, include `<cuda_profiler_api.h>`, link `-lcudart`, and call
`cudaProfilerStart()` / `cudaDeviceSynchronize()` / `cudaProfilerStop()`.
From Python, call the driver entry points through
`ctypes.CDLL("libcuda.so.1")` on a thread with a current context, and check
the return: a nonzero result means the edge was recorded but the driver
refused the call.

### capture-status.json

- `phase_receipt`: one entry per range edge (`edges`), and for a single
  start/stop pair the measured window: `duration_nanos`, `kernel_count`,
  `device_busy_nanos`, `device_idle_nanos`. Its `status` is `UNREAD` with a
  `reason` unless every check passed: `edge_pair_invalid` means the target
  never completed a pair; `multiple_phases` means several, each in `phases`.
- Each edge's `gpu`: SM and memory clocks, temperature and power, read
  through NVML by cudaprof after it sees the edge, with `lag_nanos` saying
  how long after. The driver refreshes these on its own schedule (hundreds
  of ms on GB10), so a short range can read the same value at both edges.
- `wall_clock`: a paired CLOCK_MONOTONIC and Unix time reading, which maps
  the edges to wall time.
- `cudaprof`: the build that wrote the status (module, revision, time,
  modified, go_version), the same fields `cudaprof version` prints.
- `target_environment`: command, environment digest, and the SHA-256 of
  the executable and of the agent.

Other flags worth knowing: `--api-activity=false` cuts journal volume by
about 10x when you only need the kernel timeline; `--pcsampling` adds
per-line samples at 25-27 us of host cost per launch; `--stamp-env` records
named environment variables. See `cudaprof help capture`.

## Gaps

```sh
cudaprof gaps --warmup-until=rms_norm run.pprof
cudaprof gaps --diff_base=base.pprof --warmup-until=rms_norm new.pprof
```

Reports, per device, timeline span, device active and idle time, duty
cycle, the idle-gap distribution, host concurrency at each gap, and idle
attribution (host behind, driver submission, hardware queue). The warmup
is cut at the first kernel matching `--warmup-until`, or after
`--warmup-ms` (default 200). With `--diff_base` every figure is a delta,
followed by a per-kernel census: launch count moved, same count but moved
in the run, or untouched. Use `--json` for machine output.

## Compare

```sh
cudaprof compare -warmup-until=rms_norm -ops ops.txt -stages stages.txt base.pprof new.pprof
```

Decomposes the window of each capture in exact nanoseconds and checks the
identity before printing:

	Δwindow = ΣΔop + Δunmatched + Δmemcpy + Δmemset + Δidle + Δunresolved

`-ops` maps kernel names to ops. One `regex op` per line; the first match
wins, the pattern ends at the first space unless it is quoted as a Go
string (`"a b"` or `` `a\s+b` ``), and `#` starts a comment. Kernels no rule matches are `(unmatched)`.

	# regex op
	rms_norm            rmsnorm
	mul_mat_q|gemm      matmul
	flash_attn|fattn    attention

`-ops=default` uses the built-in groups in [defaultops.txt](defaultops.txt):
optimizer, attention, softmax, gemm, rope, layernorm, loss, mask,
activation, copy, reduce, gather, elementwise and other, for the kernel
names of MLX, cuBLAS and cuBLASLt (nvjet), CUTLASS, MAGMA and cuDNN.

A line `name = op[:glob,...] ...` combines: it adds a row, outside the
identity, that sums op `name` and the kernels each term takes, so work
fused into one kernel on one side reads against the kernels it replaced
on the other. A term takes all of its op, or with globs the kernels whose
PyTorch elementwise functor (`MulFunctor`, `GeluCUDAKernelImpl`) or match
key a glob matches, ignoring case. No kernel may be taken twice. See
`go doc` for the format, and `testdata/ops/laya-torch.txt` for laya
against its PyTorch reference.

	geglu = elementwise:gelu*

`-ops-check` lists, per rule, the kernels and launches it won on each
side and the kernels it matched that an earlier rule won, plus the
launches each combine term took, and exits nonzero when a rule wins
nothing in either capture or a term takes nothing. Without it, rules that
matched nothing appear only in the JSON `unused_rules`.

`-stages` names positions in a repeating sequence, such as the stages of a
model layer. Same line format; the boundaries are matched in file order,
cyclically, and each kernel is charged to the stage whose boundary it last
passed. Kernels before the first boundary are `(before first boundary)`.

	# regex stage
	rms_norm            norm
	flash_attn|fattn    attention
	mul_mat.*ffn|moe    ffn

`-stage-entries` (needs `-stages`) also prints each entry of each stage (entry i of a
one-cycle-per-layer file is layer i), and is refused when a stage began a
different number of times in the two captures. With `-ops`, `-stages` and
`-json`, each stage row is broken down by op.

`-by phase` groups kernels by the NVTX ranges open on the host thread
when each was launched: `forward` (a range name holding `:fwd` or
`forward`), `backward` (`:bwd`, `backward`, as in autograd's
`MulBackward0`), `optimizer` (`optim`), `(other range)`, `(no range)`, or
`(no launch record)` when the launch has no host API record. The
outermost range that names a phase decides. A thread with no ranges of
its own is placed by every thread's ranges, and the table header counts
how many launches that was. `-by range` groups by the innermost range
name instead, and `-by module` by that name with its numeric path
elements folded, so `encoder.layers.3:bwd` and `encoder.layers.17:bwd`
share the row `encoder.layers.*:bwd`. Rows sum to kernel exclusive and
carry each op's cells.

The capture needs NVTX ranges. To compare a Go trainer with a PyTorch
one module by module, name the ranges on both sides by PyTorch module
path, `PATH:fwd` and `PATH:bwd`. In Go, push them with
`nvtx.RangePush`; in PyTorch, [torchnvtx.py](torchnvtx.py) adds them
with module hooks, including under non-reentrant gradient checkpointing,
whose backward hooks do not fire:

	import torchnvtx
	torchnvtx.install(model, r"^(encoder\.layers\.\d+|head)$",
	                  checkpointed=r"^encoder\.layers\.\d+$")
	loss.backward()
	torchnvtx.close_backward()
	with torchnvtx.range("optimizer"):
	    opt.step()

	cudaprof compare -ops ops.txt -by module go.pprof torch.pprof

`-phase i` picks the capture-range phase when a capture has several.
`-warmup-until` and `-warmup-ms` cut a prologue as gaps does. `-templates`
keeps template arguments, so each instantiation is its own op.

`-iteration-marker=RE` instead narrows each window to whole iterations,
as `cudaprof iterations` cuts them: from the marker launch that begins
iteration `-skip-iterations` (default 1) to the one that ends the last
steady iteration. Two captures holding different numbers of steady
iterations are refused unless `-iteration-count=N` names how many to
compare; a note gives the window, busy and idle time per iteration.

The header gives each side's window and `gpu at range edges`: the SM and
memory clocks, temperature and power at each edge, or why they were not
recorded. Two sides at different clocks differ by that ratio before any
code does. The per-op table has launches, exclusive ns, the exclusive
delta split in two, mean and median launch duration, the summed delta and
a config column. The split is exact: `count` is the launches gained or
lost priced at the base's exclusive ns per launch, `per-launch` the rest.
A fused pair shows as a count drop on one row and a per-launch change on
another. After it:

- `launch configuration`: ops whose grid, block and dynamic shared memory
  sets are `SAME`, `CHANGED` (both sets printed with launch counts),
  `UNREAD` (not recorded) or `ONE_SIDED`.
- `launch count changes`: every op whose count moved, largest Δ from
  count first; `*` marks a ratio of 1.5x or more, or an op that ran on
  one side only (JSON `launch_count_changed`).
- `per-launch exclusive ns`: p10, p50 and p90 for the ops of largest
  |Δ|. A p50 that moved with a tight spread is a slower kernel; a p90 that
  moved alone is a few slow launches.
- `kernels replaced`: kernels that ran on one side only, paired when
  they ran equally often and share an op (with `-ops`) or, without one, a
  five-byte name prefix. Paired kernels form one row, so a fusion of two
  kernels into one is a 2:1 row, and `per launch` is each side's summed
  ns over the launches per kernel: `LayerNormBackward +
  LayerNormParamPartial -> LayerNormBackwardParam` at 424812 -> 374825.
  With `-ops`, an op's other one-sided kernels form one more row.
- with `-stages`, the per-stage table, whose `entries` columns say how
  often each stage began; counts that differ mean the boundaries did not
  align.

`-json` carries all of it, plus the edge readings (`clocks`) and each
edge's SM clock ratio (`clock_ratios`).

## Iterations

```sh
cudaprof iterations -iteration-marker='^Embed$' -expect-markers=11 -ops ops.txt run.pprof
cudaprof iterations -iteration-marker='^mlx::core::cu::gather<float, int, 1, 2' train.pprof
cudaprof iterations -nvtx -iteration-marker='^step$' train.pprof
```

Cuts the timeline at each launch of the marker kernel, matched against its
demangled name with template arguments and without parameters, or with
`-nvtx` at each NVTX range or marker of that name. Iteration i runs from
marker i to marker i+1; the time after the last marker is not an
iteration. The first `-skip-iterations` (default 1) are warm-up, and the
rest are the steady state, reported per iteration: window, busy and idle,
exclusive kernel, memcpy and memset time, launches, the ops of `-ops`
(the file compare reads; the built-in groups by default, and each kernel
its own op with `-ops=`), the top kernels, and every host API call made.
A record overlapping a marker is split between the iterations it
overlaps; it is launched in the one it starts in.
A capture made with `--memsets=false` has no memset records, so their
device time is idle; iterations says so and counts the memset calls each
iteration made, a lower bound when CUDA graphs replay captured memsets.

The gate fails, and the command exits 2, when the marker fired other than
`-expect-markers` times, left no steady iteration, or the steady
iterations launched different numbers of kernels. A capture that is not
complete is refused, as compare refuses it.

`-gaps=1ms` lists the device idle gaps of at least that length in the
steady iterations, the host API calls begun in them, and the longest.
`-perf=host.txt` labels each gap with the host stacks sampled during it,
from perf recorded on the capture's clock, CLOCK_MONOTONIC:

```sh
perf record -k CLOCK_MONOTONIC --call-graph fp -e cpu-clock --all-user -o perf.data -- \
	cudaprof capture -o run.pprof -- ./train
perf script -i perf.data --ns -F comm,tid,time,ip,sym > host.txt
cudaprof iterations -iteration-marker=RE -perf=host.txt run.pprof
```

A gap with no samples is one the host spent off the CPU: blocked,
sleeping or waiting on a lock.

The advice ranks four costs repeated in most steady iterations, by time
per iteration: device allocation and free, CUDA graph capture and
instantiation, host synchronization the device sits idle across, and an
idle share of 10% or more.
`cudaprof advise -iteration-marker=RE` takes the same flags and adds
these findings to its report, after the findings to read first. The
whole-capture idle findings, which count warm-up with the loop, drop to
context beneath them. Without the marker too, a rule that fires once
per kernel or kernel pair (IMAD.WIDE, occupancy, fusion pairs) keeps
its three largest findings and folds the rest into one.
`cudaprof capture -o run.pprof --iteration-marker=RE` runs the gate on the
capture it wrote, prints a one-line summary, and exits 2 when the gate
fails, so a scripted capture cannot silently record the wrong window.

## GEMMs

```sh
cudaprof capture --gemms -o run.pprof -- python train.py
cudaprof gemms -warmup-until=rms_norm run.pprof
cudaprof compare -gemms -warmup-until=rms_norm base.pprof new.pprof
```

`capture --gemms` installs a cuBLASLt logger callback in the target and
records one line per matmul, including cuBLAS's internal Lt calls (PyTorch's
`a @ w.t()` arrives as `cublasLtTSTMatmul`). It works whether the target
links libcublasLt or opens it with dlopen. The logger's file output is off,
so the target's own output is unchanged. Each call is joined to the kernels
it launched: a launch whose stack passes through libcublasLt belongs to the
last call on that thread.

`gemms` groups the calls in the window by shape, op, types, compute type,
epilogue, beta and algorithm, then lists the kernel of each group:

	calls: 2 in window, 2 joined to kernels, 0 unparsed; kernel time 0.087 ms

	  calls  total ms  us/call   shape MxNxK  op                  types  compute  epilogue  beta                               algo
	      1     0.063     62.6  768x512x1024  TN  R_16BF,R_16BF->R_16BF      32F      BIAS     0                      21 16x16/32x1
	      1     0.024     23.9  768x512x1024  TN  R_16BF,R_16BF->R_16BF      32F   DEFAULT     0  67 64x184/64xAUTO customOption=29

	kernels:
	  1  768x512x1024       cublasLtMatmul  cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_32x1_tn_align8>
	  2  768x512x1024       cublasLtTSTMatmul  nvjet_sm121_tst_mma_64x184x64_3_16x184x64_tmaAB_bz_TNNN

M, N and K are cuBLAS's column-major view: M×N is D's rows×cols. Fields the
logger leaves out take their defaults (op N, epilogue DEFAULT). alpha is
not logged, and beta is `?` where the logger prints it so, as it does for
out-of-place calls whose beta it did not read. `compare -gemms` joins two
captures by shape, op and types and sorts the shapes by |Δ total|, then
shows the algorithm and kernel each side chose for them. The join is
exact, so two training runs whose batches hold different token counts
share few shapes.

A call is unjoined when its kernels were recorded into a CUDA graph, or
when the frame-pointer walk cannot reach libcublasLt (libraries built
without frame pointers, as on some x86_64 systems). A capture made without
`--gemms` is refused.

## A/B

```sh
gpu-lock -- cudaprof ab -o ab -control \
	-base 'env DATA=long-train ./train-main' -target 'env DATA=long-train ./train-new' \
	-quiet 'wait_quiet.sh quiet.log 120' \
	-compare '-warmup-until=rms_norm -ops ops.txt' \
	-- --inject cupti_agent.so --capture-range=cudaProfilerApi --finalize=profiler-stop
```

Captures the base and the target back to back with the same capture flags
(everything after `--`), and with `-control` the base again, then compares
base with target and base with control. `-order` sets the run order
(default `base,target,control`). The directory gets each arm's capture and
log, `base-target.txt` and `base-control.txt` (compare's text, with `.json`
beside them), and the summary in `ab.txt` and `ab.json`:

	      arm  window ms  busy ms  kernel ms  idle ms     sm MHz  mem MHz  temp C  power W  capture s
	     base     81.096   78.375     78.375    2.721  1488/1488      -/-   55/55      8/8        1.1
	   target     61.443   58.801     58.801    2.642    253/253      -/-   55/55      6/6        1.1
	  control     83.177   80.381     80.381    2.795    838/838      -/-   56/56      8/8        1.3

	                   Δwindow ms  Δbusy ms  Δkernel ms  Δidle ms
	   base -> target     -19.653   -19.574     -19.574    -0.079
	  base -> control      +2.080    +2.006      +2.006    +0.074  noise floor
	|Δwindow| is 9.4x the control's

The commands are split into words, with quotes, and run without a shell:
capture reads the range edges from its target process's journal, so a
shell wrapper would leave the captures without a range. Use `env` for
variables. `-compare` takes compare's flags; compare reads each capture
under the limits it records, so capture's `--max-records` and byte limits
need no repeating there. Each run's GPU
columns are NVML's readings at the start and stop edges, and a warning
names any edge whose SM clock differs from the base's by more than
`-clock-tolerance` percent (default 2) or whose temperature differs by more
than `-temp-tolerance` degrees (default 5). NVML refreshes these values
every few hundred milliseconds, so a short range can read a stale clock.

Why a control: a delta between two runs is the change plus whatever the
runs did differently on their own. The control is the same program twice,
so its delta is that drift with no change in it, measured by the same
arithmetic over the same range; read base→target only against it. In the
laya long-train captures the control moved 49 ms against a 528 ms gain.
Running the control last puts the target between the two base runs, so the
control also carries drift across the whole run.

Why one hold: captures in separate lock holds are separated by other
work and by the GPU's thermal history. ab takes no lock; run it inside
one, `gpu-lock -- cudaprof ab ...`, so nothing runs between its captures.
A lock keeps out only processes that take it, so `-quiet` runs a command
through `sh -c` before each capture and stops on a nonzero exit; a script
that waits until `nvidia-smi --query-compute-apps` is empty is the usual
check. For a verdict with a pre-registered noise gate over several blocks,
use qualify.

## Kernels

```sh
cudaprof kernels -grep 'LayerNorm' -phase 0 capture.pprof
```

Lists the kernels of one capture's window whose match key (the name
compare joins by) matches `-grep`: launches, summed and exclusive ns,
mean, median and p90 launch duration, and the launch configurations. It
takes compare's window flags.

## Qualify

Scores an ABBA campaign: blocks of four captures, run in the order C1, G1,
G2, C2, where C is the reference and G the arm under test. The manifest
has one `block arm path` per line; relative paths resolve against the
manifest's directory.

	# block arm path
	1 C1 base-1a.pprof
	1 G1 new-1a.pprof
	1 G2 new-1b.pprof
	1 C2 base-1b.pprof
	2 C1 base-2a.pprof
	...

```sh
cudaprof qualify -warmup-until=rms_norm -violations load.log manifest.txt
```

A warmup option is required. Each capture is measured by its steady-state
span, the window gaps reports. Every capture of an arm must record the same
executable SHA-256, and each block's ranges must have started in C1, G1,
G2, C2 order, or the campaign is refused. Per block the pairs are G1-C1
and G2-C2:

- effect: median over blocks of the pair mean (positive: G is slower);
- noise floor: max(`-noise-floor-ms`, MAD(C2-C1), MAD(G2-G1));
- noise gate: passes when |effect| > `-noise-multiplier` x noise floor
  (defaults 1 ms and 3);
- sign gate: passes when every pair has the same nonzero sign.

`-ops` and `-stages` take the files compare reads and add a breakdown by
op or stage: medians of C, G and G - C exclusive time, the least and
greatest G - C, and the range of launch counts. It is not gated.

`-violations` checks each arm's wall-clock window (first to last range
edge, widened by `-violation-margin`, default 2s) against a log kept by a
host watcher. One violation or heartbeat per line, fields separated by
spaces or tabs:

	# time pid comm cpu gpu holder
	2026-09-24T03:12:07Z heartbeat qualify
	2026-09-24T03:12:07.5Z 41822 python3 327.0 0 qualify
	2026-09-24T03:12:08Z heartbeat qualify

`time` is RFC 3339, `cpu` percent of one core, `gpu` device memory in
MiB, `holder` the lock holder's label or `-`. An arm is covered when a
heartbeat lies at or before its widened window, one at or after it, and no
two consecutive heartbeats between are further apart than the margin. An
arm with a violation, no wall clock, no closed range, or no heartbeat
coverage is excluded, and its whole block is dropped. `-min-blocks` is the
fewest blocks that may remain (default: all).

Exit status: 0 when both gates pass, 2 when either fails, 1 when the
verdict is `REFUSED` or could not be computed.

## Export

```sh
cudaprof export -o run.pftrace run.pprof
```

Writes the device timeline as a Perfetto trace for https://ui.perfetto.dev:
one track per device and stream for kernels and memcpys, one per thread for
API calls. A capture whose journal dropped records is refused.

## Viewing

```sh
cudaprof -top run.pprof                        # kernel totals
cudaprof -http=127.0.0.1:8087 run.pprof        # upstream pprof web UI
cudaprof view -html report.html run.pprof      # source, PTX and SASS side by side
```

`cudaprof help` lists the rest: `diff`, `lines`, `advise`, `cfg`,
`pressure`, `inspect`, `explain`.

## Version and help

```sh
cudaprof version            # module, revision, time, modified, go
cudaprof help               # every command
cudaprof help compare       # one command's flags
```

## Traps

- A Go target without `--finalize=profiler-stop` or `--finalize=exit-group`
  is never finalized and its journal is empty: the agent's exit handler
  never runs.
- Ops join by kernel name. When the two sides split the same work into
  different kernels, a fused kernel on one side and three on the other are
  separate rows that each read as a regression or a win. Group them with
  `-ops`, or compare by position with `-stages`.
- An exported trace's timestamps are CUPTI's, rebased on its earliest record,
  with no clock sync packet. Two exports do not share a time axis, and
  neither lines up with a host trace from another clock.
- A GPU test that calls `t.Skip` when it finds no device reports `ok`.
  Run it with `-v` and check for `SKIP` before trusting a capture of it.
- The summed column counts overlapping launches more than once; exclusive
  charges each busy instant to one record. Only exclusive adds up to the
  window.
- Medians in the qualify breakdown are each of their own column, so the
  rows' medians do not sum to the total's median.

## Workload evidence

`shape-compare` joins instrumented work by explicit application metadata and
launch geometry. `correlate` exposes retained submission and NVTX links.
`memory` reports transfer and allocation-call evidence with residency limits.

- [Existing Gemma4 diagnostic demonstration](evidence-demo.md)
- [Controlled follow-up validation](evidence-validation.md)
- [Token and hierarchy overview](overview.md)
- [Comparison summary](comparison_summary.md)
- [Dependency evidence](dependency.md)
- [Repeated campaign runner](campaign.md)
- [Native capability probe](probe.md)
- [Counter receipt qualification](counter_receipt.md)
- [Application marker convention](../../cudaprof/marker/README.md)
- [Shape comparison](shape_compare.md)
- [Submission links](correlation.md)
- [Go submission stacks and host profiles](../../cudaprof/go_stacks.md)
- [Python submission stacks and wall samples](python_stacks.md)
- [Memory evidence](memory.md)
- [Gap accounting limits](../../cudaprof/gaps/README.md)
- [Campaign sample counts and spread](qualify-campaign.md)
- [Targeted counter receipts](../../cudaprof/counters/README.md)
- [Capture-edge telemetry](telemetry.md)

These facilities preserve missing observations and separate collector
completeness, workload invariants, stationarity, and shape qualification.
Counter collection and managed-memory observation need separate controlled
experiments; no causal interpretation follows from matched timing alone.

Append raw-record diagnostic findings to an export:

```sh
cudaprof export -findings -o capture.pftrace capture.pprof
```

Findings distinguish kernel records retained in the artifact from a complete
collector capture population. Missing or contradictory counts and unknown
loss withhold the second finding. These counts do not establish total workload
launches, clock synchronization, or a causal diagnosis. Without `-findings`,
export retains its raw trace behavior.
