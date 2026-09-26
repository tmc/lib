# cudaprof

Cudaprof records a CUDA program's device timeline with CUPTI and reads
the result: kernel totals and source through upstream pprof, idle gaps,
a per-op decomposition of why one capture took longer than another, the
verdict of an ABBA timing campaign, and a Perfetto trace.

```sh
cd nvidia
go install ./x/cmd/cudaprof
go generate ./x/cmd/cudaprof   # Linux with CUDA only: builds cupti_agent.so into GOBIN or GOPATH/bin
```

| Command | Needs |
| --- | --- |
| `capture` | Linux, an NVIDIA driver, and `cupti_agent.so` built against the CUDA toolkit's CUPTI (12.x or 13.x headers; the agent refuses a runtime of another major version). Tested on an NVIDIA GB10 (arm64, SM 12.1) with CUPTI 13.0. Driver version: not recorded in the code. |
| `gaps`, `compare`, `iterations`, `qualify`, `export`, `view`, `-top`, `-http`, `version`, `help` | Any OS. No GPU, no CUDA. |

A capture is one `.pprof` file: a pprof profile of device time that also
embeds `capture-status.json`, the per-record journals, the module images
and `capture-provenance.json`. `cudaprof -extract newdir run.pprof` writes
them out.

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
wins, the pattern ends at the first space (use `\s`), and `#` starts a
comment. Kernels no rule matches are `(unmatched)`.

	# regex op
	rms_norm            rmsnorm
	mul_mat_q|gemm      matmul
	flash_attn|fattn    attention

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
code does. The per-op table has launches, exclusive, summed and mean ns,
and a config column. After it:

- `launch configuration`: ops whose grid, block and dynamic shared memory
  sets are `SAME`, `CHANGED` (both sets printed with launch counts),
  `UNREAD` (not recorded) or `ONE_SIDED`.
- `launch count changes`: ops whose counts differ by 1.5x or more, or ran
  on one side only.
- `per-launch exclusive ns`: p10, p50 and p90 for the ops of largest
  |Δ|. A p50 that moved with a tight spread is a slower kernel; a p90 that
  moved alone is a few slow launches.
- with `-stages`, the per-stage table, whose `entries` columns say how
  often each stage began; counts that differ mean the boundaries did not
  align.

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
(the file compare reads), the top kernels, and every host API call made.
A record overlapping a marker is split between the iterations it
overlaps; it is launched in the one it starts in.

The gate fails, and the command exits 2, when the marker fired other than
`-expect-markers` times, left no steady iteration, or the steady
iterations launched different numbers of kernels. A capture that is not
complete is refused, as compare refuses it.

The advice ranks four costs repeated in most steady iterations, by time
per iteration: device allocation and free, CUDA graph capture and
instantiation, host synchronization the device sits idle across, and an
idle share of 10% or more.
`cudaprof advise -iteration-marker=RE` takes the same flags and adds
these findings to its report, after the findings to read first.

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
