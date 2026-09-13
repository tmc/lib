# Profiling ds4-min against the ds4.c oracle

How to get a Go CPU profile and a Perfetto timeline out of `examples/ds4-min`,
and the matching pair out of the `ds4_bench.c` oracle, over one fixed workload:
a 2048-token prefill of identical token IDs followed by exactly two decoded
tokens.

Every number and command below was observed on the dates named. Claims that
have not been measured are marked **unverified** rather than omitted; do not
promote one to fact without running it.

## The workload

Use token IDs, not text. `-prompt-tokens` reads a JSON array of integers and
applies no chat template and no special tokens, so both runtimes see the same
input without depending on tokenizer agreement:

    scripts/testdata/oracle-workload-tokens2048.json

`-tokens 2` bounds decode. `-ctx` must leave room for prefill plus the decode
slots: `preparePrompt` rejects a context that cannot hold both, so 2048 prefill
plus 2 decode needs at least 2050.

To bind numerics across a pair of runs, pass the digests rather than trusting
the rates:

    -expect-prefill-logits HEX   -expect-final-logits HEX

Both hash finite float32 bits as little-endian 32-bit words, and
`-expect-final-logits` requires exactly `-tokens` successful accepts, so an
early stop cannot satisfy it. These flags landed in `d6e25f78`; a binary built
before that commit does not have them.

## Go, CUDA, on the DGX

### Run without the profiler first

The Go profiles cost almost nothing and the run is the baseline the Nsight run
is compared against.

    ds4-min -model "$MODEL" -backend cuda \
      -prompt-tokens scripts/testdata/oracle-workload-tokens2048.json \
      -tokens 2 -ctx 4096 \
      -cpuprofile go.cpu.pprof -memprofile go.heap.pprof \
      -trace go.trace -spans go.spans.jsonl -span-level layer

`-span-level layer` writes a `gputrace.host-event/v1` JSONL sidecar. Since
8d5baa4d this workload produces **11,354 spans** in a 2.0 MiB file: one
`prefill`, one `generation`, two `token`, 256 `batch` — 2048 tokens at a batch
width of 8 — and 11,094 `layer`, of which 11,008 are prefill (256 x 43) and 86
are decode (2 x 43). The recorder is bounded at 100,000 records and 8 MiB and
drops silently past either, so check the count against what the shape predicts
rather than trusting the file's existence.

### Then run under Nsight

On this DGX only one of the four installed `nsys` binaries accepts
`-t cuda-sw`; the others fail with `Illegal --trace argument 'cuda-sw'`,
exit 1, and no report. Pin the absolute versioned path — `/usr/local/cuda` is
alternatives-managed and its target has drifted:

| binary | version | `-t cuda-sw` |
| --- | --- | --- |
| `/usr/local/bin/nsys` | 2024.5.1 | rejects |
| `/usr/local/cuda/bin/nsys` | 2025.3.2 | rejects |
| `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys` | 2025.3.2 | rejects |
| `/opt/nvidia/nsight-systems/2026.1.1/bin/nsys` | 2026.1.1 | **accepts** |

    /opt/nvidia/nsight-systems/2026.1.1/bin/nsys profile \
      -t cuda-sw,nvtx,osrt --cuda-graph-trace=node --cuda-memory-usage=true \
      --cuda-flush-interval=100 \
      -f true -o go-prefill2048-decode2 \
      ds4-min -model "$MODEL" -backend cuda \
        -prompt-tokens scripts/testdata/oracle-workload-tokens2048.json \
        -tokens 2 -ctx 4096 -cuda-profiler-stop \
        -cpuprofile go.cpu.pprof -trace go.trace \
        -spans go.spans.jsonl -span-level layer

**`--cuda-flush-interval=100` and `-cuda-profiler-stop` are both required, and
neither is optional.** Without them the capture silently loses its tail — with
default settings that means all of decode and a percent or two of prefill,
with no error and a report that opens normally. Why, and what each one buys,
is in "The missing tail was never flushed" below; the short version is that
`ds4-min` has to ask the profiler to flush before the process leaves, because
Go's `runtime.exit` will not.

One thing this does not give you: `nsys` reports CPU sampling unavailable on
this host, so the Nsight report carries no host samples — the Go CPU profile is
the host-side instrument, not a fallback.

### NVTX, and what the export does to it

`ds4-min` emits an NVTX range for every span it records, so the device timeline
arrives labelled by phase, batch, token and layer. The ranges bypass the
recorder's caps on purpose: a run that drops sidecar records still has a
complete Nsight timeline. Add `-calibration <file>` to bracket two uniquely
named ranges between reads of the span clock; `ds4-min` refuses the flag when no
provider is attached rather than paying a model load to find out.

Three things about the provider, measured with a standalone Go binary under
`nsys -t nvtx` — no model, no GPU:

`nvtx.Enabled()` tracks the real condition. Bare, `NVTX_INJECTION64_PATH` is
unset and `RangeStart` returns 0. Under the 2026.1.1 binary the variable points
at `target-linux-sbsa-armv8/libToolsInjection64.so`, `Enabled()` is true, and
handles come back nonzero.

A process that exits within a millisecond of its last range exports **no
`NVTX_EVENTS` table at all**. The same program with sleeps, running about 2.5 s,
exported all 15 of its events. This is a flush boundary at exit, not a
capability limit — but it means a trivial smoke test reports "NVTX does not
work" for the wrong reason.

One row came out malformed in all three probes: the last
`RangePush`/`RangePop` pair carries `start == end`, both holding an absolute
`systemClockNs` value instead of the session-relative one every other row uses.
Subtracting the session start puts it 2,624 ns after the preceding mark, where
the program's emission order says it belongs — but the 50 ms duration is gone,
not rescaled. **The cause is not established.** The probe pushed and popped
across a `time.Sleep` without `runtime.LockOSThread`, and push/pop are
current-thread stacks, so a migrated goroutine popping a different stack is not
excluded; `ScopedRange` pins for exactly this reason. Treat it as a warning
about unpinned push/pop, not as a known Nsight defect.

It does not reach the calibration path. `calibrate` uses
`RangeStart`/`RangeEnd`, and in the qualification capture below the end
calibration is both an `eventType` 60 row and the last range emitted before
exit: it is normalized, carries its true duration, and no row in the capture has
an absolute-magnitude timestamp. The probes alone could not have shown that —
they never put a `RangeStart`/`RangeEnd` last — so the capture, not the probe,
is what closes it.

`NVTX_EVENTS.text` carries the range string literally and `textId` is NULL for
these, which is why `calibrate` puts the run ID in the range name. The schema
does populate `rangeId`, but a unique name is still the better key: it is chosen
by the program and survives into any other receipt.


Export to SQLite, which is what the converter reads:

    nsys export --type=sqlite go-prefill2048-decode2.nsys-rep

## Go, Metal, on the Mac

Same command with `-backend metal` and no `nsys`. **Unverified for this
workload:** whether the Metal path emits the same span inventory, and what a
2048-token prefill costs there. Run the Metal model load under `memguard` —
this workload has panicked the host through VM compressor exhaustion before.

## The ds4.c oracle

`ds4_bench.c` has no NVTX and one clock, `bench_now_sec()` on `CLOCK_MONOTONIC`.
It gates the CUDA profiler range on two environment variables, read from the
built binary on the DGX:

    DS4_BENCH_CUDA_PROFILE_PREFILL=1          # profile window around prefill
    DS4_BENCH_CUDA_PROFILE_RANGE=0:2          # START_TOKEN:TOKEN_COUNT

`cudaProfilerStart` / `cudaProfilerStop` are weak symbols under
`DS4_BENCH_HAVE_CUDA_PROFILER`; in the DGX build they resolve to
`libcudart.so.13`, so the windows are live.

**These are two separate capture ranges, not one.** `ds4_bench.c:824` starts
and `:829` stops around prefill; `:913` starts again for decode. Nsight's own
help for the pinned 2026.1.1 binary says `--capture-range-end=stop` ends
collection at the first range end and *ignores every subsequent range*, so with
both env hooks enabled that flag captures prefill only and silently drops
decode. Use `repeat-shutdown:2`:

    nsys profile -t cuda-sw,nvtx,osrt \
      -c cudaProfilerApi --capture-range-end=repeat-shutdown:2 \
      -o c-prefill2048-decode2 ds4-bench <args>

**Unverified:** this capture has not been run. The env-var names, the weak
symbols, and the two range boundaries are read from the source and the built
binary; the `repeat-shutdown:2` behaviour is read from `nsys profile --help`.
No resulting report exists, so none of this is yet evidence about a capture.

### Compare accepted token IDs, not counts

The oracle picks `ds4_session_argmax_excluding(session, eos)` at
`ds4_bench.c:906`, so it never emits EOS; `ds4-min` stops when it does. Two runs
can agree on "two tokens decoded" and have decoded different tokens. Compare the
accepted IDs — `-expect-final-logits` covers the Go side, and the oracle's
`teacher_forced_decode` path at `:904` takes the next token from the prompt
instead of sampling, which removes the disagreement entirely when both sides
use it.

**Unverified:** there is no pprof equivalent for the C host side, and whether
`perf record` works on this DGX has not been probed — `nsys`'s refusal to
sample CPU there is a reason for doubt, not an answer.

## Converting to Perfetto

`examples/cmd/profile2trace` writes native `.pftrace` from three inputs. It
needs `sqlite3` on PATH and no GPU.

    profile2trace -nsys capture.sqlite  > device.pftrace
    profile2trace -spans go.spans.jsonl > host.pftrace

Open the result with **Open trace file** at <https://ui.perfetto.dev>.

A whole-run export of this workload holds 4,069,537 kernel rows.
`-nsys-window START:END` keeps only the intervals whose start falls in the
half-open range, in source-clock nanoseconds, so adjoining windows partition a
capture without duplicating or dropping a row, and stderr names how many rows
were omitted. `-max-slices` (default 500000) refuses a larger selection rather
than emitting silently.

Find the bounds from the capture, never by guessing:

    sqlite3 capture.sqlite 'select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL'

### Do not align two clocks by eye

`profile2trace` refuses to merge sources without an explicit
`-spans-offset-ns` or `-metal-offset-ns`. The Nsight timeline is `nsys_ns`; the
span sidecar is `cpu_uptime_ns`; the oracle is `CLOCK_MONOTONIC`. An offset is
**added** to the source timestamp, so for a measured pair
`(source_ns, reference_ns)` the offset is `reference_ns - source_ns`. Until
such a pair is measured for this workload, keep the device trace and the host
trace as separate files and read them side by side.

### The pair has now been measured, and it refutes a constant offset

Run `nvtx-20260912-DF6906F6` (below) bracketed two calibration ranges 160.1 s
apart. Each bracket is the interval `[nsight_start - host_after,
nsight_start - host_before]`, the range of offsets consistent with that reading:

| bracket | offset interval (ns) | width |
| --- | --- | --- |
| start | `-496790522283921 .. -496790522271313` | 12,608 ns |
| end | `-496790524064271 .. -496790524063023` | 1,248 ns |

**They are disjoint.** The whole end interval lies 1,779,102 ns below the start
interval — 1,400 times the tighter bracket's own width, so this is not
measurement slack. Over the 160.107485342 s the span clock counted, Nsight
counted 160.105706240 s: the two clocks differ in *rate* by 11.11 ppm. One
constant offset cannot relate them, so **no value is passed to
`-spans-offset-ns` and merged-clock qualification stays open.** Do not split the
difference; an averaged offset would be wrong at both ends and defensible at
neither.

The *cause* is unestablished. An NTP-slewed `CLOCK_MONOTONIC` read against an
undisciplined counter would look like this, but nothing here measures the
discipline of either clock, and 11.11 ppm is also within the range of plain
crystal error. The gate fails on the arithmetic alone; naming a mechanism is not
required to fail it, and would not be evidence if it were offered.

The same measurement refutes the obvious guess about the Nsight timebase. If
`TARGET_INFO_SESSION_START_TIME.systemClockNs` were the report's zero in the
span clock's domain, the offset would be `-496805755631696`; the measured
brackets sit 15.233 s away from it. The prediction was written down before the
capture precisely so it could fail visibly. Treat `systemClockNs` as an
unidentified clock until something measures it.

What would close this: a third bracket mid-run. Two points fix a line with no
residual left over, so a linear fit through these two is unfalsifiable as it
stands; a third says whether the drift is linear or whether the clocks are
related by something worse.

## Reading the Go profiles

    go tool pprof -http=: go.cpu.pprof
    go tool trace go.trace

The execution trace is large — 228 MB for this workload — and shows goroutine
scheduling and GC, not GPU work. For where device time goes, query the SQLite
directly; it is faster than any viewer:

    select s.value, count(*), sum(k.end-k.start)/1e6 as ms
    from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
    group by s.value order by ms desc limit 15;

## Measured CUDA digests, DGX spark-31db, 2026-09-12

Binary built at `a02c5c62`, sha256 `4aa8f6dc…ff4910`, `-backend cuda`,
`-ctx 4096`, `-prompt-tokens scripts/testdata/oracle-workload-tokens2048.json`,
`-tokens 2`:

    token_ids_sha256  5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380
    prefill logits    4eee1b147ac1b839cf80148919e598dd4623e862fbd6a7d9dbc8d5e06b80fd4a
    final logits      837a6b8716431254516705085ace5d59393c4fffe92fe44069a38720995ea6a3
    accepted IDs      15342, then 201

Three runs produced these, not one. `checkLogits` returns early on an empty
expectation (`logits_check.go:24`), so a digest is only computed when one is
supplied, and the prefill check returns on mismatch before decode
(`main.go:157`). So: measure prefill against a placeholder; measure final while
re-checking prefill against the measured value; then supply both. Only the
third run exits 0, and until it does the pair is one sample rather than a
constant.

The token-ID digest is byte-identical to `oracleWorkloadTokenHash` at
`oracle_workload_bench_test.go:25`, so this is provably the same input the
bench harness names.

**These are not comparable to the registered logits constants in that file.**
`oracleWorkloadPrefillLogitsSHA256` and `oracleWorkloadFinalLogitsSHA256` belong
to a harness that runs at `oracleWorkloadContext = 65665` and decodes 128
tokens (`:121`), excluding EOS (`:86`). Same tokens, different context and
different decode. A disagreement between these digests and those constants
would say nothing about either. Do not edit a constant to match a run.

## Reference run, DGX spark-31db, 2026-09-12

Commit `8fd3f4e9`, go1.26.7 linux/arm64, nsys 2026.1.1.204, model
`DeepSeek-V4-Flash-IQ2XXS-...-imatrix-0731.gguf` (86,720,111,488 B).
Both runs exited 0 and printed `" alla"`.

| | prefill | generation |
| --- | --- | --- |
| without nsys | 12.94 t/s | 12.03 t/s |
| under nsys | 12.82 t/s | 11.79 t/s |

Device inventory of the Nsight export:

| | |
| --- | --- |
| kernel rows | 4,069,537 |
| device span | 158.784187136 s |
| device busy | 148.324597024 s |
| distinct kernel names | 34 |
| NVTX table | absent |

Converting all 4,069,537 rows took **22.05 s wall and 5.03 GiB peak RSS**, and
produced a **1.11 GB** `.pftrace`. That file loads: `trace_processor` v57.2 read
it in 24.55 s at 3.4 GiB RSS and reported 4,069,537 slices summing to
148,324,597,024 ns — the SQLite totals exactly. A full-capture conversion of
this workload is feasible end to end; only the browser UI's own limits are
untested. Windows are a diagnostic convenience, not a necessity.

Two equal 200 ms windows convert to 1.4 MB files that load with slice counts
and summed durations matching the SQLite exactly and zero parser errors:

| window | slices | summed ns | SQLite |
| --- | --- | --- | --- |
| device tail, last 200 ms | 5,105 | 188,293,216 | 5,105 / 188,293,216 |
| mid-prefill, 200 ms at t=80 s | 5,172 | 187,731,328 | 5,172 / 187,731,328 |

**The tail window is a device-tail diagnostic, not "the two decode steps".**
Naming it after decode would assert a clock mapping nobody has measured. The
host sidecar puts run2's `generation` at 169,667,558 ns (run1: 166,260,292), and
the 200 ms tail was chosen to cover that with margin — but `cpu_uptime_ns` and
`nsys_ns` are different domains, so which device kernels belong to decode is
still inferred, not established.

A second reason not to name it after decode has since been measured: this
capture lost device events, so its device tail may end before the workload
did. The window holds 5,105 kernels across 32 distinct kernel names, so it is
populated — but with no NVTX table in this run, nothing in it attributes to a
phase, and the tail's own completeness is not established.

A concrete lead, not yet a mapping: the export's `TARGET_INFO_SESSION_START_TIME`
carries `utcEpochNs=1789252196696595551` and `systemClockNs=491666116248656`.
The clock backing `systemClockNs` has to be established before either is
subtracted from anything — done blindly it drives the first host timestamp
negative, which is the tell that the domains were mismatched.

### Batching prefill buys almost nothing here

Activity is close to uniform across the 158.78 s device span, with two
exceptions that were missed the first time this was written. Bucketed into
5-second bins, the interior bins hold 127k-131k launches over the same 32
kernel kinds, but the first bin holds 122,904, the bin at 45-50 s holds
109,126, and one 801,297,888 ns gap sits at t=50.355 s. The earlier claim here
that *every* bucket holds 127k-131k launches was wrong; it was asserted from
the steady-state bins without bucketing the whole span.

The final-two-seconds claim does survive measurement: the largest inter-kernel
gap in the last 2 s is 120,768 ns across 50,777 kernels, well under the 5 ms
that was claimed. That window uses 29 distinct kernel kinds against prefill's
32.

It is called "the tail window" and not "the decode window" throughout, and one
sentence here previously slipped and called it the latter. The slip now has
evidence against it as well as the argument above: both NVTX-bearing captures,
where containment settles the question, stop recording device events more than
a second *before* decode begins, and hold zero decode kernels (see "The two
captures compared"). This reference capture has no NVTX table, so its tail
cannot be attributed either way — but the prior should be that it is prefill,
not decode.

**This reference capture is itself incomplete, and by the same mechanism
described under "What the capture drops on exit" below.** Its diagnostics
carry `Not all CUDA events might have been collected` and no
`Number of CUPTI events produced` line at all. That missing line is a
shutdown-path diagnostic, not a completeness gate — the threaded `_exit`
control arm retained all 1,500,000 of its kernels without it — so read it as a
reason to go check the source counts and NVTX containment, which is what the
first sentence of this paragraph rests on. So the 801 ms gap is not evidence that
the GPU was idle at t=50.355 s; it is evidence that the capture stopped
recording there. Nothing in this section's numbers should be read as a device
occupancy measurement.

Prefill *is* batched. `EvalBatch` at `cuda_session_linux.go:308` runs chunks of
`cudaPrefillBatchWidth = 8` (`:72`), and the `*_batch` kernels are in the
capture to prove that path ran: `ds4_q8_0_matvec_q8_1_batch` 32,643 launches,
`ds4_quant_q8_1_batch` 21,762. It simply does not pay — 12.96 t/s prefill
against 11.92 t/s generation, a 9% spread where batched prefill would normally
run several times faster than decode.

So what a prefill comparison against the oracle has to account for is not a
missing batch path but an ineffective one. That is a property of the runtime,
not of the profiler.

### Where the prefill layer spans come from

`beginLayer` sits in two places. `evalLayerState` carries the decode one, on
the single-token path under `evalPosition`. `evalLayerBatch` carries prefill's,
and spans the whole chunk rather than each token in it: the batched front end
evaluates all eight together, so a per-token interval would report one duration
eight times. The projected tail both paths share opens no span, which is why
before 8d5baa4d a 2048-token prefill emitted no layer intervals at all.

Measured on 2026-09-12 with both logits digests supplied and reproduced:

| span | n | total |
| --- | --- | --- |
| prefill | 1 | 158,155,050,285 ns |
| batch | 256 | 158,154,800,221 |
| layer | 11,094 | 147,733,008,877 |
| generation | 1 | 166,664,130 |
| token | 2 | 166,412,034 |

The `layer` row is every layer interval in the file. Split it by which phase
span contains it before comparing against either phase, or the decode layers
inflate the prefill figure:

| | n | total |
| --- | --- | --- |
| layer, within `prefill` | 11,008 | 147,647,267,982 ns |
| layer, within `generation` | 86 | 85,740,895 |

So prefill's layer intervals are 93.36% of the prefill span, and the residual
is 10,507,782,303 ns — 10.508 s the layer spans do not cover. Decode's are
51.4% of its span, a much larger share left over, which is what you would
expect when 43 layers are amortized over one token instead of eight.

That residual is host time outside the layer spans. It is not a measured
attribution: nothing here says which of the per-token embedding rows, the
group tail, the vocabulary projection, or the per-chunk stream synchronization
it went to. Naming the candidates is not measuring them, and a finer span
level is what would settle it.

Do not reach for `markCUDAPhase` to get finer detail into this sidecar. The
shallow-regime phase count is `(tokens/8)*(43+1) + tokens*43*8`
(`cuda_phase_profile_linux_test.go:95`), which is 715,776 intervals at 2048
tokens — seven times the record cap and far past the byte cap, so most of
prefill and probably all of decode would be dropped, and the sidecar would
still be written. Detailed phase names belong in NVTX, not here.

## NVTX qualification run, DGX spark-31db, 2026-09-12

Run ID `nvtx-20260912-DF6906F6`, commit `80f8fcb5`, `ds4-min-nvtx` built
go1.27.0 linux/arm64 with `vcs.modified=false`, nsys 2026.1.1, model content
sha256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
2048 prompt tokens, 2 decoded, `-ctx 4096`, both digest expectations passed on
the binary's own check, accepted IDs 15342 and 201, stdout `" alla"`.

The instrumentation is close to free. Prefill 12.81 t/s and generation
11.84 t/s, against 12.96 and 11.92 for the same workload with spans but no
NVTX and no profiler — about 1.2% and 0.7%.

Every span has its range and every range closed:

| | |
| --- | --- |
| `NVTX_EVENTS` rows | 11,356 |
| unfinished (`end IS NULL`) | 0 |
| unclosed (`end > 1e12`, see below) | 0 |
| rows not of `eventType` 60 | 0 |
| rows with an absolute-magnitude start | 0 |
| host sidecar spans | 11,354 |
| CUDA kernel rows | 3,973,441 |

The name histogram matches the sidecar inventory exactly — 11,094 `layer`,
256 `batch`, 2 `token`, 1 `prefill`, 1 `generation`, summing to the 11,354
spans, plus the 2 calibration ranges.

`end IS NULL` is the obvious test for an unclosed range and it is the wrong
one: Nsight does not leave an unclosed range NULL, it backfills the end with
the session's end stamp taken on the **boot clock**, while every legitimate
end is in the session domain, which starts at 0. `end IS NULL` therefore
returns 0 whether or not ranges were lost.

The test is *"is this end outside the session domain"*, and the session
domain's upper bound comes from the capture itself — the session duration, or
the last timestamp in the device tables. `end > 1e12` is the instance of that
test for these runs, where the session is ~163 s and the boot clock is ~5e14;
it is not a universal threshold, because a legitimately long capture would
cross 1e12 on its own. Bound it against the session, not a constant.

Both rows are recorded above because for *this* capture the two agree — its
NVTX side did close cleanly, both calibration ranges survived, and the loss
here is device events only. The P3a capture below is where they disagree.

**What this run does not carry:** NVTX names below the layer, and the clock
mapping. The ranges mirror the recorded spans, so a device kernel still cannot
be attributed to a phase name finer than `layer`; `markCUDAPhase` granularity
is a separate capture, and putting those names in the host sidecar as well
needs a streaming writer, not a larger cap. And the calibration brackets came
back disjoint — see "Do not align two clocks by eye" — so the device and host
traces still have to be read side by side.

Receipts on the DGX in `~/tmp/nvtx-qual` with `remote.sha256`, copied to
`~/tmp/ds4-capture-local/nvtx-qual` with digests verified against it. The
1.23 GB SQLite export and the 466 MB `.nsys-rep` stay on the DGX; everything
else, including the binary, is local.

## What the capture drops on exit

Both captures above lose CUDA device events, and the loss is large enough to
change what you may conclude from either. This section records what was
measured, what the measurements refute, and what remains open — the mechanism
is *not* fully established, and the guide should not be read as if it were.

### The loss, measured

In the NVTX qualification capture, NVTX tells you exactly where the device
record is missing, because every prefill batch is a named range and containment
is checkable. Counting kernels whose start falls inside a batch range
(`batch.start <= k.start < batch.end` — "any kernel at or after the batch
start" is the wrong query and reports 255/256):

| | |
| --- | --- |
| prefill batches with at least one contained kernel | 253 of 256 |
| empty batch ordinals | 7, 227, 256 |
| kernels inside the `generation` range | 0 |
| largest interior gaps | 1.765 s @ 5.223 s, 1.215 s @ 142.553 s, 0.804 s @ 35.395 s, 0.709 s @ 73.555 s |

The three empty batches each hold **43 finished NVTX layer ranges and zero CUDA
runtime rows**. A batch whose 43 layers all ran and closed cannot have issued
zero CUDA API calls, so this is capture loss, not an idle device.

It is also not a clock-domain artifact. The two calibration brackets bound the
host-to-report rate difference at 11.11 ppm (see above), which accumulates to
1.78 ms over the whole 160 s run — three orders of magnitude short of the
0.709-1.765 s gaps it would have to explain.

### What a matched control refutes

`~/tmp/cupti-tail/tail.cu` and `threads.cu` on the DGX reproduce the capture's
*shape* with no model: a long stream of small launches under NVTX ranges. Every
arm is measured, not asserted.

| arm | kernels | span | rate /s | ranges nonempty | min contained |
| --- | --- | --- | --- | --- | --- |
| baseline | 275,000 | 0.702 s | 391,522 | 11/11 | 25,000 |
| `--cuda-flush-interval=200` | 275,000 | — | — | 11/11 | 25,000 |
| `cudaDeviceReset` | 275,000 | — | — | 11/11 | 24,999 |
| long | 4,025,000 | 10.443 s | 385,434 | 161/161 | 24,999 |
| long + 3 s linger | 4,025,000 | 10.439 s | 385,585 | 161/161 | — |
| **paced** | 4,025,000 | **154.420 s** | **26,065** | 161/161 | 25,000 |
| **per-range pthread** | 4,000,000 | **153.914 s** | 25,988 | 160/160 | 25,000 |

The paced arm matches the real capture on volume, rate and duration at once and
loses nothing. The threaded arm additionally creates and destroys an OS thread
per range — the Go runtime's thread churn, isolated — and also loses nothing.

So launch volume, launch rate, run duration, a trailing linger,
`cudaDeviceReset`, a 200 ms flush interval, and thread retirement are each
**insufficient to produce the loss** in a C program. That is the exact claim
they support. It is *not* the claim that `--cuda-flush-interval` would fail as
a mitigation for the Go loss — that is untested.

### The exit path reproduces the diagnostic signature

The discriminator that survived is in Nsight's own diagnostics. A capture that
lost nothing reports both sides of its accounting:

```
Number of CUDA events collected: 	12075169.
Number of CUPTI events produced: 	12075623, CUPTI buffers: 50.
```

Both captures in this guide report only the first line. The second is printed
during process teardown, and Go's `runtime.exit` on linux/arm64 goes straight
to `SYS_exit_group` (`src/runtime/sys_linux_arm64.s:53`), skipping the libc
teardown every C control arm above performs by returning from `main`. Which
part of that teardown emits the line is unestablished: an `atexit` handler is
the obvious candidate, but nothing here reads CUPTI's source or intercepts its
registration, and `_exit` removes all of libc's teardown rather than one
handler.

Changing only that in the control reproduces the signature:

| arm | ranges | kernels expected | kernels seen | short range | produced line |
| --- | --- | --- | --- | --- | --- |
| threaded, return from `main` | 160 | 4,000,000 | 4,000,000 | none | present |
| threaded, `_exit(0)` | 60 | 1,500,000 | 1,500,000 | none | **absent** |
| single-thread, `_exit(0)` | 60 | 1,500,000 | **1,499,944** | last range, 24,944 | **absent** |

The return-from-`main` row is the 160-iteration threaded arm from the table
above; the `_exit` arms were run at 60 iterations. The arms differ in length,
so read the last three columns, not the kernel totals across rows.

`_exit(0)` skips `atexit` exactly as `exit_group` does. It removes the produced
counter from the report in both arms, and in one arm it also drops 56 kernels,
all of them from the final range.

**This is a reproduced signature, not a proven mechanism.** What it explains is
the missing counter line and a truncated tail. What it does not explain on its
own is the interior loss — batches 7 and 227 are nowhere near exit. A
supporting observation, not a proof: in the real capture, five CUDA runtime
thread IDs stop emitting 0.694-1.304 s before their own last NVTX boundary, and
a sixth carries 122 NVTX boundaries with no runtime rows at all — the shape a
per-thread buffer that was filled but never flushed would leave. The control
does not yet reproduce *that*.

### What to do about it meanwhile

Treat gaps in a capture as unproven until the diagnostics are read. Concretely:

- Check for the `Number of CUPTI events produced` line in every capture before
  drawing a conclusion from device timing. Its absence means the capture shut
  down on a path that skipped the accounting, which is a reason to distrust the
  totals — **not** a completeness gate: the threaded `_exit` arm above retained
  all 1,500,000 kernels with no produced line. The gates are source counts and
  NVTX containment; the missing line only tells you to go check them.
- Do not use `Not all CUDA events might have been collected` as the signal. A
  control arm printed `Not all NVTX events might have been collected` on a
  capture whose NVTX and kernel accounting is exactly complete; the warning
  fires in both directions.
- Check NVTX range containment rather than eyeballing a timeline. A batch with
  finished layer ranges and no CUDA runtime rows is missing data, and only the
  containment query shows it.

## P3a qualification run, DGX spark-31db, 2026-09-12

Run ID `p3a-20260912-DF6906F6`, revision `ed2bf798dff21580b7c01f58865a4b50e0dda375`
with `vcs.modified=false`, same workload and model as above. This is the run
that carries the fine phase names in the host sidecar (`-span-level op
-spans-stream`) and four interior clock markers (`-calibration-every 64`).
Both digest expectations were asserted on the command line and both
reproduced bit for bit; the run exited 0. Prefill 12.81 t/s, generation
11.83 t/s — 1.2% and 0.8% against the uninstrumented 12.96 and 11.92.

The host half is complete, and its record count was predicted before the run:

| | |
| --- | --- |
| `spans_begun` = `spans_ended` = `spans_encoded` | 727,732 |
| `spans_unended`, `spans_dropped` | 0, 0 |
| `open_peak` | 4 |
| sidecar bytes | 148,725,448 |
| `complete` | true |

The per-name histogram matches the source formula for every name, with
prefill and decode counted separately (L=43 layers, T=2048 prefill tokens,
W=8 batch width, D=2 decoded):

| name | count | formula |
| --- | --- | --- |
| `attention`, `attention-output`, `ffn-pre`, `shared-ffn`, `expert-ffn`, `ffn-post` | 88,150 each | T·L + D·L |
| `qkv-normalization` | 88,064 | T·L, batched path only |
| `compression` | 84,050 | T·(L−2) + D·(L−2) |
| `compression-raw` | 4,100 | 2T + 2D |
| `dense-front` | 11,008 | (T/W)·L, batched path only |
| `group-tail` | 256 | T/W, batched path only |
| `layer N` (×43) | 258 each | 11,094 total |
| `batch`, `token`, `prefill`, `generation` | 256, 2, 1, 1 | |

716,378 fine + 11,354 coarse = 727,732. The 602 decode intervals are the
seven phases `evalLayerProjected` marks per layer per token, 2·43·7. No
indexer phases appear, which is correct for 2048+2: the compressed indexer
emits on `(position+1)%4==0` and its 513th emit is at position 2051, which
this workload never reaches.

**The device half of this capture is not usable, and the failure is worse
than the reference run's.** Two losses, both measured:

*Tail.* Device recording stops at 161.438 s (last kernel end; last CUDA
runtime row 161.397 s) while the process ran to about 162.909 s. Kernel
density is flat at ~6,300 per 250 ms right up to the cut, then one partial
bin and nothing. All 3,919,871 of the 3,919,873 captured kernels fall inside
the `prefill` range; `generation`, `token 1` and `token 2` each contain
**zero**.

A backfilled range must not be used as an ordinary containment bound, and
two of those three are backfilled. Here it costs nothing, because the
backfill only ever moves the end *later*: the interval counted is a superset
of the true one, so a count of zero inside it proves zero inside the true
range. `token 1` closed normally and also holds zero. Anywhere the count is
nonzero, re-derive the bound before attributing it. The three last NVTX calls the process made are also gone — `token`
2's `RangeEnd`, `generation`'s `RangeEnd`, and the whole `end` calibration
range. The `end` range is absent in the strongest available sense: its text
was never interned, so `StringIds` holds five of the six marker names.

*Interior.* Per-batch kernel counts are otherwise exact — 219 batches hold
16,083 and 15 hold 16,103 — so the baseline is measured, not estimated. The
other 22 batches are damaged, and the damage comes in adjacent pairs:
(55,56), (65,66), (138,139), (158,159), (186,187), (226,227), (238,239),
each pair losing about one batch's worth across a two-batch window, plus
ordinal 195 empty on its own and 256 truncated by the tail cut. The 22 hold
156,149 where the baseline predicts ~353,826: a deficit of ~197,677 kernels,
about 4.8% of prefill.

Adding the fine names did not cause this. They are host-only in P3a — no
NVTX range is opened for them, which `TestPhaseSpansOpenNoNVTXRange` pins —
and the NVTX inventory here is the same 11,354 coarse ranges as the
qualification run above.

### The clock is not affine, and this is a proof

The point of the interior markers was to decide whether the residual between
the span clock and the Nsight domain is a step or a drift. Five markers
reached the capture:

| label | host bracket (ns) | Nsight start |
| --- | --- | --- |
| `start` | 501367652635873..501367652644689 | 2,060,603,232 |
| `batch-64` | 501406611549429..501406611550485 | 41,019,136,928 |
| `batch-128` | 501446322257679..501446322258879 | 80,729,473,472 |
| `batch-192` | 501486467469733..501486467471317 | 120,874,319,312 |
| `batch-256` | 501526899856810..501526899858346 | 161,306,346,944 |

Do not answer this with a least-squares fit. A fit that misses a bracket is
not evidence that no mapping exists, and a fit through every point has no
held-out validation to offer. The question is feasibility, and it is a
two-variable linear program: is there an (a, b) with
`lo_i ≤ a + b·x_i ≤ hi_i` for every marker? Enumerate the 45 vertices formed
by pairs of the 10 constraints, in exact rational arithmetic, and test each
against all 10.

| model | verdict |
| --- | --- |
| constant offset (b = 1) | infeasible by 1,468,409 ns |
| affine, five recorded markers | infeasible |
| affine, four interior markers alone | infeasible |

Infeasibility is a proof rather than a failure to fit: no line stabs all five
segments. It holds on the four interior markers alone, so it is not an
artifact of the first NVTX call's wider bracket (8,816 ns, against
1,056–1,584 ns for the others). This settles what the two-bracket 11.11 ppm
result could not: the residual is neither a step nor a rate offset, so no
constant and no linear rescaling maps one clock onto the other. What it does
not establish is that the clocks are non-affine in general — this is one
capture, and a truncated one.

### What this capture is good for

Host phase timing at fine granularity, numerics identity, the prefill device
timeline through 161.438 s minus the 4.8% noted above, and the clock result.
Not decode on the device. The stated goal is prefill *and two decode tokens*;
on the device side this run delivers only the first, so the capture-loss
mechanism is now blocking a deliverable rather than qualifying a footnote.


## The two captures compared, and what the comparison shows

Both captures were put through the same per-batch containment query. In both,
every batch is classified and the classifications account for all 256. The
*sums* differ in what they prove, and an earlier draft of this section
overstated it: in the qualification capture the batch sum equals the capture's
total kernel count exactly, so nothing lies outside a prefill batch range; in
P3a the batch sum is 3,919,871 against a total of 3,919,873, leaving **two
kernels outside every batch range**. Two is not a rounding error, it is two
rows with a location, and the claim that nothing lies outside holds for one
run and not the other. Bound the query by `min(start)` and `max(end)` over
the batch ranges rather than with a `NOT EXISTS` over 3.9M rows, which does
not finish, and the two rows come back named:

```
1951625856|1951628064|ds4_probe|before first batch
1953446272|1953449952|ds4_read_u16|before first batch
```

Both start before the first batch range opens, so they are warm-up rather
than loss. That is now a measurement and not an assumption.

| | NVTX qualification | P3a |
| --- | --- | --- |
| batches at 16,083 | 228 | 219 |
| batches at 16,103 | 15 | 15 |
| ordinal 1 | 16,020 | 16,020 |
| below baseline, excl. ordinal 1 | 12 | 21 |
| of those, under 15,000 | 12 | 20 |
| accounting | 228+15+1+12 = 256 | 219+15+1+21 = 256 |
| sum over batches | 3,973,441 | 3,919,871 |
| total kernels in capture | 3,973,441 | 3,919,873 |
| last kernel end | 160.807 s | 161.438 s |
| `prefill` range closes | 161.897 s | 161.952 s |
| decode kernels | 0 | 0 |
| NVTX ranges lost | none | last 3 calls |

Ordinal 1 holds exactly 16,020 in *both* runs — 63 below baseline, the same
number twice. That is deterministic first-batch warm-up, not loss, and it is
excluded from the damage counts below.

### We have never captured decode on the device

This is the finding that matters, and it was visible in the first capture
without being stated. Its last kernel lands at 160.807 s while `prefill`
closes at 161.897 s and decode runs to 162.067 s — so device recording had
already stopped more than a second before decode began. P3a repeats it with a
1.47 s cut. The batch accounting says the same thing independently: in the
qualification run every one of its 3,973,441 kernels is inside a prefill
batch, and in P3a all but two are.

The workload is "prefill 2048, then decode two tokens". On the device side we
have only ever had the first half. Any statement in this guide or elsewhere
about decode *device* behaviour on CUDA rests on no data.

### The loss recurs at similar positions across runs

The damaged batches are not scattered. They cluster into short runs of
adjacent ordinals, and the runs partly repeat across two independent captures:

```
NVTX qualification:  [6,7,8]  [55,56]  [116,117]  [226,227,228]  [255,256]
P3a:                 [54,55,56]  [65,66]  [138,139,140]  [158,159]
                     [186,187]  [194,195,196]  [226,227]  [229]
                     [238,239]  [256]
```

Three of the five qualification windows overlap a P3a window. Treating the
P3a windows plus one ordinal of adjacency slack as covering ~43 of 256
positions, independent placement would put ~0.84 of the five on top of one.
Three landed there. That is a rough expectation, not a test.

**This does not narrow the mechanism, and an earlier draft of this section
wrongly said it pointed away from buffer pressure.** It does not. The workload
is deterministic: the same launches and the same API volume happen at the same
token positions every run, so a buffer that fills from launch volume fills at
reproducible positions too. Recurrence at the same ordinals is exactly what
buffer pressure predicts here, not evidence against it. What the recurrence
does buy is a *predicted* place to look — a control can now be pointed at
batch 55 or 226 rather than at a run's worth of timeline.

### It is loss, not an async-launch shift

A CUPTI kernel row is stamped with *device execution* time, not launch time,
so a deficit in one batch range could in principle be work that executed
inside a neighbour's range. It is not. No batch in either of these two
captures exceeds the baseline, so there is no surplus anywhere to receive
shifted work, and each damaged window is a net deficit of roughly 22,800
kernels against a baseline that 243 of 256 batches hit exactly in one run and
234 of 256 in the other.

The scope matters: later captures do hold a batch above baseline. With the
exit flush in place the final batch comes in at 16,112, above both 16,083 and
16,103, because it carries the tail work that earlier captures were losing.
Re-derive the surplus before reusing this argument on a different run.

## The cut is a fixed residual at exit, not a ceiling

Three full-length captures stopped within a 0.63 s window of each other, at
160.807 s, 161.323 s and 161.438 s, each having recorded close to four
million kernels. Two readings fit that: a *time* ceiling somewhere near 161 s,
or a *volume* ceiling somewhere near 4M kernels. The two are confounded in
every full-length run, because the workload always takes about the same time
to emit about the same number of kernels.

A shorter prompt separates them. `probe512` is the same binary, the same
flags and the same `--cuda-flush-interval=1000` as `p3aflush`, with a 512-token
prompt instead of 2048 — a quarter of the duration and a quarter of the
kernels. If either ceiling were real the run would finish under it and record
to the end.

| run | prompt | flush interval | kernels | last kernel | `prefill` closes | cut |
| --- | --- | --- | --- | --- | --- | --- |
| nvtx-qual | 2048 | none | 3,973,441 | 160.807 s | 161.897 s | 1.091 s |
| P3a | 2048 | none | 3,919,873 | 161.438 s | 161.952 s | 0.514 s |
| p3aflush | 2048 | 1000 ms | 4,049,488 | 161.323 s | 162.159 s | 0.836 s |
| probe512 | 512 | 1000 ms | 993,771 | 41.205 s | 41.885 s | 0.680 s |

**Both ceilings are refuted.** `probe512` cuts at 41.205 s with 993,771
kernels recorded — a quarter of the way to either proposed ceiling — and it
cuts for the same reason, 0.680 s before its own `prefill` range closes, with
decode again holding zero device rows. The run length varies by a factor of
four and the kernel count by a factor of four; the cut stays between 0.514 s
and 1.091 s. It scales with neither.

The hypothesis was stated in writing before the probe ran, and the probe
killed it. It is recorded here rather than deleted because the shape of the
refutation is the useful part: two candidate ceilings that a single workload
length cannot tell apart, separated by changing the one input that moves them
independently.

### What the cut is instead

What survives is a residual measured from the *end*: recording stops a
fraction of a second before the process does, by an amount that looks like
buffered data that was never written out. Under a periodic flush the residual
is bounded by the interval — at 1000 ms, `p3aflush` lost 0.836 s of prefill
plus decode's 0.170 s, and `probe512` lost 0.680 s plus the same 0.170 s, both
just under one interval. With no interval set at all the residual is bounded
instead by however much CUPTI buffers before it fills and drains on its own,
which lands in the same range at this launch rate.

Decode is the casualty in every run for a structural reason rather than a
coincidence: decode is the last 0.17 s of the process, and the residual is
larger than that in all four captures. Decode has never been outside the lost
window.

This predicts something cheap to test. Shrink the flush interval well below
decode's duration and the residual should shrink with it, and decode should
start appearing on the device timeline. It also says what the real fix is: a
flush at exit. The capture's own diagnostics name it —

```
Buffers holding CUDA trace data will be flushed on CudaProfilerStop() call.
```

— and `ds4-min` never calls `cudaProfilerStop`; there is no binding for it
anywhere in the tree. Go's `runtime.exit` on linux/arm64 goes straight to
`SYS_exit_group` (`src/runtime/sys_linux_arm64.s:53`), so no libc teardown
runs either. Nothing in the process ever asks for that final flush.

A successful short-interval run would qualify a *mitigation*, not confirm the
mechanism. Confirming the mechanism means making the flush happen and watching
the residual go to zero.

## The missing tail was never flushed, and the fix is one call

Every capture carries this line in its own diagnostics:

```
Buffers holding CUDA trace data will be flushed on CudaProfilerStop() call.
```

`ds4-min` never made that call — there was no binding for it anywhere in the
tree — and nothing made it on the way out either, because `runtime.exit` on
linux/arm64 issues `exit_group` directly (`src/runtime/sys_linux_arm64.s:53`)
without running libc teardown. Whatever the profiler had buffered when the
process left was discarded.

`-cuda-profiler-stop` binds `cuProfilerStop` from `libcuda.so.1` through
purego and calls it after the last kernel and after the last NVTX range
closes. `probestop` is `probe512` with that one flag added and nothing else
changed — the two run scripts differ by one line, and `--cuda-flush-interval`
stays at 1000 ms in both:

| | probe512 | probestop |
| --- | --- | --- |
| decode device kernels | 0 | 4,480 (2,240 + 2,240) |
| last kernel to end of decode | 844.9 ms | 0.119 ms |
| kernels | 993,771 | 1,033,820 |
| last batch | empty | 16,112 |
| batches below 15,000 | 5 | 0 |
| `ds4_get_row_f16` | 1,984 | 2,056 |
| `ds4_router_hash` | 1,488 | 1,542 |
| `ds4_router_biased` | 19,769 | 20,560 |
| prefill / decode | 12.81 / 11.83 t/s | 12.91 / 12.11 t/s |

The throughput row matters as much as the others: the call happens once, at
the end, so it costs nothing during the run. Compare `--cuda-flush-interval=10`,
which also recovers some decode and inflates prefill wall time from 39.6 s to
88.1 s doing it.

### Why the counts are the gate and the symmetry is not

Two decode tokens holding 2,240 kernels each is suggestive and proves
nothing; a loss that happened to be symmetric would look the same. What
settles it is a count derived from the source rather than from the capture.
Three kernel families issue a fixed number of launches per *evaluated*
token — prefill tokens and decode tokens alike, so a run of `n` prompt
tokens plus two generated ones has `n + 2` evaluations:

| kernel | launches per evaluation | n=512 | n=2048 |
| --- | --- | --- | --- |
| `ds4_get_row_f16` | 4 | 2,056 | 8,200 |
| `ds4_router_hash` | 3 | 1,542 | 6,150 |
| `ds4_router_biased` | 40 | 20,560 | 82,000 |

The four come from `encodeEmbeddingRow` being called four times per token on
both the batch path (`cuda_session_linux.go:377`) and the single-token path
(`:581`), each issuing exactly one launch (`cuda_runtime_linux.go:1246`). The
three and the forty come from the shared layer tail across 43 layers, where
`index < 3` selects `encodeRouterHash` and every other layer selects
`encodeRouterBiased` (`:708`).

`probestop` hits all three exactly, and no capture before it did. These are
**necessary conditions, not a completeness proof**: a capture can satisfy
three sentinel families and still lose others. They are the strongest gate
available without enumerating every launch in the program.

### At 2048 the call recovers most of the loss, not all of it

`p3astop` is the production workload — prefill 2048, two decode tokens, the
registered logits digests asserted by the run itself — with the same one-flag
change against `p3a-flush-run.sh`. It passes every gate the 512-token arm
passes except the one that matters most:

| | p3aflush (no stop) | p3astop (stop) | required |
| --- | --- | --- | --- |
| `ds4_get_row_f16` | 8,032 | 8,168 | 8,200 |
| `ds4_router_hash` | 6,024 | 6,126 | 6,150 |
| `ds4_router_biased` | 80,585 | 81,488 | 82,000 |
| last kernel to end of decode | 514 ms | 35.4 ms | 0 |
| decode kernels | 0 | 3,602 | 4,480 |
| decode token 1 | 0 | 2,240 | 2,240 |
| decode token 2 | 0 | 1,362 | 2,240 |
| batches below 15,000 | 10 | 4 | 0 |
| empty batches | 1 | 0 | 0 |
| prefill / decode | 12.90 / 12.10 t/s | 12.79 / 11.85 t/s | — |

Both logits digests reproduce bit-for-bit and the host sidecar is complete at
727,732 spans with nothing unended or dropped, so the run itself is sound;
what is short is the device recording. Decode token 1 is complete and token 2
holds 61% of its launches, consistent with the 35.4 ms residual against its
84.2 ms span.

The improvement is large — the deficit falls from 42 token-equivalents to
8 on both `get_row_f16` and `router_hash` — and it is not zero, where the
512-token arm's is.

Do not read that as "the flush is necessary but insufficient". Insufficient at
this scale is measured. Necessary is not: no control has excluded some other
intervention recovering the same data, and an intervention that recovers
something is evidence about the mechanism, not proof of it. What is
established is that this call recovers most of the loss at 2048 and all of it
at 512.

The launch counts also stop moving together: `router_biased` is short by 12.8
token-equivalents against the other two families' 8, which rules out reading
the remaining deficit as some whole number of lost tokens.

### Both levers together close the inventory

`p3astop100` is `p3astop` with `--cuda-flush-interval=100` instead of 1000 —
one line different, both levers on. It is the first capture of the production
workload whose *inventory* is complete — every count the project knows how to
derive comes out exact. It does not pass every gate: the host-to-device clock
gate still fails, and that failure is the subject of the next subsection.

| inventory gate | required | p3astop100 |
| --- | --- | --- |
| prefill logits sha256 | `4eee1b14…` | matches |
| final logits sha256 | `837a6b87…` | matches |
| accepted tokens | 15342, 201 | 15342, 201 |
| exit status | 0 | 0 |
| host spans begun = ended = encoded | 727,732 | 727,732 |
| host spans unended / dropped | 0 / 0 | 0 / 0 |
| `ds4_get_row_f16` | 8,200 | 8,200 |
| `ds4_router_hash` | 6,150 | 6,150 |
| `ds4_router_biased` | 82,000 | 82,000 |
| decode token 1 / token 2 | 2,240 / 2,240 | 2,240 / 2,240 |
| batches below 15,000 | 0 | 0 |
| empty batches | 0 | 0 |
| NVTX calibration markers | 6 | 6 |
| unclosed NVTX ranges | 0 | 0 |

The kernel accounting closes exactly, which no earlier capture managed:

```
total kernels                4,121,996
prefill batch sum            4,117,514   = 239*16,083 + 15*16,103 + 16,020 + 16,112
decode kernels                   4,480
unaccounted                          2   the two pre-prefill warm-up probes
```

The residual from the last kernel to the close of `generation` is 131,936 ns.
Throughput is 12.78 / 11.69 t/s against 12.96 / 11.92 uninstrumented, so the
whole instrumented capture costs about 1.4% of prefill and 1.9% of decode.

Neither lever does this alone. The exit flush alone leaves 8 token-equivalents
and 39% of decode token 2 missing; a 100 ms interval alone leaves all of
decode missing. **Use both.** The recipe is in the CUDA section above.

### The clock gate still fails, and now it means something

A complete inventory is not an aligned trace. Lining a host span up against
the device kernels it launched needs a mapping between the two clocks, and on
this capture there is none: with the full six-marker inventory present, the
constant-offset intersection is empty by 433,684 ns, and exact rational
feasibility for *any* positive-rate affine map is infeasible — on all six
markers and independently on the four interior ones. Witness brackets:

```
all six:  lower 39013512071/39013402240 (start, batch 64)
        > upper 20685395543/20685341560 (batch 192, end)
interior: lower  7966498707/7966476848  (batch 64, batch 128)
        > upper 20274519825/20274466784 (batch 192, batch 256)
```

The earlier captures showed the same infeasibility, and it could always be
waved away: they were missing markers, they were missing their tails, so
perhaps the surviving brackets were simply wrong. Here all six bracket
records are present, and the device-tail cutoff seen in every earlier arm is
absent under the checks above. **The two confounds we had actually observed
are gone from this run.** That is the whole of it: records outside the
enumerated checks, and timestamps assigned wrongly rather than lost, are
untested either way. This is not a blanket exclusion of loss.

Until that gate passes, this capture supports device-side reasoning and
host-side reasoning separately. It does not yet support putting a host span
and a device kernel on one timeline, which is the thing the guide exists to
produce.

### What the inventory closure does and does not prove

Every inventory gate is a necessary condition that something could still slip
past — three sentinel kernel families are not an enumeration of every launch
in the program, and per-batch equality against an empirical baseline is not a
derivation of what each batch should contain. What can be said is that every
independently derived count this project knows how to compute now comes out
exact, the accounting closes on the capture total with the two warm-up rows
named, and the workload's own numerics gate passed bit-for-bit on the same
run. That is the strongest statement the inventory gates support. It is not a
qualified aligned trace, and an earlier draft of this section called it one.

### What this does not establish

The call is an intervention, and an intervention that recovers the missing
data is strong evidence about the mechanism without being a proof of it: the
binary changed, and the profiler's interception of the call can itself alter
buffering. The honest claim is that the tail is recoverable by flushing at
exit and that nothing else tried — flush intervals from 1000 ms down to
10 ms, a `_exit` control, a `cudaDeviceReset` control — recovered it.

Two smaller notes, both corrections to earlier drafts of this guide:

`probestop` is not the first capture to hold any decode kernel. The 10 ms
flush arm held 725, in token 1 only, with token 2 empty. `probestop` is the
first *symmetric two-token* coverage.

Interior batch damage is run-variable, not deterministic. The first
`-cuda-profiler-stop` run had five batches below 15,000 and the second, with
the same flag, had none. That rules out reading the interior loss as a
separate defect that the flush leaves untouched; what can be said is that it
persists across some runs under this intervention and vanishes in others.

### One live bug found by the fix

The first `-cuda-profiler-stop` run exited 1: `cuProfilerStop` returned
`CUDA_ERROR_INVALID_CONTEXT`. The call operates on the calling thread's
current context, and the goroutine that reaches the end of a run is not in
general on the thread that established one. The fix pins the thread and makes
device 0's primary context current first.

The flush had happened anyway, because it is the profiler's interception of
the call that flushes rather than the driver's own return path. That is not
something to rely on. The error was real, the process exited 1, and a run that
exits 1 is not a run — the corrected control reproduces the capture with the
call returning success.

## The C oracle eats the same tokens, and the input path needed no patch

The other half of the comparison needs both runtimes to consume identical
content. That looked like a problem: `ds4-min` takes a JSON array of token
IDs (`-prompt-tokens`), while `ds4-bench` takes only prose
(`--prompt-file` / `--chat-prompt-file`). The obvious move was to patch a
third input arm into the oracle — which would have made every capture a
measurement of a modified oracle.

It is unnecessary, because the fixture was never ds4-min's tokenization in
the first place. `docs/oracle-workload-bench.md` records that
`scripts/testdata/oracle-workload-tokens2048.json` "comes from the pristine
b0a147a CPU token dump of promessi_sposi.txt" — it is the *C* tokenizer's
output, which the Go side consumes verbatim. So feeding `ds4-bench` that
same text should reproduce it exactly. That is a claim, and it was checked
rather than assumed:

    ./ds4 --cpu -m $MODEL --ctx 4096 --raw \
      --prompt-file speed-bench/promessi_sposi.txt --dump-tokens

The first 2048 IDs, packed little-endian uint32 (8192 bytes) and hashed:

    5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380

which is the registered fixture hash, exactly. `promessi_sposi.txt` is
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`, and all
eight copies of it on the DGX are that same file.

Three seams make the dump stand in for what `ds4-bench` actually prefills,
each read in source rather than inferred:

- `ds4_cli.c:522-523` — under `--raw`, the CLI tokenizes with
  `ds4_tokenize_text(engine, prompt, out)`.
- `ds4_bench.c:737` — under `--prompt-file`, the bench calls the same
  `ds4_tokenize_text`. (The chat arm at `:735` does not; `--raw` is what
  matches it, and `--chat-prompt-file` would not.)
- `ds4_bench.c:810-814` — the prefilled prefix is `{.v = prompt.v, .len =
  frontier}`, the first `frontier` IDs with nothing prepended.

The two binaries are separate links over the same `ds4.o`, so this is a
same-function argument, not a same-process observation. The stronger form —
capturing the prefix inside the timed bench process — is what
`docs/oracle-workload-bench.md` already flags as missing on the CUDA side,
and this does not supply it.

`ds4.c` here hashes to
`2403b881be7adbdb50f5901aabe2a08f35e7b34ed09f5b57f1b608843894393b`, the same
post-patch source as the qualified oracle references, so the tokenizer that
produced this dump is the one those references used.

### What still has to match, and what teacher forcing would not prove

Identical prefill input is not identical decode. `ds4-bench.c:904-906`
chooses `prompt.v[frontier + gen_done]` under `--teacher-forced-decode` and
`ds4_session_argmax_excluding(session, eos)` otherwise. ds4-min is
autoregressive and accepted 15342 then 201. The comparison worth capturing is
the autoregressive one, and its gate was that the C side emit those same two
IDs. It does; see the receipt below. Teacher forcing would pin the execution trajectory while proving nothing
about selection, so if it is ever used it gets its own label and its own row —
never a substitute for the argmax result.

## Both sides, same tokens: where the time actually goes

With input identity established above, the oracle was captured on the same
workload — prefill 2048, two decode tokens, context allocation 4096:

    DS4_BENCH_CUDA_PROFILE_PREFILL=1 DS4_BENCH_CUDA_PROFILE_RANGE=0:2 \
    nsys profile -t cuda-sw,nvtx,osrt \
      -c cudaProfilerApi --capture-range-end=repeat-shutdown:2 \
      --cuda-flush-interval=100 --flush-on-cudaprofilerstop=true \
      ./ds4-bench --cuda -m $MODEL \
        --prompt-file speed-bench/promessi_sposi.txt \
        --ctx-start 2048 --ctx-max 2048 --ctx-alloc 4096 --gen-tokens 2

`ds4-bench` calls `cudaProfilerStart`/`Stop` twice — around prefill
(`ds4_bench.c:822,827`) and around the decode range (`:914,963`) — so
`repeat-shutdown:2` yields two reports, `.1` prefill and `.2` decode. Both
runs are instrumented; neither number below is an uninstrumented rate.

| | C oracle | Go ds4-min | ratio |
| --- | ---: | ---: | ---: |
| prefill kernels | 3,472 | 4,117,514 | **1186x** |
| prefill device time | 3,280.79 ms | 150,476.96 ms | 45.87x † |
| prefill kernel envelope | 3,744.5 ms | 160,463.3 ms | 42.9x |
| envelope occupancy | 87.6% | 93.8% | — |
| decode kernels (2 tokens) | 3,536 | 4,480 | 1.27x |
| decode device time | 99.8 ms | 160.2 ms | **1.60x** |

† **This is one sample and it is not the ratio.** Two more C runs of this
exact workload, with the identical 3472 launches, put it at 58.53x and 31.69x.
Read *One C sample was not enough* below before using this row for anything.
The decode row is stable across the samples; the prefill row is not.

Read the row labels carefully. Device time is summed kernel duration; the
envelope is first-kernel-start to last-kernel-end, which is neither the
profiler range nor the function's wall interval. Occupancy is the ratio of the
two and describes how little dead air sits between kernels — it is not
evidence for a compute-bound, memory-bound, or launch-bound diagnosis, and
summed kernel time is not an operation count.

### The gap is not launch overhead, and not one constant either

The 1186x launch ratio invites the conclusion that ds4-min is drowning in
launch overhead. It is not: at 87.6% and 93.8% envelope occupancy there is no
idle gap for the extra launches to hide in, and the summed kernel time
differs by tens of times on its own — 31.7x at the narrowest sample. Whatever is wrong is happening *inside* the kernels.

The oracle's heaviest launches are small multiples of the 43 layers:

    gateup_iq2_swiglu_q8_d2r_kernel     43 launches    846.1 ms
    dense_q8_d2r_kernel                172 launches    658.4 ms
    down_q2k_d2r_kernel                 43 launches    330.8 ms
    attention_tokentile_hmma_kernel     41 launches    227.1 ms

Forty-three gate-up launches for the entire 2048-token prefill is one per
layer: the oracle's `prefill_chunk` defaults to 4096 on CUDA, so all 2048
tokens cross each layer in a single **matrix-matrix** product, and each
layer's weights are read once for the whole chunk.

An earlier draft of this section concluded that raising
`cudaPrefillBatchWidth` from 8 would therefore close the gap — "one constant,
not 43 slow kernels". **That is wrong, and the source refutes it.**
`cuda_session_linux.go:543-556` batches only the dense front
(`encodeQ80MatvecQ81Batch(..., n)`) and then loops `for i := range rows`
calling `evalLayerProjected` per token. Everything downstream of that loop —
the shared FFN and the whole expert path at `:717-762` — is `Matvec`: matrix-
*vector* products on one token's state, `encodeSwiGLU(..., 1)`, one token at a
time. Widening the batch changes dense-front batching and buffer sizes; it
does not turn expert gate/up/down into cross-token launches.

So the difference is algorithmic, not a tuning constant: **the oracle batches
expert evaluation across tokens, and ds4-min executes token-specific
matvecs.**

Resist the next step, which an earlier draft took — that the oracle reads each
expert's weights once where ds4-min reads them 2048 times. Launch structure is
not memory traffic. Tiling and cache behaviour can reread data inside a single
matmul, and experts are routed *per token*, so ds4-min does not touch every
expert on every one of the 2048 passes. Batched evaluation creates the
opportunity for weight reuse; actual bytes moved need counters or an explicit
access model, and neither has been collected. Batching the expert path is a
real piece of work, and any proposed optimization gets measured on its own
rather than inherited from this observation.

### What qualifies this comparison, and what does not

The Go side carries source-derived completeness checks that came out exact —
8200 / 6150 / 82000 across three kernel families. The C side now has an
equivalent, derived from `ds4_cuda.cu` and `cuda/mmq/ds4_mmq_d2r.cu` against
this build's CUDA identity, and every one of them passes. They hold only for
this source revision, these build defines, the default env toggles, and the
selected D2R path; change any of those and the expectations have to be
re-derived.

| family | expected | prefill | | expected | decode |
| --- | ---: | ---: | --- | ---: | ---: |
| `router_select_warp_topk` | 43 | **43** | | 86 | **86** |
| `moe_mmq_sum` | 43 | **43** | | 86 | **86** |
| `store_raw_kv_batch` | 43 | **43** | | — | — |
| `gateup_iq2_swiglu_q8_d2r` | 43 | **43** | | — | — |
| `down_q2k_d2r` | 43 | **43** | | — | — |
| `d2r_build_worklist<32,64>` | 86 | **86** | | — | — |
| `swiglu` | 43 | **43** | | 86 | **86** |
| `embed_token(s)_hc` | 1 | **1** | | 2 | **2** |
| `output_hc_weights` | 1 | **1** | | 2 | **2** |
| `hc_weighted_sum` | — | — | | 2 | **2** |

Ten rows, sixteen individual counts (ten in prefill, six in decode) over
twelve distinct kernel symbols: the `d2r_build_worklist` row carries two
template instantiations, and prefill's `embed_tokens_hc` is a different symbol
from decode's `embed_token_hc`.

One launch per layer for a single layer-major prefill call over all 2048
tokens, and 43x2 for two single-token decode Evals. `d2r_build_worklist`
reaching 86 is the `<32>` and `<64>` instantiations at 43 each.

This is the same *kind* of check the Go side has, with the same limit: these
are necessary conditions on named families, not an enumeration of every
launch. Several families are deliberately excluded because their counts are
not reducible to a fixed per-layer number without deriving the position and
chunk branches — compressor pool/set-rows, fp8 quantization, the indexer
hadamard/scores/topk/masks, attention token-tile, and the generic
quantize/fill/sanitize/rms/cuBLAS/CUTLASS families. The captured multiples for
those (41, 42, 62, 84, 87, 129, 169, 172, 344) stay observations.

Both exported captures still carry Nsight's `Not all CUDA events might have
been collected` in `DIAGNOSTIC_EVENT` (cap1 13,149 produced / 12,492
collected; cap2 12,893 / 12,195 — different counter domains, not subtractable
as missing-kernel counts). Passing sentinels detects deficits in the named
families; it does not erase that diagnostic for the families nobody has
accounted for.

### One C sample was not enough, and the ratio is a range

The 45.87x above is one draw. Two further runs of the same workload on the
same host, each with the identical 3472 prefill launches, give summed kernel
durations of **2570.92 ms** and **4748.80 ms** against the published
3280.79 ms. Holding the single Go sample fixed, that is:

| C prefill sample | summed kernel duration | Go / C |
| --- | ---: | ---: |
| run 3 (pristine binary) | 2,570.92 ms | **58.53x** |
| run 1 (pristine binary, published above) | 3,280.79 ms | **45.87x** |
| run 2 (receipt-patched binary) | 4,748.80 ms | **31.69x** |

The binary is not the variable: the pristine binary produced both the lowest
and the middle figure, and the patched binary's added host work is an 8 KB
write outside the prefill capture range entirely. Nor is it clocks or
thermals, which would move everything together. It is concentrated:

| kernel | launches | run 3 | run 1 | run 2 | spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense_q8_d2r` | 172 | 300.3 ms | 658.4 ms | 1512.5 ms | **5.04x** |
| `gateup_iq2_swiglu_q8_d2r` | 43 | 665.5 ms | 846.1 ms | 1294.4 ms | 1.94x |
| `down_q2k_d2r` | 43 | 286.2 ms | 330.8 ms | 355.0 ms | 1.24x |
| `attention_tokentile_hmma` | 41 | 204.8 ms | 227.1 ms | 246.1 ms | 1.20x |

`dense_q8_d2r` spans a factor of five across 172 launches that are identical
in number and in the work they are asked to do. What drives it is not
established here; the ordering of the three runs does not follow the ordering
of anything we recorded about host state.

All three states pass the sentinel sweep identically — the same twelve kernel
symbols at the same counts as the published capture, in both the new prefill
states and the new decode state — so the spread is not a retention artefact in
any family anyone has expectations for. All three also still carry the
`Not all CUDA events might have been collected` diagnostic, exactly as the
published capture does. (Prefill's `hc_weighted_sum` is 1 in every state; no
source-derived expectation has been established for it, so that is an
observation.)

So: **treat 45.87x as one sample of a quantity whose observed range is
31.7x to 58.5x, and do not quote it as the ratio.** Three samples are
observed variability, not a bound and not a qualified ratio; the true spread
could be wider in either direction. The direction survives
this easily — every sample is a factor of thirty or more, and no amount of
C-side variance closes a gap that size. The figure does not survive it. The
Go side has exactly one sample and its own variance is unmeasured, so the
range above is a lower bound on the total spread, not a confidence interval.

Anything measuring a future optimization against 45.87x would be measuring
against noise. Re-measure both sides, repeatedly, before crediting a change.

### The retention risk, and two withdrawn bounds

Note which way that cuts, conditionally: *if* C duration is undercounted and
the Go retained duration is held fixed, the observed ratio is inflated. That
is not an unconditional upper bound, because Go completeness is itself only
tested by necessary sentinels.

An earlier draft offered two bounds against that risk. Neither is a bound, and
both are withdrawn:

- **Divisibility by 43 constrains nothing.** Losing an entire 43-launch family,
  or 43 launches out of 86, preserves divisibility exactly. Whole-buffer loss
  need not look ragged. The multiples are a pattern worth deriving expectations
  from, not a retention measurement.
- **The registered 661.18 t/s is not a replication.** It is wall prefill
  throughput; 45.87x is a ratio of summed instrumented kernel durations — a
  different metric. It also ran at context allocation 65665 against this run's
  4096 (`docs/oracle-workload-bench.md:48-49`), on a different binary, host
  load and profiling condition, and that document says in its own words that
  one repetition establishes "neither statistical significance nor a
  port/oracle ratio" (`:70`). Two numbers computed different ways from
  unmatched runs do not corroborate each other by landing nearby.

What the evidence supports is narrower than the earlier draft claimed: this
capture reports 45.87x on summed kernel duration, an unmatched historical
observation reports 51.0x on wall throughput, and both motivate work on the Go
prefill path. Neither establishes a qualified ratio: C retention is now
checked against source-derived expectations for twelve kernel symbols and
passes, but the remaining families are unaccounted and the run conditions are
still unmatched.

### The decode arm is close, and its gate is now closed

Decode is 1.60x on device time — a different regime from prefill entirely, and
the first measurement putting a number on that difference.

That comparison assumed both runtimes decoded the same two tokens, and for a
while nothing established it. ds4-min accepted 15342 then 201, recorded by the
binary's own check in its capture's `run.stderr`. The C side's IDs were not
retained and could not be recovered from that capture:

- `--show-output` converts `gen_token_buf` to text only (`ds4_bench.c:985-997`);
  no integer IDs are printed, and deriving IDs from text is not a receipt.
- `--dump-frontier-logits-dir` writes *before* decode (`:841`), and its
  `argmax_id` is `ds4_session_argmax`, which is EOS-**inclusive** (`:502`),
  while decode uses `argmax_excluding(eos)` (`:906`). It answers a different
  question.
- `--capture-range-end=repeat-shutdown:2` terminates the process when the
  second range closes, so that run printed no CSV row either.

"Emit after the loop" does not work under this recipe: the second
`cudaProfilerStop` at `:958-965` fires *before* the final token is accumulated
at `:968-976`, and terminates the process. So the receipt has to be on disk
before the stop, and the IDs accepted before the receipt. That is the patch
that was written, and it is in `DF6906F6-c-receipt/receipt.patch`:

- `bench_write_ids_le32` writes IDs as little-endian int32, active only when
  `DS4_BENCH_ID_RECEIPT_DIR` is set. A failed write fails the run.
- `token_t1` and the acceptance loop move above the `cudaProfilerStop` block —
  with no profiler range set, a pure reordering of two blocks that do not
  touch each other. The accepted-ID receipt is flushed immediately before the
  stop, once.
- `gen_token_buf` is allocated whenever `--gen-tokens` is positive rather than
  being gated on `--show-output`, because the receipt needs it either way.
- The stop condition becomes `gen_done >= start + tokens` after accumulation,
  plus `generation_stop`, so an EOS-truncated run still stops.

Only accepted IDs are ever written. Run it by setting the env var alongside
the capture recipe:

```
DS4_BENCH_ID_RECEIPT_DIR=$OUT/ids \
DS4_BENCH_CUDA_PROFILE_PREFILL=1 DS4_BENCH_CUDA_PROFILE_RANGE=0:2 \
$NSYS profile ... ./ds4-bench --cuda -m "$M" ...
```

Under the full recipe it produces:

| receipt | bytes | contents |
| --- | ---: | --- |
| `prefix-2048.le32.bin` | 8192 | sha256 `5fdf3b77…c80380` |
| `accepted-2048.le32.bin` | 8 | **15342, 201** |

**Both gates close at once.** The accepted IDs are the same two ds4-min
accepted, each side's recorded by its own profiled process — so the decode row
compares two runs demonstrably doing the same thing. And the prefix hash is
the registered fixture hash exactly, emitted in-process by the binary under
capture from the prefix it actually handed `ds4_session_sync`. That is
stronger than the same-function argument the prefill row rested on, which
spanned two binaries over one `ds4.o`.

What the receipt does not do: the env guard preserves the disabled-path
behaviour, not binary byte identity — the validated binary is a relink, and
its hash is in the bundle's `binary-identity.sha256`, distinct from the one
that produced the published capture. And `RUN_RC=0` remains Nsight's exit, not
the application's: `repeat-shutdown` terminated `ds4-bench` deliberately, so
the run still demonstrates no clean application completion or cleanup.

## Fine phase names, and what they do not isolate

`-span-level op` with `-spans-stream` carries the `markCUDAPhase` names into
the host sidecar: `dense-front`, `qkv-normalization`, `attention`,
`attention-output`, `compression` (or `compression-raw` on the first two
layers), `ffn-pre`, `shared-ffn`, `expert-ffn`, `ffn-post`, `group-tail`.

The counts are fixed by the source. `validateCUDAPhaseCounts`
(`cuda_phase_profile_linux_test.go:91`) mandates them, and at 2048 tokens,
43 layers and `cudaPrefillBatchWidth` 8 they sum to **715,776** prefill
intervals. The two decode tokens add **602** more: the shared path is
`evalLayerProjected` (`cuda_session_linux.go:645`), which marks seven phases
per layer per token, so 2 x 43 x 7. With the coarse levels `op` subsumes
(1 prefill, 1 generation, 2 token, 256 batch, 11,094 layer) a complete
sidecar holds **727,732** records. Anything else means the workload or the
instrumentation changed, and the accounting receipt must independently show
`spans_begun == spans_ended == spans_encoded` with `complete` true.

**A phase interval is the gap between two marks, which on the decode path is
not the same as an operator's cost.** `dense-front`, `qkv-normalization` and
`group-tail` are marked in `evalLayerBatch` and `evalBatch` only. The
single-token path has no front mark, so a decode `ffn-post` interval runs
until the next layer's first mark and absorbs that layer's front work — and
the last layer's runs into the final output and stream sync. The counts can
be exactly right while the decode durations attribute work to the wrong
name. Use the prefill intervals for per-operator cost; treat the decode ones
as a timeline, not an attribution, until the single-token path carries its
own front mark.

The spans are host-only by design and open no NVTX range: at this
granularity that would add about 715,000 ranges to a capture that carries
11,356, which changes the thing being measured.
