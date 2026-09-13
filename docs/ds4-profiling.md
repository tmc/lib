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
      -f true -o go-prefill2048-decode2 \
      ds4-min -model "$MODEL" -backend cuda \
        -prompt-tokens scripts/testdata/oracle-workload-tokens2048.json \
        -tokens 2 -ctx 4096 \
        -cpuprofile go.cpu.pprof -trace go.trace \
        -spans go.spans.jsonl -span-level layer

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
counted 160.105706240 s: the two clocks differ in *rate* by 11.11 ppm, which is
what an NTP-slewed `CLOCK_MONOTONIC` against an undisciplined counter looks
like. One constant offset cannot relate them, so **no value is passed to
`-spans-offset-ns` and merged-clock qualification stays open.** Do not split the
difference; an averaged offset would be wrong at both ends and defensible at
neither.

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

A concrete lead, not yet a mapping: the export's `TARGET_INFO_SESSION_START_TIME`
carries `utcEpochNs=1789252196696595551` and `systemClockNs=491666116248656`.
The clock backing `systemClockNs` has to be established before either is
subtracted from anything — done blindly it drives the first host timestamp
negative, which is the tell that the domains were mismatched.

### Batching prefill buys almost nothing here

Activity is uniform across the entire 158.78 s device span: every 5-second
bucket holds 127k-131k launches over the same 32 kernel kinds, and there is no
gap longer than 5 ms anywhere in the final two seconds. The decode window uses
29 distinct kernel kinds against prefill's 32.

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
| rows not of `eventType` 60 | 0 |
| rows with an absolute-magnitude start | 0 |
| host sidecar spans | 11,354 |
| CUDA kernel rows | 3,973,441 |

The name histogram matches the sidecar inventory exactly — 11,094 `layer`,
256 `batch`, 2 `token`, 1 `prefill`, 1 `generation`, summing to the 11,354
spans, plus the 2 calibration ranges.

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
