# profile2trace

Convert an Nsight Systems SQLite export, ds4-min host sidecar, or Metal region
receipt to native Perfetto protobuf using `nvidia/x/perfetto`. The command runs
without a GPU. Nsight input requires `sqlite3` with `-json` support on PATH;
no Go SQLite driver or module changes are needed.

From `examples/`:

```sh
GOWORK=off go run ./cmd/profile2trace -nsys capture.sqlite > capture.pftrace
GOWORK=off go run ./cmd/profile2trace -spans go.spans.jsonl > host.pftrace
GOWORK=off go run ./cmd/profile2trace -metal regions.json > metal.pftrace
```

Open the output with **Open trace file** at <https://ui.perfetto.dev>.
The command writes counts and diagnostics to stderr. Check its exit status
before using the output; a write failure can leave a partial trace.

## Nsight input

Export an existing capture with `nsys export --type=sqlite`. The reader expects
`CUPTI_ACTIVITY_KIND_KERNEL` with `start`, `end`, `deviceId`, `contextId`,
`streamId`, `globalPid`, `correlationId`, and `demangledName`, plus `StringIds`
with `id` and `value`. These are nanosecond intervals. Every kernel row becomes
one slice, including rows whose name is missing from `StringIds`. Tracks
separate device, process, context, and stream identities.

When `NVTX_EVENTS` exists, its `start`, `end`, `globalTid`, `endGlobalTid`,
`text`, and `textId` fields supply host ranges. Direct and registered strings
are supported. NULL-end records (marks or unfinished ranges) are omitted and
counted explicitly. Missing kernel tables, empty kernel tables, unsupported
schemas, NULL required fields, and reversed intervals fail conversion.
The reader does not include memcpy, runtime API, or CPU sample tables.

The profiling coordinator's latest owner-run probe on this DGX found that
only `/opt/nvidia/nsight-systems/2026.1.1/bin/nsys` accepts `-t cuda-sw`.
Bare `nsys` (`/usr/local/bin/nsys`, 2024.5.1), `/usr/local/cuda/bin/nsys`
(2025.3.2), and `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys` all reject it.
Earlier advice naming `/usr/local/cuda/bin/nsys` was withdrawn: that path is
alternatives-managed and its target has drifted since the retained experiments.

For this host, pin the absolute versioned path. Retain the executable hash,
version, full arguments, exit status and exported table inventory with every
capture. A previous capture with plain `-t cuda` returned success without a
kernel table; the converter checks the table regardless of exit status. These
are attributed host-specific findings, not general Nsight requirements.

## Bounding a large capture

A whole-run Nsight export of a long workload holds far more kernel rows than
Perfetto can usefully display: a 159-second ds4-min CUDA run exported 4,069,537.
`-nsys-window START:END` keeps only the Nsight intervals whose **start** falls in
the half-open range, in source-clock nanoseconds, so adjoining windows partition
a capture without duplicating or dropping a row. Either side may be empty.
A windowed run reports on stderr how many rows it kept and how many it omitted.

```sh
GOWORK=off go run ./cmd/profile2trace \
  -nsys capture.sqlite -nsys-window 160300000000: > tail.pftrace
```

`-max-slices` (default 500000) refuses a window selecting more kernel rows than
the limit, naming the selected and total counts, rather than emitting a trace no
viewer will open. Raise it deliberately.

Find window bounds from the source itself, not by guessing:

```sh
sqlite3 capture.sqlite \
  'select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL'
```

Schema reference: [NVIDIA Nsight Systems User Guide](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html).

## Combining clocks

An offset is **added** to the corresponding source timestamp. For a measured
pair `(source_ns, reference_ns)`, use `reference_ns - source_ns`. Nsight is the
reference when present; otherwise host spans are the reference. A single
source needs no offset. Multiple sources require explicit offsets:

```sh
GOWORK=off go run ./cmd/profile2trace \
  -nsys capture.sqlite -spans go.spans.jsonl \
  -spans-offset-ns "$HOST_TO_NSYS_NS" > combined.pftrace

GOWORK=off go run ./cmd/profile2trace \
  -spans go.spans.jsonl -metal regions.json \
  -metal-offset-ns "$METAL_TO_HOST_NS" > combined.pftrace
```

Use zero only when the capture evidence establishes a shared clock. The
converter cannot establish calibration from unrelated start times, match runs
across files, or correct clock drift. In particular, a Metal test receipt and
a separate binary's host sidecar are separate executions; aligning their first
intervals would not make them a simultaneous capture.

After applying offsets, the earliest timestamp becomes zero. Each slice retains
its aligned timestamp, trace origin, and source clock domain as annotations.
The existing writer's slice API does not select a per-event clock domain, so
conversion applies the offsets before writing, rather than emitting an
uncorrelated clock snapshot. Nanoseconds remain integer timestamps; Metal's
seconds are rounded to nanoseconds.

Host sidecars must use `gputrace.host-event/v1`, `interval`, and
`cpu_uptime_ns`, with one nonempty run ID. They have no recorded thread IDs.
Metal `regions` contain command-buffer GPU intervals, not individual kernels.
Overlapping ranges receive additional tracks, preserving intervals without
inventing nesting or host thread identities. Input arrays and output event
ordering are held in memory; this is intended for bounded profiling captures.

## CPU checks

```sh
GOWORK=off TMPDIR="$HOME/tmp" go test ./cmd/profile2trace -count=1
```

Tests build synthetic SQLite databases, check both NVTX name forms and missing
kernel names, reject missing/empty/invalid captures, validate sidecar schemas,
and independently decode the emitted protobuf to check durations and balanced
slices. Synthetic fixtures do not establish GPU parity or qualification of a
particular live Nsight export.

## Offline Perfetto queries

With the official [trace processor](https://perfetto.dev/docs/reference/trace-processor-cli)
installed, run the retained SQL without a browser:

```sh
trace_processor query -f cmd/profile2trace/check.sql combined.pftrace
sqlite3 capture.sqlite 'SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;'
```

The `cuda_kernel` count must equal the SQLite count. The query also reports
unfinished slices and nonzero parser error statistics. On the synthetic mixed
fixture, Perfetto UI v58.3 and offline trace processor v58.2 both returned two
CUDA slices, two NVTX slices, one host slice, and one Metal slice. Offline
parsing reported zero unfinished slices and no nonzero parser errors.
