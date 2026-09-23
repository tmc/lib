# Metal profiling with Go tools

Run and validate a Metal workload, recording hardware stage timestamps and a
portable pprof capture:

```sh
MTL_CAPTURE_ENABLED=1 go run ./examples/metal-trace -capture -enrich \
  -out="$HOME/tmp/metal-example"
```

Run this from the apple module directory. Enrichment requires an installed
`gputrace` and the macOS Metal replay tools. The example prints any enrichment
failure; a saved live capture alone does not establish successful replay.

Extract the capture, then use the standard Go tools:

```sh
go run ./cmd/metalprof -extract "$HOME/tmp/metal-example/files" \
  "$HOME/tmp/metal-example/capture.pprof"
cd "$HOME/tmp/metal-example/files"
go tool pprof -http=127.0.0.1:8099 -symbolize=none gpu.pprof
# Run separately:
go tool trace -http=127.0.0.1:8098 go.trace
```

The extracted directory contains MSL source for pprof's source view. Source lines
are navigation anchors for whole-stage measurements, not measured instruction
hotness. `-symbolize=none` avoids looking for a native executable for these
source-only samples. AIR is an offline compilation artifact, not proof of the
native instructions used by the runtime pipeline.

`manifest.json` contains typed observations and workload validation results.
`replay_export.json` retains the full gputrace export. Compiler registers,
instruction mixes, spill sizes, timing spans and archive coverage remain JSON
observations; they are not additive pprof sample columns.

`gpu.pprof` contains live `gpu_time`. `replay.pprof` contains
`inferred_dispatch_time`, derived from replay cumulative offsets and including
captured warmup. Do not compare those values as equivalent measurements or merge
them. The replay profile does not inherit the live CPU-reference validation
claim. Xcode-processed pipeline compute timing, when separately collected, also
belongs in a separate replay profile.

Compiler statistics require the corresponding `recorded_statistics` entry.
An absent value is not a measured zero. Counter archive coverage distinguishes
records tied to capture encoders from machine-wide records. Raw counter records
retain their native owners; they are not attributed to kernels by array order.

On the validated Apple M4 Max, stage timestamps are available; dispatch-boundary
counter sampling is not. Native ISA/PC attribution and some derived counters
remain unresolved or unavailable in the installed toolchain. Those gaps must not
be filled with estimated instruction costs or zero-valued measurements.

The example also records host command-buffer events for measured passes (warmup
excluded). Select them with standard pprof sample indexes:

```sh
go tool pprof -sample_index=command_buffer_count gpu.pprof
go tool pprof -sample_index=command_buffer_commit_count gpu.pprof
go tool pprof -sample_index=command_buffer_create_time gpu.pprof
```

Creation count records successful creations; commit count records submissions,
not successful GPU completion. Creation time measures the host CommandBuffer API
call using Go's monotonic clock, excluding stack collection and queue lookup.
Each event has its Go call stack and kernel/pass/operation labels. Counts are per
command buffer, not per dispatch. GPU time remains the default sample type.
Go stack names are included; Go source files are not bundled by this example.
