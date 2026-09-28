# Isolated launch reproducers

Generate optional capture methods alongside ordinary typed launches:

```
go run github.com/tmc/lib/nvidia/x/cmd/cudabind -type Module -repro kernel.cu
```

`CaptureWriteAt(stream, grid, block, limit, out, alias, index, value)` has the
same kernel arguments as `WriteAt`, plus a maximum number of allocation bytes
to retain. It synchronizes the stream and returns a `*cudarepro.Launch` without
launching the kernel. Explicitly save that record with `record.Write(file)`
before submitting suspect work. Capture is intended for a quiescent diagnostic
run: callers keep allocations alive and exclude writes from other streams.

The record contains the native CUBIN and digest, argument names and C types,
captured ABI widths, original addresses, full backing allocation contents,
launch geometry, stream handle and Go call site. Generated captures also retain
the embedded source files. Their target names the capture device; source files
are the generator inputs, excluding its appended ABI assertions. Compiler
options are optional provenance, not reconstructed from the native image.
Replay loads the retained CUBIN; it does not recompile those source files.

Full allocations are captured once even when several arguments alias them.
Replay allocates them again and relocates each argument by its original offset.
The allocation-byte limit counts complete backing allocations, not only slice
lengths. A capture includes application data; save it only where that data may
be stored. Use records from trusted producers: replay executes their GPU code,
and the schema is producer metadata rather than an authenticated ABI proof.

Build the runner, then execute a saved capture in a fresh process:

```
go build -o "$HOME/tmp/cudarepro" github.com/tmc/lib/nvidia/x/cmd/cudarepro
compute-sanitizer --tool memcheck --error-exitcode 66 \
  "$HOME/tmp/cudarepro" -record "$HOME/tmp/replay.json" capture.json
```

The required `-record` writes relocated addresses **before** submission, so a
fault cannot prevent attribution metadata from being saved. Files must not
already exist. `-output` optionally writes output buffer bytes as JSON. Replay
checks argument count against the captured ABI and each argument width against
the loaded function before driver submission.

For a saved sanitizer log, print findings and candidate arguments without a GPU:

```
"$HOME/tmp/cudarepro" -attribute sanitizer.log "$HOME/tmp/replay.json"
```

Or parse the sanitizer output with `cudasanitizer.ParseLog`, then pass findings and
the relocated record to `cudarepro.Attribute`. Results are **candidates**: aliased
arguments, address reuse, or several possible launches remain ambiguous. The
join uses contained addresses or the sanitizer's explicitly identified nearest
allocation, never symbol-name similarity. Unsupported report formats retain
text and produce no attribution. Restrict records to the failing process and
execution interval using independent evidence.

This first implementation supports ordinary native JIT-loaded cudabind modules
with scalar values and direct allocation pointers. Capture from precompiled
loaders, graph capture, mutable module globals, pointers inside records or
allocation contents, texture/surface objects and concurrent execution are
unsupported. It does not reproduce races or a whole stream dependency graph.
Capture synchronizes its stream; replay runs that isolated saved state in a new
stream. The CUBIN requires compatible hardware, and scalar bytes require the
same host ABI.

## Qualification

The maintained opt-in hardware test builds a generated binding fixture, captures
an out-of-bounds write, and replays it under real memcheck in a fresh process.
It attributes the fault to the named `out` argument, checks alias relocation,
then captures a corrected index and verifies both clean memcheck and output
bytes. CPU tests cover unknown report formats, aliases, reused addresses,
invalid capture metadata, byte limits and generated-name collisions.

Run from the NVIDIA module under the shared GPU serializer:

```
GPU_LANE=cudarepro "$HOME/ml-explore/bin/gpu-slot" -- \
  env GOWORK=off CUDABIND_REPRO_TEST=1 \
  go test -count=1 -v -run '^TestReproducer$' ./x/cmd/cudabind
```

The fixture is `x/cmd/cudabind/testdata/repro`; regeneration requires compatible
libclang CUDA headers. Its normal test path explicitly skips hardware work
unless requested. This qualification establishes the isolated native path,
not full sanitizer coverage of an application's asynchronous execution.
