# Benchmark counters with the code that produced them

A `cuptibench` run exports a pprof profile of hardware counters and a JSON file
of results beside it. Neither carries the compiled code, so a reader can see
that a range executed 258 thread blocks but cannot get from that number back to
a source line. This example packages the two together with the exact source,
PTX, CUBIN and SASS of the module the benchmark loaded, as a portable capture
`x/cmd/cudaprof` opens without a GPU.

It composes existing packages and adds nothing to them:

| Service | Package |
|---|---|
| Counter collection, benchmark timing, export | `x/cuptibench` |
| Compilation and retained PTX/CUBIN | `x/nvrtcx` |
| Disassembly | `x/cudasass` |
| Capture container, code parsing, module identity | `x/cudacapture` |

Compilation, disassembly and viewing stay in the consumer. `cuptibench.Plan`
does none of them.

## What the capture asserts, and what it does not

- **Module identity is a digest, not a name.** The benchmark records the sha256
  of the image it handed the driver. `cudacapture.VerifyModule` recomputes it
  over the retained CUBIN, so an offline reader learns the artifacts are the
  image the producer loaded. It is a producer record checked against retained
  bytes, not a driver attestation. A kernel name that resolves proves nothing
  about which generation of the module ran.
- **Timing and counters are separate executions.** `cuptibench` times an
  uninstrumented run and collects counters by replay. The capture keeps
  cuptibench's own `timing_capture_id` for the timed execution and gives it
  descriptors of its own, so nothing invites adding a replay counter to a
  measured duration.
- **A region counter is not an instruction counter.** Counters are scoped to
  the ranges cuptibench selected for that capture's mode. The descriptors say
  so, and metrics whose additivity or unit was never verified are recorded as
  observations rather than pushed into pprof sample values.
- **Nothing partial is published.** A collection with dropped ranges, passes
  that did not all stop, or undecoded counters is refused, as are artifacts
  whose digest is not the loaded image and kernels the retained code does not
  define.

## Run

`Compose` needs no device. Producing its inputs does. From the `nvidia` module
directory, on an exclusively owned CUDA device with NVRTC, nvdisasm and CUPTI
counter permission:

```sh
capture_dir=$(mktemp -d "$HOME/tmp/cuptibench-capture.XXXXXX")
GOWORK=off go test ./x/examples/cuptibench-capture \
  -run '^$' -bench '^BenchmarkComposeFromDevice$' -benchtime=1x -count=1 -v \
  -cuptiprofile="$capture_dir/device.pprof"
GOWORK=off go run ./x/cmd/cudaprof "$capture_dir/device.pprof.capture"
```

Inspect the retained capture without a GPU:

```sh
GOWORK=off go run ./x/cmd/cudaprof -top "$capture_dir/device.pprof.capture"
GOWORK=off go run ./x/cmd/cudaprof -list=compute "$capture_dir/device.pprof.capture"
GOWORK=off go run ./x/cmd/cudaprof -disasm=compute "$capture_dir/device.pprof.capture"
```

The disassembly view shows retained PTX and SASS without instruction costs.
These counters measure a region; they contain no sampled instruction addresses.
For this unannotated view, use `-disasm` and the capture path alone; measurement
filters do not apply to the retained code listing.

The benchmark skips without a device, NVRTC or nvdisasm. Check for a
`BenchmarkComposeFromDevice` result line; a zero exit status alone does not
prove it ran. The single iteration checks composition, not performance.

An explicit `-cuptiprofile` retains the profile, its `.json` results, and a
`.capture` directory containing the portable capture. Without it, the outputs
are temporary and are removed when the benchmark exits. Use a literal path;
this example does not expand `{package}`. Run it once per process (`-count=1`
and one `-cpu` value), because it calls the process-wide `cuptibench.Finish`
before reading the exported files. It checks all 1,024 output values before
publishing.

To compose from a run of another benchmark, export it first and pass the two
files it wrote together with the artifacts that benchmark compiled:

```sh
profile_dir=$(mktemp -d "$HOME/tmp/cuptibench-example.XXXXXX")
CUPTIBENCH_EXAMPLE=1 GOWORK=off go test ./x/examples/cuptibench \
  -run '^$' -bench '^BenchmarkPipeline$' -benchtime=1x -count=1 \
  -cuptiprofile="$profile_dir/pipeline.pprof"
```

```go
c, err := cuptibenchcapture.Compose(
	cuptibenchcapture.Export{Name: path, Profile: profileBytes, Results: resultBytes},
	cuptibenchcapture.Artifacts{
		Prefix: "kernels", Source: src, PTX: ptx, CUBIN: cubin, SASS: sass,
		ImageSHA256: imageDigest, Kernels: []string{"rmsnorm", "project", "silu"},
	})
```

Then open it:

```sh
GOWORK=off go run ./x/cmd/cudaprof capture.pprof
```

## Offline tests

The tests that do not need a device use a fixture module, with its CUBIN stored
hex encoded so the only ELF input in git stays reviewable text. They cover the
four things the capture claims: the raw counter profile and the packaged
capture both parse, the artifact digest matches the recorded image, source and
disassembly resolve from the capture alone, and a failed or partial capture is
refused.
