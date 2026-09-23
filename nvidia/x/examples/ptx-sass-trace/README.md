# Go → CUDA → PTX → SASS

This example runs a DeepSeek-V4 (ds4) inference micro-workload:
RMSNorm, Q8_0 quantized matrix-vector multiplication with warp reductions,
SwiGLU activation, and sparse MoE expert dispatch with indirect indexing,
along with sub-byte IQ2 SIMD vs scalar dot products and shared-memory staging.
It checks every output against a CPU reference after each stage. The defaults
use 1,048,576 elements, one warmup, and 12 measured passes through the pipeline.

NVRTC produces PTX and CUBIN; `nvdisasm` decodes that CUBIN, and the program
executes the same binary. The capture includes a Go execution trace, a pprof
profile, source, both assembly forms, the CUBIN, and a manifest with checksums.
The program also writes `capture.pprof`, a single file containing the profile
and all recorded files in pprof metadata. That is the file to share.

From the NVIDIA module directory on a CUDA host with NVRTC:

```sh
GOWORK=off go run ./x/examples/ptx-sass-trace
```

`nvdisasm` is found on `PATH`; `-nvdisasm` names another one. It must
understand the GPU's architecture.

The output is a new directory under `~/tmp`. Set `-out` to choose a new directory,
or use `-elements`, `-rounds`, and `-passes` to change the workload.

Copy just `capture.pprof` to your viewing machine, then run:

```sh
GOWORK=off go run ./x/cmd/cudaprof -http=127.0.0.1:8087 ~/tmp/capture.pprof
GOWORK=off go run ./x/cmd/cudaprof -disasm=compute ~/tmp/capture.pprof
```

The viewer uses upstream pprof with an object reader for saved NVIDIA SASS.
**Source** shows CUDA, PTX, SASS, and the Go launch site. **Disassemble** supports
both compiler stages: filter `compute.PTX` for virtual instructions or
`compute.SASS` for native instructions from the captured binary. Viewing needs no GPU or CUDA
installation. Stock `go tool pprof` can read the profile; NVIDIA disassembly
requires this viewer. It unpacks embedded files into a private directory under
`~/tmp` while reading them. Ordinary pprof tools preserve the embedded metadata
when rewriting the profile, but do not interpret it as source or disassembly.

To embed an older capture, or extract recorded files for inspection:

```sh
GOWORK=off go run ./x/cmd/cudaprof -pack ~/tmp/capture.pprof ~/tmp/capture/gpu.pprof
GOWORK=off go run ./x/cmd/cudaprof -extract ~/tmp/extracted ~/tmp/capture.pprof
```

For Go trace, use the producer's Go version, recorded as `go_version` in
`manifest.json`:

```sh
GOTOOLCHAIN=$(sed -n 's/.*"go_version": "\([^"]*\)".*/\1/p' ~/tmp/extracted/manifest.json) \
  go tool trace ~/tmp/extracted/go.trace
```

Open **User-defined tasks → CUDA pipeline** for numbered CUDA/PTX/SASS events.
Timeline regions measure host launch-and-wait time. `cuda.event` logs record
separate CUDA event intervals.

To make a shareable directory and `.tar.gz` with macOS/Linux viewers for arm64
and amd64, plus buildable source and vendored dependencies:

```sh
./x/examples/ptx-sass-trace/package.sh ~/tmp/capture ~/tmp/go-cuda-profile
```

The recipient extracts the archive and runs `./view`. `./trace` uses Go to open
the execution trace. The included `src/BUILD.md` explains rebuilding offline.

`gpu_time` measures whole kernel launches with CUDA events. The SASS entry and
PTX arithmetic rows are navigation anchors, not instruction-level costs or
sampled PCs. The displayed Go/CUDA/PTX/SASS stack expresses the launch and
compilation relationship. Unmeasured hardware counters are omitted. Addresses
are CUBIN file offsets for SASS and text byte offsets for PTX, with separate
logical image bases. Neither is a sampled GPU address. PC sampling would
record instruction addresses during execution to identify instruction hot
spots and stalls; this capture records whole-launch CUDA event intervals.

## `-activity`: CUPTI kernel activity records

`-activity` collects one CUPTI activity record per kernel the device ran, which
is the only tier here that establishes a launch count and per-launch device
intervals. A CUDA event interval covers only the launches this example chose to
bracket, and a PC sample says where execution was but never for how long, so a
share of samples is a share of samples and never a share of elapsed time.

It is off by default. `cuptiactivity.Start` validates the native record size of
**every** activity kind the package can decode against the Go layouts this
build was generated for -- not only the kinds the session asks to collect --
and refuses to start on a CUPTI whose layouts it does not know. The error names
the kind and says every decodable kind is checked; it is a toolkit mismatch,
not a fault in the tier.

The tier fails closed: any status other than `collected` refuses the capture
rather than writing a bundle with a plausible zero in it, because a zero record
count and a successful empty collection are otherwise the same output. Derived
quantities are labelled: the union of kernel intervals is what the span is
comparable against, the sum is reported beside it with the overlap, and SM
occupancy is the literal string `UNMEASURED` rather than a number this tier
cannot produce.

## Go benchmark metrics

For `go test -bench`, full and isolated subbenchmarks, and `-cuptiprofile`, see
[the cuptibench example](../cuptibench/README.md). It demonstrates the benchmark
and replay lifecycle with normalization, dense projection, and activation.
