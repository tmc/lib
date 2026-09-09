# Go → CUDA → PTX → SASS

This example runs vector addition followed by 512 dependent fused multiply-add
operations per element. It checks every output against a CPU reference after
each stage. The defaults use 1,048,576 elements, one warmup, and 12 measured
passes through both kernels.

NVRTC produces PTX and CUBIN; `nvdisasm` decodes that CUBIN, and the program
executes the same binary. The capture includes a Go execution trace, a pprof
profile, source, both assembly forms, the CUBIN, and a manifest with checksums.
The program also writes `capture.pprof`, a single file containing the profile
and all recorded files in pprof metadata. That is the file to share.

From the NVIDIA module directory on a CUDA host with NVRTC:

```sh
GOWORK=off go run ./examples/ptx-sass-trace \
  -nvdisasm /usr/local/cuda-13/bin/nvdisasm
```

The output is a new directory under `~/tmp`. Set `-out` to choose a new directory,
or use `-elements`, `-rounds`, and `-passes` to change the workload.

Copy just `capture.pprof` to your viewing machine, then run:

```sh
GOWORK=off go run ./cmd/cudaprof -http=127.0.0.1:8087 ~/tmp/capture.pprof
GOWORK=off go run ./cmd/cudaprof -disasm=compute ~/tmp/capture.pprof
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
GOWORK=off go run ./cmd/cudaprof -pack ~/tmp/capture.pprof ~/tmp/capture/gpu.pprof
GOWORK=off go run ./cmd/cudaprof -extract ~/tmp/extracted ~/tmp/capture.pprof
```

For Go trace, use the producer's Go version recorded in `manifest.json`:

```sh
GOTOOLCHAIN=go1.26.7 go tool trace ~/tmp/extracted/go.trace
```

Open **User-defined tasks → CUDA pipeline** for numbered CUDA/PTX/SASS events.
Timeline regions measure host launch-and-wait time. `cuda.event` logs record
separate CUDA event intervals.

To make a shareable directory and `.tar.gz` with macOS/Linux viewers for arm64
and amd64, plus buildable source and vendored dependencies:

```sh
./examples/ptx-sass-trace/package.sh ~/tmp/capture ~/tmp/go-cuda-profile
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
