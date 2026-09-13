# Go → CUDA → PTX → SASS

This is a real GPU capture of a DeepSeek-V4 (ds4) inference pipeline:
RMSNorm, Q8_0 quantized matrix-vector multiplication with warp reductions,
SwiGLU activation, and sparse MoE expert dispatch with indirect indexing,
along with sub-byte IQ2 SIMD vs scalar dot products and shared-memory staging.
NVRTC compiled the code, nvdisasm decoded the CUBIN, and the program executed
that same CUBIN on NVIDIA hardware. Every output was checked against a CPU
reference after every stage.

## Open the demo

Share just `capture.pprof`. It embeds the source, PTX, SASS, CUBIN, Go trace,
and capture manifest in standard pprof metadata. With the NVIDIA-aware viewer:

```sh
cudaprof -http=127.0.0.1:8087 capture.pprof
```

No CUDA installation or GPU is needed to view this recording. In the optional
share bundle with prebuilt viewers, `./view` opens the same pprof interface.

- **Flame Graph:** compare `vector_add` with `compute`.
- **Source:** search for `compute` to see CUDA, PTX, and SASS. Click a source
  row to expand its assembly instructions.
- **Disassemble:** search for `compute.PTX` or `compute.SASS` to choose a
  compiler stage. Search for `compute` to see both.
- **Details:** capture parameters and timing interpretation.

Command-line views: `cudaprof -top capture.pprof`,
`cudaprof -list=compute capture.pprof`, and `cudaprof -disasm=compute capture.pprof`.
Stock `go tool pprof` reads the samples and preserves the embedded files, but
needs the NVIDIA-aware viewer to interpret them as source and disassembly.

The recording includes a Go execution trace. Run
`cudaprof -extract recording capture.pprof` to extract it, then use the Go
version in `recording/manifest.json` to open `recording/go.trace` with
`go tool trace`. In the optional share bundle, `./trace` does this directly.
Open **User-defined tasks → CUDA pipeline**. PTX and SASS appear as
numbered task events; timeline regions show host launch-and-wait intervals.

## What the numbers mean

`gpu_time` is the CUDA event interval around an entire kernel launch. Profile
cost is attached to each kernel's SASS entry; PTX arithmetic rows provide source
navigation. It is not a cost
for that individual instruction. The Go/CUDA/PTX/SASS stack expresses the
launch and compilation relationship, rather than a sampled hardware stack.
There are no sampled PCs or hardware memory/stall counters in this capture.

SASS addresses in Disassemble are CUBIN file offsets: each function's .text
section offset plus its instruction offset. PTX addresses are byte offsets in
the recorded text. Profile addresses add separate logical image bases.
These are reproducible file coordinates, not live device addresses.
The viewer reads saved NVIDIA assembly; it does not substitute CPU opcodes.

## Reproduce and inspect

`manifest.json` records the device, compiler, Go version, parameters, per-launch
times, validation result, and SHA-256 checksums. The viewer checks the bundled
files before displaying them. Original CUDA source, PTX, CUBIN, complete SASS,
Go source, and compiler/disassembler identity are included.

The `src` directory in the share bundle contains the buildable source and
vendored dependencies. See `BUILD.md` there for capture and rebuild commands.
View code and recordings locally; nothing is uploaded by the demo.
