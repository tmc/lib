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

`gpu_time` is the CUDA event interval around an entire kernel launch, and it is
the default view. `samples` counts launches, one per recorded launch. Profile
cost is attached to each kernel's SASS entry; PTX arithmetic rows provide source
navigation. It is not a cost for that individual instruction.

Whether this capture holds sampled hardware PCs depends on how it was recorded.
Without `-pcsampling`, the Go/CUDA/PTX/SASS stack expresses the launch and
compilation relationship rather than a sampled hardware stack, and the capture
carries no sampled PCs or stall counters; the sample types it advertises are
only the two it can fill. With `-pcsampling`, the manifest's `pc_sampling`
object records the period, how many samples were collected, how many resolved
to a PTX statement, and what was dropped -- read those counters before trusting
the attribution, because a high resolved fraction over a batch that dropped
most of its samples is not a working measurement.

`pass`, `layer`, `moe.expert` and `moe.slot` are numeric labels, so pprof's
range filters apply: `-tagfocus='layer=0:3'` selects the first four layers.

SASS addresses in Disassemble are CUBIN file offsets: each function's .text
section offset plus its instruction offset. PTX addresses are byte offsets in
the recorded text, assigned at capture time against a second logical image, so
any pprof consumer sees them and not only `cudaprof`. Profile addresses add
separate logical image bases.
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
