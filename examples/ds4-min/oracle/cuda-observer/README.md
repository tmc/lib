# Pinned CUDA oracle observers

These patches add optional scalar and windowed raw-prefill observers to
`antirez/ds4` at `b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd`. They belong to the
C reference oracle, not the Go implementation. They preserve the original
entry points and kernels. There is no cuBLAS observer: its operand loads occur
inside library kernels and have no source-owned observation boundary here.

Apply in order to a fresh checkout of the pinned commit:

1. `compatibility.patch`: the existing Linux compatibility fixes in `ds4.c`.
2. `scalar.patch`: optional scalar raw-prefill observer, site 1.
3. `window.patch`: optional windowed raw-prefill observer, site 2.

`source-hashes.json` identifies the exact resulting files. The header patch
documents the 56-byte descriptor, 8-byte state and 32-byte record ABI, bounds,
sticky overflow, allocation ownership and graph lifetime requirements.

The suffixed APIs delegate null/disabled observers to the original entry point.
Each preserves the original route predicate and delegates unsupported routes
without observing them. Site 1 records a selected scalar KV score load. Site 2
records a selected component of a global `float4` load before its shared-memory
store. That producer serves eight heads; the requested head selects a group,
not an independent per-head load. Neither observer captures a whole tensor.

## CPU validation

From this directory, with Python 3, Git, and C/C++ compilers available:

```sh
python3 validate.py /path/to/local/antirez/ds4
```

This creates an isolated checkout and artifacts under `~/tmp`, applies the
patches, verifies the resulting hashes, compiles the public ABI as C11/C++11,
and runs source and synthetic append/selection controls. It does not use the
GPU or change the supplied checkout. Controls include bounds, overflow,
concurrent reservations, guard bytes, adjacent-coordinate negatives and a
capacity-validation mutant that must fail.

## CUDA build and hardware gate

`window-harness.cu` includes the patched CUDA source by include path. It calls
only the original and observed window kernels with device-allocated Q, KV,
sinks and output. It does not exercise a public dispatcher or model resolver.
Build in a separate directory, substituting the patched oracle include path:

```sh
nvcc -O2 -g -lineinfo --use_fast_math \
  -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1 \
  -I/path/to/patched/oracle -Xcompiler -ffunction-sections \
  -Xcompiler -fdata-sections -Xlinker --gc-sections \
  -o window-harness /path/to/window-harness.cu -lcublas
```

GPU execution requires separate authorization and the host's `gpu-lock`.
The harness requires `GPU_LOCK_HELD`, one GB10 device and CUDA runtime 13.4.
It writes fixture/output/record files in its working directory and stops on
any error or mismatch. Do not run it in this source directory. The lock wrapper
executes its arguments; `gpu-lock --help` is not a help command.

On 2026-09-20, one locked run on `spark-31db` (GB10, driver 580.173.02,
CUDA 13.4.59, sm_121a) passed six kernel launches: original baseline, two
identical site-2 positives, adjacent out-of-window negative, zero-capacity
overflow and sticky-overflow preservation. All 665600 output floats matched
byte-for-byte across baseline and both positive runs; record hashes matched.
The retained harness used a relative include of the same CUDA source; this
copy changes only that include to support the build command above. The
portable copy has not been rebuilt or rerun on GPU.

Historical output SHA256:
`287ec8ad17c786ec787749ed5d6d080e35ca770e91c40e4d1c472fad8c933102`.
Historical site-2 record SHA256:
`4e2b061af6a4109b86d9e7cf8790bfaa59901fdcf9835c601998f506caad1050`.

Raw evidence remains outside Git, following `../../manual-gates.md`:

- `~/tmp/agent-collab/ds4-min/20260920-DAA8BAB9-window-observer-compile-receipt.md`
- `~/tmp/agent-collab/ds4-min/20260920-C8332C52-windowed-prefill-observer-gpu-receipt.md`
- `~/tmp/agent-collab/ds4-min/window-gpu-DAA8BAB9/artifacts/`
- `~/tmp/agent-collab/ds4-min/20260920-C8332C52-cublas-observer-compile-receipt.md`

These narrow fixture results do not establish full-model or Go/C parity,
performance, public-dispatch behavior, or cuBLAS/async/graph/file-mapping
qualification. Those paths remain UNREAD for the window harness. Requalify
source, compiler, device and lease before reusing historical evidence.
