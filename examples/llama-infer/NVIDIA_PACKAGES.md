# nvidia/x packages and this example

A record of which packages in `nvidia/x` this example uses, and why the rest
are not used. Written 2026-09-10 against lib HEAD; a package that changes may
deserve a different answer.

The Where column names the API a tracked file actually calls, not what the
package offers: an import establishes a dependency, never a feature. Three rows
first published in af01edb described capabilities the tracked code does not
reach, corrected here.

Nothing below has been executed on a GPU. Adoption here means the code is
written, typechecked for linux, and unit-tested where the logic is
platform-neutral. It does not mean it has run.

Rows marked *gated* are off unless their environment variable is set, leave
the default path unchanged, and carry an UNQUALIFIED header in the file that
adds them. They were written on 2026-09-10 after the user chose
implemented-but-unqualified over not-implemented for every deferred package;
the measurement each one needs is unchanged and still unmade.

## Used

| Package | Where |
| --- | --- |
| cuda, cudalaunch, cudamem | the CUDA engine and every kernel launch |
| nvrtcx | `CompileGPUModulesAs` compiles `forward.cu` at run time |
| cudaattn | `Compile` and `DefaultConfig`: the dense `CausalFlashAttentionDecode` kernel. `CompilePaged` and `PagedSharedMemoryBytes` are called only by `paged_kv_cuda.go`, which nothing in the engine reaches (see its header). |
| cudakv | `NewTracked`: the dense per-sequence and single-stream KV caches. `NewPagedCache` is called only by `paged_kv_cuda.go`, which nothing in the engine reaches. `NewFP8` is not called. |
| cudasafetensors | checkpoint loading |
| cudasample | sampling |
| perfetto, cudatrace | `-cuda-trace`: host submission and device-interval tracks |
| nvtx | `ScopedRange` in the inferbench path, not in `-cuda-trace` |
| cudacapture | `gpuprof` reads cudaprof captures and qualifies them |
| cudaptx, cudasass | `-kernel-report`: what NVRTC produced for the forward kernels |
| cudatest | the hardware-test gate |
| cudagraph | *gated* `LLAMA_INFER_CUDA_GRAPH=1`: `Capture`, `Instantiate`, `Exec.Update`, `Exec.Launch` replay the single-stream decode step (`cudagraph_cuda.go`) |
| mxfp4 | *gated* `LLAMA_INFER_MXFP4=1`: `QuantizeMatrix` at load, `Engine.LinearGEMVOn` for the seven per-layer projections and the LM head of the single-stream decode (`mxfp4_cuda.go`). The full-precision copies stay resident; this is not a memory saving. |

## Not used, with the reason

**cublasltx** — deliberate, and a regression if adopted as it stands.
`cudaMatrixProjection` precomputes a per-batch plan once, including the
`MatmulAlgoGetHeuristic` call. `cublasltx.Engine.MatMul` rebuilds the
descriptor, three layouts, and the preference, and re-runs the heuristic on
every call. Adopting it would move that work into the per-token path. It would
become the better choice if the engine grew a persistent plan.

**cudalayer** — the closest call here: its doc says it exists for this example.
It offers a fused `ProjectionResidualRMSNorm`. `forward.cu` already fuses
`FusedRMSNormQKV` and `FusedSwiGLUDown`, so the gain is not obviously positive,
and the package says its API and kernel layout are not stable. Substituting a
kernel in the decode path is exactly the change that cannot be justified
without measuring it on hardware. Not adopted for lack of a GPU, not for lack
of merit.

**cudaquant** — Q4_0/Q8_0 dequantization on the host, with no device kernel.
There is nothing for the CUDA path to call; a Q4_0 checkpoint would be
dequantized to float32 at load and uploaded as such, which is a loader feature
this example does not have. mxfp4 has a device kernel and is adopted above,
gated, without the accuracy control its own doc says adoption requires.

**cudasanitizer** — runs a program under compute-sanitizer. Belongs in a test
harness on a CUDA host, and would report nothing here.

**cuptiactivity, cuptirange, cuptipcsampling** — in-process CUPTI collection.
This example gets its GPU telemetry from cudaprof captures instead, which is
one collector rather than two competing for the same CUPTI subscriber slot.

**cudasass, from the capture side** — measured, not assumed. A cudaprof
capture can carry the CUBIN, PTX, SASS, and source for the kernels it observed,
and `cudacapture.LoadCode` parses them, so `gpuprof` could report a register
floor per kernel instead of leaving `HardwareReplayData.PeakGPRs` at zero. All
three retained captures checked on 2026-09-10 (`serving-20260910b/sglang`,
`serving-20260910b/sglang-perkind`, `20260909-native-cuda13-final`) record every
code role as explicitly unavailable, so nothing available here could exercise
that path. Not written: an unexercisable path is worse than a documented zero.
The compile-side use of cudasass in `-kernel-report` stands, since it
disassembles the cubin NVRTC produced.

**cudaobjdump** — inspects cubins in host binaries. The kernels here are
compiled at run time by NVRTC, so the cubin is in memory and `cudasass`
disassembles it directly; there is no host binary to dump.

**nvfatbinx, nvjitlinkx, nvvmx** — multi-image fatbinaries, device linking, and
NVVM IR. All are for a build pipeline with separately compiled device code.
This example compiles one translation unit for the device it is running on.

**nvlshm, moonep** — cross-process NVLink shared buffers and expert-parallel
MoE communication. This example is single-process and dense.

**cudamlx** — a port of MLX's CUDA backend runtime. Adopting it would replace
the engine, not extend it.

**egraph** — term rewriting for kernel fusion. It operates on an IR this
example does not have. See `nvidia/x/MODELIR_DESIGN.md`: no ModelIR-to-CUDA
lowering exists yet.
