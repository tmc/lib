# DS4 kernel hot-swap integration

This is an integration guide for ds4-min at
`8dee8a2c9d7a6ad1efc04fc2a5823fd0a3bb52c5`, not a claim that ds4-min
already uses the service. Its model source is read-only in this lane. The
service integration and the native header-precedence control require separate
qualification before use in a model process.

## Replace retained raw handles

`cudaFoundation.loadMathKernels` compiles a translation unit, loads a module,
and writes raw functions into the destinations returned by `cudaKernels`.
`newCUDAHCBatch` also retains five functions from that module. The quantizer
has a separate module. Updating only the HC fields would leave attention,
batched HC and quantization call sites using the old functions.

Create one `NewNVRTCLoader(device, srcDir)` and `NewService(loader)` for the
owning CUDA device. Register each replaceable entry by its actual CUDA symbol
using `Service.Register(ctx, name, Source{Code, Filename, Entry, Flags}, label)`.
Use the complete translation unit and its real prerequisites; `cuda_hc.cu`
calls helpers supplied by earlier math fragments. Preserve the selected
architecture, fast-math, line-info and vectorization settings for model
comparisons. An ABI-compatible candidate can still be numerically wrong.

Route every dispatch of a registered entry through `Service.Launch`. Adopt a
nonnil `PendingLaunch` before checking the accompanying error, and call its
`Wait` to establish completion. Keep all arguments, streams and device
allocations alive until that succeeds. Cancellation alone is not completion.
Do not release an owner because a request returned an error.

For a forward pass that needs raw handles, acquire `Service.Pin` for each entry
it uses, read `Pin.Function`, and keep the pin until the pass's stream has
completed. `Pin.Release(ctx)` must succeed before dropping the pin. This also
applies to the function fields in `cudaHC` and `cudaHCBatch`, all destinations
in `cudaKernels`, the separately loaded quantizer, and alternate attention
paths. Retaining a function value without a pin bypasses the service's module
lifetime protection. A pass using multiple independently selected entries
needs an application-level mutation boundary if it requires a coherent set;
independent pins do not provide a multi-kernel transaction.

Close graph/launch/fixture owners and the service successfully before closing
the model's stream, arenas or device. A failed close retains its owners for
retry. A poisoned service requires process-level recovery rather than reuse.

## Register the actual captured nodes

`cudaRuntime.evalLayerState` captures QKV work with `cudagraph.Capture`,
instantiates it, then caches the exec and calls raw `Exec.Launch`. Capture
itself does not register those nodes with the service.

Before destroying the source graph, enumerate `Graph.Nodes`, inspect each
kernel node's `Node.Params`, and record the mapping from the node's actual
function to the registered entry and its real argument order. Do not assume
all captured nodes belong to the same kernel or infer arguments from a name.
Bind each relevant node using `Service.RegisterGraph(ctx, name, exec, node,
stream, grid, args...)`. A nonzero returned State with an error can retain a
possibly applied patch; keep the caller's exec, node, stream and arguments
alive until service cleanup succeeds. Grid is a `driver.Dim3` here.

Use `Service.LaunchGraph(ctx, exec)` for subsequent graph submissions, adopt
its nonnil `GraphLaunch` even on error, and call `Wait(ctx)`. Replace both the
initial cached-exec dispatch and later dispatches in `evalLayerState`. Raw
`Exec.Launch` bypasses generation pins and patch coordination. The argument
slice is copied at registration, but its pointed-to values and allocations
remain caller-owned. A graph patch needs its own A-to-B output control; an
accepted topology update is not numerical evidence.

## Declare real replay footprints

`FixtureConfig.Args` contains the scalar arguments and `nil` at each declared
pointer position. `FixtureBuffer.Arg` supplies that pointer with a complete
byte extent and an owned `cudamem.Slice[byte]`. Use
`cudamem.Reinterpret[byte]` to obtain a byte view of an existing owned typed
slice; check its error. Outputs use `FixtureOutput{Arg: index, Elem: "f32"}`.
Launch grid, block, shared memory and owning input stream must match the
original dispatch. The selected geometry override is applied at recording.

Record each kernel's real stage input after its prerequisites have completed.
The recording runs the selected baseline with original and poisoned outputs;
it is not a snapshot of an arbitrary buffer before its producer has run.
Choose explicit finite noise/output tolerances and timing rounds, rather than
assuming `DefaultReplayPolicy`'s exact, zero-noise defaults fit fast math.

### Scalar cached HC (FP32)

Arguments below are zero-based. Read-only inputs are scratch copies unless
explicitly marked stable. A stable binding is still lifetime-borrowed, not
owned by the service; keep it alive and exclude concurrent writes until the
fixture closes. The `head=0` split writes all 24 coefficients.

| Entry | Arguments and declared bytes | Output | Grid / block |
| --- | --- | --- | --- |
| `ds4_hc_cached_split` | 0 mix: 24×4; 1 scale: 3×4; 2 base: 24×4; 3 split: 24×4; 4 `uint32(0)` | arg 3 FP32 | (1,1,1) / (32,1,1) |
| `ds4_hc_cached_collapse` | 0 state: 16384×4; 1 split: 24×4; 2 out: 4096×4 | arg 2 FP32 | (16,1,1) / (256,1,1) |
| `ds4_hc_cached_expand` | 0 state: 16384×4; 1 block: 4096×4; 2 split: 24×4; 3 out: 16384×4 | arg 3 FP32 | (64,1,1) / (256,1,1) |

The production pre path subsequently writes RMS-normalized data into the
second 4096 floats of its 8192-float output. Record the collapse's written
4096-float slice, not that unwritten suffix. For `head=1`, split writes only
the first four values: declare that written slice separately or refuse a
fixture that declares all 24 outputs.

Batched split has mix/split extents `n×24×4`, shared scale/base extents as above,
and grid (1,n,1). Batched expand has state/out `n×16384×4`, block `n×4096×4`,
split `n×24×4`, and grid (64,n,1). Batched collapse uses `out_stride=8192`
and writes only the first 4096 floats of each interleaved row. The current
fixture output contract compares complete contiguous declared bindings;
do not present the untouched row gaps as write-only output. Use a dedicated
packed diagnostic launch or an explicitly supported output-view prerequisite
before claiming that strided production fixture is admitted.

### Attention-output fallback

`evalLayerFFNPre` projects eight head groups to `attnLow` (8192 FP32), then
projects that to `attnOut` (4096 FP32), followed by HC expand. The native
fallback quantizes each 32-float activation block to 36 bytes. Its Q8_0
weight blocks occupy 34 bytes. These are different formats.

| Entry / production stage | Pointer arguments and scalar arguments | Output bytes | Grid / block |
| --- | --- | --- | --- |
| `ds4_quant_q8_0`, first projection | 0 heads: 32768×4; 1 quantized: 1024×36 | byte output; current FP32 replay output contract cannot directly compare this | (1024,1,1) / (32,1,1) |
| `ds4_q8_0_matvec_quantized_blockdiag`, OutA | 0 weights: 8192×128×34; 1 quantized: 8×128×36; 2 low: 8192×4; 3 `uint32(128)`; 4 `uint32(1024)` | 8192×4 | (128,8,1) / (256,1,1) |
| `ds4_quant_q8_0`, second projection | 0 low: 8192×4; 1 quantized: 256×36 | byte output, same limitation | (256,1,1) / (32,1,1) |
| `ds4_q8_0_matvec_quantized`, OutB | 0 weights: 4096×256×34; 1 quantized: 256×36; 2 out: 4096×4; 3 `uint32(256)`; 4 `uint32(4096)` | 4096×4 | (512,1,1) / (256,1,1) |

These rows describe the single-token raw-block fallback, not the aligned
artifact or batched/D2R variants. Inspect the selected route before recording:
`encodeQ80Native`/`encodeQ80NativeBlockDiag` can choose different entries when
an aligned artifact is present. A kernel replay is not a whole attention
stage comparison; preserve upstream quantized inputs and retain a separate
application logits check.

### Mapped-weight prerequisite

`cudaModelTensor.device` is a raw `driver.DevicePtr`. `modelRange` may return
an HMM host-mapping address, a registered mapping, or a resident offset. Those
raw addresses are not an owned `cudamem.Slice`. `FixtureBuffer` does not expose
a constructor that imports that ownership. Do not manufacture a slice or free
the model mapping as though it were a CUDA allocation.

For a bounded fixture, copy the exact tensor bytes to an owned `cudamem`
allocation and record that allocation, retaining its model/tensor identity.
This is an explicit diagnostic input adaptation, not proof of the original
mapped-weight lifetime path. Owned resident or artifact slices may be used
when their actual extents match the selected kernel. Production replay of
borrowed mapped tensors needs an independently specified external-lifetime
binding prerequisite. This also affects HC scale/base tensors.

## Switch from embedded headers to disk hot mode

The pinned source does **not** use `WithHeadersFS` for math compilation.
`cudaMathHeaderOptions` calls `WithHeader("/ds4-min/"+name, contents)` for each
embedded fragment. `cudaMathCompilationSource` includes those absolute
virtual names. Merely adding `WithIncludePaths(srcDir)` leaves that path intact.

In an explicit hot mode, export the same fragments to a controlled source
directory, omit all embedded `WithHeader` options, and generate relative
includes such as `#include "cuda_hc.cu"`. Use the disk directory with
`NewNVRTCLoader`; use a real source filename and preserve fragment order.
Do not leave `/ds4-min/...` includes that cannot resolve on the GPU host.
Keep normal embedded mode unchanged outside this opt-in consumer adaptation.
Use the compiler's `Dependencies` digests for header fan-out; a source change
must not be identified only by mtime.

`TestDS4IncludePrecedenceNative` prepares competing in-memory value 7 and disk
value 11 headers, then edits the disk value to 13. It tests both relative and
absolute names, retaining each source/header/PTX/CUBIN/compiler-flag receipt
and guarded device output. The expected controls are memory cases = 7,
disk-only cases = 11 then 13. They are an unqualified hypothesis until the
reviewed test actually runs under the coordinator's bounded GPU lock. A
compile-only success or a skipped test does not establish precedence.

## Application Checker: C teacher to candidate

Retain logits from an actual C teacher run, not a second Go run. Keep the C
source/binary, model digest, tokenizer/token-ID stream, forced-token hash,
prompt, accepted decode step and numerical settings with those bytes.
`dsv4eval.CompareTeacherStates(CDir, candidateDir)` validates matching forced
tokens, full token-ID hashes, contiguous steps, vocabulary lengths and finite
logits, then reports directional `KLLeftRight`: C teacher to candidate.
It does not itself establish that the left directory was produced by C, or
bind the model/compiler identity; validate that retained provenance first.

The application hook must reset its session and then `Sync` the same prompt
for every candidate. `Sync` reuses a matching token prefix; without `Reset`,
the same prompt can perform no evaluation. Copy returned `Runtime.Logits`
before subsequent evaluation changes the backing slice. Finish all submitted
GPU work before returning, or retain its completion owner as required by the
hook contract. Register the hook with `Service.SetCheck`; select replay plus
Checker, or Checker alone, and choose the failure policy explicitly.

The integration sequence below is illustrative: session construction,
teacher provenance loading, candidate snapshot writing and the KL threshold
are application-owned. The symbols are the actual public interfaces, not a
complete executable model adapter.

```go
check := cudahotswap.Checker(func(ctx context.Context, state cudahotswap.State) error {
    if err := session.Reset(); err != nil { return err }
    if err := session.Sync(prompt); err != nil { return err }
    // Copy logits, finish GPU work, write the candidate's provenance-bound
    // state, then compare retained C states on the left against it.
    return compareRetainedCTeacher(ctx, state)
})
```

For a runnable host demonstration of the arithmetic, run from `nvidia`:

```sh
GOWORK=off go test -run '^Example_ds4DirectionalKL$' -v ./x/cudahotswap
```

Observed example output:

```text
KL(teacher || candidate) = 0.433781, error = <nil>
```

The companion test covers equal logits, constant shifts, wrong logits,
direction reversal, extreme finite values, empty/unequal vocabularies and
nonfinite input. It uses max-centered log probabilities; it is a host
arithmetic demonstration, not retained C-teacher evidence or a model verdict.

## Remaining qualification

Validate consumer handle coverage, captured-node mapping, real mapped-weight
ownership or the explicit diagnostic-copy adaptation, fixture output views,
header precedence and retained-C provenance before a model process uses this
guide. Full Push/revert/graph and 20-edit latency controls remain separate
native service gates. No model parity or performance conclusion follows
from the host KL example.
