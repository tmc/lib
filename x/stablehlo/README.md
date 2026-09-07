# StableHLO

This module emits StableHLO functions and modules as MLIR text. The builder
checks ranks, axes and known dimensions before emitting operations. It includes
RMSNorm, stable softmax and scaled dot-product attention for static
shapes. GroupedQueryAttention supports unequal Q and KV head counts, including
multi-query attention, without explicitly replicating K/V tensors. Its masked
variant supports causal and padding masks over query-head score dimensions.
Attention accepts f32, f16 and bf16 inputs with matching batch dimensions.
For f16 and bf16, arithmetic uses f32 and the output is converted back to the
input type. This favors numerical range over reduced-precision compute throughput.
Masked attention accepts boolean causal and padding masks, with true meaning
allowed. Fully masked rows return zero for finite inputs. CausalMask supports
an offset for cached decoding, and And combines masks after broadcasting.
Integer constants, checked dynamic slice updates, and runtime-position causal
masks support fixed-capacity decode graphs. Dynamic updates use StableHLO
clamping. ScatterRows instead replaces indexed rows and ignores out-of-range
indices. BatchedCausalMask supports independent runtime positions and valid
query counts. Cache callers validate counts and capacity before dispatch.

GatherRows selects embedding rows with scalar or tensor i32/i64 indices.
Runtime indices clamp to the available rows; model runtimes must reject invalid
token IDs before dispatch. SiLU and SwiGLU compute activations in f32 for
f16/bf16 inputs and convert only the final result. RotaryEmbedding uses caller-
supplied f32 cosine/sine tables and split-half feature pairs, with trailing
broadcasting and optional unrotated tail features. These operations provide
model-building primitives; they do not constitute a complete inference engine.

No dropout is applied. Mask operations follow the
[StableHLO specification](https://openxla.org/stablehlo/spec).

Run the Go tests with `go test ./...`. To also compile and execute emitted
matmul, RMSNorm, softmax and attention graphs against NumPy references, set
`STABLEHLO_PYTHON` to a Python interpreter with JAX and NumPy installed:

```
STABLEHLO_PYTHON=/path/to/venv/bin/python go test -run TestNativeCPU -v ./builder
```

That test selects the CPU backend and requires an intentional attention-scale
mutation to fail. It validates emitted graphs, not the Go PJRT bindings or TPU
hardware. Without the environment variable, this optional test is skipped.

The optional `TestNativeModelOps` gate covers row lookup, scalar/strided slices,
concatenation, activations and full/partial rotary embeddings with independent
references and a rotation-sign mutation. Run both native gates with:

```
STABLEHLO_PYTHON=/path/to/venv/bin/python go test -run TestNative -v ./builder
```
