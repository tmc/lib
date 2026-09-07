# StableHLO

This module emits StableHLO functions and modules as MLIR text. The builder
checks ranks, axes and known dimensions before emitting operations. It includes
RMSNorm, stable softmax and scaled dot-product attention for static
shapes. Attention accepts f32, f16 and bf16 inputs with matching batch dimensions.
For f16 and bf16, arithmetic uses f32 and the output is converted back to the
input type. This favors numerical range over reduced-precision compute throughput.
Masked attention accepts boolean causal and padding masks, with true meaning
allowed. Fully masked rows return zero for finite inputs. CausalMask supports
an offset for cached decoding, and And combines masks after broadcasting.
Integer constants, checked dynamic slice updates, and runtime-position causal
masks support fixed-capacity decode graphs. Dynamic updates use StableHLO
clamping; the cache wrapper checks capacity before dispatch.

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
