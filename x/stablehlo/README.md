# StableHLO

This module emits StableHLO functions and modules as MLIR text. The builder
checks ranks, axes and known dimensions before emitting operations. It includes
RMSNorm, stable softmax and unmasked scaled dot-product attention for static
shapes. Attention currently requires f32 inputs and matching batch dimensions.
It applies neither dropout nor a causal mask.

Run the Go tests with `go test ./...`. To also compile and execute emitted
matmul, RMSNorm, softmax and attention graphs against NumPy references, set
`STABLEHLO_PYTHON` to a Python interpreter with JAX and NumPy installed:

```
STABLEHLO_PYTHON=/path/to/venv/bin/python go test -run TestNativeCPU -v ./builder
```

That test selects the CPU backend and requires an intentional attention-scale
mutation to fail. It validates emitted graphs, not the Go PJRT bindings or TPU
hardware. Without the environment variable, this optional test is skipped.
