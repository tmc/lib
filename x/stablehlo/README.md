# StableHLO

This module emits StableHLO functions and modules as MLIR text. The builder
checks ranks, axes and known dimensions before emitting operations. It includes
RMSNorm, stable softmax and unmasked scaled dot-product attention for static
shapes. Attention accepts f32, f16 and bf16 inputs with matching batch dimensions.
For f16 and bf16, arithmetic uses f32 and the output is converted back to the
input type. This favors numerical range over reduced-precision compute throughput.
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
