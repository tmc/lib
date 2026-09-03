# Symbolic Llama decoder graph

This example constructs and validates the shapes of a single-head attention and
SwiGLU decoder layer. It does not run inference. The current symbolic IR lacks
transpose, causal masking, and softmax operations, and its compiler does not
lower matrix multiplication or RMS normalization.

Run from the `examples` module:

```sh
GOWORK=off go run ./apple/llama-ir
```

Output:

```text
validated symbolic decoder graph: nodes=23 output=[8 64] scratch=16384 bytes
lowering unsupported: compile: node 2: operation "rmsnorm" unsupported
```
