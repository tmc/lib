# Application FFN planning

The host-only `ffnPlan` checks a single dense float32 decoder FFN:
RMSNorm, gate/up projections, SwiGLU, down projection, and residual addition.
Matrices are output-major. Input vectors and matrices require exact lengths.
Dimensions and each matrix/scratch byte count must fit int; epsilon must be
finite and positive. The caller supplies a scratch byte budget.

Scratch contains distinct normalized, gate, up, activation, and down-result
regions: 2*hidden + 3*intermediate float32 words. The reference owns this
scratch and a separate hidden-size output until return; inputs are read-only.
Output memory is not counted in the scratch budget. That budget does not bound
weights, process memory, or device memory. Plans describe logical dependencies,
not launch admission, GPU barriers, residency, or cancellation.

Tests compare the reference with the existing CPU RMSNorm/matmul primitives
using asymmetric weights and odd shapes. They refuse invalid dimensions,
byte-size overflow, insufficient budget, invalid epsilon, length mismatch,
zero plans, and inconsistent storage counts. They check detached dependency
metadata, unchanged inputs, and independent outputs.

No CUDA source, compiler invocation, GPU execution, CLI integration, or
performance result is provided. The existing llama GPU path retains its
partial QKV and SwiGLU/down fusion. A full persistent decoder still needs
operator/storage/runtime support and independent physical qualification.
