# LLaMA decoder runtime

`llama.New` validates and uploads f32 weights for a decoder with grouped-query
attention, unscaled split-half rotary embeddings, a gated MLP and final logits.
Linear weights use `[output, input]` order. A nil LM head shares the embedding
buffer. Nil Q/K/V biases mean zero. Dimensions and weights are validated before
device allocation; the runtime does not infer checkpoint conventions.

Weight shapes use H=HiddenSize, I=IntermediateSize, Q=QueryHeads*HeadSize,
K=KVHeads*HeadSize and V=VocabSize:

| Weight | Shape |
| --- | --- |
| Embedding, LMHead | [V, H] |
| AttentionNorm, MLPNorm, Norm | [H] |
| Query | [Q, H] |
| Key, Value | [K, H] |
| AttentionOutput | [H, Q] |
| Gate, Up | [I, H] |
| Down | [H, I] |
| QueryBias | [Q] |
| KeyBias, ValueBias | [K] |

`Forward` accepts row-major token IDs for `[BatchSize, chunk]`. Pass nil counts
for a full chunk in every batch, or one valid prefix count per member. Valid IDs
must belong to the vocabulary; padding IDs are ignored. The returned device
buffer contains `[BatchSize, chunk, VocabSize]` logits. Read only valid prefix
rows and close the buffer when finished. `Lengths` returns each cached length.

One compiled graph evaluates all layers and returns replacement histories plus
logits. The runtime installs histories only after the entire execution succeeds.
Validation, compilation, execution and cancellation errors retain previous
history. Metadata, replaced-history or program-eviction cleanup failures close
the model; native destruction failures can still leave plugin allocations.

The model retains a decode graph and at most one other chunk shape. `Reset`
clears logical lengths and keeps device storage and programs. There is no
history eviction, buffer donation or guaranteed in-place update. Attention
uses full-capacity tensors; performance and memory use depend on the compiler
and device. Rotary tables are computed on the host for each chunk and shared
across layers. Calls must be externally synchronized. Close the model before
closing its borrowed engine.

This package does not load checkpoints, tokenize, sample, support scaled RoPE
variants, or qualify a trained model's output quality. The TPU example adapter handles checkpoint loading and tokenization separately.

Run `go test ./...` for host checks. Set `XLA_PJRT_CPU_PLUGIN` to an installed
CPU plugin for the native two-layer logit/history reference tests, independent
batch lengths, padding, cancellation, reset, tied weights and program reuse.
These tests demonstrate CPU correctness, not TPU throughput or memory usage.

`NewEncoded` accepts `EncodedWeights` whose tensors each carry an explicit
`ElementType` and little-endian payload. Storage types may differ; they are
validated before device allocation and retained on device without a host-wide
float32 conversion. A nil encoded LM head shares the embedding buffer.

`Config.ElementType` selects F32 (also the zero default), F16 or BF16 activations,
projections and KV history. Parameters are converted in the graph when their
source type differs. Normalization, rotary operations, SiLU/SwiGLU and attention
use F32 intermediates and round at their output boundaries. Residual additions
use the execution type. The LM projection rounds to the execution type before
its logits are converted to F32 for the caller. These boundaries need not match
Transformers' half-precision rounding; they are not a bitwise parity promise.
F16 and BF16 may also choose different greedy tokens from F32: a small overall
logit error does not guarantee the same ordering of the highest scores.

Native encoded-model tests check mixed storage types, half-precision history,
independent batch lengths, reset, tied weights and host-source reuse. Their
small-fixture numerical comparison uses a float64 reference with documented
half-precision tolerances; it does not qualify pretrained half-model output.
