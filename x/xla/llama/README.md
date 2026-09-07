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
variants, or qualify a trained model's output quality. It currently uses f32
storage and arithmetic. The TPU example adapter is separate work.

Run `go test ./...` for host checks. Set `XLA_PJRT_CPU_PLUGIN` to an installed
CPU plugin for the native two-layer logit/history reference tests, independent
batch lengths, padding, cancellation, reset, tied weights and program reuse.
These tests demonstrate CPU correctness, not TPU throughput or memory usage.
