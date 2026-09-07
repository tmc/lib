// Package llama runs LLaMA decoder graphs through PJRT.
//
// A model implements embedding lookup, pre-norm grouped-query attention with
// unscaled split-half rotary embeddings, a gated MLP, residual connections and
// final normalization/projection. It owns device weights and bounded history.
// Each forward graph updates all layers atomically with functional buffers;
// it does not donate inputs or promise in-place updates.
//
// New accepts float32 weights; NewEncoded accepts explicitly typed F32, F16 or
// BF16 payloads. Config.ElementType selects execution and history precision.
// Selected intermediates compute in F32; Forward always returns F32 logits.
//
// Supply validated model weights and close the model before its borrowed
// engine. Calls must be externally synchronized. Forward returns caller-owned
// logits; only the valid count prefix of each batch member is meaningful.
// Tokenization, checkpoint loading, sampling and model-specific RoPE scaling
// are outside this package. It does not infer checkpoint conventions.
package llama
