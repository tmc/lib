// Package llama runs f32 LLaMA decoder graphs through PJRT.
//
// A model implements embedding lookup, pre-norm grouped-query attention with
// unscaled split-half rotary embeddings, a gated MLP, residual connections and
// final normalization/projection. It owns device weights and bounded history.
// Each forward graph updates all layers atomically with functional buffers;
// it does not donate inputs or promise in-place updates.
//
// Supply validated model weights to New and close the model before its borrowed
// engine. Calls must be externally synchronized. Forward returns caller-owned
// logits; only the valid count prefix of each batch member is meaningful.
// Tokenization, checkpoint loading, sampling and model-specific RoPE scaling
// are outside this package. It does not infer checkpoint conventions.
package llama
