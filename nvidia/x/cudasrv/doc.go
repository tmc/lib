// Package cudasrv provides an OpenAI-compatible HTTP inference server
// with Server-Sent Events (SSE) streaming, Prometheus metrics, and graceful cancellation.
//
// Completion usage is omitted unless the engine implements [UsageEngine].
// gpu_output_chunks_total counts output chunks, including empty chunks, rather
// than tokenizer tokens; it replaces the incorrectly named gpu_tokens_generated_total.
package cudasrv
