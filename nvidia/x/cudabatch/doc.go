// Package cudabatch provides continuous batch scheduling, admission control,
// token and memory budgeting, fairness policies, validated shape binding,
// and KV cache lifetime tracking for LLM inference.
//
// Continuous batching coschedules chunked prefill tasks and autoregressive decode
// steps across iterations. Requests transition through deterministic states:
// Waiting, Prefilling, Decoding, Completed, and Canceled.
//
// # Lifetime and KV Ownership
//
// When a request is canceled, its cancellation does not immediately release physical
// KV cache blocks or reuse output buffers. The scheduler retains KV block ownership
// and output buffers until the active GPU or host execution step completes.
//
// # Live Protocol Continuation Tracking
//
// Multi-turn protocols (such as OpenAI /v1/responses and Anthropic /v1/messages)
// require preserving hidden reasoning tokens and tool-call outputs across turns
// without re-prefilling from scratch or stripping reasoning tokens. Live continuations
// are tracked via distinct cache sources (ResponsesVisible, ThinkingVisible),
// with exact suffix tokenization on prompt and continuation boundaries to prevent
// BPE boundary corruption.
package cudabatch
