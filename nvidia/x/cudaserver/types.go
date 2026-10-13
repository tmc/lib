package cudaserver

// ChatMessage represents a single turn in a chat conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest represents an OpenAI-compatible /v1/chat/completions payload.
type ChatCompletionRequest struct {
	Model            string        `json:"model"`
	Messages         []ChatMessage `json:"messages"`
	Stream           bool          `json:"stream"`
	Temperature      float32       `json:"temperature,omitempty"`
	TopP             float32       `json:"top_p,omitempty"`
	MaxTokens        int           `json:"max_tokens,omitempty"`
	SessionID        string        `json:"session_id,omitempty"`
	ResponsesVisible bool          `json:"responses_visible,omitempty"`
	ThinkingVisible  bool          `json:"thinking_visible,omitempty"`
}

// ChatChoice represents a single completion choice.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// UsageStats records actual engine tokenizer counts for a completion.
// TotalTokens must equal PromptTokens plus CompletionTokens.
type UsageStats struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse represents the full non-streaming response.
type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	// Usage is nil when the engine does not report tokenizer counts.
	Usage *UsageStats `json:"usage,omitempty"`
}

// StreamDelta represents the incremental content in an SSE chunk.
type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// StreamChoice represents a choice in an SSE chunk.
type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// ChatCompletionChunk represents an SSE streaming data chunk.
type ChatCompletionChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// Metrics represents runtime performance metrics.
type Metrics struct {
	RequestsTotal       uint64
	OutputChunksTotal   uint64
	AvgTimeToFirstToken float64
	AvgInterTokenTime   float64
}
