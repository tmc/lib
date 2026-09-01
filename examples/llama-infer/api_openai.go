package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// ChatMessage represents a single message in the chat history.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest is the incoming payload for /v1/chat/completions.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Prompt      string        `json:"prompt,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float32       `json:"temperature,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

// ChatCompletionChoice is a single generation candidate.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatCompletionResponse is the non-streaming response format.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   map[string]int         `json:"usage"`
}

// ChatCompletionChunkChoice represents a delta chunk in SSE streaming.
type ChatCompletionChunkChoice struct {
	Index        int         `json:"index"`
	Delta        ChatMessage `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// ChatCompletionChunk is the SSE stream chunk format.
type ChatCompletionChunk struct {
	ID      string                      `json:"id"`
	Object  string                      `json:"object"`
	Created int64                       `json:"created"`
	Model   string                      `json:"model"`
	Choices []ChatCompletionChunkChoice `json:"choices"`
}

// ModelObject represents a model descriptor in /v1/models.
type ModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse is the response payload for /v1/models.
type ModelsResponse struct {
	Object string        `json:"object"`
	Data   []ModelObject `json:"data"`
}

func (s *TelemetryServer) handleV1Models(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	resp := ModelsResponse{
		Object: "list",
		Data: []ModelObject{
			{
				ID:      s.cfg.Name,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "tmc/lib",
			},
		},
	}
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *TelemetryServer) handleV1ChatCompletions(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON payload: %v", err), http.StatusBadRequest)
		return
	}

	// Format prompt from messages
	var promptBuilder strings.Builder
	for _, m := range req.Messages {
		promptBuilder.WriteString(fmt.Sprintf("%s: %s\n", m.Role, m.Content))
	}
	prompt := promptBuilder.String()
	if prompt == "" {
		prompt = req.Prompt
	}
	if prompt == "" {
		prompt = "The key to scalable AI architectures is"
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 64
	}

	reqID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	if req.Stream {
		// Server-Sent Events (SSE) Streaming
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		if s.engine != nil {
			_ = s.engine.Generate(r.Context(), s.cfg, prompt, maxTokens, func(tokenID int32, tokenText string) bool {
				chunk := ChatCompletionChunk{
					ID:      reqID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   s.cfg.Name,
					Choices: []ChatCompletionChunkChoice{
						{
							Index: 0,
							Delta: ChatMessage{
								Role:    "assistant",
								Content: tokenText,
							},
						},
					},
				}
				bytes, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", bytes)
				flusher.Flush()
				return true
			})
		}

		// Send stream terminator
		finish := "stop"
		finalChunk := ChatCompletionChunk{
			ID:      reqID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   s.cfg.Name,
			Choices: []ChatCompletionChunkChoice{
				{
					Index:        0,
					Delta:        ChatMessage{},
					FinishReason: &finish,
				},
			},
		}
		finalBytes, _ := json.Marshal(finalChunk)
		fmt.Fprintf(w, "data: %s\n\n", finalBytes)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	// Non-streaming response
	var fullText strings.Builder
	tokenCount := 0
	if s.engine != nil {
		_ = s.engine.Generate(r.Context(), s.cfg, prompt, maxTokens, func(tokenID int32, tokenText string) bool {
			fullText.WriteString(tokenText)
			tokenCount++
			return true
		})
	}

	resp := ChatCompletionResponse{
		ID:      reqID,
		Object:  "chat.completion",
		Created: created,
		Model:   s.cfg.Name,
		Choices: []ChatCompletionChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: fullText.String(),
				},
				FinishReason: "stop",
			},
		},
		Usage: map[string]int{
			"prompt_tokens":     len(prompt) / 4,
			"completion_tokens": tokenCount,
			"total_tokens":      (len(prompt) / 4) + tokenCount,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
