package cudasrv

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// Engine is the backend inference engine driving model generation.
type Engine interface {
	Generate(ctx context.Context, req ChatCompletionRequest, tokenChan chan<- string) error
}

// Tokenizer encodes text strings into token IDs and decodes token IDs into text.
type Tokenizer interface {
	Encode(text string) ([]int32, error)
	Decode(tokens []int32) (string, error)
}


// Server provides an HTTP handler for OpenAI-compatible LLM inference.
type Server struct {
	engine        Engine
	modelName     string
	requestsTotal uint64
	tokensTotal   uint64
}

// New creates a new inference server wrapping engine.
func New(engine Engine, modelName string) *Server {
	return &Server{
		engine:    engine,
		modelName: modelName,
	}
}

// Handler returns the http.Handler serving API routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/metrics", s.handleMetrics)
	return mux
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	resp := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{
				"id":       s.modelName,
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "nvidia-x",
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON payload: %v", err), http.StatusBadRequest)
		return
	}

	atomic.AddUint64(&s.requestsTotal, 1)

	if req.Stream {
		s.handleStreaming(w, r, req)
	} else {
		s.handleNonStreaming(w, r, req)
	}
}

func (s *Server) handleStreaming(w http.ResponseWriter, r *http.Request, req ChatCompletionRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	reqID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	tokenChan := make(chan string, 32)
	errChan := make(chan error, 1)

	go func() {
		defer close(tokenChan)
		errChan <- s.engine.Generate(r.Context(), req, tokenChan)
	}()

	for tok := range tokenChan {
		atomic.AddUint64(&s.tokensTotal, 1)
		chunk := ChatCompletionChunk{
			ID:      reqID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   s.modelName,
			Choices: []StreamChoice{
				{
					Index: 0,
					Delta: StreamDelta{Content: tok},
				},
			},
		}
		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	finishReason := "stop"
	finalChunk := ChatCompletionChunk{
		ID:      reqID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []StreamChoice{
			{
				Index:        0,
				FinishReason: &finishReason,
			},
		},
	}
	finalData, _ := json.Marshal(finalChunk)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", finalData)
	_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) handleNonStreaming(w http.ResponseWriter, r *http.Request, req ChatCompletionRequest) {
	tokenChan := make(chan string, 32)
	errChan := make(chan error, 1)

	go func() {
		defer close(tokenChan)
		errChan <- s.engine.Generate(r.Context(), req, tokenChan)
	}()

	var sb strings.Builder
	tokenCount := 0
	for tok := range tokenChan {
		sb.WriteString(tok)
		tokenCount++
	}

	if err := <-errChan; err != nil {
		http.Error(w, fmt.Sprintf("generation error: %v", err), http.StatusInternalServerError)
		return
	}

	atomic.AddUint64(&s.tokensTotal, uint64(tokenCount))

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []ChatChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: sb.String(),
				},
				FinishReason: "stop",
			},
		},
		Usage: UsageStats{
			PromptTokens:     10,
			CompletionTokens: tokenCount,
			TotalTokens:      10 + tokenCount,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	reqs := atomic.LoadUint64(&s.requestsTotal)
	toks := atomic.LoadUint64(&s.tokensTotal)

	_, _ = fmt.Fprintf(w, "# HELP gpu_requests_total Total number of chat completion requests received.\n")
	_, _ = fmt.Fprintf(w, "# TYPE gpu_requests_total counter\ngpu_requests_total %d\n", reqs)
	_, _ = fmt.Fprintf(w, "# HELP gpu_tokens_generated_total Total number of tokens generated.\n")
	_, _ = fmt.Fprintf(w, "# TYPE gpu_tokens_generated_total counter\ngpu_tokens_generated_total %d\n", toks)
}
