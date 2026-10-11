package cudasrv

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// Engine is the backend inference engine driving model generation.
// Generate must stop when ctx is canceled and must not close tokenChan.
type Engine interface {
	Generate(ctx context.Context, req ChatCompletionRequest, tokenChan chan<- string) error
}

// UsageEngine reports the engine's actual tokenizer counts for the rendered
// prompt and completed output. The server validates arithmetic, not the engine
// tokenizer implementation. Without it, responses omit usage; token stream
// chunks are not tokenizer tokens. GenerateWithUsage must obey Engine's rules.
type UsageEngine interface {
	Engine
	GenerateWithUsage(context.Context, ChatCompletionRequest, chan<- string) (UsageStats, error)
}

func (s *Server) generate(ctx context.Context, req ChatCompletionRequest, tokens chan<- string) (*UsageStats, error) {
	engine, ok := s.engine.(UsageEngine)
	if !ok {
		return nil, s.engine.Generate(ctx, req, tokens)
	}
	usage, err := engine.GenerateWithUsage(ctx, req, tokens)
	if err != nil {
		return nil, err
	}
	maxInt := int(^uint(0) >> 1)
	if usage.PromptTokens < 0 || usage.CompletionTokens < 0 || usage.PromptTokens > maxInt-usage.CompletionTokens || usage.TotalTokens != usage.PromptTokens+usage.CompletionTokens {
		return nil, errors.New("invalid engine token usage")
	}
	return &usage, nil
}

// Tokenizer encodes text strings into token IDs and decodes token IDs into text.
type Tokenizer interface {
	Encode(text string) ([]int32, error)
	Decode(tokens []int32) (string, error)
}

// Server provides an HTTP handler for OpenAI-compatible LLM inference.
type Server struct {
	engine         Engine
	modelName      string
	flightRecorder *cudaprof.FlightRecorder
	requestsTotal  uint64
	chunksTotal    uint64
}

// New creates a new inference server wrapping engine.
func New(engine Engine, modelName string) *Server {
	return &Server{
		engine:    engine,
		modelName: modelName,
	}
}

// SetFlightRecorder configures an in-memory GPU flight recorder for live trace export.
func (s *Server) SetFlightRecorder(fr *cudaprof.FlightRecorder) {
	s.flightRecorder = fr
}

// Handler returns the http.Handler serving API routes.
// Chat completion requests must contain one JSON value in at most 1 MiB.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/debug/trace.pftrace", s.handlePerfettoTrace)
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

	if s.engine == nil {
		http.Error(w, "generation unavailable", http.StatusServiceUnavailable)
		return
	}
	// Bound the entire body, including whitespace after the JSON value.
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	decoder := json.NewDecoder(r.Body)
	var req ChatCompletionRequest
	err := decoder.Decode(&req)
	if err == nil {
		var extra any
		if err = decoder.Decode(&extra); err == io.EOF {
			err = nil
		} else if err == nil {
			err = errors.New("multiple JSON values")
		}
	}
	if err != nil {
		status := http.StatusBadRequest
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			status = http.StatusRequestEntityTooLarge
		}
		http.Error(w, "invalid JSON payload", status)
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

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	reqID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	tokenChan := make(chan string, 32)
	errChan := make(chan error, 1)

	go func() {
		defer close(tokenChan)
		_, err := s.generate(ctx, req, tokenChan)
		errChan <- err
	}()

stream:
	for {
		var tok string
		var open bool
		select {
		case <-ctx.Done():
			return
		case tok, open = <-tokenChan:
			if !open {
				break stream
			}
		}
		atomic.AddUint64(&s.chunksTotal, 1)
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
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			return
		}
	}

	if err := <-errChan; err != nil {
		// A failed generation must never carry a successful stop or DONE.
		_, _ = fmt.Fprint(w, "event: error\ndata: {\"error\":\"generation failed\"}\n\n")
		flusher.Flush()
		return
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
	if _, err := fmt.Fprintf(w, "data: %s\n\n", finalData); err != nil {
		return
	}
	if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
		return
	}
	flusher.Flush()
}

func (s *Server) handleNonStreaming(w http.ResponseWriter, r *http.Request, req ChatCompletionRequest) {
	tokenChan := make(chan string, 32)
	type result struct {
		usage *UsageStats
		err   error
	}
	resultChan := make(chan result, 1)
	go func() {
		defer close(tokenChan)
		usage, err := s.generate(r.Context(), req, tokenChan)
		resultChan <- result{usage, err}
	}()

	var sb strings.Builder
	chunkCount := 0
	for tok := range tokenChan {
		sb.WriteString(tok)
		chunkCount++
	}

	generated := <-resultChan
	if generated.err != nil {
		http.Error(w, fmt.Sprintf("generation error: %v", generated.err), http.StatusInternalServerError)
		return
	}

	atomic.AddUint64(&s.chunksTotal, uint64(chunkCount))

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
		Usage: generated.usage,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	reqs := atomic.LoadUint64(&s.requestsTotal)
	toks := atomic.LoadUint64(&s.chunksTotal)

	_, _ = fmt.Fprintf(w, "# HELP gpu_requests_total Total number of chat completion requests received.\n")
	_, _ = fmt.Fprintf(w, "# TYPE gpu_requests_total counter\ngpu_requests_total %d\n", reqs)
	_, _ = fmt.Fprintf(w, "# HELP gpu_output_chunks_total Total number of output stream chunks generated.\n")
	_, _ = fmt.Fprintf(w, "# TYPE gpu_output_chunks_total counter\ngpu_output_chunks_total %d\n", toks)
}

func (s *Server) handlePerfettoTrace(w http.ResponseWriter, r *http.Request) {
	if s.flightRecorder == nil {
		http.Error(w, "Flight recorder not configured", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", "attachment; filename=\"trace.pftrace\"")
	if err := s.flightRecorder.WritePerfettoTo(w); err != nil {
		http.Error(w, fmt.Sprintf("Error writing Perfetto trace: %v", err), http.StatusInternalServerError)
	}
}
