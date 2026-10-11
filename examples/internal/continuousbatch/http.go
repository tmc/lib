package continuousbatch

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// Handler exposes the scheduler through the OpenAI chat-completions response shape.
// It accepts one JSON request of at most 1 MiB, with only the fields needed
// by the scheduler prototype. A nil Scheduler responds with service unavailable.
type Handler struct {
	Scheduler *Scheduler
	Model     string
	nextID    atomic.Uint64
}

type chatRequest struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
	Stream    bool   `json:"stream"`
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if h.Scheduler == nil {
		http.Error(w, "scheduler unavailable", http.StatusServiceUnavailable)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	dec := json.NewDecoder(r.Body)
	var req chatRequest
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	id := fmt.Sprintf("chatcmpl-%d", h.nextID.Add(1))
	response, err := h.Scheduler.Submit(r.Context(), Request{ID: id, Prompt: req.Prompt, MaxTokens: req.MaxTokens})
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer response.Cancel()
	if req.Stream {
		h.stream(w, r, id, response)
		return
	}
	var text strings.Builder
	for {
		select {
		case <-r.Context().Done():
			return
		case token, ok := <-response.Tokens:
			if !ok {
				goto complete
			}
			text.WriteString(token.Text)
		}
	}
complete:
	select {
	case <-r.Context().Done():
		return
	case err := <-response.Done:
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": h.Model,
		"choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": text.String()}, "finish_reason": "stop"}},
	})
}

func (h *Handler) stream(w http.ResponseWriter, r *http.Request, id string, response *Response) {
	defer response.Cancel()
	if _, ok := w.(http.Flusher); !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	controller := http.NewResponseController(w)
	w.Header().Set("Content-Type", "text/event-stream")
	for {
		select {
		case <-r.Context().Done():
			return
		case token, ok := <-response.Tokens:
			if !ok {
				goto complete
			}
			chunk := map[string]any{"id": id, "object": "chat.completion.chunk", "model": h.Model, "choices": []any{map[string]any{"index": 0, "delta": map[string]string{"content": token.Text}}}}
			data, err := json.Marshal(chunk)
			if err != nil {
				return
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
				return
			}
			if err := controller.Flush(); err != nil {
				return
			}
		}
	}
complete:
	select {
	case <-r.Context().Done():
		return
	case err := <-response.Done:
		if err != nil {
			return
		}
	}
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		return
	}
	_ = controller.Flush()
}
