package continuousbatch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// Handler exposes the scheduler through the OpenAI chat-completions response shape.
// It intentionally accepts only the fields needed by the scheduler prototype.
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
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	id := fmt.Sprintf("chatcmpl-%d", h.nextID.Add(1))
	response, err := h.Scheduler.Submit(r.Context(), Request{ID: id, Prompt: req.Prompt, MaxTokens: req.MaxTokens})
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	if req.Stream {
		h.stream(w, r, id, response)
		return
	}
	var text strings.Builder
	for token := range response.Tokens {
		text.WriteString(token.Text)
	}
	if err := <-response.Done; err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": h.Model,
		"choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": text.String()}, "finish_reason": "stop"}},
	})
}

func (h *Handler) stream(w http.ResponseWriter, r *http.Request, id string, response *Response) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		response.Cancel()
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	for token := range response.Tokens {
		chunk := map[string]any{"id": id, "object": "chat.completion.chunk", "model": h.Model, "choices": []any{map[string]any{"index": 0, "delta": map[string]string{"content": token.Text}}}}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		if err := r.Context().Err(); err != nil {
			response.Cancel()
			return
		}
	}
	if err := <-response.Done; err != nil {
		return
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
