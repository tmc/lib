package cudasrv_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasrv"
)

type mockEngine struct{}

func (m *mockEngine) Generate(ctx context.Context, req cudasrv.ChatCompletionRequest, tokenChan chan<- string) error {
	tokenChan <- "Hello"
	tokenChan <- " world"
	tokenChan <- "!"
	return nil
}

func TestServerModels(t *testing.T) {
	srv := cudasrv.New(&mockEngine{}, "llama-3.2-1b")
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", w.Code)
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal models resp: %v", err)
	}
}

func TestServerNonStreaming(t *testing.T) {
	srv := cudasrv.New(&mockEngine{}, "llama-3.2-1b")
	payload := `{"model":"llama-3.2-1b","messages":[{"role":"user","content":"Hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(payload))
	w := httptest.NewRecorder()

	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d (body: %s)", w.Code, w.Body.String())
	}

	var resp cudasrv.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal completion resp: %v", err)
	}

	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "Hello world!" {
		t.Errorf("unexpected content: %+v", resp)
	}
}

func TestServerStreamingSSE(t *testing.T) {
	srv := cudasrv.New(&mockEngine{}, "llama-3.2-1b")
	payload := `{"model":"llama-3.2-1b","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(payload))
	w := httptest.NewRecorder()

	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "data: ") || !strings.Contains(body, "[DONE]") {
		t.Errorf("expected SSE streaming chunks in body:\n%s", body)
	}
}

func TestServerMetrics(t *testing.T) {
	srv := cudasrv.New(&mockEngine{}, "llama-3.2-1b")
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()

	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", w.Code)
	}

	if !strings.Contains(w.Body.String(), "gpu_requests_total") {
		t.Errorf("expected gpu_requests_total in metrics output:\n%s", w.Body.String())
	}
}
