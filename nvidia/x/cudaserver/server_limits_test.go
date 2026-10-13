package cudaserver_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaserver"
)

type engineFunc func(context.Context, cudaserver.ChatCompletionRequest, chan<- string) error

func (f engineFunc) Generate(ctx context.Context, req cudaserver.ChatCompletionRequest, tokens chan<- string) error {
	return f(ctx, req, tokens)
}

func TestRequestLimits(t *testing.T) {
	for _, tt := range []struct {
		name, body string
		want       int
	}{
		{"trailing value", `{}` + `{}`, http.StatusBadRequest},
		{"trailing junk", `{}x`, http.StatusBadRequest},
		{"oversized JSON", `{"model":"` + strings.Repeat("x", 1<<20) + `"}`, http.StatusRequestEntityTooLarge},
		{"oversized whitespace", `{}` + strings.Repeat(" ", 1<<20), http.StatusRequestEntityTooLarge},
		{"one value", `{} `, http.StatusOK},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			engine := engineFunc(func(context.Context, cudaserver.ChatCompletionRequest, chan<- string) error { calls++; return nil })
			w := httptest.NewRecorder()
			cudaserver.New(engine, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tt.body)))
			if w.Code != tt.want {
				t.Fatalf("status=%d want=%d: %s", w.Code, tt.want, w.Body.String())
			}
			if tt.want != http.StatusOK && calls != 0 {
				t.Fatalf("engine called %d times", calls)
			}
		})
	}
}

func TestNilEngineRefused(t *testing.T) {
	w := httptest.NewRecorder()
	cudaserver.New(nil, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d", w.Code)
	}
}

func TestStreamGenerationFailure(t *testing.T) {
	engine := engineFunc(func(context.Context, cudaserver.ChatCompletionRequest, chan<- string) error {
		return errors.New("private upstream credentials")
	})
	w := httptest.NewRecorder()
	cudaserver.New(engine, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"stream":true}`)))
	body := w.Body.String()
	if !strings.Contains(body, "event: error") || strings.Contains(body, "[DONE]") || strings.Contains(body, "private") || strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("false successful or leaking stream: %s", body)
	}
}

type failedWriter struct{ header http.Header }

func (w *failedWriter) Header() http.Header     { return w.header }
func (*failedWriter) WriteHeader(int)           {}
func (*failedWriter) Write([]byte) (int, error) { return 0, errors.New("disconnected") }
func (*failedWriter) Flush()                    {}

func TestStreamWriteFailureCancelsEngine(t *testing.T) {
	stopped := make(chan struct{})
	engine := engineFunc(func(ctx context.Context, _ cudaserver.ChatCompletionRequest, tokens chan<- string) error {
		defer close(stopped)
		select {
		case tokens <- "first":
		case <-ctx.Done():
			return ctx.Err()
		}
		<-ctx.Done()
		return ctx.Err()
	})
	w := &failedWriter{header: make(http.Header)}
	cudaserver.New(engine, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"stream":true}`)))
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("engine did not receive cancellation")
	}
}

type failedFlusher struct{ failedWriter }

func (*failedFlusher) Write(p []byte) (int, error) { return len(p), nil }
func (*failedFlusher) FlushError() error           { return errors.New("flush disconnected") }

func TestStreamFlushFailureCancelsEngine(t *testing.T) {
	stopped := make(chan struct{})
	engine := engineFunc(func(ctx context.Context, _ cudaserver.ChatCompletionRequest, tokens chan<- string) error {
		defer close(stopped)
		select {
		case tokens <- "first":
		case <-ctx.Done():
			return ctx.Err()
		}
		<-ctx.Done()
		return ctx.Err()
	})
	w := &failedFlusher{failedWriter{header: make(http.Header)}}
	cudaserver.New(engine, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"stream":true}`)))
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("engine did not receive cancellation")
	}
}
