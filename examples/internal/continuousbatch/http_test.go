package continuousbatch

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHandlerUnavailable(t *testing.T) {
	w := httptest.NewRecorder()
	new(Handler).ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{}`)))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d", w.Code)
	}
}

func TestHandlerRejectsInvalidBodyBeforeSubmission(t *testing.T) {
	s := newTestScheduler(t, new(fakeBackend), Config{})
	h := &Handler{Scheduler: s}
	for _, body := range []string{`{} {}`, `{} garbage`, `{"prompt":"` + strings.Repeat("x", 1<<20) + `"}`, `{}` + strings.Repeat(" ", 1<<20)} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d", w.Code)
		}
	}
	if got := s.Metrics().Submitted; got != 0 {
		t.Fatalf("submitted = %d", got)
	}
}

type failingStreamWriter struct {
	header             http.Header
	writeErr, flushErr error
	writes, flushes    int
}

func (w *failingStreamWriter) Header() http.Header { return w.header }
func (*failingStreamWriter) WriteHeader(int)       {}
func (w *failingStreamWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writeErr != nil {
		return 0, w.writeErr
	}
	return len(p), nil
}
func (w *failingStreamWriter) Flush()            { w.flushes++ }
func (w *failingStreamWriter) FlushError() error { w.flushes++; return w.flushErr }

func TestStreamFailureCancelsResponse(t *testing.T) {
	for _, tc := range []struct {
		name               string
		writeErr, flushErr error
	}{
		{name: "write", writeErr: errors.New("disconnected")},
		{name: "flush", flushErr: errors.New("disconnected")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			tokens := make(chan Token, 2)
			tokens <- Token{Text: "first"}
			tokens <- Token{Text: "second"}
			// Keep channels open: the handler must stop without awaiting producer shutdown.
			response := &Response{Tokens: tokens, Done: make(chan error), cancel: cancel}
			w := &failingStreamWriter{header: make(http.Header), writeErr: tc.writeErr, flushErr: tc.flushErr}
			done := make(chan struct{})
			go func() {
				new(Handler).stream(w, httptest.NewRequest(http.MethodPost, "/", nil), "id", response)
				close(done)
			}()
			select {
			case <-done:
			case <-time.After(time.Second):
				cancel()
				t.Fatal("handler waited after transport failure")
			}
			if ctx.Err() == nil {
				t.Fatal("response was not canceled")
			}
			if w.writes != 1 {
				t.Fatalf("writes = %d", w.writes)
			}
			if tc.writeErr != nil && w.flushes != 0 {
				t.Fatalf("flushes = %d", w.flushes)
			}
		})
	}
}

func TestStreamCancellationWithoutTokens(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	responseCtx, responseCancel := context.WithCancel(context.Background())
	defer responseCancel()
	r := httptest.NewRequest(http.MethodPost, "/", nil).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		new(Handler).stream(httptest.NewRecorder(), r, "id", &Response{Tokens: make(chan Token), Done: make(chan error), cancel: responseCancel})
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("handler blocked on tokens after cancellation")
	}
	if responseCtx.Err() == nil {
		t.Fatal("response not canceled")
	}
}
