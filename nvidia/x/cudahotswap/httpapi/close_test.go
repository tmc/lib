package httpapi

import (
	"context"
	"errors"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type blockedWatchService struct{ fakeService }

func (*blockedWatchService) Watch(ctx context.Context, _ cudahotswap.WatchConfig, emit func(string, *cudahotswap.PushResult, error) error) error {
	return emit("kernel", &cudahotswap.PushResult{Report: cudahotswap.Report{Outcome: cudahotswap.OutcomeOK}}, nil)
}

type blockedResponseWriter struct {
	header           http.Header
	entered, release chan struct{}
	once             sync.Once
}

func (w *blockedResponseWriter) Header() http.Header { return w.header }
func (w *blockedResponseWriter) WriteHeader(int)     {}
func (w *blockedResponseWriter) Flush()              {}
func (w *blockedResponseWriter) Write(b []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	return 0, errors.New("owned connection released")
}
func TestCloseCancelsBlockedTransportWrite(t *testing.T) {
	h, e := New(&blockedWatchService{}, Resources{})
	if e != nil {
		t.Fatal(e)
	}
	w := &blockedResponseWriter{header: make(http.Header), entered: make(chan struct{}), release: make(chan struct{})}
	serveDone := make(chan struct{})
	go func() {
		h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/watch", strings.NewReader(`{"label":"own"}`)))
		close(serveDone)
	}()
	select {
	case <-w.entered:
	case <-time.After(time.Second):
		t.Fatal("write never entered")
	}
	ctx, c := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer c()
	closed := make(chan error, 1)
	go func() { closed <- h.Close(ctx) }()
	select {
	case err := <-closed:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Close = %v", err)
		}
	case <-time.After(200 * time.Millisecond):
		t.Error("Close ignored cancellation while transport Write blocks")
	}
	h.mu.Lock()
	active := h.active
	h.mu.Unlock()
	if active != 1 {
		t.Errorf("lost active request: %d", active)
	}
	service := h.service.(*blockedWatchService)
	service.mu.Lock()
	for _, call := range service.calls {
		if call == "Close" {
			t.Error("service prematurely closed")
		}
	}
	service.mu.Unlock()
	close(w.release)
	select {
	case <-serveDone:
	case <-time.After(time.Second):
		t.Fatal("handler failed to retire after release")
	}
	if err := h.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if len(service.calls) == 0 || service.calls[len(service.calls)-1] != "Close" {
		t.Error("retry failed to close service")
	}
}

func TestCloseRetainsOwnersAndSerializesRetry(t *testing.T) {
	service := new(fakeService)
	h, err := New(service, Resources{})
	if err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	h.handles["pending"] = retained{wait: func(context.Context) error { once.Do(func() { close(entered) }); <-release; return nil }}
	done := make(chan error, 1)
	go func() { done <- h.Close(context.Background()) }()
	<-entered
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := h.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("cancel = %v", err)
	}
	ctx, cancel = context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := h.Close(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("deadline = %v", err)
	}
	h.mu.Lock()
	_, kept := h.handles["pending"]
	h.mu.Unlock()
	if !kept {
		t.Error("lost owner")
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/list", strings.NewReader(`{"label":"late"}`)))
	if !strings.Contains(w.Body.String(), "handler is closed") {
		t.Error("new request admitted")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	n := len(h.handles)
	h.mu.Unlock()
	if n != 0 {
		t.Errorf("remaining owners %d", n)
	}
}
