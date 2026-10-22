package dsv4server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type concurrentModel struct {
	next    atomic.Int32
	entered atomic.Int32
	release chan struct{}
	once    sync.Once
}

func (*concurrentModel) IndependentSessions() bool         { return true }
func (m *concurrentModel) Render([]Message) ([]int, error) { return []int{1}, nil }
func (m *concurrentModel) Decode(ids []int) (string, error) {
	if len(ids) != 1 {
		return "", errors.New("expected one generated token")
	}
	return strconv.Itoa(ids[0]), nil
}
func (m *concurrentModel) NewSession() (*dsv4.Session, error) {
	id := int(m.next.Add(1))
	return dsv4.NewSession(concurrentBackend{model: m, id: id}, 32, dsv4.NewStopSet(99))
}

type concurrentBackend struct {
	model *concurrentModel
	id    int
}

func (b concurrentBackend) NewState(int) (dsv4.State, error) {
	return &concurrentState{model: b.model, id: b.id}, nil
}
func (b concurrentBackend) Eval(state dsv4.State, token, position int) error {
	s := state.(*concurrentState)
	if position == 0 {
		if n := s.model.entered.Add(1); n == 4 {
			s.model.once.Do(func() { close(s.model.release) })
		}
		select {
		case <-s.model.release:
		case <-time.After(2 * time.Second):
			return errors.New("concurrent requests did not acquire four sessions")
		}
	}
	s.tokens = append(s.tokens, token)
	return nil
}
func (b concurrentBackend) Logits(state dsv4.State) []float32 {
	s := state.(*concurrentState)
	logits := make([]float32, s.id+1)
	logits[s.id] = float32(s.id)
	return logits
}
func (concurrentBackend) Close() error { return nil }

type concurrentState struct {
	model  *concurrentModel
	id     int
	tokens []int
}

func (s *concurrentState) Reset() error {
	s.tokens = nil
	return nil
}
func (*concurrentState) Snapshot() (dsv4.Snapshot, error) { return concurrentSnapshot{}, nil }
func (*concurrentState) Restore(dsv4.Snapshot) error      { return nil }
func (*concurrentState) Save(io.Writer) error             { return nil }
func (*concurrentState) Load(io.Reader) error             { return nil }
func (*concurrentState) Close() error                     { return nil }

type concurrentSnapshot struct{}

func (concurrentSnapshot) Close() error { return nil }

func TestHandlerConcurrentIndependentSlots(t *testing.T) {
	model := &concurrentModel{release: make(chan struct{})}
	h, err := New(model, Config{Model: "test", MaxTokens: 1, SessionSlots: 4})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(h)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	results := make(chan string, 4)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0}`))
			if err != nil {
				results <- "request: " + err.Error()
				return
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				results <- "request: " + err.Error()
				return
			}
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body)
			results <- strconv.Itoa(resp.StatusCode) + ":" + string(body)
		}()
	}
	wg.Wait()
	close(results)
	seen := make(map[string]bool)
	for result := range results {
		if !strings.HasPrefix(result, "200:") {
			t.Fatalf("concurrent request = %s", result)
		}
		seen[result] = true
	}
	if len(seen) != 4 {
		t.Fatalf("responses shared mutable state: got %d distinct responses", len(seen))
	}
}

type nonIndependentModel struct{ testModel }

func (nonIndependentModel) IndependentSessions() bool { return false }

func TestHandlerRejectsNonIndependentSessionSlots(t *testing.T) {
	if _, err := New(nonIndependentModel{}, Config{SessionSlots: 4}); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("New error = %v, want %v", err, ErrUnsupported)
	}
}
