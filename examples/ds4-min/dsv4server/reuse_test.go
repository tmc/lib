package dsv4server

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

type reuseModel struct {
	mu    sync.Mutex
	evals int
}

func (m *reuseModel) Render(messages []Message) ([]int, error) {
	if len(messages) != 1 || messages[0].Role != "user" {
		return nil, errors.New("reuse model wants one user message")
	}
	prompt := make([]int, len(messages[0].Content))
	for i := range prompt {
		prompt[i] = int(messages[0].Content[i])
	}
	return prompt, nil
}

func (*reuseModel) Decode([]int) (string, error) { return "", nil }

func (m *reuseModel) NewSession() (*dsv4.Session, error) {
	return dsv4.NewSession(reuseBackend{model: m}, 64, dsv4.NewStopSet(1))
}

func (m *reuseModel) IndependentSessions() bool { return true }

type reuseBackend struct{ model *reuseModel }

func (b reuseBackend) NewState(int) (dsv4.State, error) { return &reuseState{}, nil }
func (b reuseBackend) Eval(state dsv4.State, token, position int) error {
	s := state.(*reuseState)
	if position != len(s.tokens) {
		return errors.New("reuse position mismatch")
	}
	s.tokens = append(s.tokens, token)
	b.model.mu.Lock()
	b.model.evals++
	b.model.mu.Unlock()
	return nil
}
func (reuseBackend) Logits(dsv4.State) []float32 { return []float32{0, 1} }
func (reuseBackend) Close() error                { return nil }

type reuseState struct{ tokens []int }

func (s *reuseState) Reset() error {
	s.tokens = nil
	return nil
}
func (s *reuseState) Snapshot() (dsv4.Snapshot, error) { return reuseSnapshot{}, nil }
func (s *reuseState) Restore(dsv4.Snapshot) error      { return nil }
func (s *reuseState) Save(_ io.Writer) error           { return nil }
func (s *reuseState) Load(_ io.Reader) error           { return nil }
func (s *reuseState) Close() error                     { return nil }

type reuseSnapshot struct{}

func (reuseSnapshot) Close() error { return nil }

func TestHandlerSessionReuseContracts(t *testing.T) {
	model := &reuseModel{}
	h, err := New(model, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, prompt string
		want         int
	}{
		{"initial", "abc", 3},
		{"common-prefix continuation after stop", "abcd", 4},
		{"changed trailing text resets", "abd", 7},
		{"continuation after reset reuses prefix", "abdz", 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"test","messages":[{"role":"user","content":"` + tc.prompt + `"}],"max_tokens":1,"temperature":0}`
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
			}
			model.mu.Lock()
			got := model.evals
			model.mu.Unlock()
			if got != tc.want {
				t.Fatalf("Eval count = %d, want %d", got, tc.want)
			}
		})
	}
}
