package dsv4server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
	"github.com/tmc/lib/examples/ds4-min/sampler"
)

type testModel struct{}

type historyTestModel struct{ testModel }

type defaultThinkingTestModel struct{ testModel }

type thinkingTestModel struct {
	testModel
	mode ThinkMode
}

type taggedOutputModel struct {
	testModel
	text string
}

type streamingThinkingModel struct {
	testModel
	parts []string
	pos   int
}

type defaultStreamingThinkingModel struct{ streamingThinkingModel }

func (*defaultStreamingThinkingModel) DefaultThinkMode() ThinkMode { return ThinkHigh }

func (m *streamingThinkingModel) RenderThinking(messages []Message, mode ThinkMode) ([]int, error) {
	m.pos = 0
	return m.Render(messages)
}

func (m *streamingThinkingModel) Decode([]int) (string, error) {
	if m.pos >= len(m.parts) {
		return "", errors.New("too many decode calls")
	}
	part := m.parts[m.pos]
	m.pos++
	return part, nil
}

func (m *taggedOutputModel) Decode([]int) (string, error) { return m.text, nil }

func (m *thinkingTestModel) RenderThinking(messages []Message, mode ThinkMode) ([]int, error) {
	m.mode = mode
	return m.Render(messages)
}

type contextModel struct {
	testModel
	n int
}

func (m contextModel) Render([]Message) ([]int, error) { return make([]int, m.n), nil }

type completionPromptModel struct {
	testModel
	got string
}

func (m *completionPromptModel) Render(messages []Message) ([]int, error) {
	if len(messages) != 2 || messages[0].Role != "system" || messages[0].Content != "You are a helpful assistant" || messages[1].Role != "user" {
		return nil, errors.New("want default system and user messages")
	}
	m.got = messages[1].Content
	return []int{1}, nil
}

type toolAwareTestModel struct {
	testModel
	calls  int
	schema []dsv4tools.Schema
}

type toolTaggedOutputModel struct {
	toolAwareTestModel
	text string
}

func (m *toolTaggedOutputModel) Decode([]int) (string, error) { return m.text, nil }

func (m *toolAwareTestModel) RenderTools(messages []Message, schema []dsv4tools.Schema) ([]int, error) {
	m.calls++
	m.schema = append([]dsv4tools.Schema(nil), schema...)
	return m.Render(messages)
}

func (testModel) IndependentSessions() bool { return true }

func (testModel) Render(messages []Message) ([]int, error) {
	if len(messages) == 2 && messages[0].Role == "system" && messages[0].Content == "You are a helpful assistant" {
		messages = messages[1:]
	}
	if len(messages) != 1 || messages[0].Role != "user" {
		return nil, errors.New("test model wants one user message")
	}
	return []int{1}, nil
}

func (historyTestModel) Render(messages []Message) ([]int, error) {
	if len(messages) < 2 {
		return nil, errors.New("history test model wants a conversation")
	}
	return []int{1}, nil
}

func (defaultThinkingTestModel) DefaultThinkMode() ThinkMode { return ThinkHigh }

func (testModel) Decode(ids []int) (string, error) {
	var b strings.Builder
	for _, id := range ids {
		b.WriteByte(byte('a' + id))
	}
	return b.String(), nil
}

func (testModel) NewSession() (*dsv4.Session, error) {
	return dsv4.NewSession(testBackend{}, 16, dsv4.NewStopSet(9))
}

type testBackend struct{}

func (testBackend) NewState(int) (dsv4.State, error) { return &testState{}, nil }
func (testBackend) Eval(state dsv4.State, token, position int) error {
	s := state.(*testState)
	if position != len(s.tokens) {
		return errors.New("position mismatch")
	}
	s.tokens = append(s.tokens, token)
	return nil
}
func (testBackend) Logits(state dsv4.State) []float32 {
	// Token 2 is the deterministic next token until the generated sequence
	// reaches the test limit. The handler therefore exercises Accept and usage.
	return []float32{0, 1, 2}
}
func (testBackend) Close() error { return nil }

type testState struct {
	tokens []int
	closed bool
}

func (s *testState) Reset() error {
	if s.closed {
		return dsv4.ErrClosed
	}
	s.tokens = nil
	return nil
}
func (s *testState) Snapshot() (dsv4.Snapshot, error) { return testSnapshot{}, nil }
func (s *testState) Restore(dsv4.Snapshot) error      { return nil }
func (s *testState) Save(io.Writer) error             { return nil }
func (s *testState) Load(io.Reader) error             { return nil }
func (s *testState) Close() error {
	s.closed = true
	return nil
}

type testSnapshot struct{}

func (testSnapshot) Close() error { return nil }

func TestHandlerCompletionAndModelList(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("completion status = %d, body %s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), `"content":"cc"`) || !strings.Contains(rec.Body.String(), `"prompt_tokens":1`) {
		t.Fatalf("completion body = %s", rec.Body)
	}

	req = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"id":"test"`) {
		t.Fatalf("models response = %d %s", rec.Code, rec.Body)
	}
	req = httptest.NewRequest(http.MethodGet, "/v1/models/test", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"id":"test"`) {
		t.Fatalf("model response = %d %s", rec.Code, rec.Body)
	}
	var model map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &model); err != nil {
		t.Fatal(err)
	}
	if model["context_length"] != float64(32768) {
		t.Fatalf("context_length = %v", model["context_length"])
	}
	provider, ok := model["top_provider"].(map[string]any)
	if !ok || provider["max_completion_tokens"] != float64(2) {
		t.Fatalf("top_provider = %v", model["top_provider"])
	}
	req = httptest.NewRequest(http.MethodGet, "/v1/models/deepseek-chat", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("alias model response = %d, want 404", rec.Code)
	}
}

func TestHandlerDefaultBodyLimit(t *testing.T) {
	h, err := New(testModel{}, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if h.cfg.MaxBody != 64<<20 {
		t.Fatalf("default body limit = %d, want %d", h.cfg.MaxBody, 64<<20)
	}
}

func TestHandlerCloseRejectsNewRequests(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}],"max_tokens":1}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("closed request status = %d, want %d; body=%s", rec.Code, http.StatusServiceUnavailable, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "server shutting down") {
		t.Fatalf("closed request body = %s", rec.Body)
	}
}

func TestStopSequencesValidationAndCut(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want string
		ok   bool
	}{
		{"absent", nil, "", true},
		{"string", "STOP", "answer ", true},
		{"array", []any{"END", "STOP"}, "answer ", true},
		{"empty", "", "", false},
		{"wrong type", 1, "", false},
		{"too many", []any{"a", "b", "c", "d", "e"}, "", false},
		{"empty array", []any{}, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stops, err := parseStop(tt.in)
			if (err == nil) != tt.ok {
				t.Fatalf("parseStop(%v) error = %v, want success %v", tt.in, err, tt.ok)
			}
			if err == nil && tt.want != "" {
				got, stopped := stops.cut("answer STOP tail")
				if !stopped || got != tt.want {
					t.Fatalf("cut = %q, %v; want %q, true", got, stopped, tt.want)
				}
			}
		})
	}
	stops, _ := parseStop([]any{"é!"})
	var pending string
	if got, stopped := stops.add(&pending, "é"); stopped || got != "" {
		t.Fatalf("partial UTF-8 stop = %q, %v", got, stopped)
	}
	if got, stopped := stops.add(&pending, "!"); !stopped || got != "" {
		t.Fatalf("split stop = %q, %v", got, stopped)
	}
}

func TestStopSequencesHTTP(t *testing.T) {
	for _, tt := range []struct {
		name string
		path string
		body string
		want string
	}{
		{"chat", "/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0,"stop":"STOP"}`, `"content":"answer "`},
		{"completion", "/v1/completions", `{"prompt":"x","max_tokens":1,"temperature":0,"stop":["STOP"]}`, `"text":"answer "`},
		{"messages", "/v1/messages", `{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0,"stop_sequences":["STOP"]}`, `"text":"answer "`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h, err := New(&taggedOutputModel{text: "answer STOP tail"}, Config{Model: "test", MaxTokens: 1})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), tt.want) || strings.Contains(rec.Body.String(), "STOP") {
				t.Fatalf("response = %d %s", rec.Code, rec.Body)
			}
		})
	}
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"x"}],"stop":true}`,
		`{"messages":[{"role":"user","content":"x"}],"stop":["a","b","c","d","e"]}`,
	} {
		h, _ := New(testModel{}, Config{Model: "test", MaxTokens: 1})
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("invalid stop status = %d, body %s", rec.Code, rec.Body)
		}
	}
}

func TestHandlerSplitsThinkingOutput(t *testing.T) {
	model := &taggedOutputModel{text: "<think>private work</think>public answer"}
	h, err := New(model, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"reasoning_content":"private work"`) || !strings.Contains(body, `"content":"public answer"`) || strings.Contains(body, `<think>`) {
		t.Fatalf("split response = %s", body)
	}

	req = httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"type":"thinking"`) || !strings.Contains(rec.Body.String(), `"type":"text"`) {
		t.Fatalf("anthropic response = %d %s", rec.Code, rec.Body)
	}
}

func TestHandlerRejectsMalformedThinkingOutput(t *testing.T) {
	model := &taggedOutputModel{text: "<think>unterminated"}
	h, err := New(model, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "unterminated") {
		t.Fatalf("malformed response = %d %s", rec.Code, rec.Body)
	}
}

func TestHandlerThinkingStreaming(t *testing.T) {
	tests := []struct {
		name string
		path string
		body string
		want []string
	}{
		{name: "chat", path: "/v1/chat/completions", body: `{"messages":[{"role":"user","content":"x"}],"max_tokens":3,"stream":true,"thinking":{"type":"enabled"}}`, want: []string{`"reasoning_content":"work"`, `"content":"answer"`}},
		{name: "responses", path: "/v1/responses", body: `{"input":"x","max_output_tokens":3,"stream":true,"reasoning":{"effort":"low"}}`, want: []string{"response.reasoning_summary_text.delta", `"delta":"answer"`}},
		{name: "messages", path: "/v1/messages", body: `{"messages":[{"role":"user","content":"x"}],"max_tokens":3,"stream":true,"thinking":{"type":"enabled"}}`, want: []string{"thinking_delta", "text_delta", `"thinking":"work"`, `"text":"answer"`}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := &streamingThinkingModel{parts: []string{"<th", "ink>work</", "think>answer"}}
			h, err := New(model, Config{Model: "test", MaxTokens: 3})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
			body := rec.Body.String()
			for _, want := range test.want {
				if !strings.Contains(body, want) {
					t.Errorf("stream missing %q: %s", want, body)
				}
			}
			if strings.Contains(body, "event: error") || strings.Contains(body, "<think>") || strings.Contains(body, "</think>") {
				t.Fatalf("malformed thinking stream: %s", body)
			}
		})
	}
}

func TestHandlerThinkingStreamingCoalescesReasoningDelta(t *testing.T) {
	model := &streamingThinkingModel{parts: []string{"<think>1", ".", "</think>answer"}}
	h, err := New(model, Config{Model: "test", MaxTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":3,"stream":true,"thinking":{"type":"enabled"}}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if got := strings.Count(body, "data: "); got != 5 {
		t.Fatalf("stream data frames = %d, want role, reasoning, content, finish, done: %s", got, body)
	}
	if !strings.Contains(body, `"reasoning_content":"1."`) || strings.Contains(body, `"reasoning_content":"1"`) {
		t.Fatalf("reasoning delta was not coalesced: %s", body)
	}
	if !strings.Contains(body, `"content":"answer"`) || !strings.Contains(body, `"finish_reason":"length"`) {
		t.Fatalf("terminal stream fields missing: %s", body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("stream did not terminate with DONE: %s", body)
	}
}

func TestHandlerDefaultThinkingStreamingCoalescesReasoningDelta(t *testing.T) {
	for _, test := range []struct {
		name        string
		parts       []string
		wantContent bool
	}{
		{"tagged", []string{"<think>1", ".", "</think>answer"}, true},
		{"untagged", []string{"1", ".", ""}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := &defaultStreamingThinkingModel{streamingThinkingModel{parts: test.parts}}
			h, err := New(model, Config{Model: "test", MaxTokens: 3})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"max_tokens":3,"stream":true}`))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			body := rec.Body.String()
			wantFrames := 4
			if test.wantContent {
				wantFrames++
			}
			if rec.Code != http.StatusOK || strings.Count(body, "data: ") != wantFrames || strings.Count(body, `"reasoning_content":"1."`) != 1 || strings.Contains(body, `"reasoning_content":"1"`) || strings.Contains(body, "<think>") {
				t.Fatalf("default thinking stream = %d %s", rec.Code, body)
			}
			if strings.Contains(body, `"content":"answer"`) != test.wantContent || !strings.Contains(body, `"finish_reason":"length"`) || !strings.HasSuffix(body, "data: [DONE]\n\n") {
				t.Fatalf("default thinking terminal fields = %s", body)
			}
		})
	}
}

func TestSamplingRequestValidationAndReplay(t *testing.T) {
	_, got, _, err := (samplingRequest{}).sampler()
	if err != nil {
		t.Fatal(err)
	}
	if got != sampler.Default {
		t.Fatalf("omitted sampling = %+v, want %+v", got, sampler.Default)
	}
	_, got, _, err = (samplingRequest{Temperature: ptr(float32(0))}).sampler()
	if err != nil {
		t.Fatal(err)
	}
	if got.Temperature != 0 || got.TopP != sampler.Default.TopP || got.MinP != sampler.Default.MinP {
		t.Fatalf("explicit zero temperature = %+v, want temperature 0 with default filters", got)
	}

	cases := []struct {
		name string
		req  samplingRequest
		want string
	}{
		{"temperature", samplingRequest{Temperature: ptr(float32(0.7))}, ""},
		{"top-p", samplingRequest{TopP: ptr(float32(0.9))}, ""},
		{"top-k", samplingRequest{TopK: ptr(4)}, ""},
		{"seed-needs-sampling", samplingRequest{Seed: ptr(uint64(9))}, "seed requires"},
		{"zero-seed", samplingRequest{Temperature: ptr(float32(1)), Seed: ptr(uint64(0))}, "seed must"},
		{"bad-top-p", samplingRequest{TopP: ptr(float32(2))}, "top_p"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a, p, r, err := tc.req.sampler()
			if tc.want != "" {
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("error = %v, want %q", err, tc.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if a == nil || r == nil {
				t.Fatal("sampler returned nil state")
			}
			if tc.req.Temperature != nil && p.Temperature != *tc.req.Temperature {
				t.Fatalf("temperature = %v", p.Temperature)
			}
		})
	}

	req := samplingRequest{Temperature: ptr(float32(1)), TopP: ptr(float32(1)), Seed: ptr(uint64(17))}
	a, p, r, err := req.sampler()
	if err != nil {
		t.Fatal(err)
	}
	first := sampleToken([]float32{0, 0, 1, 0}, a, p, r)
	// Rebuild the sampler and RNG so an explicit seed is a reproducible stream.
	b, q, replayRNG, err := req.sampler()
	if err != nil {
		t.Fatal(err)
	}
	second := sampleToken([]float32{0, 0, 1, 0}, b, q, replayRNG)
	if first != second {
		t.Fatalf("seeded samples differ: %d != %d", first, second)
	}
}

func TestSamplingControlsReachEveryEndpoint(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, path, body string
	}{
		{"chat", "/v1/chat/completions", `{"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0.7,"top_p":1,"min_p":0,"top_k":2,"seed":17}`},
		{"completion", "/v1/completions", `{"model":"test","prompt":"x","max_tokens":1,"temperature":0.7,"top_p":1,"min_p":0,"top_k":2,"seed":17}`},
		{"responses", "/v1/responses", `{"model":"test","input":"x","max_output_tokens":1,"temperature":0.7,"seed":17}`},
		{"messages", "/v1/messages", `{"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0.7,"top_p":1,"top_k":2,"seed":17}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
			}
		})
	}
}

func TestSamplingControlsFailClosed(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"x"}],"temperature":-1}`,
		`{"messages":[{"role":"user","content":"x"}],"top_p":2}`,
		`{"messages":[{"role":"user","content":"x"}],"top_k":1025}`,
		`{"messages":[{"role":"user","content":"x"}],"temperature":1,"seed":0}`,
		`{"messages":[{"role":"user","content":"x"}],"temperature":1,"thinking":{"type":"enabled"}}`,
	} {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, want 400", body, rec.Code)
		}
	}
}

func TestThinkingModeDispatchesOnlyThroughOptInModel(t *testing.T) {
	model := &thinkingTestModel{}
	h, err := New(model, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		body string
		want ThinkMode
	}{
		{"object", `{"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled"}}`, ThinkHigh},
		{"effort", `{"messages":[{"role":"user","content":"x"}],"reasoning_effort":"max"}`, ThinkMax},
		{"alias", `{"messages":[{"role":"user","content":"x"}],"think":true}`, ThinkHigh},
		{"disabled", `{"messages":[{"role":"user","content":"x"}],"thinking":{"type":"disabled"}}`, ThinkNone},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
			}
			if model.mode != tc.want {
				t.Fatalf("mode = %d, want %d", model.mode, tc.want)
			}
		})
	}
	if _, _, err := requestThinkMode(nil, true, "high"); err == nil {
		t.Fatal("conflicting thinking controls unexpectedly accepted")
	}
	legacy, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled"}}`))
	rec := httptest.NewRecorder()
	legacy.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "thinking is unsupported") {
		t.Fatalf("legacy thinking response = %d %s", rec.Code, rec.Body)
	}
}

func ptr[T any](v T) *T { return &v }

func TestCompletionsAcceptsFirstStringInPromptArray(t *testing.T) {
	model := &completionPromptModel{}
	h, err := New(model, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":["first",{"ignored":true}],"max_tokens":1,"temperature":0}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if model.got != "first" {
		t.Fatalf("rendered prompt = %q, want first array element", model.got)
	}
}

func TestCompletionsRejectsEmptyPromptArray(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":[],"max_tokens":1}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "missing prompt") {
		t.Fatalf("status = %d, body = %s, want missing prompt", rec.Code, rec.Body)
	}
}

func TestHandlerRejectsUnknownRequestModels(t *testing.T) {
	cases := []struct {
		name, path, body string
	}{
		{"chat", "/v1/chat/completions", `{"model":"not-loaded","messages":[{"role":"user","content":"hello"}]}`},
		{"completions", "/v1/completions", `{"model":"not-loaded","prompt":"hello"}`},
		{"responses", "/v1/responses", `{"model":"not-loaded","input":"hello"}`},
		{"messages", "/v1/messages", `{"model":"not-loaded","messages":[{"role":"user","content":"hello"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "unknown model") {
				t.Fatalf("status=%d body=%s, want unknown-model rejection", rec.Code, rec.Body)
			}
		})
	}
}

func TestHandlerAcceptsDeepSeekAliases(t *testing.T) {
	for _, model := range []string{"deepseek-chat", "deepseek-reasoner"} {
		h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
		if err != nil {
			t.Fatal(err)
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"`+model+`","messages":[{"role":"user","content":"hello"}]}`))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"model":"`+model+`"`) {
			t.Fatalf("model=%s status=%d body=%s, want accepted alias", model, rec.Code, rec.Body)
		}
	}
}

func TestHandlerModelCatalogAliases(t *testing.T) {
	h, err := New(testModel{}, Config{
		Model:                    "deepseek-v4-flash",
		ModelAliases:             []string{"deepseek-v4-pro"},
		ModelCreated:             1767225600,
		ModelOwner:               "ds4.c",
		ModelName:                "DeepSeek V4 Flash",
		ModelMaxCompletion:       32768,
		ModelSupportedParameters: []string{"tools", "temperature"},
		ContextLength:            32768,
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("models status = %d, body = %s", rec.Code, rec.Body)
	}
	var got struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 2 || got.Data[0]["id"] != "deepseek-v4-flash" || got.Data[1]["id"] != "deepseek-v4-pro" {
		t.Fatalf("model catalog = %#v", got.Data)
	}
	for _, model := range got.Data {
		if model["owned_by"] != "ds4.c" || model["name"] != "DeepSeek V4 Flash" || model["created"] != float64(1767225600) {
			t.Fatalf("model metadata = %#v", model)
		}
		provider := model["top_provider"].(map[string]any)
		if provider["max_completion_tokens"] != float64(32768) {
			t.Fatalf("max completion = %#v", provider)
		}
	}
	for _, id := range []string{"deepseek-v4-flash", "deepseek-v4-pro"} {
		req = httptest.NewRequest(http.MethodGet, "/v1/models/"+id, nil)
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("model %s status = %d, body = %s", id, rec.Code, rec.Body)
		}
	}
	for _, id := range []string{"deepseek-v4-flash", "deepseek-v4-pro"} {
		req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"`+id+`","messages":[{"role":"user","content":"hello"}],"max_tokens":1}`))
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("request %s status = %d, body = %s", id, rec.Code, rec.Body)
		}
	}
}

func TestChatToolChoiceNoneDisablesToolPrompt(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup"}}],"tool_choice":"none","max_tokens":1}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s, want tool-disabled request to use ordinary model path", rec.Code, rec.Body)
	}
}

func TestChatRejectsUnsupportedToolChoice(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}],"tool_choice":"required"}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "unsupported tool_choice") {
		t.Fatalf("status=%d body=%s, want unsupported tool_choice", rec.Code, rec.Body)
	}
}

func TestHandlerContextLimitErrors(t *testing.T) {
	cases := []struct {
		name, path, body, param string
	}{
		{"chat", "/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}]}`, "messages"},
		{"completions", "/v1/completions", `{"prompt":"x"}`, "prompt"},
		{"responses", "/v1/responses", `{"input":"x"}`, "input"},
		{"messages", "/v1/messages", `{"messages":[{"role":"user","content":"x"}]}`, "messages"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, err := New(contextModel{n: 3}, Config{Model: "test", MaxTokens: 1, ContextLength: 2})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
			}
			var response struct {
				Error struct {
					Message string `json:"message"`
					Type    string `json:"type"`
					Param   string `json:"param"`
					Code    string `json:"code"`
					Prompt  int    `json:"n_prompt_tokens"`
					Ctx     int    `json:"n_ctx"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if response.Error.Type != "invalid_request_error" || response.Error.Code != "context_length_exceeded" || response.Error.Param != tc.param || response.Error.Prompt != 3 || response.Error.Ctx != 2 {
				t.Fatalf("error = %+v", response.Error)
			}
		})
	}
}

func TestHandlerAliasesAndCORS(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1, CORS: true})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"deepseek-chat","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"model":"deepseek-chat"`) {
		t.Fatalf("alias response = %d %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("cors origin = %q", got)
	}
	req = httptest.NewRequest(http.MethodOptions, "/v1/anything", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent || rec.Body.Len() != 0 {
		t.Fatalf("options = %d %q", rec.Code, rec.Body.String())
	}
}

func TestHandlerStreamingCompletion(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0,"stream":true}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream status = %d, body %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("stream content type = %q", got)
	}
	body := rec.Body.String()
	for _, want := range []string{"\"role\":\"assistant\"", "\"content\":\"c\"", "data: [DONE]"} {
		if !strings.Contains(body, want) {
			t.Errorf("stream body missing %q: %s", want, body)
		}
	}
	if got := strings.Count(body, "data: "); got != 5 {
		t.Fatalf("stream data frames = %d, want role + 2 tokens + finish + done", got)
	}
}

func TestHandlerStreamingUsageAndStreamOptions(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream status = %d, body %s", rec.Code, rec.Body)
	}
	if frames := strings.Count(rec.Body.String(), "data: "); frames != 5 { // role, token, finish, usage, done
		t.Fatalf("frames = %d, body %s", frames, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), `"choices":[]`) || !strings.Contains(rec.Body.String(), `"completion_tokens":1`) {
		t.Fatalf("usage frame missing: %s", rec.Body)
	}

	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream_options":{"include_usage":true}}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("non-stream stream_options status = %d, body %s", rec.Code, rec.Body)
	}
}

func TestHandlerMessagesStreamingContract(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0,"stream":true}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status/content type = %d/%q, body=%s", rec.Code, rec.Header().Get("Content-Type"), rec.Body)
	}
	body := rec.Body.String()
	for _, want := range []string{"event: message_start", "event: content_block_start", "event: content_block_delta", "text_delta", "event: content_block_stop", "event: message_delta", "event: message_stop"} {
		if !strings.Contains(body, want) {
			t.Errorf("stream missing %q: %s", want, body)
		}
	}
	if strings.Contains(body, "event: error") {
		t.Fatalf("unexpected stream error: %s", body)
	}

	req = httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hello"}],"tools":[],"max_tokens":1,"temperature":0,"stream":true}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("empty tools should preserve text stream, status=%d body=%s", rec.Code, rec.Body)
	}
}

func TestHandlerResponsesStreamingContract(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test","input":"hello","max_output_tokens":2,"temperature":0,"stream":true}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status/content type = %d/%q, body=%s", rec.Code, rec.Header().Get("Content-Type"), rec.Body)
	}
	body := rec.Body.String()
	for _, want := range []string{"event: response.created", "event: response.output_item.added", "event: response.content_part.added", "event: response.output_text.delta", "event: response.output_text.done", "event: response.content_part.done", "event: response.output_item.done", "event: response.incomplete", `"status":"incomplete"`} {
		if !strings.Contains(body, want) {
			t.Errorf("stream missing %q: %s", want, body)
		}
	}
	if strings.Contains(body, "event: response.failed") {
		t.Fatalf("unexpected stream failure: %s", body)
	}

	req = httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test","input":"hello","max_output_tokens":1,"stream":true,"tools":[{"type":"function","function":{"name":"f"}}]}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "streaming tool calls are unsupported") {
		t.Fatalf("streamed tools status/body = %d/%s", rec.Code, rec.Body)
	}
}

func TestHandlerLegacyCompletion(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"model":"test","prompt":"hello","max_tokens":2,"temperature":0}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("completion status = %d, body %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"object":"text_completion"`) || !strings.Contains(body, `"text":"cc"`) {
		t.Fatalf("completion body = %s", body)
	}
}

func TestHandlerResponsesAndMessages(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		path string
		body string
		want []string
	}{
		{"responses", "/v1/responses", `{"model":"test","input":"hello","max_output_tokens":2,"temperature":0}`, []string{`"object":"response"`, `"type":"output_text"`, `"text":"cc"`}},
		{"messages", "/v1/messages", `{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0}`, []string{`"type":"message"`, `"role":"assistant"`, `"text":"cc"`}},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
			}
			for _, want := range test.want {
				if !strings.Contains(rec.Body.String(), want) {
					t.Errorf("body missing %q: %s", want, rec.Body)
				}
			}
		})
	}
}

func TestHandlerUsageMetadata(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		path  string
		body  string
		check func(t *testing.T, usage map[string]any)
	}{
		{
			name: "chat",
			path: "/v1/chat/completions",
			body: `{"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":1}`,
			check: func(t *testing.T, usage map[string]any) {
				details, ok := usage["prompt_tokens_details"].(map[string]any)
				if !ok || details["cached_tokens"] != float64(0) || details["cache_write_tokens"] != float64(1) {
					t.Fatalf("prompt token details = %#v", usage["prompt_tokens_details"])
				}
			},
		},
		{
			name: "responses",
			path: "/v1/responses",
			body: `{"model":"test","input":"x","max_output_tokens":1}`,
			check: func(t *testing.T, usage map[string]any) {
				details, ok := usage["input_tokens_details"].(map[string]any)
				if !ok || details["cached_tokens"] != float64(0) {
					t.Fatalf("input token details = %#v", usage["input_tokens_details"])
				}
				if _, ok := usage["output_tokens_details"].(map[string]any); !ok {
					t.Fatalf("missing output token details: %#v", usage)
				}
			},
		},
		{
			name: "messages",
			path: "/v1/messages",
			body: `{"model":"test","messages":[{"role":"user","content":"x"}],"max_tokens":1}`,
			check: func(t *testing.T, usage map[string]any) {
				if usage["cache_read_input_tokens"] != float64(0) || usage["cache_creation_input_tokens"] != float64(1) {
					t.Fatalf("cache token details = %#v", usage)
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status/body = %d/%s", rec.Code, rec.Body)
			}
			var body map[string]any
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			usage, ok := body["usage"].(map[string]any)
			if !ok {
				t.Fatalf("missing usage: %s", rec.Body)
			}
			test.check(t, usage)
		})
	}
}

func TestHandlerRejectsUnsupportedFields(t *testing.T) {
	h, err := New(historyTestModel{}, Config{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"x"}],"unknown":1}`,
	} {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, want 400", body, rec.Code)
		}
	}
}

func TestHandlerAcceptsAssistantHistory(t *testing.T) {
	h, err := New(historyTestModel{}, Config{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		body string
	}{
		{"assistant text", `{"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"two"},{"role":"user","content":"three"}]}`},
		{"assistant empty text", `{"messages":[{"role":"user","content":"one"},{"role":"assistant","content":""},{"role":"user","content":"three"}]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(test.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
			}
		})
	}
}

func TestClassifyDefaultThinkingOutput(t *testing.T) {
	defaultHandler, err := New(defaultThinkingTestModel{}, Config{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	plainHandler, err := New(testModel{}, Config{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name      string
		h         *Handler
		text      string
		requested ThinkMode
		explicit  bool
		reasoning string
		content   string
		present   bool
	}{
		{"default unmarked", defaultHandler, "1.", ThinkNone, false, "1.", "", true},
		{"default marked", defaultHandler, "<think>work</think>answer", ThinkNone, false, "work", "answer", true},
		{"explicit none", defaultHandler, "1.", ThinkNone, true, "", "1.", false},
		{"plain model", plainHandler, "1.", ThinkNone, false, "", "1.", false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.h.classifyThinkingOutput(test.text, test.requested, test.explicit)
			if err != nil {
				t.Fatal(err)
			}
			if got.reasoning != test.reasoning || got.content != test.content || got.present != test.present {
				t.Fatalf("output = %+v, want reasoning=%q content=%q present=%v", got, test.reasoning, test.content, test.present)
			}
		})
	}
}

func TestHandlerRejectsToolHistoryWithoutTools(t *testing.T) {
	h, err := New(testModel{}, Config{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"one"},{"role":"tool","tool_call_id":"call_1","content":"two"}]}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "unsupported message role") {
		t.Fatalf("status/body = %d %s", rec.Code, rec.Body)
	}
}

func TestHandlerRejectsTrailingJSON(t *testing.T) {
	h, err := New(testModel{}, Config{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		path string
		body string
	}{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}]} {"messages":[]}`},
		{"/v1/completions", `{"prompt":"x"} {"prompt":"y"}`},
		{"/v1/responses", `{"input":"x"} {"input":"y"}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"x"}]} {"messages":[]}`},
	} {
		t.Run(test.path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "invalid JSON request") {
				t.Fatalf("status/body = %d %s, want trailing JSON rejection", rec.Code, rec.Body)
			}
		})
	}
}

func TestHandlerRejectsUnknownEndpoint(t *testing.T) {
	h, err := New(testModel{}, Config{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/completions", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
}

func TestHandlerCompletionRejectsUnsupportedFields(t *testing.T) {
	h, err := New(testModel{}, Config{})
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{
		`{"prompt":"x","stream_options":{"include_usage":true}}`,
		`{"prompt":"x","tools":[]}`,
		`{"prompt":"x","unknown":1}`,
	} {
		req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, want 400", body, rec.Code)
		}
	}
}

func TestHandlerCompletionsStreamingContract(t *testing.T) {
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"model":"test","prompt":"hello","max_tokens":2,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status/content type = %d/%q, body=%s", rec.Code, rec.Header().Get("Content-Type"), rec.Body)
	}
	body := rec.Body.String()
	for _, want := range []string{"\"object\":\"text_completion\"", "\"text\":\"c\"", "\"finish_reason\":\"length\"", "\"choices\":[]", "data: [DONE]"} {
		if !strings.Contains(body, want) {
			t.Errorf("stream missing %q: %s", want, body)
		}
	}
	if !strings.Contains(body, `"completion_tokens":2`) {
		t.Errorf("usage missing: %s", body)
	}
}

func TestHandlerToolRequestsValidateThenFailClosed(t *testing.T) {
	h, err := New(testModel{}, Config{})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		body string
		want string
	}{
		{"malformed", `{"messages":[{"role":"user","content":"x"}],"tools":{}}`, "tools must be an array"},
		{"invalid schema", `{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"x","parameters":[]}}]}`, "invalid tool schema"},
		{"valid schema", `{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"x","parameters":{"type":"object"}}}]}`, "tools are unsupported"},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(test.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), test.want) {
				t.Fatalf("status/body = %d %s, want %q", rec.Code, rec.Body, test.want)
			}
		})
	}
}

func TestHandlerToolAwareModelReceivesValidatedSchemas(t *testing.T) {
	for _, test := range []struct {
		name string
		path string
		body string
	}{
		{"chat", "/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"max_tokens":1}`},
		{"responses", "/v1/responses", `{"input":"x","tools":[{"name":"lookup","parameters":{"type":"object"}}],"max_output_tokens":1}`},
		{"messages", "/v1/messages", `{"messages":[{"role":"user","content":"x"}],"tools":[{"name":"lookup","parameters":{"type":"object"}}],"max_tokens":1}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := &toolAwareTestModel{}
			h, err := New(model, Config{Model: "test", MaxTokens: 1})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status/body = %d %s", rec.Code, rec.Body)
			}
			if model.calls != 1 || len(model.schema) != 1 || model.schema[0].Name != "lookup" {
				t.Fatalf("tool render calls/schema = %d/%v", model.calls, model.schema)
			}
		})
	}
}

func TestHandlerToolOutputAcceptsEmptyThinkingMarker(t *testing.T) {
	model := &toolTaggedOutputModel{text: "</think>The"}
	h, err := New(model, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"messages":[{"role":"user","content":"weather"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"max_tokens":1}`,
	))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status/body = %d %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"content":"The"`) || strings.Contains(body, "reasoning_content") || strings.Contains(body, "</think>") {
		t.Fatalf("tool output = %s", body)
	}
}

func TestHandlerToolAwareModelStillRejectsMalformedSchemas(t *testing.T) {
	model := &toolAwareTestModel{}
	h, err := New(model, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"messages":[{"role":"user","content":"x"}],"tools":[{"name":"lookup","parameters":[]}]}`,
	))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "invalid tool schema") {
		t.Fatalf("status/body = %d %s", rec.Code, rec.Body)
	}
	if model.calls != 0 {
		t.Fatalf("tool renderer called for malformed schema: %d", model.calls)
	}
}

func TestHandlerResponsesAndMessagesRejectUnsupportedFields(t *testing.T) {
	h, err := New(testModel{}, Config{})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		path string
		body string
	}{
		{"/v1/responses", `{"input":"x","reasoning":{"effort":"low"}}`},
		{"/v1/responses", `{"input":"x","tools":[]}`},
		{"/v1/responses", `{"input":"x","input":[{"type":"message"}]}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled"}}`},
		{"/v1/messages", `{"messages":[{"role":"assistant","content":"x"}]}`},
		{"/v1/messages", `{"messages":[{"role":"user","content":"x"}],"unknown":1}`},
	} {
		req := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s body %s: status = %d, want 400", test.path, test.body, rec.Code)
		}
	}
}
