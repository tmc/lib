package dsv4server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type messagesContractModel struct {
	thinkingTestModel
	text   string
	called bool
}

func (m *messagesContractModel) DefaultThinkMode() ThinkMode  { return ThinkHigh }
func (m *messagesContractModel) Decode([]int) (string, error) { return m.text, nil }
func (m *messagesContractModel) RenderThinking(messages []Message, mode ThinkMode) ([]int, error) {
	m.called = true
	return m.thinkingTestModel.RenderThinking(messages, mode)
}

// G6 records an unmarked default reasoning fragment and an explicit disabled
// request. The response signature must refer to its own id, and prompt tokens
// are accounted for as cache creation rather than input tokens.
func TestMessagesG6Contract(t *testing.T) {
	for _, tc := range []struct {
		name, control, text string
		thinking            bool
	}{
		{"default", "", "1.", true},
		{"disabled", `,"thinking":{"type":"disabled"}`, "Paris", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &messagesContractModel{text: tc.text}
			h, err := New(m, Config{Model: "test", MaxTokens: 2})
			if err != nil {
				t.Fatal(err)
			}
			defer h.Close()
			body := `{"messages":[{"role":"user","content":"capital"}],"max_tokens":2,"temperature":0` + tc.control + `}`
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
			var response struct {
				ID      string              `json:"id"`
				Content []map[string]string `json:"content"`
				Usage   map[string]int      `json:"usage"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if response.Usage["input_tokens"] != 0 || response.Usage["cache_creation_input_tokens"] != 1 || response.Usage["output_tokens"] != 2 {
				t.Fatalf("usage = %v", response.Usage)
			}
			if tc.thinking {
				if len(response.Content) != 2 || response.Content[0]["thinking"] != tc.text || response.Content[0]["signature"] != response.ID {
					t.Fatalf("content = %v, id = %q", response.Content, response.ID)
				}
			} else if !m.called || m.mode != ThinkNone || len(response.Content) != 1 || response.Content[0]["text"] != tc.text {
				t.Fatalf("disabled: called=%v mode=%v content=%v", m.called, m.mode, response.Content)
			}
		})
	}
}
