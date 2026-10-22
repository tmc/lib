package dsv4server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// transcriptCase describes the stable portion of one API response. IDs and
// timestamps are deliberately excluded: they identify a request, not its
// wire contract.
type transcriptCase struct {
	name      string
	path      string
	body      string
	object    string
	objectKey string
	text      string
	usage     map[string]any
}

func TestDeterministicTranscripts(t *testing.T) {
	cases := []transcriptCase{
		{
			name: "chat", path: "/v1/chat/completions",
			body:   `{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0}`,
			object: "chat.completion", text: "cc",
			usage: map[string]any{"prompt_tokens": float64(1), "completion_tokens": float64(2), "total_tokens": float64(3), "prompt_tokens_details": map[string]any{"cached_tokens": float64(0), "cache_write_tokens": float64(1)}},
		},
		{
			name: "completions", path: "/v1/completions",
			body:   `{"model":"test","prompt":"hello","max_tokens":2,"temperature":0}`,
			object: "text_completion", text: "cc",
			usage: map[string]any{"prompt_tokens": float64(1), "completion_tokens": float64(2), "total_tokens": float64(3), "prompt_tokens_details": map[string]any{"cached_tokens": float64(0), "cache_write_tokens": float64(1)}},
		},
		{
			name: "responses", path: "/v1/responses",
			body:   `{"model":"test","input":"hello","max_output_tokens":2,"temperature":0}`,
			object: "response", text: "cc",
			usage: map[string]any{"input_tokens": float64(1), "output_tokens": float64(2), "total_tokens": float64(3), "input_tokens_details": map[string]any{"cached_tokens": float64(0), "cache_write_tokens": float64(1)}, "output_tokens_details": map[string]any{"reasoning_tokens": float64(0)}},
		},
		{
			name: "messages", path: "/v1/messages",
			body:   `{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0}`,
			object: "message", objectKey: "type", text: "cc",
			usage: map[string]any{"input_tokens": float64(0), "output_tokens": float64(2), "cache_read_input_tokens": float64(0), "cache_creation_input_tokens": float64(1)},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			first := runTranscript(t, tc)
			second := runTranscript(t, tc)
			if !reflect.DeepEqual(first, second) {
				t.Fatalf("normalized response changed between runs:\nfirst=%v\nsecond=%v", first, second)
			}
			objectKey := tc.objectKey
			if objectKey == "" {
				objectKey = "object"
			}
			if got := first[objectKey]; got != tc.object {
				t.Fatalf("object = %v, want %q", got, tc.object)
			}
			if got := transcriptText(first); got != tc.text {
				t.Fatalf("text = %q, want %q", got, tc.text)
			}
			if got := first["usage"]; !reflect.DeepEqual(got, tc.usage) {
				t.Fatalf("usage = %v, want %v", got, tc.usage)
			}
		})
	}
}

func TestLengthFinishContracts(t *testing.T) {
	cases := []struct {
		name, path, body, want string
	}{
		{"chat", "/v1/chat/completions", `{"messages":[{"role":"user","content":"hello"}],"max_tokens":1}`, `"finish_reason":"length"`},
		{"completions", "/v1/completions", `{"prompt":"hello","max_tokens":1}`, `"finish_reason":"length"`},
		{"responses", "/v1/responses", `{"input":"hello","max_output_tokens":1}`, `"status":"incomplete"`},
		{"messages", "/v1/messages", `{"messages":[{"role":"user","content":"hello"}],"max_tokens":1}`, `"stop_reason":"max_tokens"`},
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
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), tc.want) {
				t.Fatalf("status=%d body=%s, want %q", rec.Code, rec.Body, tc.want)
			}
		})
	}
}

func TestTranscriptMutationsFailClosed(t *testing.T) {
	cases := []struct {
		name string
		path string
		body string
	}{
		{"chat unknown field", "/v1/chat/completions", `{"messages":[{"role":"user","content":"hello"}],"max_tokens":1,"mutated":true}`},
		{"completions unknown field", "/v1/completions", `{"prompt":"hello","max_tokens":1,"mutated":true}`},
		{"responses unsupported reasoning", "/v1/responses", `{"input":"hello","reasoning":{"effort":"low"}}`},
		{"messages unsupported thinking", "/v1/messages", `{"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"enabled"}}`},
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
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body)
			}
			var body map[string]any
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			errBody, ok := body["error"].(map[string]any)
			if !ok || errBody["type"] != "invalid_request_error" {
				t.Fatalf("error envelope = %v", body)
			}
		})
	}
}

func runTranscript(t *testing.T, tc transcriptCase) map[string]any {
	t.Helper()
	h, err := New(testModel{}, Config{Model: "test", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("content type = %q, want application/json", got)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	delete(body, "id")
	delete(body, "created")
	delete(body, "created_at")
	return body
}

func transcriptText(body map[string]any) string {
	if choices, ok := body["choices"].([]any); ok && len(choices) != 0 {
		choice, _ := choices[0].(map[string]any)
		if message, ok := choice["message"].(map[string]any); ok {
			text, _ := message["content"].(string)
			return text
		}
		text, _ := choice["text"].(string)
		return text
	}
	if output, ok := body["output"].([]any); ok && len(output) != 0 {
		item, _ := output[0].(map[string]any)
		content, _ := item["content"].([]any)
		if len(content) != 0 {
			part, _ := content[0].(map[string]any)
			text, _ := part["text"].(string)
			return text
		}
	}
	if content, ok := body["content"].([]any); ok && len(content) != 0 {
		part, _ := content[0].(map[string]any)
		text, _ := part["text"].(string)
		return text
	}
	return ""
}
