package dsv4server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCompletionKeepsDefaultThinkingInText(t *testing.T) {
	h, err := New(&defaultStreamingThinkingModel{streamingThinkingModel{parts: []string{"1."}}}, Config{Model: "test", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":"x","max_tokens":1,"temperature":0}`))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"text":"1."`) || strings.Contains(w.Body.String(), "reasoning_content") {
		t.Fatalf("completion = %d %s", w.Code, w.Body)
	}
}
