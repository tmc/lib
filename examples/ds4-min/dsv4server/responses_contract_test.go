package dsv4server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestResponsesTokenLimitContract(t *testing.T) {
	for _, tt := range []struct {
		name       string
		model      Model
		wantOutput int
	}{
		{"default reasoning", defaultThinkingTestModel{}, 0},
		{"answer", testModel{}, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h, err := New(tt.model, Config{Model: "test", MaxTokens: 2})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"hello","max_output_tokens":2,"temperature":0}`))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body %s", rec.Code, rec.Body)
			}
			var got struct {
				Status            string            `json:"status"`
				Output            []json.RawMessage `json:"output"`
				IncompleteDetails struct {
					Reason string `json:"reason"`
				} `json:"incomplete_details"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got.Status != "incomplete" || got.IncompleteDetails.Reason != "max_tokens" || got.Output == nil || len(got.Output) != tt.wantOutput {
				t.Fatalf("response = %s", rec.Body)
			}
		})
	}
}
