package cudaserver_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaserver"
)

type usageEngine struct{ usage cudaserver.UsageStats }

func (e usageEngine) Generate(context.Context, cudaserver.ChatCompletionRequest, chan<- string) error {
	panic("usage engine must use GenerateWithUsage")
}
func (e usageEngine) GenerateWithUsage(_ context.Context, _ cudaserver.ChatCompletionRequest, ch chan<- string) (cudaserver.UsageStats, error) {
	ch <- "many words in one chunk"
	ch <- ""
	return e.usage, nil
}

func TestUsageCountsEngineTokensNotChunks(t *testing.T) {
	for _, want := range []cudaserver.UsageStats{{PromptTokens: 14, CompletionTokens: 7, TotalTokens: 21}, {PromptTokens: 0, CompletionTokens: 0, TotalTokens: 0}} {
		w := httptest.NewRecorder()
		cudaserver.New(usageEngine{want}, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
		var got cudaserver.ChatCompletionResponse
		if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Usage == nil || *got.Usage != want {
			t.Fatalf("usage=%+v want=%+v", got.Usage, want)
		}
	}
}

func TestUnknownUsageOmitted(t *testing.T) {
	engine := engineFunc(func(_ context.Context, _ cudaserver.ChatCompletionRequest, ch chan<- string) error {
		ch <- "many words in one chunk"
		return nil
	})
	w := httptest.NewRecorder()
	cudaserver.New(engine, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	var got map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if _, ok := got["usage"]; ok {
		t.Fatal("unknown tokenizer usage must be omitted")
	}
}

func TestInvalidUsageRefused(t *testing.T) {
	for _, bad := range []cudaserver.UsageStats{{PromptTokens: -1}, {CompletionTokens: -1}, {PromptTokens: 2, CompletionTokens: 3, TotalTokens: 2}, {PromptTokens: int(^uint(0) >> 1), CompletionTokens: 1}} {
		w := httptest.NewRecorder()
		cudaserver.New(usageEngine{bad}, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
		if w.Code != http.StatusInternalServerError {
			t.Fatalf("invalid usage %+v status=%d", bad, w.Code)
		}
	}
}

func ExampleUsageEngine() {
	server := cudaserver.New(usageEngine{cudaserver.UsageStats{PromptTokens: 14, CompletionTokens: 7, TotalTokens: 21}}, "test")
	response := httptest.NewRecorder()
	server.Handler().ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	var completion cudaserver.ChatCompletionResponse
	_ = json.Unmarshal(response.Body.Bytes(), &completion)
	fmt.Println(completion.Usage.TotalTokens)
	// Output: 21
}

func TestMetricsCountOutputChunks(t *testing.T) {
	server := cudaserver.New(usageEngine{cudaserver.UsageStats{PromptTokens: 14, CompletionTokens: 7, TotalTokens: 21}}, "test")
	handler := server.Handler()
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	metrics := httptest.NewRecorder()
	handler.ServeHTTP(metrics, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	text := metrics.Body.String()
	if !strings.Contains(text, "gpu_output_chunks_total 2\n") || strings.Contains(text, "gpu_tokens_generated_total") {
		t.Fatalf("metrics mislabel chunks as tokens: %s", text)
	}
}
