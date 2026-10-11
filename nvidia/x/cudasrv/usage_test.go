package cudasrv_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasrv"
)

type usageEngine struct{ usage cudasrv.UsageStats }

func (e usageEngine) Generate(context.Context, cudasrv.ChatCompletionRequest, chan<- string) error {
	panic("usage engine must use GenerateWithUsage")
}
func (e usageEngine) GenerateWithUsage(_ context.Context, _ cudasrv.ChatCompletionRequest, ch chan<- string) (cudasrv.UsageStats, error) {
	ch <- "many words in one chunk"
	ch <- ""
	return e.usage, nil
}

func TestUsageCountsEngineTokensNotChunks(t *testing.T) {
	for _, want := range []cudasrv.UsageStats{{PromptTokens: 14, CompletionTokens: 7, TotalTokens: 21}, {PromptTokens: 0, CompletionTokens: 0, TotalTokens: 0}} {
		w := httptest.NewRecorder()
		cudasrv.New(usageEngine{want}, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
		var got cudasrv.ChatCompletionResponse
		if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Usage == nil || *got.Usage != want {
			t.Fatalf("usage=%+v want=%+v", got.Usage, want)
		}
	}
}

func TestUnknownUsageOmitted(t *testing.T) {
	engine := engineFunc(func(_ context.Context, _ cudasrv.ChatCompletionRequest, ch chan<- string) error {
		ch <- "many words in one chunk"
		return nil
	})
	w := httptest.NewRecorder()
	cudasrv.New(engine, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	var got map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if _, ok := got["usage"]; ok {
		t.Fatal("unknown tokenizer usage must be omitted")
	}
}

func TestInvalidUsageRefused(t *testing.T) {
	for _, bad := range []cudasrv.UsageStats{{PromptTokens: -1}, {CompletionTokens: -1}, {PromptTokens: 2, CompletionTokens: 3, TotalTokens: 2}, {PromptTokens: int(^uint(0) >> 1), CompletionTokens: 1}} {
		w := httptest.NewRecorder()
		cudasrv.New(usageEngine{bad}, "test").Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
		if w.Code != http.StatusInternalServerError {
			t.Fatalf("invalid usage %+v status=%d", bad, w.Code)
		}
	}
}

func ExampleUsageEngine() {
	server := cudasrv.New(usageEngine{cudasrv.UsageStats{PromptTokens: 14, CompletionTokens: 7, TotalTokens: 21}}, "test")
	response := httptest.NewRecorder()
	server.Handler().ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	var completion cudasrv.ChatCompletionResponse
	_ = json.Unmarshal(response.Body.Bytes(), &completion)
	fmt.Println(completion.Usage.TotalTokens)
	// Output: 21
}

func TestMetricsCountOutputChunks(t *testing.T) {
	server := cudasrv.New(usageEngine{cudasrv.UsageStats{PromptTokens: 14, CompletionTokens: 7, TotalTokens: 21}}, "test")
	handler := server.Handler()
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	metrics := httptest.NewRecorder()
	handler.ServeHTTP(metrics, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	text := metrics.Body.String()
	if !strings.Contains(text, "gpu_output_chunks_total 2\n") || strings.Contains(text, "gpu_tokens_generated_total") {
		t.Fatalf("metrics mislabel chunks as tokens: %s", text)
	}
}
