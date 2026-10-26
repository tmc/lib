package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type endpointVLLMV1Transport struct{ calls int }

func (r *endpointVLLMV1Transport) RoundTrip(*http.Request) (*http.Response, error) {
	r.calls++
	return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("# TYPE vllm:prompt_tokens_total counter\nvllm:prompt_tokens_total{model_name=\"m\",engine=\"0\"} 7\n"))}, nil
}

func TestEndpointVLLMV1Config(t *testing.T) {
	for _, engine := range []string{"v1", "V1", " v1", "v0"} {
		t.Run(engine, func(t *testing.T) {
			c := endpointConfig{URL: "https://control.invalid/metrics", Timeout: time.Second, MaxBytes: 4096, Workload: &nvtelemetry.WorkloadSource{Provider: "vllm", Version: "0.8.5", Engine: engine, RunID: "control"}}
			config, client, request, err := readEndpointConfig(endpointConfigFile(t, c))
			if err != nil {
				t.Fatal(err)
			}
			defer client.CloseIdleConnections()
			if config.Workload.Engine != engine {
				t.Fatal("config lost engine")
			}
			raw, err := json.Marshal(config)
			if err != nil {
				t.Fatal(err)
			}
			var round endpointConfig
			if err := json.Unmarshal(raw, &round); err != nil {
				t.Fatal(err)
			}
			if round.Workload.Engine != engine {
				t.Fatal("round trip lost engine")
			}
			transport := new(endpointVLLMV1Transport)
			client.Transport = transport
			result, err := nvtelemetry.FetchWorkload(context.Background(), client, request, *config.Workload, nvtelemetry.WorkloadLimits{})
			if engine == "v1" {
				if err != nil || transport.calls != 1 || result.Export.Source.Engine != "v1" || result.Export.Metrics[0].Labels["engine"] != "0" {
					t.Fatalf("configured fetch: %+v %v", result, err)
				}
			} else if err == nil || transport.calls != 0 {
				t.Fatalf("invalid engine transmitted %q calls=%d error=%v", engine, transport.calls, err)
			}
		})
	}
	// Case aliases cannot smuggle two conflicting selectors into one source.
	path := filepath.Join(t.TempDir(), "duplicate.json")
	raw := `{"url":"https://control.invalid/metrics","timeout_nanos":1000000000,"max_bytes":4096,"workload":{"Provider":"vllm","Version":"0.8.5","Engine":"v1","engine":"V1","RunID":"control"}}`
	if err := os.WriteFile(path, []byte(raw), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := readEndpointConfig(path); err == nil {
		t.Fatal("accepted duplicate engine alias")
	}
}
