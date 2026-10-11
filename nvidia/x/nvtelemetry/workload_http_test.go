package nvtelemetry

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

type workloadRoundTrip func(*http.Request) (*http.Response, error)

func (f workloadRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestFetchWorkloadReaders(t *testing.T) {
	for _, tc := range []struct{ provider, version, file string }{
		{"triton", "25.06", "triton-before.prom"}, {"vllm", "0.8.5", "vllm.prom"}, {"ray", "2.44.0", "ray.prom"},
		{"mlflow", "2.22.0", "mlflow.json"}, {"wandb", "0.19.11", "wandb.jsonl"}, {"clearml", "1.18.0/api-2.9", "clearml.json"}, {"pytorch", "2.7.0", "pytorch.json"},
	} {
		t.Run(tc.provider, func(t *testing.T) {
			raw, err := os.ReadFile("testdata/workload/" + tc.file)
			if err != nil {
				t.Fatal(err)
			}
			client := &http.Client{Transport: workloadRoundTrip(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(string(raw)))}, nil
			})}
			request, _ := http.NewRequest("GET", "https://example.invalid/export", nil)
			run := "run"
			if tc.provider == "clearml" {
				run = "train1"
			}
			result, err := FetchWorkload(context.Background(), client, request, WorkloadSource{Provider: tc.provider, Version: tc.version, RunID: run}, WorkloadLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if string(result.Export.Raw) != string(raw) {
				t.Fatal("raw lost")
			}
			if tc.provider == "pytorch" {
				if len(result.Annotations) == 0 {
					t.Fatal("annotations lost")
				}
			} else if len(result.Export.Metrics) == 0 {
				t.Fatal("metrics lost")
			}
			if client.Timeout != 0 || client.CheckRedirect != nil {
				t.Fatal("client mutated")
			}
		})
	}
}
func TestFetchWorkloadRefusals(t *testing.T) {
	s := WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}
	for _, status := range []int{302, 401, 200} {
		calls := 0
		client := &http.Client{Transport: workloadRoundTrip(func(*http.Request) (*http.Response, error) {
			calls++
			return &http.Response{StatusCode: status, Header: http.Header{"Location": []string{"https://secret.invalid"}}, Body: io.NopCloser(strings.NewReader(strings.Repeat("secret", 10)))}, nil
		})}
		req, _ := http.NewRequest("GET", "https://example.invalid?credential=secret", nil)
		_, err := FetchWorkload(context.Background(), client, req, s, WorkloadLimits{MaxBytes: 32})
		if err == nil || strings.Contains(err.Error(), "secret") || calls != 1 {
			t.Fatalf("refusal %v calls %d", err, calls)
		}
	}
	client := &http.Client{Transport: workloadRoundTrip(func(*http.Request) (*http.Response, error) { return nil, errors.New("secret transport") })}
	req, _ := http.NewRequest("GET", "https://example.invalid", nil)
	if _, err := FetchWorkload(context.Background(), client, req, s, WorkloadLimits{}); err == nil || strings.Contains(err.Error(), "secret") {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := FetchWorkload(ctx, client, req, s, WorkloadLimits{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
func ExampleFetchWorkload() {
	client := &http.Client{Transport: workloadRoundTrip(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"metrics":[{"key":"loss","value":0.5,"timestamp":1,"step":0}]}`))}, nil
	})}
	request, _ := http.NewRequest("GET", "https://example.invalid/api/2.0/mlflow/metrics/get-history", nil)
	result, err := FetchWorkload(context.Background(), client, request, WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}, WorkloadLimits{})
	fmt.Println(len(result.Export.Metrics), err)
	// Output: 1 <nil>
}
