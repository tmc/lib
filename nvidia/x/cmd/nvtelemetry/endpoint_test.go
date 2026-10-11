package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func endpointConfigFile(t *testing.T, c endpointConfig) string {
	t.Helper()
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "endpoint.json")
	if err := os.WriteFile(p, b, 0600); err != nil {
		t.Fatal(err)
	}
	return p
}
func TestFetchConfiguredManagedEndpoint(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer private-token" {
			t.Error("missing auth")
		}
		fmt.Fprint(w, "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} 12.5\n")
	}))
	defer server.Close()
	token := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(token, []byte("private-token\n"), 0600); err != nil {
		t.Fatal(err)
	}
	c := endpointConfig{URL: server.URL, BearerTokenFile: token, Timeout: time.Second, MaxBytes: 4096, Managed: &nvtelemetry.ManagedImport{SourceID: "configured", Producer: "dcgm-exporter", ProducerVersion: "4.6.1", NativeOrigin: "dcgm", Fields: nvtelemetry.DCGMExporterFields()}}
	var out, stderr bytes.Buffer
	if err := runFetchEndpoint([]string{"-config", endpointConfigFile(t, c)}, &out, &stderr); err != nil {
		t.Fatal(err)
	}
	var sample nvtelemetry.Sample
	if err := json.Unmarshal(out.Bytes(), &sample); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(sample.Readings) != 1 || sample.Readings[0].Value != 12500 || sample.Readings[0].HasSourceTimestamp {
		t.Fatal(sample)
	}
	c.Timeout = 0
	if err := runFetchEndpoint([]string{"-config", endpointConfigFile(t, c)}, &out, &stderr); err == nil || calls != 1 {
		t.Fatal("invalid bounds made a request")
	}
}
func TestEndpointConfigRefusals(t *testing.T) {
	for _, raw := range []string{`{"url":"http://private.invalid","url":"http://other.invalid"}`, `{"url":null}`, `{"url":"one","URL":"two"}`, `{"unknown":1}`, `{} {}`} {
		p := filepath.Join(t.TempDir(), "bad.json")
		if err := os.WriteFile(p, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
		_, _, _, err := readEndpointConfig(p)
		if err == nil || strings.Contains(err.Error(), "private") || strings.Contains(err.Error(), "other") {
			t.Fatal(err)
		}
	}
	c := endpointConfig{URL: "http://private.invalid", Timeout: time.Second, MaxBytes: 4096, Workload: &nvtelemetry.WorkloadSource{Provider: "triton", Version: "25.06"}}
	if _, _, _, err := readEndpointConfig(endpointConfigFile(t, c)); err == nil {
		t.Fatal("remote plaintext silently allowed")
	}
}
func TestManagedSamplerIdentityRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-other\"} 12.5\n")
	}))
	defer server.Close()
	req, _ := http.NewRequest("GET", server.URL, nil)
	s := managedEndpointSampler{client: server.Client(), request: req, clock: nvtelemetry.RealClock{}, config: endpointConfig{MaxBytes: 4096, Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-a"}}, Managed: &nvtelemetry.ManagedImport{SourceID: "configured", Producer: "dcgm-exporter", ProducerVersion: "4.6.1", NativeOrigin: "dcgm", Fields: nvtelemetry.DCGMExporterFields()}}}
	if _, err := s.Sample(t.Context()); err == nil {
		t.Fatal("unselected device admitted")
	}
}

func TestConfiguredMLflowPaging(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Query().Get("page_token") == "" {
			fmt.Fprint(w, `{"metrics":[{"key":"loss","value":0.5}],"next_page_token":"next"}`)
		} else {
			fmt.Fprint(w, `{"metrics":[{"key":"loss","value":0.25}]}`)
		}
	}))
	defer server.Close()
	c := endpointConfig{URL: server.URL + "/api/2.0/mlflow/metrics/get-history?run_id=run&metric_key=loss", Timeout: time.Second, MaxBytes: 4096, MaxPages: 4, Workload: &nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}}
	var out bytes.Buffer
	if err := runFetchEndpoint([]string{"-config", endpointConfigFile(t, c)}, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	var pages []nvtelemetry.WorkloadHTTPResult
	if err := json.Unmarshal(out.Bytes(), &pages); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || len(pages) != 2 || pages[0].Export.RawSHA256 == pages[1].Export.RawSHA256 {
		t.Fatal("pages lost")
	}
}

func TestEndpointProviderHeaderFiles(t *testing.T) {
	for _, raw := range []string{`{"DD-API-KEY":"private"}`, `{"Authorization":"Basic caller-encoded"}`, `{"signoz-access-token":"private"}`} {
		path := filepath.Join(t.TempDir(), "headers.json")
		if err := os.WriteFile(path, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
		headers, err := readEndpointHeaders(path)
		if err != nil || len(headers) != 1 {
			t.Fatal(err, headers)
		}
	}
	for _, raw := range []string{`{"X-Auth":"one","x-auth":"two"}`, `{"Authorization":"private\nheader"}`, `{"Host":"private"}`, `{"Authorization":null}`, `{"Authorization":123}`, `{"bad name":"private"}`, `{} {}`} {
		path := filepath.Join(t.TempDir(), "bad-headers.json")
		if err := os.WriteFile(path, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := readEndpointHeaders(path)
		if err == nil || strings.Contains(err.Error(), "private") {
			t.Fatal(err)
		}
	}
}

func TestConfiguredClearMLPaging(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body map[string]any
		if r.Method != http.MethodPost || json.NewDecoder(r.Body).Decode(&body) != nil || body["task"] != "task" {
			t.Error("invalid task request")
		}
		if body["scroll_id"] == nil {
			fmt.Fprint(w, `{"meta":{"result_code":200},"data":{"events":[{"type":"training_stats_scalar","task":"task","metric":"loss","variant":"train","value":0.5,"iter":1,"timestamp":1000}],"returned":1,"total":2,"scroll_id":"next"}}`)
		} else {
			fmt.Fprint(w, `{"meta":{"result_code":200},"data":{"events":[{"type":"training_stats_scalar","task":"task","metric":"loss","variant":"train","value":0.25,"iter":2,"timestamp":1010}],"returned":1,"total":2,"scroll_id":"next"}}`)
		}
	}))
	defer server.Close()
	body := filepath.Join(t.TempDir(), "body.json")
	if err := os.WriteFile(body, []byte(`{"task":"task","event_type":"training_stats_scalar"}`), 0600); err != nil {
		t.Fatal(err)
	}
	c := endpointConfig{URL: server.URL + "/events.get_task_events", Method: http.MethodPost, BodyFile: body, Timeout: time.Second, MaxBytes: 4096, MaxPages: 4, Workload: &nvtelemetry.WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "task"}}
	var out bytes.Buffer
	if err := runFetchEndpoint([]string{"-config", endpointConfigFile(t, c)}, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	var pages []nvtelemetry.WorkloadHTTPResult
	if err := json.Unmarshal(out.Bytes(), &pages); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || len(pages) != 2 || pages[0].Export.RawSHA256 == pages[1].Export.RawSHA256 {
		t.Fatal("pages lost")
	}
}
