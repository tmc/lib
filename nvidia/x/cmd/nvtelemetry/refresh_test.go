package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const refreshVLLM = "# TYPE vllm:generation_tokens_total counter\nvllm:generation_tokens_total{model_name=\"example\"} 40\n"

type refreshTestLine struct {
	Cycle     int    `json:"cycle"`
	Source    string `json:"source"`
	Ready     bool   `json:"ready"`
	Error     string `json:"error"`
	RawSHA256 string `json:"raw_sha256"`
	Status    struct {
		Cycles     int64 `json:"cycles"`
		Failures   int64 `json:"failures"`
		HasSuccess bool  `json:"has_success"`
		Records    int   `json:"records"`
		Withheld   int   `json:"withheld"`
	} `json:"status"`
}

func writeRefreshFile(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	return p
}

func refreshLines(t *testing.T, out []byte) []refreshTestLine {
	t.Helper()
	var lines []refreshTestLine
	s := bufio.NewScanner(bytes.NewReader(out))
	for s.Scan() {
		var l refreshTestLine
		d := json.NewDecoder(strings.NewReader(s.Text()))
		if err := d.Decode(&l); err != nil {
			t.Fatalf("line %q: %v", s.Text(), err)
		}
		lines = append(lines, l)
	}
	return lines
}

func refreshServer(t *testing.T, status *atomic.Int64, hits *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if s := int(status.Load()); s != 200 {
			w.WriteHeader(s)
			return
		}
		fmt.Fprint(w, refreshVLLM)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// runWithin runs the command and fails the test if it has not returned
// within 10s, so a refresh that ignores its bound fails promptly instead of
// hanging until the package timeout.
func runWithin(t *testing.T, args []string, stdout, stderr *bytes.Buffer) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- run(args, stdout, stderr) }()
	select {
	case err := <-done:
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("refresh did not return within 10s")
		return nil
	}
}

func vllmEndpoint(t *testing.T, url string) string {
	return endpointConfigFile(t, endpointConfig{URL: url, Timeout: time.Second, MaxBytes: 4096,
		Workload: &nvtelemetry.WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "run"}})
}

func TestRefreshWorkloadLoopback(t *testing.T) {
	var status, hits atomic.Int64
	status.Store(200)
	srv := refreshServer(t, &status, &hits)
	cfg := writeRefreshFile(t, "refresh.json", fmt.Sprintf(`{"interval_nanos":1000000,"timeout_nanos":1000000000,"max_age_nanos":60000000000,"workloads":[%q]}`, vllmEndpoint(t, srv.URL)))
	var out, stderr bytes.Buffer
	if err := runWithin(t, []string{"refresh", "-config", cfg, "-count", "3"}, &out, &stderr); err != nil {
		t.Fatalf("refresh: %v\n%s", err, stderr.String())
	}
	lines := refreshLines(t, out.Bytes())
	if len(lines) != 3 || hits.Load() != 3 {
		t.Fatalf("%d lines, %d requests, want 3 and 3:\n%s", len(lines), hits.Load(), out.String())
	}
	for i, l := range lines {
		if l.Cycle != i+1 || l.Source != "workload[0]" || !l.Ready || l.Error != "" || l.RawSHA256 == "" || l.Status.Cycles != int64(i+1) || l.Status.Records != 1 {
			t.Fatalf("line %d = %+v", i, l)
		}
	}
}

func TestRefreshWorkloadFailureReported(t *testing.T) {
	var status, hits atomic.Int64
	status.Store(500)
	srv := refreshServer(t, &status, &hits)
	cfg := writeRefreshFile(t, "refresh.json", fmt.Sprintf(`{"interval_nanos":1000000,"timeout_nanos":1000000000,"max_age_nanos":60000000000,"workloads":[%q]}`, vllmEndpoint(t, srv.URL)))
	var out bytes.Buffer
	if err := run([]string{"refresh", "-config", cfg, "-count", "2"}, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	lines := refreshLines(t, out.Bytes())
	if len(lines) != 2 {
		t.Fatalf("lines:\n%s", out.String())
	}
	for _, l := range lines {
		if l.Ready || l.Error == "" || l.Status.HasSuccess || strings.Contains(l.Error, srv.URL) {
			t.Fatalf("failed cycle reported as %+v", l)
		}
	}
}

func TestRefreshErrorsOmitEndpointURL(t *testing.T) {
	const secret = "S3CRETtok"
	var status, hits atomic.Int64
	status.Store(500)
	srv := refreshServer(t, &status, &hits)
	host := strings.TrimPrefix(srv.URL, "http://")
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	refused := l.Addr().String()
	l.Close()
	tests := []struct {
		name string
		url  string
		host string
	}{
		{"query token status", srv.URL + "/metrics?token=" + secret, host},
		{"query token refused", "http://" + refused + "/metrics?token=" + secret, refused},
		{"userinfo", "http://user:" + secret + "@" + host + "/metrics", host},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := writeRefreshFile(t, "refresh.json", fmt.Sprintf(`{"interval_nanos":1000000,"timeout_nanos":1000000000,"max_age_nanos":60000000000,"workloads":[%q]}`, vllmEndpoint(t, tt.url)))
			var out, stderr bytes.Buffer
			err := runWithin(t, []string{"refresh", "-config", cfg, "-count", "2"}, &out, &stderr)
			all := out.String() + stderr.String()
			if err != nil {
				all += err.Error()
			}
			if err == nil {
				lines := refreshLines(t, out.Bytes())
				if len(lines) != 2 {
					t.Fatalf("lines:\n%s", all)
				}
				for _, l := range lines {
					if l.Ready || l.Error == "" {
						t.Fatalf("failed cycle reported as %+v", l)
					}
				}
			}
			if strings.Contains(all, secret) || strings.Contains(all, tt.host) {
				t.Fatalf("output leaks endpoint URL:\n%s", all)
			}
		})
	}
}

func TestRefreshDuration(t *testing.T) {
	var status, hits atomic.Int64
	status.Store(200)
	srv := refreshServer(t, &status, &hits)
	cfg := writeRefreshFile(t, "refresh.json", fmt.Sprintf(`{"interval_nanos":5000000,"timeout_nanos":1000000000,"max_age_nanos":60000000000,"workloads":[%q]}`, vllmEndpoint(t, srv.URL)))
	var out bytes.Buffer
	start := time.Now()
	if err := run([]string{"refresh", "-config", cfg, "-duration", "40ms"}, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Fatalf("duration-bounded refresh ran %v", d)
	}
	if n := len(refreshLines(t, out.Bytes())); n < 1 {
		t.Fatal("no cycles")
	}
}

func TestRefreshRefusals(t *testing.T) {
	var status, hits atomic.Int64
	status.Store(200)
	srv := refreshServer(t, &status, &hits)
	good := vllmEndpoint(t, srv.URL)
	managed := endpointConfigFile(t, endpointConfig{URL: srv.URL, Timeout: time.Second, MaxBytes: 4096,
		Managed: &nvtelemetry.ManagedImport{SourceID: "configured", Producer: "dcgm-exporter", ProducerVersion: "4.6.1", NativeOrigin: "dcgm", Fields: nvtelemetry.DCGMExporterFields()}})
	paged := endpointConfigFile(t, endpointConfig{URL: srv.URL, Timeout: time.Second, MaxBytes: 4096, MaxPages: 2,
		Workload: &nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}})
	remote := endpointConfigFile(t, endpointConfig{URL: "http://example.invalid/metrics", Timeout: time.Second, MaxBytes: 4096,
		Workload: &nvtelemetry.WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "run"}})
	noSource := endpointConfigFile(t, endpointConfig{URL: srv.URL, Timeout: time.Second, MaxBytes: 4096})
	bounds := `"interval_nanos":1000000,"timeout_nanos":1000000000,"max_age_nanos":1000000000`
	attribution := func(snapshot string) string {
		return fmt.Sprintf(`{%s,"attribution":{"root":%q,"squeue":"squeue.json","mappings":"m.json","executions":"e.json","proc_dir":"proc","pids":[1],"boot_id":"b","node_id":"n","pid_namespace":"p","snapshot":%s,"table_capacity":16,"table_ttl_nanos":1000000000}}`, bounds, t.TempDir(), snapshot)
	}
	tests := []struct {
		name   string
		args   []string
		config string
		want   string
	}{
		{"no config", []string{"-count", "1"}, "", "requires a config file"},
		{"no bound", nil, fmt.Sprintf(`{%s,"workloads":[%q]}`, bounds, good), "requires -count or -duration"},
		{"negative count", []string{"-count", "-1"}, fmt.Sprintf(`{%s,"workloads":[%q]}`, bounds, good), "requires -count or -duration"},
		{"unknown field", []string{"-count", "1"}, fmt.Sprintf(`{%s,"workloads":[%q],"extra":1}`, bounds, good), "invalid refresh configuration fields"},
		{"duplicate key", []string{"-count", "1"}, fmt.Sprintf(`{%s,"workloads":[%q],"Workloads":[]}`, bounds, good), "invalid refresh configuration JSON"},
		{"trailing", []string{"-count", "1"}, fmt.Sprintf(`{%s,"workloads":[%q]} {}`, bounds, good), "trailing"},
		{"zero interval", []string{"-count", "1"}, fmt.Sprintf(`{"timeout_nanos":1,"max_age_nanos":1,"workloads":[%q]}`, good), "invalid refresh interval"},
		{"long timeout", []string{"-count", "1"}, fmt.Sprintf(`{"interval_nanos":1,"timeout_nanos":31000000000,"max_age_nanos":1,"workloads":[%q]}`, good), "invalid refresh interval"},
		{"no sources", []string{"-count", "1"}, fmt.Sprintf(`{%s}`, bounds), "requires an attribution source"},
		{"managed", []string{"-count", "1"}, fmt.Sprintf(`{%s,"workloads":[%q,%q]}`, bounds, good, managed), "managed"},
		{"paged", []string{"-count", "1"}, fmt.Sprintf(`{%s,"workloads":[%q,%q]}`, bounds, good, paged), "page"},
		{"remote http", []string{"-count", "1"}, fmt.Sprintf(`{%s,"workloads":[%q,%q]}`, bounds, good, remote), "allow_http"},
		{"no workload source", []string{"-count", "1"}, fmt.Sprintf(`{%s,"workloads":[%q,%q]}`, bounds, good, noSource), "no workload source"},
		{"unpinned snapshot", []string{"-count", "1"}, attribution(`{"node_id":"n","clock_domain":"d","clock_epoch":"e","start_nanos":1,"end_nanos":2}`), "unpinned"},
		{"zero table capacity", []string{"-count", "1"}, strings.Replace(attribution(`{"source_revision":"r","node_id":"n","clock_domain":"d","clock_epoch":"e","start_nanos":1,"end_nanos":2}`), `"table_capacity":16`, `"table_capacity":0`, 1), "capacity"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := []string{"refresh"}
			if tt.config != "" {
				args = append(args, "-config", writeRefreshFile(t, "refresh.json", tt.config))
			}
			args = append(args, tt.args...)
			var out bytes.Buffer
			err := runWithin(t, args, &out, &bytes.Buffer{})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
			if out.Len() != 0 {
				t.Fatalf("refused config produced output %q", out.String())
			}
		})
	}
	if hits.Load() != 0 {
		t.Fatalf("refused configurations made %d requests", hits.Load())
	}
}

func TestRefreshUsage(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"help"}, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "refresh") {
		t.Fatal("usage omits refresh")
	}
}
