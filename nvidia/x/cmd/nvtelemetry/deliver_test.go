package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestDeliverSignalsAndPartialSuccess(t *testing.T) {
	for _, signal := range []string{"metrics", "logs"} {
		t.Run(signal, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Header.Get("Content-Type") != "application/json" {
					t.Error("wrong content type")
				}
				io.Copy(io.Discard, r.Body)
				fmt.Fprint(w, `{}`)
			}))
			defer server.Close()
			var payload bytes.Buffer
			readings := []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU-a", Evidence: nvtelemetry.EvidenceDerived, Value: 12345, TimestampUnix: 1000}}
			var err error
			if signal == "logs" {
				err = nvtelemetry.WriteOTLPLogs(&payload, readings, nvtelemetry.ExportOptions{})
			} else {
				err = nvtelemetry.WriteOTLPMetrics(&payload, readings, nvtelemetry.ExportOptions{})
			}
			if err != nil {
				t.Fatal(err)
			}
			input := filepath.Join(t.TempDir(), "request.json")
			if err := os.WriteFile(input, payload.Bytes(), 0600); err != nil {
				t.Fatal(err)
			}
			config := endpointConfigFile(t, endpointConfig{URL: server.URL, Timeout: time.Second, MaxBytes: 4096})
			var out bytes.Buffer
			if err := runDeliver([]string{"-config", config, "-input", input, "-signal", signal}, &out, io.Discard); err != nil {
				t.Fatal(err)
			}
			var stats nvtelemetry.DeliveryStats
			if err := json.Unmarshal(out.Bytes(), &stats); err != nil {
				t.Fatal(err)
			}
			if stats.SucceededBatches != 1 || stats.PendingBatches != 0 || calls.Load() != 1 {
				t.Fatal(stats, calls.Load())
			}
			other := "logs"
			if signal == "logs" {
				other = "metrics"
			}
			out.Reset()
			if err := runDeliver([]string{"-config", config, "-input", input, "-signal", other}, &out, io.Discard); err == nil || calls.Load() != 1 {
				t.Fatal("mixed signals transmitted")
			}
		})
	}
}
func TestServeOTLPBothSignalsDrain(t *testing.T) {
	var metrics, logs atomic.Int32
	completed := make(chan string, 2)
	receiver := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var signal string
		var first bool
		if r.URL.Path == "/v1/logs" {
			signal = "logs"
			first = logs.Add(1) == 1
		} else if r.URL.Path == "/v1/metrics" {
			signal = "metrics"
			first = metrics.Add(1) == 1
		} else {
			t.Error("unexpected route")
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Error(err)
			return
		}
		if _, err := fmt.Fprint(w, `{}`); err != nil {
			t.Error(err)
			return
		}
		if first {
			completed <- signal
		}
	}))
	defer receiver.Close()
	metricConfig := endpointConfigFile(t, endpointConfig{URL: receiver.URL + "/v1/metrics", Timeout: time.Second, MaxBytes: 4096})
	logConfig := endpointConfigFile(t, endpointConfig{URL: receiver.URL + "/v1/logs", Timeout: time.Second, MaxBytes: 4096})
	guard, stop := context.WithTimeout(t.Context(), 10*time.Second)
	defer stop()
	ctx, cancel := context.WithCancel(guard)
	defer cancel()
	var output bytes.Buffer
	done := make(chan struct{})
	var serveErr error
	go func() {
		serveErr = runServeContext(ctx, []string{"-mock", "-listen", "127.0.0.1:0", "-interval", "10ms", "-otlp-metrics-config", metricConfig, "-otlp-logs-config", logConfig}, &output, io.Discard)
		close(done)
	}()
	defer func() {
		cancel()
		<-done
		if t.Failed() {
			t.Logf("serve result: %v\n%s", serveErr, output.String())
		}
	}()

	// The deadline bounds a missing prerequisite; it does not establish success.
	seen := make(map[string]bool)
	for len(seen) != 2 {
		select {
		case signal := <-completed:
			seen[signal] = true
		case <-done:
			t.Fatalf("serve stopped before both exports completed: %v", serveErr)
		case <-guard.Done():
			t.Fatal("both exports did not complete before the test deadline")
		}
	}
	cancel()
	select {
	case <-done:
		if serveErr != nil {
			t.Fatal(serveErr)
		}
	case <-guard.Done():
		t.Fatal("serve did not drain after both exports completed")
	}
	if metrics.Load() == 0 || metrics.Load() != logs.Load() {
		t.Fatal("signals were not paired/drained", metrics.Load(), logs.Load())
	}
	stats := make(map[string]nvtelemetry.DeliveryStats)
	for _, line := range bytes.Split(output.Bytes(), []byte("\n")) {
		if !bytes.HasPrefix(line, []byte("{")) {
			continue
		}
		var record struct {
			Signal string                    `json:"signal"`
			Stats  nvtelemetry.DeliveryStats `json:"stats"`
		}
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		stats[record.Signal] = record.Stats
	}
	for _, signal := range []string{"metrics", "logs"} {
		st, ok := stats[signal]
		if !ok || st.EnqueuedBatches == 0 || st.SucceededBatches != st.EnqueuedBatches || st.PendingBatches != 0 || st.PendingBytes != 0 || st.FailedBatches != 0 || st.DroppedBatches != 0 {
			t.Fatalf("%s delivery did not fully drain: %+v", signal, st)
		}
	}
	if stats["metrics"].SucceededBatches != stats["logs"].SucceededBatches || stats["metrics"].SucceededBatches != uint64(metrics.Load()) || stats["logs"].SucceededBatches != uint64(logs.Load()) {
		t.Fatal("delivery receipts differ from paired receiver acknowledgements", stats, metrics.Load(), logs.Load())
	}
}

func TestDeliverPartialLogsReturnsIncomplete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"partialSuccess":{"rejectedLogRecords":"1"}}`)
	}))
	defer server.Close()
	input := filepath.Join(t.TempDir(), "request.json")
	if err := os.WriteFile(input, []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{}]}]}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	config := endpointConfigFile(t, endpointConfig{URL: server.URL, Timeout: time.Second, MaxBytes: 4096})
	var out bytes.Buffer
	if err := runDeliver([]string{"-config", config, "-input", input, "-signal", "logs"}, &out, io.Discard); !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
		t.Fatal(err)
	}
	var stats nvtelemetry.DeliveryStats
	if err := json.Unmarshal(out.Bytes(), &stats); err != nil {
		t.Fatal(err)
	}
	if stats.RejectedLogRecords != 1 || stats.RejectedDataPoints != 0 || stats.SucceededBatches != 0 || stats.FailedBatches != 1 {
		t.Fatal(stats)
	}
}

func TestDeliverConfiguredProviderAuthentication(t *testing.T) {
	var accepted atomic.Int32
	receiver := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("DD-API-KEY") != "private-test-key" {
			w.WriteHeader(401)
			return
		}
		accepted.Add(1)
		fmt.Fprint(w, `{}`)
	}))
	defer receiver.Close()
	headers := filepath.Join(t.TempDir(), "headers.json")
	if err := os.WriteFile(headers, []byte(`{"DD-API-KEY":"private-test-key"}`), 0600); err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(t.TempDir(), "request.json")
	if err := os.WriteFile(input, []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{}]}]}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	c := endpointConfig{URL: receiver.URL, HeadersFile: headers, Timeout: time.Second, MaxBytes: 4096}
	var out bytes.Buffer
	if err := runDeliver([]string{"-config", endpointConfigFile(t, c), "-input", input, "-signal", "logs"}, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if accepted.Load() != 1 || bytes.Contains(out.Bytes(), []byte("private-test-key")) {
		t.Fatal("authentication or privacy failed")
	}
	c.HeadersFile = ""
	out.Reset()
	if err := runDeliver([]string{"-config", endpointConfigFile(t, c), "-input", input, "-signal", "logs"}, &out, io.Discard); !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
		t.Fatal(err)
	}
	var stats nvtelemetry.DeliveryStats
	if err := json.Unmarshal(out.Bytes(), &stats); err != nil {
		t.Fatal(err)
	}
	if stats.AuthFailures != 1 || stats.SucceededBatches != 0 || accepted.Load() != 1 {
		t.Fatal(stats)
	}
}
