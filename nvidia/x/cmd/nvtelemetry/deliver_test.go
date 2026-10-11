package main

import (
	"bytes"
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
	receiver := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/logs" {
			logs.Add(1)
		} else if r.URL.Path == "/v1/metrics" {
			metrics.Add(1)
		} else {
			t.Error("unexpected route")
		}
		io.Copy(io.Discard, r.Body)
		fmt.Fprint(w, `{}`)
	}))
	defer receiver.Close()
	metricConfig := endpointConfigFile(t, endpointConfig{URL: receiver.URL + "/v1/metrics", Timeout: time.Second, MaxBytes: 4096})
	logConfig := endpointConfigFile(t, endpointConfig{URL: receiver.URL + "/v1/logs", Timeout: time.Second, MaxBytes: 4096})
	if err := runServeContext(t.Context(), []string{"-mock", "-listen", "127.0.0.1:0", "-interval", "10ms", "-duration", "100ms", "-otlp-metrics-config", metricConfig, "-otlp-logs-config", logConfig}, io.Discard, io.Discard); err != nil {
		t.Fatal(err)
	}
	if metrics.Load() == 0 || metrics.Load() != logs.Load() {
		t.Fatal("signals were not paired/drained", metrics.Load(), logs.Load())
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
