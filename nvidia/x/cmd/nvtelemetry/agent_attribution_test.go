package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// This control runs the real Agent collector hook, encoding and bounded delivery
// worker. Only the HTTP dial boundary is declared: net.Pipe carries actual HTTP
// requests to a local decoded receiver, with no listener or network connection.
func TestAgentAttributedOTLPDelivery(t *testing.T) {
	type payload struct {
		path string
		body []byte
		err  error
	}
	received := make(chan payload, 4)
	var peers sync.WaitGroup
	transport := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		client, server := net.Pipe()
		peers.Add(1)
		go func() {
			defer peers.Done()
			defer server.Close()
			server.SetDeadline(time.Now().Add(3 * time.Second))
			request, err := http.ReadRequest(bufio.NewReader(server))
			if err != nil {
				received <- payload{err: err}
				return
			}
			raw, err := io.ReadAll(io.LimitReader(request.Body, 1<<20+1))
			request.Body.Close()
			if err != nil || len(raw) > 1<<20 {
				received <- payload{err: fmt.Errorf("receiver body: %v", err)}
				return
			}
			received <- payload{path: request.URL.Path, body: raw}
			fmt.Fprint(server, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
		}()
		return client, nil
	}}
	oldTransport := http.DefaultTransport
	http.DefaultTransport = transport
	defer func() { transport.CloseIdleConnections(); peers.Wait(); http.DefaultTransport = oldTransport }()
	dir := t.TempDir()
	endpoint := func(signal string) string {
		t.Helper()
		b, err := json.Marshal(endpointConfig{URL: "http://receiver.invalid/v1/" + signal, AllowHTTP: true, Timeout: time.Second, MaxBytes: 1 << 20})
		if err != nil {
			t.Fatal(err)
		}
		p := filepath.Join(dir, signal+".json")
		if err := os.WriteFile(p, b, 0600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	cfg := &ParsedConfig{Listen: "127.0.0.1:0", Interval: time.Hour, MaxAge: time.Second, Budget: 100, ShutdownTimeout: 2 * time.Second, Mock: true, IsLoopback: true, OTLPMetricsConfig: endpoint("metrics"), OTLPLogsConfig: endpoint("logs")}
	agent, err := NewAgent(cfg, "", io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	agent.ctx, agent.cancel = context.WithCancel(t.Context())
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := agent.Shutdown(ctx); err != nil {
			t.Error(err)
		}
	}()
	readings := []nvtelemetry.Reading{}
	for i, job := range []string{"job-a", "job-b"} {
		value := uint64(9007199254740993 + 2*i)
		zero := int64(0)
		pid := uint32(12 + i)
		a := &nvtelemetry.EntityAssignment{Process: nvtelemetry.EntityProcess{Process: cudaworkload.ProcessGenID{PID: pid, Generation: uint64(400 + i)}, BootID: "boot", NodeID: "node", PIDNamespace: "pid:[1]", HasExecGeneration: true, ExecGeneration: 9}, Device: nvtelemetry.DeviceMetadata{UUID: "GPU-a"}, Owner: nvtelemetry.EntityOwner{JobID: job, AllocationID: job + "-allocation", Attempt: "1"}, Source: "synthetic-authority", SourceRevision: "fixture-1", Authoritative: true, State: "assigned", ClockDomain: "monotonic", ClockEpoch: "boot", StartNanos: 0, EndNanos: 100, Sharing: "time_slice"}
		readings = append(readings, nvtelemetry.Reading{Name: nvtelemetry.MetricMemoryUsedBytes, Unit: nvtelemetry.UnitBytes, DeviceUUID: "GPU-a", Evidence: nvtelemetry.EvidenceMeasured, TimestampUnix: 1000, Number: &nvtelemetry.Number{Uint: &value}, Observation: &nvtelemetry.Observation{Assignment: a, UncertaintyNS: &zero, Scope: "process", Interval: &tracecontract.Interval{Start: 20, End: 30, Clock: "monotonic"}, ClockEpoch: "boot", Family: "device_metrics", Kind: "gauge", SourceClass: "synthetic", Producer: "agent-fixture", ProducerSchema: "test"}})
	}
	readings = append(readings, nvtelemetry.Reading{Name: nvtelemetry.MetricMemoryTotalBytes, Unit: nvtelemetry.UnitBytes, DeviceUUID: "GPU-a", Evidence: nvtelemetry.EvidenceMeasured, TimestampUnix: 1000, IsInt: true, IntValue: 99}, nvtelemetry.Reading{Name: nvtelemetry.MetricTemperatureC, Unit: nvtelemetry.UnitCelsius, DeviceUUID: "GPU-a", Evidence: nvtelemetry.EvidenceUnavailable, UnavailableReason: nvtelemetry.ReasonUnsupported, TimestampUnix: 1000})
	sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-a"})
	sampler.SetFallback(func(context.Context) ([]nvtelemetry.Reading, error) { return readings, nil })
	agent.SetSamplerFactory(func(*ParsedConfig) (nvtelemetry.Sampler, func(), error) { return sampler, nil, nil })
	if err := agent.ReloadConfig(cfg); err != nil {
		t.Fatal(err)
	}
	delivered := map[string][]byte{}
	for len(delivered) < 2 {
		select {
		case p := <-received:
			if p.err != nil {
				t.Fatal(p.err)
			}
			delivered[p.path] = p.body
		case <-time.After(5 * time.Second):
			agent.targetMu.Lock()
			target := agent.target
			agent.targetMu.Unlock()
			var cause error
			if target != nil {
				select {
				case <-target.collectorDone:
					cause = target.collectorErr
				default:
				}
			}
			t.Fatalf("Agent did not deliver attributed payloads; collector=%v metrics=%d logs=%d", cause, len(delivered["/v1/metrics"]), len(delivered["/v1/logs"]))
		}
	}
	agent.targetMu.Lock()
	target := agent.target
	agent.targetMu.Unlock()
	deadline := time.Now().Add(2 * time.Second)
	for target.metricsDelivery.Stats().SucceededBatches != 1 || target.logsDelivery.Stats().SucceededBatches != 1 {
		if time.Now().After(deadline) {
			t.Fatalf("receiver payloads not acknowledged: metrics=%+v logs=%+v", target.metricsDelivery.Stats(), target.logsDelivery.Stats())
		}
		time.Sleep(time.Millisecond)
	}
	if agent.metrics.deliveryFailures.Load() != 0 {
		t.Fatal("Agent reported a delivery failure")
	}
	query := func(data []byte, signal string) map[string][]map[string]any {
		t.Helper()
		d := json.NewDecoder(bytes.NewReader(data))
		d.UseNumber()
		var doc map[string]any
		if err := d.Decode(&doc); err != nil {
			t.Fatal(err)
		}
		resources, ok := doc["resource"+signal].([]any)
		if !ok {
			t.Fatalf("bad OTLP shape: %s", data)
		}
		got := map[string][]map[string]any{}
		for _, raw := range resources {
			r := raw.(map[string]any)
			attrs := r["resource"].(map[string]any)["attributes"].([]any)
			job := ""
			status := ""
			for _, raw := range attrs {
				a := raw.(map[string]any)
				switch a["key"] {
				case "nvtelemetry.owner.job_id":
					job = a["value"].(map[string]any)["stringValue"].(string)
				case "nvtelemetry.assignment.status":
					status = a["value"].(map[string]any)["stringValue"].(string)
				}
			}
			if job != "" && status != "assigned" || job == "" && status != "absent" {
				t.Fatal("wrong resource status", job, status)
			}
			scopeKey, itemKey := "scopeMetrics", "metrics"
			if signal == "Logs" {
				scopeKey, itemKey = "scopeLogs", "logRecords"
			}
			for _, scope := range r[scopeKey].([]any) {
				for _, item := range scope.(map[string]any)[itemKey].([]any) {
					got[job] = append(got[job], item.(map[string]any))
				}
			}
		}
		return got
	}
	metrics := query(delivered["/v1/metrics"], "Metrics")
	logs := query(delivered["/v1/logs"], "Logs")
	for i, job := range []string{"job-a", "job-b"} {
		if len(metrics[job]) != 1 || len(logs[job]) != 1 {
			t.Fatalf("missing resource population: metrics=%v logs=%v", metrics, logs)
		}
		point := metrics[job][0]["gauge"].(map[string]any)["dataPoints"].([]any)[0].(map[string]any)
		if point["asInt"] != fmt.Sprint(uint64(9007199254740993+2*i)) {
			t.Fatal("integer/resource mismatch", job, point)
		}
	}
	if len(metrics[""]) != 1 || len(logs[""]) != 2 {
		t.Fatal("device-only or unavailable population changed", metrics, logs)
	}
	foundUnavailable := false
	for _, record := range logs[""] {
		values := record["body"].(map[string]any)["kvlistValue"].(map[string]any)["values"].([]any)
		fields := map[string]string{}
		for _, v := range values {
			f := v.(map[string]any)
			fields[f["key"].(string)] = f["value"].(map[string]any)["stringValue"].(string)
		}
		if fields["name"] == nvtelemetry.MetricTemperatureC {
			foundUnavailable = true
			if fields["status"] != "unavailable" || fields["value_uint64"] != "" || fields["value_float64"] != "" {
				t.Fatal("missing numeric became zero", fields)
			}
		}
	}
	if !foundUnavailable {
		t.Fatal("unavailable diagnostic dropped")
	}
}

func TestAgentCanonicalDeviceExposition(t *testing.T) {
	cfg := &ParsedConfig{Listen: "127.0.0.1:0", Interval: time.Hour, MaxAge: time.Hour, Budget: 100, ShutdownTimeout: time.Second, Mock: true, IsLoopback: true}
	agent, err := NewAgent(cfg, "", io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	agent.ctx, agent.cancel = context.WithCancel(t.Context())
	defer func() {
		if err := agent.Shutdown(context.Background()); err != nil {
			t.Error(err)
		}
	}()
	sample, err := nvtelemetry.ReadManaged(t.Context(), strings.NewReader("# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} 12.345\n"), nvtelemetry.ManagedImport{SourceID: "synthetic-import", Producer: "dcgm", ProducerVersion: "fixture", NativeOrigin: "synthetic-import", Fields: nvtelemetry.DCGMExporterFields(), TimestampUnix: time.Now().UnixNano()})
	if err != nil {
		t.Fatal(err)
	}
	sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-a"})
	sampler.SetFallback(func(context.Context) ([]nvtelemetry.Reading, error) { return sample.Readings, nil })
	agent.SetSamplerFactory(func(*ParsedConfig) (nvtelemetry.Sampler, func(), error) { return sampler, nil, nil })
	if err := agent.ReloadConfig(cfg); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for agent.metrics.collectionSuccesses.Load() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("actual Agent hook never published device value")
		}
		time.Sleep(time.Millisecond)
	}
	request, _ := http.NewRequest(http.MethodGet, "http://test.invalid/metrics", nil)
	recorder := httptest.NewRecorder()
	agent.getMetricsHandler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), " 12345\n") {
		t.Fatalf("canonical Agent value lost: %d %s", recorder.Code, recorder.Body.String())
	}
}
