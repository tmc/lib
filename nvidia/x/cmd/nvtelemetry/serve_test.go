package main

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type endpointLine struct {
	mu   sync.Mutex
	text string
	addr chan string
}

func (w *endpointLine) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.text += string(p)
	if i := strings.IndexByte(w.text, '\n'); i >= 0 {
		select {
		case w.addr <- w.text[:i]:
		default:
		}
	}
	return len(p), nil
}

func TestServeEndpointLifecycle(t *testing.T) {
	token := "test-private-token"
	path := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(path, []byte(token), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	output := &endpointLine{addr: make(chan string, 1)}
	done := make(chan error, 1)
	go func() {
		done <- runServeContext(ctx, []string{"-mock", "-listen", "127.0.0.1:0", "-interval", "5ms", "-max-age", "1s", "-budget", "1000", "-auth-token-file", path}, output, io.Discard)
	}()
	var addr string
	select {
	case addr = <-output.addr:
	case err := <-done:
		t.Fatal(err)
	case <-time.After(3 * time.Second):
		t.Fatal("listener startup timeout")
	}
	client := &http.Client{Timeout: time.Second}
	request := func(method, url, accept, bearer string) (int, string, http.Header) {
		t.Helper()
		req, err := http.NewRequestWithContext(t.Context(), method, url, nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Accept", accept)
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		return resp.StatusCode, string(raw), resp.Header
	}
	code, body, _ := request("GET", addr, "", "")
	if code != 401 || strings.Contains(body, token) {
		t.Fatal(code, body)
	}
	deadline := time.Now().Add(time.Second)
	for {
		code, body, _ = request("GET", addr, "application/openmetrics-text;version=1.0.0", token)
		if code == 200 && strings.Contains(body, "GPU-mock") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("snapshot startup", code, body)
		}
		time.Sleep(time.Millisecond)
	}
	if !strings.Contains(body, "GPU-mock") || !strings.Contains(body, "backend=\"mock\"") || !strings.HasSuffix(body, "# EOF\n") {
		t.Fatal(body)
	}
	code, body, headers := request("HEAD", addr, "", token)
	if code != 200 || body != "" || headers.Get("Content-Length") == "" {
		t.Fatal("HEAD", code, body, headers)
	}
	ready := strings.TrimSuffix(addr, "/metrics") + "/readyz"
	code, _, _ = request("GET", ready, "", token)
	if code != 200 {
		t.Fatal("readiness", code)
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("shutdown timeout")
	}
	if resp, err := client.Get(addr); err == nil {
		resp.Body.Close()
		t.Fatal("listener retained after shutdown")
	}
}

func TestServeRefusalsBeforeNativeAdmission(t *testing.T) {
	for _, args := range [][]string{
		{}, {"-mock", "-interval", "0"}, {"-mock", "-budget", "0"}, {"-mock", "-max-age", "0"},
		{"-mock", "-listen", ":9464"}, {"-mock", "-listen", ":9464", "-allow-remote"},
		{"-mock", "-tls-cert", "missing"}, {"-mock", "-tls-cert", "missing", "-tls-key", "missing"},
		{"-mock", "extra"}, {"-auth-token-file", "missing"},
	} {
		if err := runServeContext(t.Context(), args, io.Discard, io.Discard); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
func TestServeFiniteDuration(t *testing.T) {
	if err := runServeContext(t.Context(), []string{"-mock", "-listen", "127.0.0.1:0", "-duration", "30ms", "-interval", "5ms", "-budget", "100"}, io.Discard, io.Discard); err != nil {
		t.Fatal(err)
	}
}
func TestServeRecordBudgetStops(t *testing.T) {
	if err := runServeContext(t.Context(), []string{"-mock", "-listen", "127.0.0.1:0", "-interval", "5ms", "-budget", "2"}, io.Discard, io.Discard); err == nil {
		t.Fatal("exhausted budget reported complete")
	}
}

func TestServeTLSRoundTrip(t *testing.T) {
	fixture := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer fixture.Close()
	pair := fixture.TLS.Certificates[0]
	key, err := x509.MarshalPKCS8PrivateKey(pair.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	certPath := filepath.Join(t.TempDir(), "cert.pem")
	keyPath := filepath.Join(t.TempDir(), "key.pem")
	if err := os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: pair.Certificate[0]}), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	output := &endpointLine{addr: make(chan string, 1)}
	done := make(chan error, 1)
	go func() {
		done <- runServeContext(ctx, []string{"-mock", "-listen", "127.0.0.1:0", "-tls-cert", certPath, "-tls-key", keyPath, "-interval", "5ms"}, output, io.Discard)
	}()
	var address string
	select {
	case address = <-output.addr:
	case err := <-done:
		t.Fatal(err)
	case <-time.After(time.Second):
		t.Fatal("TLS startup timeout")
	}
	client := fixture.Client()
	client.Timeout = time.Second
	response, err := client.Get(address)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.TLS == nil || len(response.TLS.VerifiedChains) == 0 {
		t.Fatal("unverified TLS")
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("TLS shutdown timeout")
	}
}

func TestServeValidatesDeliveryBeforeNativeAdmission(t *testing.T) {
	for _, flag := range []string{"-otlp-metrics-config", "-otlp-logs-config"} {
		err := runServeContext(t.Context(), []string{"-device-uuids", "GPU-selected", flag, filepath.Join(t.TempDir(), "missing.json")}, io.Discard, io.Discard)
		if err == nil || err.Error() != "open endpoint configuration file failed" {
			t.Fatal("delivery configuration did not precede native initialization", err)
		}
	}
	err := runServeContext(t.Context(), []string{"-device-uuids", "GPU-selected", "-counter-start-unix-nanos", "-1"}, io.Discard, io.Discard)
	if err == nil || err.Error() != "invalid counter epoch start" {
		t.Fatal("counter configuration did not precede native initialization", err)
	}
}

func TestServeOwnedRecoveryAndInProcessSinkLifecycle(t *testing.T) {
	var sinkHealthy atomic.Bool
	var sinkMu sync.Mutex
	var metricsReceived [][]byte
	var logsReceived [][]byte

	sink := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		if !sinkHealthy.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, "outage")
			return
		}
		sinkMu.Lock()
		switch r.URL.Path {
		case "/v1/metrics":
			metricsReceived = append(metricsReceived, body)
		case "/v1/logs":
			logsReceived = append(logsReceived, body)
		default:
			t.Errorf("unexpected sink path %s", r.URL.Path)
		}
		sinkMu.Unlock()
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer sink.Close()

	tmp := t.TempDir()
	metricsCfgPath := filepath.Join(tmp, "otlp-metrics.json")
	logsCfgPath := filepath.Join(tmp, "otlp-logs.json")
	if err := os.WriteFile(metricsCfgPath, []byte(fmt.Sprintf(`{"url":%q,"timeout_nanos":50000000,"max_bytes":65536}`, sink.URL+"/v1/metrics")), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(logsCfgPath, []byte(fmt.Sprintf(`{"url":%q,"timeout_nanos":50000000,"max_bytes":65536}`, sink.URL+"/v1/logs")), 0600); err != nil {
		t.Fatal(err)
	}

	retentionDir := filepath.Join(tmp, "retention")
	metricsSpoolDir := filepath.Join(tmp, "metrics-spool")
	logsSpoolDir := filepath.Join(tmp, "logs-spool")
	manifestRun1 := filepath.Join(tmp, "recovery-run1.json")
	manifestRun2 := filepath.Join(tmp, "recovery-run2.json")
	owner := "task-133-serve"

	// Run 1: sink is in outage (503). Batches are durably spooled to disk and retained stream 1 is published.
	run1Args := []string{
		"-mock",
		"-listen", "127.0.0.1:0",
		"-interval", "5ms",
		"-duration", "25ms",
		"-shutdown-timeout", "30ms",
		"-budget", "100",
		"-retention-dir", retentionDir,
		"-retention-owner", owner,
		"-otlp-metrics-config", metricsCfgPath,
		"-durable-metrics-spool-dir", metricsSpoolDir,
		"-otlp-logs-config", logsCfgPath,
		"-durable-logs-spool-dir", logsSpoolDir,
		"-recovery-manifest-out", manifestRun1,
	}
	run1Err := runServeContext(t.Context(), run1Args, io.Discard, io.Discard)
	if !errors.Is(run1Err, nvtelemetry.ErrDeliveryIncomplete) {
		t.Fatalf("expected ErrDeliveryIncomplete during sink outage, got %v", run1Err)
	}

	// Verify Run 1 published stream-0000000000001.jsonl and left spooled batches on disk.
	stream1Bytes, err := os.ReadFile(filepath.Join(retentionDir, "stream-0000000000001.jsonl"))
	if err != nil {
		t.Fatalf("expected stream 1 in retention dir: %v", err)
	}
	if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(stream1Bytes), 4<<20); err != nil {
		t.Fatalf("stream 1 failed strict validation: %v", err)
	}

	rawRun1Manifest, err := os.ReadFile(manifestRun1)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := ReadServeRecoveryManifest(bytes.NewReader(rawRun1Manifest))
	if err != nil {
		t.Fatalf("ReadServeRecoveryManifest run 1: %v", err)
	}
	if m1.SchemaVersion != ServeRecoverySchemaVersion || m1.Completed || m1.StoredStreams != 1 {
		t.Fatalf("unexpected run 1 manifest: %+v", m1)
	}

	// Simulate an ungraceful crash leaving stale leases and partial temp files in the owned directories.
	if err := os.Link(filepath.Join(retentionDir, "owner.json"), filepath.Join(retentionDir, "lease")); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(metricsSpoolDir, "lease"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(metricsSpoolDir, "pending-0000000000099.tmp"), []byte("partial"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(logsSpoolDir, "lease"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(logsSpoolDir, "pending-0000000000098.tmp"), []byte("partial"), 0600); err != nil {
		t.Fatal(err)
	}

	// Live-owner refusal control: if a live writer holds the retention lock, serve must refuse takeover.
	liveRetDir := filepath.Join(tmp, "live-retention")
	liveRet, err := nvtelemetry.OpenRetention(liveRetDir, nvtelemetry.RetentionConfig{
		Owner:          owner,
		MaxFiles:       4,
		MaxBytes:       1 << 20,
		MaxStreamBytes: 512 << 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := runServeContext(t.Context(), []string{
		"-mock", "-listen", "127.0.0.1:0", "-interval", "5ms", "-duration", "15ms",
		"-retention-dir", liveRetDir, "-retention-owner", owner,
	}, io.Discard, io.Discard); err == nil {
		_ = liveRet.Close()
		t.Fatal("serve admitted takeover of a live retention writer lease")
	}
	if err := liveRet.Close(); err != nil {
		t.Fatal(err)
	}

	// Run 2: sink becomes healthy. Serve verifies prior manifest, recovers stale leases/temp files,
	// flushes recovered spooled batches plus new batches to the sink, and stores stream-0000000000002.jsonl.
	sinkHealthy.Store(true)
	run2Args := []string{
		"-mock",
		"-listen", "127.0.0.1:0",
		"-interval", "5ms",
		"-duration", "25ms",
		"-shutdown-timeout", "2s",
		"-budget", "100",
		"-retention-dir", retentionDir,
		"-retention-owner", owner,
		"-otlp-metrics-config", metricsCfgPath,
		"-durable-metrics-spool-dir", metricsSpoolDir,
		"-otlp-logs-config", logsCfgPath,
		"-durable-logs-spool-dir", logsSpoolDir,
		"-recovery-manifest-in", manifestRun1,
		"-recovery-manifest-out", manifestRun2,
	}
	if err := runServeContext(t.Context(), run2Args, io.Discard, io.Discard); err != nil {
		t.Fatalf("run 2 recovery and serving failed: %v", err)
	}

	sinkMu.Lock()
	gotMetrics := len(metricsReceived)
	gotLogs := len(logsReceived)
	sinkMu.Unlock()
	if gotMetrics < 2 || gotLogs < 2 {
		t.Fatalf("expected recovered + live batches delivered to sink, got metrics=%d logs=%d", gotMetrics, gotLogs)
	}

	// Both spool directories must be completely drained with no leftover spool/pending/lease entries.
	for _, dir := range []string{metricsSpoolDir, logsSpoolDir} {
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 1 || entries[0].Name() != "owner.json" {
			t.Fatalf("expected only owner.json in drained spool %s, got %d entries", dir, len(entries))
		}
	}

	// Retention directory must now hold both stream 1 and stream 2, both strictly valid.
	stream2Bytes, err := os.ReadFile(filepath.Join(retentionDir, "stream-0000000000002.jsonl"))
	if err != nil {
		t.Fatalf("expected stream 2 in retention dir: %v", err)
	}
	if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(stream2Bytes), 4<<20); err != nil {
		t.Fatalf("stream 2 failed strict validation: %v", err)
	}

	// Verify Run 2 recovery manifest recorded versioned recovery provenance and completion.
	rawRun2Manifest, err := os.ReadFile(manifestRun2)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := ReadServeRecoveryManifest(bytes.NewReader(rawRun2Manifest))
	if err != nil {
		t.Fatalf("ReadServeRecoveryManifest run 2: %v", err)
	}
	if m2.SchemaVersion != ServeRecoverySchemaVersion || !m2.Completed || m2.StoredStreams != 1 {
		t.Fatalf("unexpected run 2 manifest top-level state: %+v", m2)
	}
	if m2.Retention == nil ||
		m2.Retention.SchemaVersion != nvtelemetry.RetentionRecoverySchemaVersion ||
		!m2.Retention.RecoveredLease ||
		m2.Retention.RetainedFiles != 1 ||
		m2.Retention.NextStreamID != 2 {
		t.Fatalf("unexpected run 2 retention recovery manifest: %+v", m2.Retention)
	}
	if m2.MetricsDelivery == nil ||
		m2.MetricsDelivery.SchemaVersion != nvtelemetry.DurableDeliveryRecoverySchemaVersion ||
		!m2.MetricsDelivery.RecoveredLease ||
		m2.MetricsDelivery.RemovedTempFiles != 1 ||
		m2.MetricsDelivery.PendingBatches < 1 {
		t.Fatalf("unexpected run 2 metrics recovery manifest: %+v", m2.MetricsDelivery)
	}
	if m2.LogsDelivery == nil ||
		m2.LogsDelivery.SchemaVersion != nvtelemetry.DurableDeliveryRecoverySchemaVersion ||
		!m2.LogsDelivery.RecoveredLease ||
		m2.LogsDelivery.RemovedTempFiles != 1 ||
		m2.LogsDelivery.PendingBatches < 1 {
		t.Fatalf("unexpected run 2 logs recovery manifest: %+v", m2.LogsDelivery)
	}
}

func TestServeRecoveryManifestVersionAndTamperRefusals(t *testing.T) {
	// 1. Direct unit checks on RetentionRecoveryManifest and DurableDeliveryRecoveryManifest schema versions.
	validRet := nvtelemetry.RetentionRecoveryManifest{
		SchemaVersion: nvtelemetry.RetentionRecoverySchemaVersion,
		MarkerVersion: 2,
		Owner:         "task-133",
		NextStreamID:  1,
	}
	var buf bytes.Buffer
	if err := nvtelemetry.WriteRetentionRecoveryManifest(&buf, validRet); err != nil {
		t.Fatal(err)
	}
	if got, err := nvtelemetry.ReadRetentionRecoveryManifest(&buf); err != nil || got != validRet {
		t.Fatalf("retention manifest round-trip: %+v %v", got, err)
	}
	for _, badVer := range []int{0, -1, 2, 99} {
		bad := validRet
		bad.SchemaVersion = badVer
		if err := nvtelemetry.WriteRetentionRecoveryManifest(io.Discard, bad); err == nil {
			t.Fatalf("accepted invalid retention recovery schema_version %d", badVer)
		}
	}
	for _, raw := range []string{
		`{"marker_version":2,"owner":"task-133","next_stream_id":1}`,
		`{"schema_version":0,"marker_version":2,"owner":"task-133","next_stream_id":1}`,
		`{"schema_version":2,"marker_version":2,"owner":"task-133","next_stream_id":1}`,
		`{"schema_version":1,"marker_version":1,"owner":"task-133","next_stream_id":1}`,
		`{"schema_version":1,"marker_version":2,"owner":"task-133","next_stream_id":1,"unknown":true}`,
	} {
		if _, err := nvtelemetry.ReadRetentionRecoveryManifest(strings.NewReader(raw)); err == nil {
			t.Fatalf("ReadRetentionRecoveryManifest accepted invalid input: %s", raw)
		}
	}

	validDeliv := nvtelemetry.DurableDeliveryRecoveryManifest{
		SchemaVersion: nvtelemetry.DurableDeliveryRecoverySchemaVersion,
		MarkerVersion: 1,
		Owner:         "task-133",
	}
	buf.Reset()
	if err := nvtelemetry.WriteDurableDeliveryRecoveryManifest(&buf, validDeliv); err != nil {
		t.Fatal(err)
	}
	if got, err := nvtelemetry.ReadDurableDeliveryRecoveryManifest(&buf); err != nil || got != validDeliv {
		t.Fatalf("durable delivery manifest round-trip: %+v %v", got, err)
	}
	for _, badVer := range []int{0, -1, 2, 99} {
		bad := validDeliv
		bad.SchemaVersion = badVer
		if err := nvtelemetry.WriteDurableDeliveryRecoveryManifest(io.Discard, bad); err == nil {
			t.Fatalf("accepted invalid durable delivery recovery schema_version %d", badVer)
		}
	}
	for _, raw := range []string{
		`{"marker_version":1,"owner":"task-133"}`,
		`{"schema_version":0,"marker_version":1,"owner":"task-133"}`,
		`{"schema_version":99,"marker_version":1,"owner":"task-133"}`,
		`{"schema_version":1,"marker_version":2,"owner":"task-133"}`,
		`{"schema_version":1,"marker_version":1,"owner":"task-133","extra":1}`,
	} {
		if _, err := nvtelemetry.ReadDurableDeliveryRecoveryManifest(strings.NewReader(raw)); err == nil {
			t.Fatalf("ReadDurableDeliveryRecoveryManifest accepted invalid input: %s", raw)
		}
	}

	// 2. ServeRecoveryManifest CLI and JSON version/tamper refusals before admission.
	tmp := t.TempDir()
	for name, payload := range map[string]string{
		"missing_version":        `{"owner":"task-133"}`,
		"zero_version":           `{"schema_version":0,"owner":"task-133"}`,
		"unknown_version":        `{"schema_version":2,"owner":"task-133"}`,
		"nested_ret_zero_ver":    `{"schema_version":1,"owner":"task-133","retention":{"schema_version":0,"marker_version":2,"owner":"task-133","next_stream_id":1}}`,
		"nested_ret_unknown_ver": `{"schema_version":1,"owner":"task-133","retention":{"schema_version":7,"marker_version":2,"owner":"task-133","next_stream_id":1}}`,
		"nested_del_zero_ver":    `{"schema_version":1,"owner":"task-133","metrics_delivery":{"schema_version":0,"marker_version":1,"owner":"task-133"}}`,
		"nested_owner_mismatch":  `{"schema_version":1,"owner":"task-133","retention":{"schema_version":1,"marker_version":2,"owner":"other","next_stream_id":1}}`,
		"unknown_field":          `{"schema_version":1,"owner":"task-133","unexpected":1}`,
		"duplicate_key":          `{"schema_version":1,"schema_version":1,"owner":"task-133"}`,
	} {
		p := filepath.Join(tmp, name+".json")
		if err := os.WriteFile(p, []byte(payload), 0600); err != nil {
			t.Fatal(err)
		}
		err := runServeContext(t.Context(), []string{
			"-mock", "-listen", "127.0.0.1:0", "-interval", "5ms", "-duration", "10ms",
			"-retention-owner", "task-133", "-recovery-manifest-in", p,
		}, io.Discard, io.Discard)
		if err == nil {
			t.Fatalf("runServeContext accepted invalid recovery manifest (%s): %s", name, payload)
		}
	}

	// 3. On-disk owner.json schema_version tampering refusal in both retention and durable delivery.
	tamperRetDir := filepath.Join(tmp, "tamper-ret")
	r, err := nvtelemetry.OpenRetention(tamperRetDir, nvtelemetry.RetentionConfig{
		Owner:          "task-133",
		MaxFiles:       2,
		MaxBytes:       1 << 20,
		MaxStreamBytes: 512 << 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tamperRetDir, "owner.json"), []byte(`{"version":2,"schema_version":99,"owner":"task-133"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := runServeContext(t.Context(), []string{
		"-mock", "-listen", "127.0.0.1:0", "-interval", "5ms", "-duration", "10ms",
		"-retention-dir", tamperRetDir, "-retention-owner", "task-133",
	}, io.Discard, io.Discard); err == nil {
		t.Fatal("runServeContext accepted retention owner.json with unknown schema_version 99")
	}

	tamperSpoolDir := filepath.Join(tmp, "tamper-spool")
	if err := os.Mkdir(tamperSpoolDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tamperSpoolDir, "owner.json"), []byte(`{"version":1,"schema_version":99,"owner":"task-133"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := nvtelemetry.RecoverDurableDeliveryManifest(tamperSpoolDir, "task-133"); err == nil {
		t.Fatal("RecoverDurableDeliveryManifest accepted owner.json with unknown schema_version 99")
	}
}
