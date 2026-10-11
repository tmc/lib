package main

import (
	"context"
	"crypto/x509"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
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
