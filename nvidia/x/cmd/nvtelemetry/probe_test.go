package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const probeTestToken = "probe-test-secret"

func TestRunProbeAuthenticatedTLS(t *testing.T) {
	certPEM, keyPEM := makeProbeTestCertificate(t)
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer "+probeTestToken {
			t.Errorf("Authorization = %q, want bearer token", got)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		switch r.URL.Path {
		case "/healthz":
			_, _ = io.WriteString(w, "ok\n")
		case "/readyz":
			_, _ = io.WriteString(w, "ready\n")
		default:
			http.NotFound(w, r)
		}
	}))
	server.TLS = &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}}
	server.StartTLS()
	defer server.Close()

	dir := t.TempDir()
	caFile := filepath.Join(dir, "ca.crt")
	if err := os.WriteFile(caFile, certPEM, 0600); err != nil {
		t.Fatal(err)
	}
	certFile := filepath.Join(dir, "tls.crt")
	if err := os.WriteFile(certFile, certPEM, 0600); err != nil {
		t.Fatal(err)
	}
	keyFile := filepath.Join(dir, "tls.key")
	if err := os.WriteFile(keyFile, keyPEM, 0600); err != nil {
		t.Fatal(err)
	}
	tokenFile := filepath.Join(dir, "bearer-token")
	if err := os.WriteFile(tokenFile, []byte(probeTestToken), 0600); err != nil {
		t.Fatal(err)
	}
	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	configFile := filepath.Join(dir, "config.json")
	config := map[string]any{
		"listen":          serverURL.Host,
		"tls_cert":        certFile,
		"tls_key":         keyFile,
		"auth_token_file": tokenFile,
	}
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configFile, data, 0600); err != nil {
		t.Fatal(err)
	}

	for _, kind := range []string{"liveness", "readiness"} {
		t.Run(kind, func(t *testing.T) {
			args := []string{"-kind=" + kind, "-config=" + configFile, "-ca-file=" + caFile, "-server-name=probe.example.test"}
			if err := runProbe(args, io.Discard, io.Discard); err != nil {
				t.Fatalf("runProbe: %v", err)
			}
		})
	}
}

func TestProbeEndpointRejectsInvalidResponses(t *testing.T) {
	certPEM, keyPEM := makeProbeTestCertificate(t)
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{"unauthorized", func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "reflected "+probeTestToken, http.StatusUnauthorized)
		}},
		{"redirect", func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, "/healthz/other", http.StatusFound)
		}},
		{"wrong status", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, "ok\n")
		}},
		{"wrong body", func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.WriteString(w, "ok\nextra")
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewUnstartedServer(tt.handler)
			server.TLS = &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}}
			server.StartTLS()
			defer server.Close()
			caFile := writeProbeTestCA(t, certPEM)
			cfg := probeTestConfig(t, server.URL)
			err := probeEndpoint(context.Background(), cfg, caFile, "probe.example.test", probeRoutes["liveness"])
			if err == nil {
				t.Fatal("probeEndpoint unexpectedly succeeded")
			}
			if strings.Contains(err.Error(), probeTestToken) || strings.Contains(err.Error(), "reflected") {
				t.Fatalf("probe error disclosed response or token: %v", err)
			}
		})
	}
}

func TestProbeEndpointRejectsUntrustedCertificateNameAndTimeout(t *testing.T) {
	certPEM, keyPEM := makeProbeTestCertificate(t)
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	server.TLS = &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}}
	server.StartTLS()
	defer server.Close()
	caFile := writeProbeTestCA(t, certPEM)
	cfg := probeTestConfig(t, server.URL)
	invalidCA := filepath.Join(t.TempDir(), "invalid-ca.crt")
	if err := os.WriteFile(invalidCA, []byte("not a PEM certificate"), 0600); err != nil {
		t.Fatal(err)
	}

	if err := probeEndpoint(context.Background(), cfg, caFile, "wrong.example.test", probeRoutes["liveness"]); err == nil {
		t.Fatal("probe with wrong TLS server name unexpectedly succeeded")
	}
	if err := probeEndpoint(context.Background(), cfg, invalidCA, "probe.example.test", probeRoutes["liveness"]); err == nil {
		t.Fatal("probe with invalid CA PEM unexpectedly succeeded")
	}
	if err := probeEndpoint(context.Background(), cfg, filepath.Join(t.TempDir(), "missing-ca"), "probe.example.test", probeRoutes["liveness"]); err == nil {
		t.Fatal("probe with missing CA unexpectedly succeeded")
	}
	start := time.Now()
	err = probeEndpoint(context.Background(), cfg, caFile, "probe.example.test", probeRoutes["liveness"])
	if err == nil {
		t.Fatal("probe with stalled response unexpectedly succeeded")
	}
	if elapsed := time.Since(start); elapsed > 3*probeTimeout {
		t.Fatalf("probe exceeded bounded timeout: %v", elapsed)
	}
	if strings.Contains(err.Error(), probeTestToken) {
		t.Fatalf("probe error disclosed token: %v", err)
	}
}

func TestRunProbeRejectsMissingConfig(t *testing.T) {
	err := runProbe([]string{
		"-kind=liveness",
		"-config=" + filepath.Join(t.TempDir(), "missing.json"),
		"-ca-file=" + filepath.Join(t.TempDir(), "ca.crt"),
		"-server-name=probe.example.test",
	}, io.Discard, io.Discard)
	if err == nil || strings.Contains(err.Error(), "missing.json") {
		t.Fatalf("runProbe error = %v, want redacted configuration error", err)
	}
}

func TestProbeEndpointRequiresAuthenticationAndTLS(t *testing.T) {
	for name, config := range map[string]*ParsedConfig{
		"nil":           nil,
		"missing token": {Listen: "127.0.0.1:9464", TLSCert: "cert", TLSKey: "key"},
		"missing TLS":   {Listen: "127.0.0.1:9464", BearerToken: probeTestToken},
	} {
		t.Run(name, func(t *testing.T) {
			if err := probeEndpoint(context.Background(), config, "ca.crt", "probe.example.test", probeRoutes["liveness"]); err == nil {
				t.Fatal("probeEndpoint unexpectedly succeeded")
			}
		})
	}
}

func TestProbeAddress(t *testing.T) {
	tests := []struct {
		listen string
		want   string
	}{
		{"127.0.0.1:9464", "127.0.0.1:9464"},
		{":9464", "127.0.0.1:9464"},
		{"0.0.0.0:9464", "127.0.0.1:9464"},
		{"[::]:9464", "[::1]:9464"},
		{"[::1]:9464", "[::1]:9464"},
		{"127.0.0.2:9464", "127.0.0.2:9464"},
	}
	for _, tt := range tests {
		t.Run(tt.listen, func(t *testing.T) {
			got, err := probeAddress(tt.listen)
			if err != nil || got != tt.want {
				t.Fatalf("probeAddress(%q) = %q, %v; want %q", tt.listen, got, err, tt.want)
			}
		})
	}
	for _, listen := range []string{"probe.example.test:9464", "192.0.2.4:9464", "127.0.0.1:0", "127.0.0.1:http", "bad"} {
		if got, err := probeAddress(listen); err == nil {
			t.Errorf("probeAddress(%q) = %q, want error", listen, got)
		}
	}
}

func TestValidProbeServerName(t *testing.T) {
	for _, name := range []string{"probe.example.test", "127.0.0.1", "2001:db8::1"} {
		if !validProbeServerName(name) {
			t.Errorf("validProbeServerName(%q) = false", name)
		}
	}
	for _, name := range []string{"", " host", "host/path", "*.example.test", "a..b", "-bad.example"} {
		if validProbeServerName(name) {
			t.Errorf("validProbeServerName(%q) = true", name)
		}
	}
}

func probeTestConfig(t *testing.T, serverURL string) *ParsedConfig {
	t.Helper()
	u, err := url.Parse(serverURL)
	if err != nil {
		t.Fatal(err)
	}
	return &ParsedConfig{Listen: u.Host, BearerToken: probeTestToken, TLSCert: "server.crt", TLSKey: "server.key"}
}

func writeProbeTestCA(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "ca.crt")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func makeProbeTestCertificate(t *testing.T) ([]byte, []byte) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	template := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "probe.example.test"},
		NotBefore:             now.Add(-time.Minute),
		NotAfter:              now.Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		DNSNames:              []string{"probe.example.test"},
		IPAddresses:           []net.IP{net.ParseIP("127.0.0.1")},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
}
