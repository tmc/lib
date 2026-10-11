package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestConfigParserStrictValidation(t *testing.T) {
	// 1. Valid minimal mock configuration
	validJSON := `{
		"listen": "127.0.0.1:9464",
		"interval": "500ms",
		"max_age": "2s",
		"budget": 5000,
		"shutdown_timeout": "3s",
		"mock": true
	}`
	cfg, err := ParseConfigJSON([]byte(validJSON))
	if err != nil {
		t.Fatalf("ParseConfigJSON valid config: %v", err)
	}
	if cfg.Listen != "127.0.0.1:9464" {
		t.Errorf("Listen = %q, want 127.0.0.1:9464", cfg.Listen)
	}
	if cfg.Interval != 500*time.Millisecond {
		t.Errorf("Interval = %v, want 500ms", cfg.Interval)
	}
	if cfg.MaxAge != 2*time.Second {
		t.Errorf("MaxAge = %v, want 2s", cfg.MaxAge)
	}
	if cfg.Budget != 5000 {
		t.Errorf("Budget = %d, want 5000", cfg.Budget)
	}
	if cfg.ShutdownTimeout != 3*time.Second {
		t.Errorf("ShutdownTimeout = %v, want 3s", cfg.ShutdownTimeout)
	}
	if !cfg.Mock {
		t.Errorf("Mock = false, want true")
	}

	// 2. Reject unknown fields
	unknownJSON := `{
		"listen": "127.0.0.1:9464",
		"mock": true,
		"unrecognized_field": "disallowed"
	}`
	if _, err := ParseConfigJSON([]byte(unknownJSON)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown field rejection, got: %v", err)
	}

	// 3. Reject duplicate keys
	duplicateJSON := `{
		"listen": "127.0.0.1:9464",
		"mock": true,
		"listen": "127.0.0.1:9465"
	}`
	if _, err := ParseConfigJSON([]byte(duplicateJSON)); err == nil || !strings.Contains(err.Error(), "duplicate key") {
		t.Fatalf("expected duplicate key rejection, got: %v", err)
	}

	// 4. Reject duplicate keys inside nested objects
	nestedDuplicateJSON := `{
		"listen": "127.0.0.1:9464",
		"mock": true,
		"raw": {
			"dup": 1,
			"dup": 2
		}
	}`
	if _, err := ParseConfigJSON([]byte(nestedDuplicateJSON)); err == nil || !strings.Contains(err.Error(), "duplicate key") {
		t.Fatalf("expected nested duplicate key rejection, got: %v", err)
	}

	// 5. Reject empty configuration
	if _, err := ParseConfigJSON([]byte("   ")); err == nil {
		t.Fatal("expected rejection of empty configuration")
	}

	// 6. Bounds checks
	table := []struct {
		name    string
		json    string
		wantErr string
	}{
		{
			name:    "interval zero",
			json:    `{"mock": true, "interval": "0s"}`,
			wantErr: "outside bounds",
		},
		{
			name:    "interval negative",
			json:    `{"mock": true, "interval": "-10ms"}`,
			wantErr: "outside bounds",
		},
		{
			name:    "interval too large",
			json:    `{"mock": true, "interval": "1h"}`,
			wantErr: "outside bounds",
		},
		{
			name:    "max_age zero",
			json:    `{"mock": true, "max_age": "0s"}`,
			wantErr: "outside bounds",
		},
		{
			name:    "budget zero",
			json:    `{"mock": true, "budget": 0}`,
			wantErr: "", // Budget 0 defaults to 100000
		},
		{
			name:    "budget negative",
			json:    `{"mock": true, "budget": -5}`,
			wantErr: "outside bounds",
		},
		{
			name:    "budget exceeds max",
			json:    `{"mock": true, "budget": 2000000}`,
			wantErr: "outside bounds",
		},
		{
			name:    "shutdown_timeout too large",
			json:    `{"mock": true, "shutdown_timeout": "2m"}`,
			wantErr: "outside bounds",
		},
		{
			name:    "negative counter start",
			json:    `{"mock": true, "counter_start_unix_nanos": -1}`,
			wantErr: "invalid counter epoch start",
		},
		{
			name:    "mock and device_uuids conflict",
			json:    `{"mock": true, "device_uuids": ["GPU-11111111-2222-3333-4444-555555555555"]}`,
			wantErr: "mock mode does not select physical UUIDs",
		},
		{
			name:    "mock and managed_config conflict",
			json:    `{"mock": true, "managed_config": "managed.json"}`,
			wantErr: "managed endpoint cannot select mock or NVML devices",
		},
		{
			name:    "invalid device UUID format",
			json:    `{"device_uuids": ["not-a-valid-uuid"]}`,
			wantErr: "invalid or duplicate device UUID",
		},
		{
			name:    "remote without allow_remote",
			json:    `{"mock": true, "listen": "0.0.0.0:9464"}`,
			wantErr: "non-loopback serving requires allow-remote",
		},
		{
			name:    "tls cert without key",
			json:    `{"mock": true, "tls_cert": "cert.pem"}`,
			wantErr: "TLS certificate and key are both required",
		},
	}

	for _, tc := range table {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseConfigJSON([]byte(tc.json))
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("got error %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestParseConfigFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.json")

	content := `{
		"listen": "127.0.0.1:9464",
		"interval": "100ms",
		"max_age": "1s",
		"mock": true
	}`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := ParseConfigFile(path)
	if err != nil {
		t.Fatalf("ParseConfigFile: %v", err)
	}
	if cfg.Interval != 100*time.Millisecond {
		t.Errorf("Interval = %v, want 100ms", cfg.Interval)
	}

	// Missing file test
	if _, err := ParseConfigFile(filepath.Join(dir, "missing.json")); err == nil {
		t.Fatal("expected error for missing config file")
	}
}
