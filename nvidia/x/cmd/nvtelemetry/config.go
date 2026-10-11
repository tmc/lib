package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"time"
)

// ServeConfig defines the JSON configuration schema for the nvtelemetry serve daemon.
// Unknown fields and duplicate object keys are strictly rejected.
type ServeConfig struct {
	Listen                string   `json:"listen"`
	Interval              string   `json:"interval"`
	MaxAge                string   `json:"max_age"`
	Duration              string   `json:"duration,omitempty"`
	Budget                int      `json:"budget"`
	ShutdownTimeout       string   `json:"shutdown_timeout,omitempty"`
	Mock                  bool     `json:"mock,omitempty"`
	DeviceUUIDs           []string `json:"device_uuids,omitempty"`
	ManagedConfig         string   `json:"managed_config,omitempty"`
	CounterStartUnixNanos int64    `json:"counter_start_unix_nanos,omitempty"`
	OTLPMetricsConfig     string   `json:"otlp_metrics_config,omitempty"`
	OTLPLogsConfig        string   `json:"otlp_logs_config,omitempty"`
	AuthTokenFile         string   `json:"auth_token_file,omitempty"`
	TLSCert               string   `json:"tls_cert,omitempty"`
	TLSKey                string   `json:"tls_key,omitempty"`
	AllowRemote           bool     `json:"allow_remote,omitempty"`
}

// ParsedConfig contains the validated in-memory settings for serving.
type ParsedConfig struct {
	Raw                   ServeConfig
	Listen                string
	Interval              time.Duration
	MaxAge                time.Duration
	Duration              time.Duration
	Budget                int
	ShutdownTimeout       time.Duration
	Mock                  bool
	DeviceUUIDs           []string
	ManagedConfig         string
	CounterStartUnixNanos int64
	OTLPMetricsConfig     string
	OTLPLogsConfig        string
	AuthTokenFile         string
	BearerToken           string
	TLSCert               string
	TLSKey                string
	TLSConfig             *tls.Config
	AllowRemote           bool
	IsLoopback            bool
}

// ParseConfigFile reads, validates, and parses a JSON configuration file.
func ParseConfigFile(path string) (*ParsedConfig, error) {
	if path == "" {
		return nil, errors.New("config file path required")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open config file: %w", err)
	}
	defer f.Close()

	// Enforce 1MB limit on config file size
	data, err := io.ReadAll(io.LimitReader(f, 1<<20+1))
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}
	if len(data) > 1<<20 {
		return nil, errors.New("config file exceeds maximum size of 1MB")
	}
	baseDir := filepath.Dir(path)
	return parseConfigBytes(data, baseDir)
}

// ParseConfigJSON parses and validates JSON configuration bytes.
func ParseConfigJSON(data []byte) (*ParsedConfig, error) {
	if len(data) > 1<<20 {
		return nil, errors.New("config data exceeds maximum size of 1MB")
	}
	return parseConfigBytes(data, "")
}

func parseConfigBytes(data []byte, baseDir string) (*ParsedConfig, error) {
	if len(data) > 1<<20 {
		return nil, errors.New("config data exceeds maximum size of 1MB")
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return nil, errors.New("empty configuration")
	}

	// 1. Strict duplicate key and null detection
	if err := checkJSONDuplicates(data); err != nil {
		return nil, fmt.Errorf("invalid configuration syntax: %w", err)
	}

	// 2. Strict decoding rejecting unknown fields
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var raw ServeConfig
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("decode configuration: %w", err)
	}
	if dec.More() {
		return nil, errors.New("extraneous data after JSON configuration object")
	}

	// 3. Resolve relative paths if baseDir is provided
	if baseDir != "" {
		raw.ManagedConfig = resolveRelative(baseDir, raw.ManagedConfig)
		raw.OTLPMetricsConfig = resolveRelative(baseDir, raw.OTLPMetricsConfig)
		raw.OTLPLogsConfig = resolveRelative(baseDir, raw.OTLPLogsConfig)
		raw.AuthTokenFile = resolveRelative(baseDir, raw.AuthTokenFile)
		raw.TLSCert = resolveRelative(baseDir, raw.TLSCert)
		raw.TLSKey = resolveRelative(baseDir, raw.TLSKey)
	}

	return validateAndBuildParsedConfig(raw)
}

func resolveRelative(baseDir, path string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(baseDir, path)
}

// checkJSONDuplicates walks tokens in data, requiring a top-level JSON object,
// rejecting null values and duplicate object keys, and checking for trailing tokens.
func checkJSONDuplicates(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if tok == nil {
		return errors.New("null configuration not permitted")
	}
	delim, ok := tok.(json.Delim)
	if !ok || delim != '{' {
		return errors.New("configuration must be a JSON object")
	}
	if err := checkObjectRecursive(dec); err != nil {
		return err
	}
	// Verify no trailing tokens
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("unexpected trailing tokens in JSON")
	}
	return nil
}

func checkObjectRecursive(dec *json.Decoder) error {
	seen := make(map[string]bool)
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return err
		}
		if keyTok == nil {
			return errors.New("null object key not permitted")
		}
		key, ok := keyTok.(string)
		if !ok {
			return fmt.Errorf("expected string object key, got %T", keyTok)
		}
		if seen[key] {
			return fmt.Errorf("duplicate key %q in JSON object", key)
		}
		seen[key] = true
		if err := checkValueRecursive(dec); err != nil {
			return err
		}
	}
	endTok, err := dec.Token()
	if err != nil {
		return err
	}
	if d, ok := endTok.(json.Delim); !ok || d != '}' {
		return fmt.Errorf("expected '}', got %v", endTok)
	}
	return nil
}

func checkValueRecursive(dec *json.Decoder) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if tok == nil {
		return errors.New("null values are not permitted in configuration")
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		return checkObjectRecursive(dec)
	case '[':
		for dec.More() {
			if err := checkValueRecursive(dec); err != nil {
				return err
			}
		}
		endTok, err := dec.Token()
		if err != nil {
			return err
		}
		if d, ok := endTok.(json.Delim); !ok || d != ']' {
			return fmt.Errorf("expected ']', got %v", endTok)
		}
		return nil
	default:
		return fmt.Errorf("unexpected delimiter %v", delim)
	}
}

func validateAndBuildParsedConfig(raw ServeConfig) (*ParsedConfig, error) {
	listen := raw.Listen
	if listen == "" {
		listen = "127.0.0.1:9464"
	}
	host, _, err := net.SplitHostPort(listen)
	if err != nil {
		return nil, fmt.Errorf("invalid listen address %q: %w", listen, err)
	}
	ip := net.ParseIP(host)
	loopback := ip != nil && ip.IsLoopback()

	interval := time.Second
	if raw.Interval != "" {
		d, err := time.ParseDuration(raw.Interval)
		if err != nil {
			return nil, fmt.Errorf("invalid interval %q: %w", raw.Interval, err)
		}
		interval = d
	}
	if interval <= 0 || interval > 10*time.Minute {
		return nil, fmt.Errorf("interval %v outside bounds (0, 10m]", interval)
	}

	maxAge := 5 * time.Second
	if raw.MaxAge != "" {
		d, err := time.ParseDuration(raw.MaxAge)
		if err != nil {
			return nil, fmt.Errorf("invalid max_age %q: %w", raw.MaxAge, err)
		}
		maxAge = d
	}
	if maxAge <= 0 || maxAge > 10*time.Minute {
		return nil, fmt.Errorf("max_age %v outside bounds (0, 10m]", maxAge)
	}

	var duration time.Duration
	if raw.Duration != "" {
		d, err := time.ParseDuration(raw.Duration)
		if err != nil {
			return nil, fmt.Errorf("invalid duration %q: %w", raw.Duration, err)
		}
		duration = d
	}
	if duration < 0 {
		return nil, errors.New("duration cannot be negative")
	}

	budget := raw.Budget
	if budget == 0 {
		budget = 100000
	}
	if budget < 1 || budget > 1<<20 {
		return nil, fmt.Errorf("budget %d outside bounds [1, %d]", budget, 1<<20)
	}

	shutdownTimeout := 5 * time.Second
	if raw.ShutdownTimeout != "" {
		d, err := time.ParseDuration(raw.ShutdownTimeout)
		if err != nil {
			return nil, fmt.Errorf("invalid shutdown_timeout %q: %w", raw.ShutdownTimeout, err)
		}
		shutdownTimeout = d
	}
	if shutdownTimeout <= 0 || shutdownTimeout > time.Minute {
		return nil, fmt.Errorf("shutdown_timeout %v outside bounds (0, 1m]", shutdownTimeout)
	}

	if raw.CounterStartUnixNanos < 0 {
		return nil, errors.New("invalid counter epoch start")
	}

	// Token validation
	token, err := readBearerToken(raw.AuthTokenFile)
	if err != nil {
		return nil, err
	}

	// TLS validation
	if (raw.TLSCert == "") != (raw.TLSKey == "") {
		return nil, errors.New("TLS certificate and key are both required")
	}
	var tlsConfig *tls.Config
	if raw.TLSCert != "" {
		pair, err := tls.LoadX509KeyPair(raw.TLSCert, raw.TLSKey)
		if err != nil {
			return nil, errors.New("invalid TLS certificate or key")
		}
		tlsConfig = &tls.Config{Certificates: []tls.Certificate{pair}, MinVersion: tls.VersionTLS12}
	}

	if !loopback && (!raw.AllowRemote || raw.TLSCert == "" || token == "") {
		return nil, errors.New("non-loopback serving requires allow-remote, TLS and bearer authentication")
	}

	// Device / Sampler mutual exclusivity
	if raw.Mock && len(raw.DeviceUUIDs) > 0 {
		return nil, errors.New("mock mode does not select physical UUIDs")
	}
	if raw.Mock && raw.ManagedConfig != "" {
		return nil, errors.New("managed endpoint cannot select mock or NVML devices")
	}
	if raw.ManagedConfig != "" && len(raw.DeviceUUIDs) > 0 {
		return nil, errors.New("managed endpoint cannot select mock or NVML devices")
	}

	// Validate OTLP configs existence / syntax if specified
	if raw.OTLPMetricsConfig != "" {
		deliv, cleanup, err := newEndpointDelivery(raw.OTLPMetricsConfig, false)
		if err != nil {
			return nil, err
		}
		if cleanup != nil {
			cleanup()
		}
		if deliv != nil {
			_ = deliv.Close(context.Background())
		}
	}
	if raw.OTLPLogsConfig != "" {
		deliv, cleanup, err := newEndpointDelivery(raw.OTLPLogsConfig, true)
		if err != nil {
			return nil, err
		}
		if cleanup != nil {
			cleanup()
		}
		if deliv != nil {
			_ = deliv.Close(context.Background())
		}
	}

	// Validate Device UUIDs if present
	var selectedUUIDs []string
	if len(raw.DeviceUUIDs) > 0 {
		if len(raw.DeviceUUIDs) > 128 {
			return nil, errors.New("device UUID allowlist exceeds device limit")
		}
		seen := make(map[string]bool, len(raw.DeviceUUIDs))
		for _, uuid := range raw.DeviceUUIDs {
			if len(uuid) > 128 || !deviceUUIDPattern.MatchString(uuid) || seen[uuid] {
				return nil, errors.New("invalid or duplicate device UUID")
			}
			seen[uuid] = true
			selectedUUIDs = append(selectedUUIDs, uuid)
		}
	}

	// Validate ManagedConfig if specified
	if raw.ManagedConfig != "" {
		c, client, _, err := readEndpointConfig(raw.ManagedConfig)
		if err != nil {
			return nil, err
		}
		client.CloseIdleConnections()
		if c.Managed == nil || c.MaxPages != 0 || c.Method != "" && c.Method != "GET" || c.BodyFile != "" || len(c.Devices) == 0 || len(c.Devices) > 256 {
			return nil, errors.New("managed serving requires explicit mappings and device identities")
		}
		seen := map[string]bool{}
		for _, d := range c.Devices {
			if d.UUID == "" || len(d.UUID) > 256 || seen[d.UUID] {
				return nil, errors.New("invalid managed device identities")
			}
			seen[d.UUID] = true
		}
	}

	return &ParsedConfig{
		Raw:                   raw,
		Listen:                listen,
		Interval:              interval,
		MaxAge:                maxAge,
		Duration:              duration,
		Budget:                budget,
		ShutdownTimeout:       shutdownTimeout,
		Mock:                  raw.Mock,
		DeviceUUIDs:           selectedUUIDs,
		ManagedConfig:         raw.ManagedConfig,
		CounterStartUnixNanos: raw.CounterStartUnixNanos,
		OTLPMetricsConfig:     raw.OTLPMetricsConfig,
		OTLPLogsConfig:        raw.OTLPLogsConfig,
		AuthTokenFile:         raw.AuthTokenFile,
		BearerToken:           token,
		TLSCert:               raw.TLSCert,
		TLSKey:                raw.TLSKey,
		TLSConfig:             tlsConfig,
		AllowRemote:           raw.AllowRemote,
		IsLoopback:            loopback,
	}, nil
}
