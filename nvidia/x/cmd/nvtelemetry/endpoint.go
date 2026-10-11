package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// endpointConfig carries file references, never inline authentication secrets.
// Workload responses are preserved as raw bytes and source hashes in the output.
type endpointConfig struct {
	Method          string                       `json:"method,omitempty"`
	BodyFile        string                       `json:"body_file,omitempty"`
	MaxPages        int                          `json:"max_pages,omitempty"`
	URL             string                       `json:"url"`
	HeadersFile     string                       `json:"headers_file,omitempty"`
	BearerTokenFile string                       `json:"bearer_token_file,omitempty"`
	CAFile          string                       `json:"ca_file,omitempty"`
	AllowHTTP       bool                         `json:"allow_http,omitempty"`
	Timeout         time.Duration                `json:"timeout_nanos"`
	MaxBytes        int64                        `json:"max_bytes"`
	Workload        *nvtelemetry.WorkloadSource  `json:"workload,omitempty"`
	Managed         *nvtelemetry.ManagedImport   `json:"managed,omitempty"`
	Devices         []nvtelemetry.DeviceMetadata `json:"devices,omitempty"`
}

func readEndpointConfig(path string) (endpointConfig, *http.Client, *http.Request, error) {
	var c endpointConfig
	raw, err := readEndpointFile(path, 64<<10)
	if err != nil {
		return c, nil, nil, err
	}
	if err := checkEndpointJSON(json.NewDecoder(bytes.NewReader(raw)), 0); err != nil {
		return c, nil, nil, errors.New("invalid endpoint configuration JSON")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return c, nil, nil, errors.New("invalid endpoint configuration fields")
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return c, nil, nil, errors.New("trailing endpoint configuration JSON")
	}
	if c.Timeout <= 0 || c.Timeout > 30*time.Second || c.MaxBytes <= 0 || c.MaxBytes > 64<<20 || c.Workload != nil && c.Managed != nil {
		return c, nil, nil, errors.New("invalid endpoint configuration bounds or source")
	}
	method := c.Method
	if method == "" {
		method = http.MethodGet
	}
	if method != http.MethodGet && method != http.MethodPost || c.MaxPages < 0 || c.MaxPages > 32 || method == http.MethodGet && c.BodyFile != "" {
		return c, nil, nil, errors.New("invalid endpoint request method or paging bounds")
	}
	var body io.Reader
	if c.BodyFile != "" {
		raw, err := readEndpointFile(c.BodyFile, 1<<20)
		if err != nil {
			return c, nil, nil, err
		}
		body = bytes.NewReader(raw)
	}
	request, err := http.NewRequest(method, c.URL, body)
	if err != nil || request.URL.Host == "" || request.URL.User != nil || request.URL.Fragment != "" || (request.URL.Scheme != "https" && request.URL.Scheme != "http") {
		return c, nil, nil, errors.New("invalid endpoint URL")
	}
	ip := net.ParseIP(request.URL.Hostname())
	if request.URL.Scheme == "http" && !c.AllowHTTP && (ip == nil || !ip.IsLoopback()) {
		return c, nil, nil, errors.New("remote HTTP endpoint requires explicit allow_http")
	}
	if c.HeadersFile != "" && c.BearerTokenFile != "" {
		return c, nil, nil, errors.New("endpoint authentication files must be selected explicitly")
	}
	headers, err := readEndpointHeaders(c.HeadersFile)
	if err != nil {
		return c, nil, nil, err
	}
	request.Header = headers
	token, err := readBearerToken(c.BearerTokenFile)
	if err != nil {
		return c, nil, nil, err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if c.CAFile != "" {
		pem, err := readEndpointFile(c.CAFile, 1<<20)
		if err != nil {
			return c, nil, nil, err
		}
		pool, err := x509.SystemCertPool()
		if err != nil {
			pool = x509.NewCertPool()
		}
		if !pool.AppendCertsFromPEM(pem) {
			return c, nil, nil, errors.New("invalid endpoint CA certificate")
		}
		transport.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
	}
	return c, &http.Client{Transport: transport, Timeout: c.Timeout}, request, nil
}

func readEndpointFile(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, errors.New("open endpoint configuration file failed")
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(raw)) > limit {
		return nil, errors.New("endpoint configuration file exceeds bounds or cannot be read")
	}
	return raw, nil
}

// Reject duplicate keys and excessive nesting before decoding into typed fields.
func checkEndpointJSON(d *json.Decoder, depth int) error {
	if depth > 32 {
		return errors.New("endpoint JSON depth limit")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[strings.ToLower(name)] {
				return errors.New("duplicate endpoint configuration key")
			}
			seen[strings.ToLower(name)] = true
			if err := checkEndpointJSON(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := checkEndpointJSON(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("invalid endpoint configuration JSON")
	}
	_, err = d.Token()
	return err
}

func runFetchEndpoint(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("fetch-endpoint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("config", "", "explicit endpoint configuration JSON file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *path == "" || fs.NArg() != 0 {
		return errors.New("fetch-endpoint requires a config file")
	}
	c, client, request, err := readEndpointConfig(*path)
	if err != nil {
		return err
	}
	defer client.CloseIdleConnections()
	if c.Workload == nil && c.Managed == nil {
		return errors.New("fetch-endpoint requires a workload or managed source")
	}
	ctx, cancel := context.WithTimeout(context.Background(), c.Timeout)
	defer cancel()
	var result any
	if c.Workload != nil {
		if c.MaxPages > 0 {
			switch c.Workload.Provider {
			case "mlflow":
				result, err = nvtelemetry.FetchMLflowHistoryPages(ctx, client, request, *c.Workload, nvtelemetry.WorkloadLimits{MaxBytes: c.MaxBytes}, c.MaxPages)
			case "clearml":
				result, err = nvtelemetry.FetchClearMLScalarPages(ctx, client, request, *c.Workload, nvtelemetry.WorkloadLimits{MaxBytes: c.MaxBytes}, c.MaxPages)
			default:
				return errors.New("workload provider does not support pagination")
			}
		} else {
			result, err = nvtelemetry.FetchWorkload(ctx, client, request, *c.Workload, nvtelemetry.WorkloadLimits{MaxBytes: c.MaxBytes})
		}
	} else {
		if c.MaxPages != 0 {
			return errors.New("managed endpoint does not support pagination")
		}
		result, err = nvtelemetry.FetchManaged(ctx, client, request, *c.Managed, nvtelemetry.RealClock{}, c.MaxBytes)
	}
	if err != nil {
		return err
	}
	return json.NewEncoder(stdout).Encode(result)
}

type managedEndpointSampler struct {
	config  endpointConfig
	client  *http.Client
	request *http.Request
	clock   nvtelemetry.Clock
}

func (s *managedEndpointSampler) Devices() []nvtelemetry.DeviceMetadata {
	return append([]nvtelemetry.DeviceMetadata(nil), s.config.Devices...)
}
func (s *managedEndpointSampler) Sample(ctx context.Context) ([]nvtelemetry.Reading, error) {
	sample, err := nvtelemetry.FetchManaged(ctx, s.client, s.request, *s.config.Managed, s.clock, s.config.MaxBytes)
	if err != nil {
		return nil, err
	}
	for _, r := range sample.Readings {
		found := false
		for _, device := range s.config.Devices {
			if r.DeviceUUID == device.UUID {
				found = true
				break
			}
		}
		if !found {
			return nil, errors.New("managed endpoint returned an unselected device")
		}
	}
	return sample.Readings, nil
}

// readEndpointHeaders supports explicit provider API-key and Basic/Bearer
// authentication without putting secrets in command arguments or config JSON.
func readEndpointHeaders(path string) (http.Header, error) {
	headers := make(http.Header)
	if path == "" {
		return headers, nil
	}
	raw, err := readEndpointFile(path, 16<<10)
	if err != nil {
		return nil, err
	}
	if err := checkEndpointJSON(json.NewDecoder(bytes.NewReader(raw)), 0); err != nil {
		return nil, errors.New("invalid endpoint headers JSON")
	}
	var values map[string]json.RawMessage
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := d.Decode(&values); err != nil || values == nil || len(values) > 32 {
		return nil, errors.New("invalid endpoint headers")
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return nil, errors.New("trailing endpoint headers JSON")
	}
	for key, encoded := range values {
		var value string
		encoded = bytes.TrimSpace(encoded)
		if len(encoded) == 0 || encoded[0] != '"' || json.Unmarshal(encoded, &value) != nil {
			return nil, errors.New("endpoint header requires a string value")
		}
		if len(key) == 0 || len(key) > 128 || len(value) > 4096 {
			return nil, errors.New("endpoint header bounds exceeded")
		}
		for _, r := range key {
			if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("!#$%&'*+-.^_`|~", r)) {
				return nil, errors.New("invalid endpoint header name")
			}
		}
		for _, r := range value {
			if r < 32 || r > 126 {
				return nil, errors.New("invalid endpoint header value")
			}
		}
		switch strings.ToLower(key) {
		case "host", "connection", "proxy-authorization", "proxy-connection", "transfer-encoding", "content-length", "trailer", "upgrade":
			return nil, errors.New("endpoint transport header override refused")
		}
		headers.Set(key, value)
	}
	return headers, nil
}
