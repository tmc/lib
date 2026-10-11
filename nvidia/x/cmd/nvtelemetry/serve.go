package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"crypto/tls"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func runServe(args []string, stdout, stderr io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runServeContext(ctx, args, stdout, stderr)
}

func runServeContext(parent context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	metricsConfig := fs.String("otlp-metrics-config", "", "optional transport-only OTLP metrics endpoint JSON")
	logsConfig := fs.String("otlp-logs-config", "", "optional transport-only OTLP diagnostic logs endpoint JSON")
	counterStart := fs.Int64("counter-start-unix-nanos", 0, "explicit counter epoch start for OTLP metrics (0 refuses counters)")
	address := fs.String("listen", "127.0.0.1:9464", "HTTP listening address")
	interval := fs.Duration("interval", time.Second, "sampling interval")
	duration := fs.Duration("duration", 0, "maximum serving duration (0 until signal or budget)")
	maxAge := fs.Duration("max-age", 5*time.Second, "maximum snapshot publication age")
	budget := fs.Int("budget", 100000, "maximum stream records before stopping")
	managedConfig := fs.String("managed-config", "", "explicit managed endpoint configuration JSON file")
	mock := fs.Bool("mock", false, "serve explicitly synthetic mock readings")
	uuids := fs.String("device-uuids", "", "exact UUID allowlist (required for live NVML)")
	tokenFile := fs.String("auth-token-file", "", "file containing an HTTP bearer token")
	cert := fs.String("tls-cert", "", "TLS certificate file")
	key := fs.String("tls-key", "", "TLS private key file")
	remote := fs.Bool("allow-remote", false, "allow a non-loopback listener with TLS and bearer authentication")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *interval <= 0 || *duration < 0 || *maxAge <= 0 || *budget < 1 || *budget > 1<<20 {
		return errors.New("invalid serving options or bounds")
	}
	host, _, err := net.SplitHostPort(*address)
	if err != nil {
		return errors.New("invalid listen address")
	}
	ip := net.ParseIP(host)
	loopback := ip != nil && ip.IsLoopback()
	token, err := readBearerToken(*tokenFile)
	if err != nil {
		return err
	}
	if (*cert == "") != (*key == "") {
		return errors.New("TLS certificate and key are both required")
	}
	if !loopback && (!*remote || *cert == "" || token == "") {
		return errors.New("non-loopback serving requires allow-remote, TLS and bearer authentication")
	}
	var tlsConfig *tls.Config
	if *cert != "" {
		pair, err := tls.LoadX509KeyPair(*cert, *key)
		if err != nil {
			return errors.New("invalid TLS certificate or key")
		}
		tlsConfig = &tls.Config{Certificates: []tls.Certificate{pair}, MinVersion: tls.VersionTLS12}
	}
	var metricsDelivery, logsDelivery *nvtelemetry.Delivery
	if *counterStart < 0 {
		return errors.New("invalid counter epoch start")
	}
	if *metricsConfig != "" {
		var cleanup func()
		metricsDelivery, cleanup, err = newEndpointDelivery(*metricsConfig, false)
		if err != nil {
			return err
		}
		defer cleanup()
		defer metricsDelivery.Close(context.Background())
	}
	if *logsConfig != "" {
		var cleanup func()
		logsDelivery, cleanup, err = newEndpointDelivery(*logsConfig, true)
		if err != nil {
			return err
		}
		defer cleanup()
		defer logsDelivery.Close(context.Background())
	}
	selected, err := parseDeviceUUIDs(*uuids)
	if err != nil {
		return err
	}
	var sampler nvtelemetry.Sampler
	if *managedConfig != "" {
		if *mock || len(selected) > 0 {
			return errors.New("managed endpoint cannot select mock or NVML devices")
		}
		c, client, request, err := readEndpointConfig(*managedConfig)
		if err != nil {
			return err
		}
		defer client.CloseIdleConnections()
		if c.Managed == nil || c.MaxPages != 0 || c.Method != "" && c.Method != "GET" || c.BodyFile != "" || len(c.Devices) == 0 || len(c.Devices) > 256 {
			return errors.New("managed serving requires explicit mappings and device identities")
		}
		seen := map[string]bool{}
		for _, d := range c.Devices {
			if d.UUID == "" || len(d.UUID) > 256 || seen[d.UUID] {
				return errors.New("invalid managed device identities")
			}
			seen[d.UUID] = true
		}
		sampler = &managedEndpointSampler{config: c, client: client, request: request, clock: nvtelemetry.RealClock{}}
	} else if *mock {
		if len(selected) > 0 {
			return errors.New("mock mode does not select physical UUIDs")
		}
		fake := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock", Model: "synthetic", Status: nvtelemetry.DeviceStatusOk})
		fake.SetFallback(func(ctx context.Context) ([]nvtelemetry.Reading, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU-mock", Evidence: nvtelemetry.EvidenceDerived, Backend: "mock", Value: 30000, TimestampUnix: time.Now().UnixNano()}}, nil
		})
		sampler = fake
	} else {
		if len(selected) == 0 {
			return errors.New("live serving requires an explicit device UUID allowlist")
		}
		if err := nvml.Init(); err != nil {
			return fmt.Errorf("initialize NVML: %w", err)
		}
		defer nvml.Shutdown()
		devices := make([]nvml.Device, 0, len(selected))
		for _, uuid := range selected {
			device, err := nvml.NewDeviceByUUID(uuid)
			if err != nil {
				return fmt.Errorf("select NVML device: %w", err)
			}
			actual, err := device.UUID()
			if err != nil || actual != uuid {
				return errors.New("selected NVML device identity mismatch")
			}
			devices = append(devices, device)
		}
		sampler, err = nvtelemetry.NewNVMLSampler(nvtelemetry.RealClock{}, devices...)
		if err != nil {
			return err
		}
	}
	handler, err := nvtelemetry.NewMetricsHandler(nvtelemetry.ExportOptions{}, *maxAge)
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.Handle("/metrics", handler)
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", 405)
			return
		}
		if !handler.Ready() {
			http.Error(w, "telemetry unavailable", 503)
			return
		}
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, "ready\n")
		}
	})
	listener, err := net.Listen("tcp", *address)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	defer listener.Close()
	server := &http.Server{Handler: bearerHandler(mux, token), TLSConfig: tlsConfig, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 10 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	if *duration > 0 {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, *duration)
		defer stop()
	}
	cfg := nvtelemetry.CollectorConfig{IntervalNanos: interval.Nanoseconds(), MaxRecordBudget: *budget, MaxByteBudget: 128 << 20, MaxBufferedRecords: 1, DropOnFull: true}
	writerConfig := cfg
	writerConfig.DropOnFull = false
	collector := nvtelemetry.NewCollector("http", sampler, nvtelemetry.RealClock{}, cfg, nvtelemetry.NewWriter(io.Discard, writerConfig))
	publishErr := make(chan error, 1)
	collector.SetSampleHook(func(s nvtelemetry.Sample) {
		// Admission is nonblocking; once admitted, both signal queues drain independently
		// of collector cancellation. Refuse a whole callback that begins after shutdown.
		if ctx.Err() != nil {
			return
		}
		numeric := make([]nvtelemetry.Reading, 0, len(s.Readings))
		for _, reading := range s.Readings {
			kind, ok := nvtelemetry.LookupMetricKind(reading.Name)
			if !ok || kind != nvtelemetry.KindEvent && kind != nvtelemetry.KindBitmask {
				numeric = append(numeric, reading)
			}
		}
		exportErr := handler.Update(numeric)
		opts := nvtelemetry.ExportOptions{MaxBytes: 4 << 20, MaxReadings: 4096, HasCounterStart: *counterStart > 0, CounterStartUnixNanos: *counterStart}
		if exportErr == nil && metricsDelivery != nil {
			available := make([]nvtelemetry.Reading, 0, len(numeric))
			for _, r := range numeric {
				if r.IsAvailable() {
					available = append(available, r)
				}
			}
			if len(available) > 0 {
				var encoded bytes.Buffer
				exportErr = nvtelemetry.WriteOTLPMetrics(&encoded, available, opts)
				if exportErr == nil {
					exportErr = metricsDelivery.Enqueue(context.Background(), encoded.Bytes())
				}
			}
		}
		if exportErr == nil && logsDelivery != nil && len(s.Readings) > 0 {
			var encoded bytes.Buffer
			exportErr = nvtelemetry.WriteOTLPLogs(&encoded, s.Readings, opts)
			if exportErr == nil {
				exportErr = logsDelivery.Enqueue(context.Background(), encoded.Bytes())
			}
		}
		if err := exportErr; err != nil {
			select {
			case publishErr <- err:
			default:
			}
			cancel()
		}
	})
	collected := make(chan error, 1)
	go func() { _, err := collector.Run(ctx); handler.Invalidate(); collected <- err }()
	served := make(chan error, 1)
	go func() {
		if *cert != "" {
			served <- server.ServeTLS(listener, "", "")
		} else {
			served <- server.Serve(listener)
		}
	}()
	scheme := "http"
	if *cert != "" {
		scheme = "https"
	}
	if _, err := fmt.Fprintf(stdout, "%s://%s/metrics\n", scheme, listener.Addr()); err != nil {
		cancel()
		_ = server.Close()
		<-collected
		<-served
		return err
	}
	var terminal error
	collectorDone, serverDone := false, false
	select {
	case terminal = <-collected:
		collectorDone = true
	case terminal = <-served:
		serverDone = true
	case <-ctx.Done():
	}
	if terminal == context.Canceled || terminal == context.DeadlineExceeded || terminal == http.ErrServerClosed {
		terminal = nil
	}
	cancel()
	handler.Invalidate()
	shutdown, stop := context.WithTimeout(context.Background(), 5*time.Second)
	defer stop()
	if err := server.Shutdown(shutdown); err != nil {
		_ = server.Close()
		terminal = errors.Join(terminal, err)
	}
	if !collectorDone {
		err := <-collected
		if err != context.Canceled && err != context.DeadlineExceeded {
			terminal = errors.Join(terminal, err)
		}
	}
	if !serverDone {
		err := <-served
		if !errors.Is(err, http.ErrServerClosed) {
			terminal = errors.Join(terminal, err)
		}
	}
	select {
	case err := <-publishErr:
		terminal = errors.Join(terminal, err)
	default:
	}
	if metricsDelivery != nil {
		terminal = errors.Join(terminal, metricsDelivery.Close(context.Background()))
		terminal = errors.Join(terminal, json.NewEncoder(stdout).Encode(struct {
			Signal string                    `json:"signal"`
			Stats  nvtelemetry.DeliveryStats `json:"stats"`
		}{"metrics", metricsDelivery.Stats()}))
	}
	if logsDelivery != nil {
		terminal = errors.Join(terminal, logsDelivery.Close(context.Background()))
		terminal = errors.Join(terminal, json.NewEncoder(stdout).Encode(struct {
			Signal string                    `json:"signal"`
			Stats  nvtelemetry.DeliveryStats `json:"stats"`
		}{"logs", logsDelivery.Stats()}))
	}
	return terminal
}

func readBearerToken(path string) (string, error) {
	if path == "" {
		return "", nil
	}
	f, err := os.Open(path)
	if err != nil {
		return "", errors.New("open bearer token file failed")
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, 4097))
	if err != nil || len(raw) > 4096 {
		return "", errors.New("invalid bearer token file")
	}
	token := strings.TrimSpace(string(raw))
	for i := range token {
		if token[i] < 33 || token[i] > 126 {
			return "", errors.New("invalid bearer token")
		}
	}
	if token == "" || strings.ContainsAny(token, "\r\n\t ") {
		return "", errors.New("invalid bearer token")
	}
	return token, nil
}
func bearerHandler(next http.Handler, token string) http.Handler {
	if token == "" {
		return next
	}
	expected := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), expected) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
