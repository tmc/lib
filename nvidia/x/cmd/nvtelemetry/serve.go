package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// ServeRecoverySchemaVersion identifies the recovery manifest schema for nvtelemetry serve.
const ServeRecoverySchemaVersion = 1

// ServeRecoveryManifest records the versioned recovery receipt and lifecycle state
// for owned retention and durable delivery in nvtelemetry serve.
type ServeRecoveryManifest struct {
	SchemaVersion     int                                          `json:"schema_version"`
	Owner             string                                       `json:"owner"`
	Retention         *nvtelemetry.RetentionRecoveryManifest       `json:"retention,omitempty"`
	MetricsDelivery   *nvtelemetry.DurableDeliveryRecoveryManifest `json:"metrics_delivery,omitempty"`
	LogsDelivery      *nvtelemetry.DurableDeliveryRecoveryManifest `json:"logs_delivery,omitempty"`
	Completed         bool                                         `json:"completed"`
	LifecycleFailures uint64                                       `json:"lifecycle_failures,omitempty"`
	MetricsFinal      *nvtelemetry.DurableDeliveryStats            `json:"metrics_final,omitempty"`
	LogsFinal         *nvtelemetry.DurableDeliveryStats            `json:"logs_final,omitempty"`
	StoredStreams     uint64                                       `json:"stored_streams"`
}

// ValidateServeRecoveryManifest checks schema versions, owner identity, and nested manifests.
func ValidateServeRecoveryManifest(m ServeRecoveryManifest) error {
	if m.SchemaVersion != ServeRecoverySchemaVersion {
		return fmt.Errorf("serve recovery manifest: unsupported schema version %d", m.SchemaVersion)
	}
	if len(m.Owner) == 0 || len(m.Owner) > 128 {
		return errors.New("serve recovery manifest: invalid owner")
	}
	for _, b := range m.Owner {
		if b < '!' || b > '~' {
			return errors.New("serve recovery manifest: invalid owner character")
		}
	}
	if m.Completed && m.LifecycleFailures != 0 {
		return errors.New("serve recovery manifest: completed with lifecycle failures")
	}
	for _, st := range []*nvtelemetry.DurableDeliveryStats{m.MetricsFinal, m.LogsFinal} {
		if st == nil {
			continue
		}
		if st.PendingSpool < 0 || st.PendingSpoolBytes < 0 || st.PendingBatches < 0 || st.PendingBytes < 0 {
			return errors.New("serve recovery manifest: negative pending delivery counts")
		}
		if m.Completed && durableOutcome(*st) != nil {
			return errors.New("serve recovery manifest: completed with incomplete delivery")
		}
	}
	if m.Retention != nil {
		if err := nvtelemetry.ValidateRetentionRecoveryManifest(*m.Retention); err != nil {
			return err
		}
		if m.Retention.Owner != m.Owner {
			return errors.New("serve recovery manifest: retention owner mismatch")
		}
	}
	if m.MetricsDelivery != nil {
		if err := nvtelemetry.ValidateDurableDeliveryRecoveryManifest(*m.MetricsDelivery); err != nil {
			return err
		}
		if m.MetricsDelivery.Owner != m.Owner {
			return errors.New("serve recovery manifest: metrics delivery owner mismatch")
		}
	}
	if m.LogsDelivery != nil {
		if err := nvtelemetry.ValidateDurableDeliveryRecoveryManifest(*m.LogsDelivery); err != nil {
			return err
		}
		if m.LogsDelivery.Owner != m.Owner {
			return errors.New("serve recovery manifest: logs delivery owner mismatch")
		}
	}
	return nil
}

// WriteServeRecoveryManifest validates and writes one JSON recovery manifest line.
func WriteServeRecoveryManifest(w io.Writer, m ServeRecoveryManifest) error {
	if w == nil {
		return errors.New("serve recovery manifest: nil writer")
	}
	if err := ValidateServeRecoveryManifest(m); err != nil {
		return err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

// ReadServeRecoveryManifest strictly decodes and validates a JSON recovery manifest.
func ReadServeRecoveryManifest(r io.Reader) (ServeRecoveryManifest, error) {
	if r == nil {
		return ServeRecoveryManifest{}, errors.New("serve recovery manifest: nil reader")
	}
	raw, err := io.ReadAll(io.LimitReader(r, 16<<10+1))
	if err != nil {
		return ServeRecoveryManifest{}, err
	}
	if len(raw) == 0 || len(raw) > 16<<10 {
		return ServeRecoveryManifest{}, errors.New("serve recovery manifest: invalid byte length")
	}
	if err := checkEndpointJSON(json.NewDecoder(bytes.NewReader(raw)), 0); err != nil {
		return ServeRecoveryManifest{}, fmt.Errorf("serve recovery manifest: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var m ServeRecoveryManifest
	if err := dec.Decode(&m); err != nil {
		return ServeRecoveryManifest{}, fmt.Errorf("serve recovery manifest: decode: %w", err)
	}
	var extra any
	if dec.Decode(&extra) != io.EOF {
		return ServeRecoveryManifest{}, errors.New("serve recovery manifest: trailing JSON")
	}
	if err := ValidateServeRecoveryManifest(m); err != nil {
		return ServeRecoveryManifest{}, err
	}
	return m, nil
}

func writeServeRecoveryManifestFile(path string, m ServeRecoveryManifest) error {
	var buf bytes.Buffer
	if err := WriteServeRecoveryManifest(&buf, m); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, buf.Bytes(), 0600); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func newDurableEndpointDelivery(path, spoolDir, owner string, maxBatches int, maxBytes int64, shutdownTimeout time.Duration, logs bool) (*nvtelemetry.DurableDelivery, func(), error) {
	c, client, request, err := readEndpointConfig(path)
	if err != nil {
		return nil, nil, err
	}
	if c.Workload != nil || c.Managed != nil || len(c.Devices) != 0 || c.Method != "" && c.Method != "GET" || c.BodyFile != "" || c.MaxPages != 0 || c.MaxBytes > 4<<20 {
		client.CloseIdleConnections()
		return nil, nil, errors.New("OTLP endpoint requires transport-only configuration and at most 4 MiB batches")
	}
	if shutdownTimeout <= 0 {
		shutdownTimeout = 5 * time.Second
	}
	cfg := nvtelemetry.DurableDeliveryConfig{
		Delivery: nvtelemetry.DeliveryConfig{
			Endpoint:           c.URL,
			Client:             client,
			Headers:            request.Header,
			MaxQueuedBatches:   min(maxBatches, 16),
			MaxQueuedBytes:     int(min(maxBytes, 8<<20)),
			MaxBatchBytes:      int(c.MaxBytes),
			MaxBatchDataPoints: 4096,
			MaxResponseBytes:   64 << 10,
			MaxAttempts:        3,
			InitialBackoff:     100 * time.Millisecond,
			MaxBackoff:         time.Second,
			BatchTimeout:       30 * time.Second,
			RequestTimeout:     c.Timeout,
			ShutdownTimeout:    shutdownTimeout,
		},
		SpoolDir:        spoolDir,
		Owner:           owner,
		MaxSpoolBatches: maxBatches,
		MaxSpoolBytes:   maxBytes,
	}
	var delivery *nvtelemetry.DurableDelivery
	if logs {
		delivery, err = nvtelemetry.NewLogsDurableDelivery(cfg)
	} else {
		delivery, err = nvtelemetry.NewDurableDelivery(cfg)
	}
	if err != nil {
		client.CloseIdleConnections()
		return nil, nil, err
	}
	return delivery, client.CloseIdleConnections, nil
}

type serveLifecycleSampler struct {
	base           nvtelemetry.Sampler
	counterStart   int64
	handoffTimeout time.Duration
	durableMetrics *nvtelemetry.DurableDelivery
	durableLogs    *nvtelemetry.DurableDelivery

	mu           sync.Mutex
	retStep      *nvtelemetry.FakeSampler
	retCollector *nvtelemetry.Collector
	sampleCount  int
	retentionErr error
	failure      func(error)
	buffer       bytes.Buffer
	closed       bool
}

func (s *serveLifecycleSampler) Devices() []nvtelemetry.DeviceMetadata {
	return s.base.Devices()
}

func (s *serveLifecycleSampler) Sample(ctx context.Context) ([]nvtelemetry.Reading, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	readings, err := s.base.Sample(ctx)
	if err != nil || len(readings) == 0 {
		return readings, err
	}

	// The source read has succeeded. Own this bounded local handoff even if
	// reload/shutdown cancels the source context before bookkeeping finishes.
	// No further source/vendor work uses it. Admission, encoding and I/O errors
	// remain errors; the separate deadline is not permission to suppress them.
	timeout := s.handoffTimeout
	if timeout <= 0 {
		timeout = time.Second
	}
	handoff, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
	defer cancel()

	var sampleErr error
	if s.retCollector != nil {
		sampleErr = s.retainSample(handoff, readings)
	}

	opts := nvtelemetry.ExportOptions{
		MaxBytes:              4 << 20,
		MaxReadings:           4096,
		HasCounterStart:       s.counterStart > 0,
		CounterStartUnixNanos: s.counterStart,
	}

	if s.durableMetrics != nil {
		numeric := make([]nvtelemetry.Reading, 0, len(readings))
		for _, r := range readings {
			if !r.IsAvailable() {
				continue
			}
			kind, ok := nvtelemetry.LookupMetricKind(r.Name)
			if !ok || kind != nvtelemetry.KindEvent && kind != nvtelemetry.KindBitmask {
				numeric = append(numeric, r)
			}
		}
		if len(numeric) > 0 {
			var encoded bytes.Buffer
			err := nvtelemetry.WriteOTLPMetrics(&encoded, numeric, opts)
			if err == nil {
				err = s.durableMetrics.Enqueue(handoff, encoded.Bytes())
			}
			if err != nil {
				sampleErr = errors.Join(sampleErr, fmt.Errorf("durable metrics: %w", err))
			}
		}
	}

	if s.durableLogs != nil {
		var encoded bytes.Buffer
		err := nvtelemetry.WriteOTLPLogs(&encoded, readings, opts)
		if err == nil {
			err = s.durableLogs.Enqueue(handoff, encoded.Bytes())
		}
		if err != nil {
			sampleErr = errors.Join(sampleErr, fmt.Errorf("durable logs: %w", err))
		}
	}

	if sampleErr != nil && s.failure != nil {
		s.failure(sampleErr)
	}
	return readings, sampleErr
}

// retainSample is the real local collector handoff. Keep a failure even when
// Step refuses before entering collection, so Stop cannot publish a healthy
// prefix that omits an already acquired source sample.
func (s *serveLifecycleSampler) retainSample(ctx context.Context, readings []nvtelemetry.Reading) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.retStep.Enqueue(readings)
	err := s.retCollector.Step(ctx)
	if err == nil {
		s.sampleCount++
	} else if s.retentionErr == nil {
		s.retentionErr = err
	}
	return err
}

// serveLifecycle owns one retained generation at a time. Agent calls each
// cleanup only after its real collector has drained, including on reload.
// Errors keep a bounded first cause and count; later generations cannot erase it.
type serveLifecycle struct {
	mu             sync.Mutex
	retention      *nvtelemetry.Retention
	maxStreamBytes int
	metrics, logs  *nvtelemetry.DurableDelivery
	active         int
	retained       *serveLifecycleSampler
	firstErr       error
	failures       uint64
}

func (l *serveLifecycle) fail(err error) {
	if err == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.firstErr == nil {
		l.firstErr = err
	}
	if l.failures != ^uint64(0) {
		l.failures++
	}
}

func (l *serveLifecycle) drained() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.active == 0
}

func (l *serveLifecycle) result() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.active != 0 {
		return errors.Join(l.firstErr, errors.New("serve generation not finalized: lifecycle ownership retained"))
	}
	return l.firstErr
}

func durableOutcome(st nvtelemetry.DurableDeliveryStats) error {
	if st.PendingSpool != 0 || st.PendingBatches != 0 || st.FailedBatches != 0 || st.DroppedBatches != 0 || st.PartialBatches != 0 || st.RejectedBatches != 0 || st.RejectedDataPoints != 0 || st.RejectedLogRecords != 0 {
		return errors.New("durable delivery incomplete: pending, failed, rejected or dropped exports")
	}
	return nil
}

func (l *serveLifecycle) sampler(cfg *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
	// Refuse replacement if the previous generation was not sealed. In normal
	// transitions Agent invokes its cleanup before calling this factory.
	l.mu.Lock()
	blocked := l.active != 0 || l.retained != nil
	l.mu.Unlock()
	if blocked {
		return nil, nil, errors.New("previous serve generation remains owned")
	}
	if cfg.Budget < 1 || cfg.Budget > 1<<20 || cfg.Interval <= 0 || cfg.ShutdownTimeout <= 0 || cfg.ShutdownTimeout > time.Minute {
		return nil, nil, errors.New("invalid serve generation bounds")
	}
	if l.retention != nil && (l.maxStreamBytes < 1 || l.maxStreamBytes > 64<<20) {
		return nil, nil, errors.New("invalid retained stream byte bound")
	}
	base, baseCleanup, err := buildBaseServeSampler(cfg)
	if err != nil {
		return nil, nil, err
	}
	s := &serveLifecycleSampler{base: base, counterStart: cfg.CounterStartUnixNanos, handoffTimeout: cfg.ShutdownTimeout, durableMetrics: l.metrics, durableLogs: l.logs, failure: l.fail}
	if l.retention != nil {
		// This is a bounded run, not a one-slot asynchronous mailbox. Collector
		// retains each sample up to the same record bound as its Writer. Actual
		// record/byte exhaustion remains terminal and cannot become complete.
		step := nvtelemetry.NewFakeSampler(base.Devices()...)
		config := nvtelemetry.CollectorConfig{IntervalNanos: cfg.Interval.Nanoseconds(), MaxRecordBudget: cfg.Budget, MaxByteBudget: l.maxStreamBytes, MaxBufferedRecords: cfg.Budget}
		s.retStep = step
		s.retCollector = nvtelemetry.NewCollector(fmt.Sprintf("serve-%d", time.Now().UnixNano()), step, nvtelemetry.RealClock{}, config, nvtelemetry.NewWriter(&s.buffer, config))
		if err := s.retCollector.InitStream(); err != nil {
			if baseCleanup != nil {
				baseCleanup()
			}
			return nil, nil, err
		}
	}
	l.mu.Lock()
	l.active++
	l.mu.Unlock()
	cleanup := func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.closed {
			return
		}
		s.closed = true
		var sealErr error
		if s.retCollector != nil {
			_, sealErr = s.retCollector.Stop()
			// A failed local handoff can precede the collector's own sampling
			// guard. Its complete prefix is not a complete source generation.
			sealErr = errors.Join(sealErr, s.retentionErr)
			if sealErr == nil && s.sampleCount > 0 {
				var id uint64
				id, sealErr = l.retention.NextStreamID()
				if sealErr == nil {
					sealErr = l.retention.Store(id, bytes.NewReader(s.buffer.Bytes()))
				}
			}
			l.fail(sealErr)
		}

		if baseCleanup != nil {
			baseCleanup()
		}
		l.mu.Lock()
		// Source cleanup has completed, even if persistence failed. Final
		// shutdown may therefore close delivery and retention safely. Keep the
		// failed generation separately so no reload can overwrite its bytes.
		l.active--
		if sealErr != nil {
			l.retained = s
		}
		l.mu.Unlock()
	}
	return s, cleanup, nil
}

func buildBaseServeSampler(cfg *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
	if cfg.ManagedConfig != "" {
		c, client, request, err := readEndpointConfig(cfg.ManagedConfig)
		if err != nil {
			return nil, nil, err
		}
		cleanup := func() { client.CloseIdleConnections() }
		return &managedEndpointSampler{config: c, client: client, request: request, clock: nvtelemetry.RealClock{}}, cleanup, nil
	}
	if cfg.Mock {
		fake := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock", Model: "synthetic", Status: nvtelemetry.DeviceStatusOk})
		fake.SetFallback(func(ctx context.Context) ([]nvtelemetry.Reading, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			now := nvtelemetry.RealClock{}
			return []nvtelemetry.Reading{{
				Name:               nvtelemetry.MetricPowerUsageMW,
				Unit:               nvtelemetry.UnitMilliwatts,
				DeviceUUID:         "GPU-mock",
				Evidence:           nvtelemetry.EvidenceDerived,
				Backend:            "mock",
				Value:              30000,
				TimestampMonotonic: now.NowMonotonic(),
				TimestampUnix:      now.NowUnix(),
			}}, nil
		})
		return fake, nil, nil
	}
	if len(cfg.DeviceUUIDs) == 0 {
		return nil, nil, errors.New("live serving requires an explicit device UUID allowlist")
	}
	if err := nvml.Init(); err != nil {
		return nil, nil, fmt.Errorf("initialize NVML: %w", err)
	}
	devices := make([]nvml.Device, 0, len(cfg.DeviceUUIDs))
	for _, uuid := range cfg.DeviceUUIDs {
		device, err := nvml.NewDeviceByUUID(uuid)
		if err != nil {
			_ = nvml.Shutdown()
			return nil, nil, fmt.Errorf("select NVML device: %w", err)
		}
		actual, err := device.UUID()
		if err != nil || actual != uuid {
			_ = nvml.Shutdown()
			return nil, nil, errors.New("selected NVML device identity mismatch")
		}
		devices = append(devices, device)
	}
	nvmlSampler, err := nvtelemetry.NewNVMLSampler(nvtelemetry.RealClock{}, devices...)
	if err != nil {
		_ = nvml.Shutdown()
		return nil, nil, err
	}
	return nvmlSampler, func() { _ = nvml.Shutdown() }, nil
}

func runServe(args []string, stdout, stderr io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runServeContext(ctx, args, stdout, stderr)
}

func runServeContext(parent context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	configFile := fs.String("config", "", "validated JSON configuration file")
	metricsConfig := fs.String("otlp-metrics-config", "", "optional transport-only OTLP metrics endpoint JSON")
	logsConfig := fs.String("otlp-logs-config", "", "optional transport-only OTLP diagnostic logs endpoint JSON")
	counterStart := fs.Int64("counter-start-unix-nanos", 0, "explicit counter epoch start for OTLP metrics (0 refuses counters)")
	address := fs.String("listen", "127.0.0.1:9464", "HTTP listening address")
	interval := fs.Duration("interval", time.Second, "sampling interval")
	duration := fs.Duration("duration", 0, "maximum serving duration (0 until signal or budget)")
	maxAge := fs.Duration("max-age", 5*time.Second, "maximum snapshot publication age")
	shutdownTimeout := fs.Duration("shutdown-timeout", 5*time.Second, "maximum graceful shutdown drain duration")
	budget := fs.Int("budget", 100000, "maximum stream records before stopping")
	managedConfig := fs.String("managed-config", "", "explicit managed endpoint configuration JSON file")
	mock := fs.Bool("mock", false, "serve explicitly synthetic mock readings")
	uuids := fs.String("device-uuids", "", "exact UUID allowlist (required for live NVML)")
	tokenFile := fs.String("auth-token-file", "", "file containing an HTTP bearer token")
	cert := fs.String("tls-cert", "", "TLS certificate file")
	key := fs.String("tls-key", "", "TLS private key file")
	remote := fs.Bool("allow-remote", false, "allow a non-loopback listener with TLS and bearer authentication")
	retentionDir := fs.String("retention-dir", "", "optional task-owned directory for bounded stream retention and recovery")
	retentionOwner := fs.String("retention-owner", "nvtelemetry-serve", "owner identifier for retention and durable delivery")
	retentionMaxFiles := fs.Int("retention-max-files", 16, "maximum retained stream files")
	retentionMaxBytes := fs.Int64("retention-max-bytes", 64<<20, "maximum total retained stream bytes")
	retentionMaxStreamBytes := fs.Int64("retention-max-stream-bytes", 4<<20, "maximum single retained stream bytes")
	durableMetricsSpoolDir := fs.String("durable-metrics-spool-dir", "", "optional task-owned directory for durable OTLP metrics spooling and recovery")
	durableLogsSpoolDir := fs.String("durable-logs-spool-dir", "", "optional task-owned directory for durable OTLP logs spooling and recovery")
	durableMaxBatches := fs.Int("durable-max-spool-batches", 64, "maximum durable spooled batches on disk")
	durableMaxBytes := fs.Int64("durable-max-spool-bytes", 16<<20, "maximum durable spooled bytes on disk")
	recoveryManifestIn := fs.String("recovery-manifest-in", "", "optional versioned recovery manifest file to verify before startup")
	recoveryManifestOut := fs.String("recovery-manifest-out", "", "optional file path to write the versioned operational recovery manifest")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("invalid serving options or bounds: extraneous arguments")
	}

	if *recoveryManifestIn != "" {
		f, err := os.Open(*recoveryManifestIn)
		if err != nil {
			return fmt.Errorf("open recovery manifest: %w", err)
		}
		prior, readErr := ReadServeRecoveryManifest(f)
		_ = f.Close()
		if readErr != nil {
			return readErr
		}
		if prior.Owner != *retentionOwner {
			return errors.New("serve recovery manifest: owner mismatch")
		}
	}

	var effectiveMetricsConfig, effectiveLogsConfig string
	normalizeInitial := func(raw ServeConfig) (ServeConfig, error) {
		effectiveMetricsConfig, effectiveLogsConfig = raw.OTLPMetricsConfig, raw.OTLPLogsConfig
		if *durableMetricsSpoolDir != "" {
			if raw.OTLPMetricsConfig == "" {
				return raw, errors.New("durable metrics spool requires -otlp-metrics-config")
			}
			raw.OTLPMetricsConfig = ""
		}
		if *durableLogsSpoolDir != "" {
			if raw.OTLPLogsConfig == "" {
				return raw, errors.New("durable logs spool requires -otlp-logs-config")
			}
			raw.OTLPLogsConfig = ""
		}
		return raw, nil
	}
	var parsed *ParsedConfig
	if *configFile != "" {
		var err error
		parsed, err = parseConfigFile(*configFile, normalizeInitial)
		if err != nil {
			return err
		}
	} else {
		// Enforce bounds matching serve CLI flag requirements
		if *interval <= 0 || *duration < 0 || *maxAge <= 0 || *shutdownTimeout <= 0 || *budget < 1 || *budget > 1<<20 {
			return errors.New("invalid serving options or bounds")
		}

		raw := ServeConfig{
			Listen:                *address,
			Interval:              interval.String(),
			MaxAge:                maxAge.String(),
			Duration:              duration.String(),
			Budget:                *budget,
			ShutdownTimeout:       shutdownTimeout.String(),
			Mock:                  *mock,
			ManagedConfig:         *managedConfig,
			CounterStartUnixNanos: *counterStart,
			OTLPMetricsConfig:     *metricsConfig,
			OTLPLogsConfig:        *logsConfig,
			AuthTokenFile:         *tokenFile,
			TLSCert:               *cert,
			TLSKey:                *key,
			AllowRemote:           *remote,
		}
		if *uuids != "" {
			raw.DeviceUUIDs = strings.Split(*uuids, ",")
		}

		var err error
		raw, err = normalizeInitial(raw)
		if err != nil {
			return err
		}
		parsed, err = validateAndBuildParsedConfig(raw)
		if err != nil {
			return err
		}
	}

	// Agent keeps configured identity, but its durable owner normalizes a
	// clone before every source/exporter acquisition, including file reload.
	parsed.OTLPMetricsConfig, parsed.OTLPLogsConfig = effectiveMetricsConfig, effectiveLogsConfig

	recManifest := ServeRecoveryManifest{
		SchemaVersion: ServeRecoverySchemaVersion,
		Owner:         *retentionOwner,
	}

	var ret *nvtelemetry.Retention
	var durableMetrics, durableLogs *nvtelemetry.DurableDelivery
	var metricsCleanup, logsCleanup func()
	cleanupOwned := func() {
		if durableLogs != nil {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_ = durableLogs.Close(ctx)
		}
		if logsCleanup != nil {
			logsCleanup()
		}
		if durableMetrics != nil {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_ = durableMetrics.Close(ctx)
		}
		if metricsCleanup != nil {
			metricsCleanup()
		}
		if ret != nil {
			_ = ret.Close()
		}
	}

	if *retentionDir != "" {
		if _, err := os.Lstat(*retentionDir); err == nil {
			retRec, err := nvtelemetry.RecoverRetentionManifest(*retentionDir, *retentionOwner)
			if err != nil {
				return err
			}
			recManifest.Retention = &retRec
		} else if !os.IsNotExist(err) {
			return err
		}
		var err error
		ret, err = nvtelemetry.OpenRetention(*retentionDir, nvtelemetry.RetentionConfig{
			Owner:          *retentionOwner,
			MaxFiles:       *retentionMaxFiles,
			MaxBytes:       *retentionMaxBytes,
			MaxStreamBytes: *retentionMaxStreamBytes,
		})
		if err != nil {
			return err
		}
		if recManifest.Retention == nil {
			nextID, err := ret.NextStreamID()
			if err != nil {
				cleanupOwned()
				return err
			}
			recManifest.Retention = &nvtelemetry.RetentionRecoveryManifest{
				SchemaVersion: nvtelemetry.RetentionRecoverySchemaVersion,
				MarkerVersion: 2,
				Owner:         *retentionOwner,
				NextStreamID:  nextID,
			}
		}
	}

	if *durableMetricsSpoolDir != "" {
		if _, err := os.Lstat(*durableMetricsSpoolDir); err == nil {
			mRec, err := nvtelemetry.RecoverDurableDeliveryManifest(*durableMetricsSpoolDir, *retentionOwner)
			if err != nil {
				cleanupOwned()
				return err
			}
			recManifest.MetricsDelivery = &mRec
		} else if !os.IsNotExist(err) {
			cleanupOwned()
			return err
		}
		var err error
		durableMetrics, metricsCleanup, err = newDurableEndpointDelivery(effectiveMetricsConfig, *durableMetricsSpoolDir, *retentionOwner, *durableMaxBatches, *durableMaxBytes, parsed.ShutdownTimeout, false)
		if err != nil {
			cleanupOwned()
			return err
		}
		if recManifest.MetricsDelivery == nil {
			st := durableMetrics.Stats()
			recManifest.MetricsDelivery = &nvtelemetry.DurableDeliveryRecoveryManifest{
				SchemaVersion:  nvtelemetry.DurableDeliveryRecoverySchemaVersion,
				MarkerVersion:  1,
				Owner:          *retentionOwner,
				PendingBatches: st.PendingSpool,
				PendingBytes:   st.PendingSpoolBytes,
			}
		}
	}

	if *durableLogsSpoolDir != "" {
		if _, err := os.Lstat(*durableLogsSpoolDir); err == nil {
			lRec, err := nvtelemetry.RecoverDurableDeliveryManifest(*durableLogsSpoolDir, *retentionOwner)
			if err != nil {
				cleanupOwned()
				return err
			}
			recManifest.LogsDelivery = &lRec
		} else if !os.IsNotExist(err) {
			cleanupOwned()
			return err
		}
		var err error
		durableLogs, logsCleanup, err = newDurableEndpointDelivery(effectiveLogsConfig, *durableLogsSpoolDir, *retentionOwner, *durableMaxBatches, *durableMaxBytes, parsed.ShutdownTimeout, true)
		if err != nil {
			cleanupOwned()
			return err
		}
		if recManifest.LogsDelivery == nil {
			st := durableLogs.Stats()
			recManifest.LogsDelivery = &nvtelemetry.DurableDeliveryRecoveryManifest{
				SchemaVersion:  nvtelemetry.DurableDeliveryRecoverySchemaVersion,
				MarkerVersion:  1,
				Owner:          *retentionOwner,
				PendingBatches: st.PendingSpool,
				PendingBytes:   st.PendingSpoolBytes,
			}
		}
	}

	if *recoveryManifestOut != "" {
		if err := writeServeRecoveryManifestFile(*recoveryManifestOut, recManifest); err != nil {
			cleanupOwned()
			return err
		}
	}

	agent, err := NewAgent(parsed, *configFile, stdout, stderr)
	if err != nil {
		cleanupOwned()
		return err
	}

	if durableMetrics != nil || durableLogs != nil {
		agent.ownedDelivery = &serveDeliveryOwnership{}
		if durableMetrics != nil {
			agent.ownedDelivery.metrics = effectiveMetricsConfig
		}
		if durableLogs != nil {
			agent.ownedDelivery.logs = effectiveLogsConfig
		}
	}
	lifecycle := &serveLifecycle{retention: ret, maxStreamBytes: int(*retentionMaxStreamBytes), metrics: durableMetrics, logs: durableLogs}
	if ret != nil || durableMetrics != nil || durableLogs != nil {
		agent.SetSamplerFactory(lifecycle.sampler)
	}

	serveErr := agent.Start(parent)

	var shutdownErr error
	shutdownCtx, cancel := context.WithTimeout(context.Background(), parsed.ShutdownTimeout)
	defer cancel()

	if durableMetrics != nil && lifecycle.drained() {
		shutdownErr = errors.Join(shutdownErr, durableMetrics.Close(shutdownCtx))
		if metricsCleanup != nil {
			metricsCleanup()
		}
	}
	if durableLogs != nil && lifecycle.drained() {
		shutdownErr = errors.Join(shutdownErr, durableLogs.Close(shutdownCtx))
		if logsCleanup != nil {
			logsCleanup()
		}
	}
	shutdownErr = errors.Join(shutdownErr, lifecycle.result())
	if durableMetrics != nil {
		st := durableMetrics.Stats()
		recManifest.MetricsFinal = &st
		shutdownErr = errors.Join(shutdownErr, durableOutcome(st))
	}
	if durableLogs != nil {
		st := durableLogs.Stats()
		recManifest.LogsFinal = &st
		shutdownErr = errors.Join(shutdownErr, durableOutcome(st))
	}
	lifecycle.mu.Lock()
	recManifest.LifecycleFailures = lifecycle.failures
	lifecycle.mu.Unlock()
	if ret != nil {
		recManifest.StoredStreams = ret.Status().Stores
		if lifecycle.drained() {
			shutdownErr = errors.Join(shutdownErr, ret.Close())
		}
	}

	if *recoveryManifestOut != "" {
		recManifest.Completed = (serveErr == nil && shutdownErr == nil)
		shutdownErr = errors.Join(shutdownErr, writeServeRecoveryManifestFile(*recoveryManifestOut, recManifest))
	}

	return errors.Join(serveErr, shutdownErr)
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
