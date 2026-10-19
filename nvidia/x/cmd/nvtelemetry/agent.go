package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type agentMetrics struct {
	collectionAttempts       atomic.Uint64
	collectionSuccesses      atomic.Uint64
	collectionFailures       atomic.Uint64
	deliveryEnqueued         atomic.Uint64
	deliverySuccesses        atomic.Uint64
	deliveryFailures         atomic.Uint64
	deliveryDropped          atomic.Uint64
	retiredDeliverySuccesses atomic.Uint64
	retiredDeliveryFailures  atomic.Uint64
	retiredDeliveryDropped   atomic.Uint64
	reloadSuccesses          atomic.Uint64
	reloadFailures           atomic.Uint64
}

type runningTarget struct {
	sampler         nvtelemetry.Sampler
	samplerCleanup  func()
	collector       *nvtelemetry.Collector
	metricsDelivery *nvtelemetry.Delivery
	metricsCleanup  func()
	logsDelivery    *nvtelemetry.Delivery
	logsCleanup     func()

	ctx           context.Context
	cancel        context.CancelFunc
	collectorDone chan struct{}
	collectorErr  error
	stopping      atomic.Bool

	targetSig string
}

func (t *runningTarget) cleanup() {
	if t.samplerCleanup != nil {
		t.samplerCleanup()
	}
	if t.metricsCleanup != nil {
		t.metricsCleanup()
	}
	if t.logsCleanup != nil {
		t.logsCleanup()
	}
}

// Agent coordinates the HTTP serving, collector sampling, reload, and shutdown lifecycle.
type Agent struct {
	mu         sync.RWMutex
	configPath string
	cfg        *ParsedConfig

	listener       net.Listener
	server         *http.Server
	handlerMu      sync.RWMutex
	metricsHandler *nvtelemetry.MetricsHandler

	targetMu       sync.Mutex
	target         *runningTarget
	targetFatalErr chan error

	metrics *agentMetrics

	ctx          context.Context
	cancel       context.CancelFunc
	shuttingDown atomic.Bool
	closed       chan struct{}
	shutdownOnce sync.Once
	shutdownErr  error

	stdout io.Writer
	stderr io.Writer

	// Test hooks
	ownedDelivery  *serveDeliveryOwnership
	samplerFactory func(cfg *ParsedConfig) (nvtelemetry.Sampler, func(), error)
}

// NewAgent initializes an Agent from a validated ParsedConfig.
func NewAgent(cfg *ParsedConfig, configPath string, stdout, stderr io.Writer) (*Agent, error) {
	if cfg == nil {
		return nil, errors.New("nil configuration")
	}
	if stdout == nil {
		stdout = io.Discard
	}
	if stderr == nil {
		stderr = io.Discard
	}

	handler, err := nvtelemetry.NewMetricsHandler(nvtelemetry.ExportOptions{}, cfg.MaxAge)
	if err != nil {
		return nil, fmt.Errorf("create metrics handler: %w", err)
	}

	return &Agent{
		configPath:     configPath,
		cfg:            cfg,
		metricsHandler: handler,
		metrics:        &agentMetrics{},
		targetFatalErr: make(chan error, 1),
		closed:         make(chan struct{}),
		stdout:         stdout,
		stderr:         stderr,
	}, nil
}

// SetSamplerFactory allows tests to inject custom samplers (e.g. fake samplers).
func (a *Agent) SetSamplerFactory(fn func(cfg *ParsedConfig) (nvtelemetry.Sampler, func(), error)) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.samplerFactory = fn
}

func (a *Agent) getMetricsHandler() *nvtelemetry.MetricsHandler {
	// Collection must not wait on the configuration lock: Reload holds that
	// lock while waiting for collection to drain.
	a.handlerMu.RLock()
	defer a.handlerMu.RUnlock()
	return a.metricsHandler
}

func (a *Agent) computeTargetSig(cfg *ParsedConfig) string {
	var uuids []string
	if len(cfg.DeviceUUIDs) > 0 {
		uuids = append([]string(nil), cfg.DeviceUUIDs...)
	}
	return fmt.Sprintf("interval=%v|budget=%d|mock=%v|uuids=%s|managed=%s|counter=%d|metrics=%s|logs=%s",
		cfg.Interval, cfg.Budget, cfg.Mock, strings.Join(uuids, ","),
		cfg.ManagedConfig, cfg.CounterStartUnixNanos, cfg.OTLPMetricsConfig, cfg.OTLPLogsConfig)
}

// serveDeliveryOwnership is immutable after startup. Config paths identify the
// endpoint configurations acquired once by serve's durable delivery owners.
// Changing or disabling one requires restart, before draining the live source.
type serveDeliveryOwnership struct{ metrics, logs string }

func (o *serveDeliveryOwnership) normalize(metrics, logs string) (string, string, error) {
	if o == nil {
		return metrics, logs, nil
	}
	if o.metrics != "" {
		if metrics != o.metrics {
			return "", "", errors.New("changing durable metrics endpoint requires restart")
		}
		metrics = ""
	}
	if o.logs != "" {
		if logs != o.logs {
			return "", "", errors.New("changing durable logs endpoint requires restart")
		}
		logs = ""
	}
	return metrics, logs, nil
}

func (o *serveDeliveryOwnership) normalizeRaw(raw ServeConfig) (ServeConfig, error) {
	var err error
	raw.OTLPMetricsConfig, raw.OTLPLogsConfig, err = o.normalize(raw.OTLPMetricsConfig, raw.OTLPLogsConfig)
	return raw, err
}

func (a *Agent) targetConfig(cfg *ParsedConfig) (*ParsedConfig, error) {
	if a.ownedDelivery == nil {
		return cfg, nil
	}
	copy := *cfg
	copy.DeviceUUIDs = append([]string(nil), cfg.DeviceUUIDs...)
	var err error
	copy.OTLPMetricsConfig, copy.OTLPLogsConfig, err = a.ownedDelivery.normalize(cfg.OTLPMetricsConfig, cfg.OTLPLogsConfig)
	if err != nil {
		return nil, err
	}
	return &copy, nil
}

func (a *Agent) readConfigFile() (*ParsedConfig, error) {
	if a.ownedDelivery == nil {
		return ParseConfigFile(a.configPath)
	}
	cfg, err := parseConfigFile(a.configPath, a.ownedDelivery.normalizeRaw)
	if err != nil {
		return nil, err
	}
	// Keep configured identity in Agent; buildTarget passes only a normalized
	// clone to samplers/exporters. Relative names were resolved before checking.
	if a.ownedDelivery.metrics != "" {
		cfg.OTLPMetricsConfig = a.ownedDelivery.metrics
	}
	if a.ownedDelivery.logs != "" {
		cfg.OTLPLogsConfig = a.ownedDelivery.logs
	}
	return cfg, nil
}

func (a *Agent) buildTarget(cfg *ParsedConfig) (*runningTarget, error) {
	targetSig := a.computeTargetSig(cfg)
	cfg, configErr := a.targetConfig(cfg)
	if configErr != nil {
		return nil, configErr
	}
	var sampler nvtelemetry.Sampler
	var samplerCleanup func()
	var err error

	if a.samplerFactory != nil {
		sampler, samplerCleanup, err = a.samplerFactory(cfg)
		if err != nil {
			return nil, err
		}
	} else if cfg.ManagedConfig != "" {
		c, client, request, err := readEndpointConfig(cfg.ManagedConfig)
		if err != nil {
			return nil, err
		}
		samplerCleanup = func() { client.CloseIdleConnections() }
		sampler = &managedEndpointSampler{config: c, client: client, request: request, clock: nvtelemetry.RealClock{}}
	} else if cfg.Mock {
		fake := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock", Model: "synthetic", Status: nvtelemetry.DeviceStatusOk})
		fake.SetFallback(func(ctx context.Context) ([]nvtelemetry.Reading, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU-mock", Evidence: nvtelemetry.EvidenceDerived, Backend: "mock", Value: 30000, TimestampUnix: time.Now().UnixNano()}}, nil
		})
		sampler = fake
	} else {
		if len(cfg.DeviceUUIDs) == 0 {
			return nil, errors.New("live serving requires an explicit device UUID allowlist")
		}
		if err := nvml.Init(); err != nil {
			return nil, fmt.Errorf("initialize NVML: %w", err)
		}
		devices := make([]nvml.Device, 0, len(cfg.DeviceUUIDs))
		for _, uuid := range cfg.DeviceUUIDs {
			device, err := nvml.NewDeviceByUUID(uuid)
			if err != nil {
				_ = nvml.Shutdown()
				return nil, fmt.Errorf("select NVML device: %w", err)
			}
			actual, err := device.UUID()
			if err != nil || actual != uuid {
				_ = nvml.Shutdown()
				return nil, errors.New("selected NVML device identity mismatch")
			}
			devices = append(devices, device)
		}
		nvmlSampler, err := nvtelemetry.NewNVMLSampler(nvtelemetry.RealClock{}, devices...)
		if err != nil {
			_ = nvml.Shutdown()
			return nil, err
		}
		sampler = nvmlSampler
		samplerCleanup = func() { _ = nvml.Shutdown() }
	}

	var metricsDelivery, logsDelivery *nvtelemetry.Delivery
	var metricsCleanup, logsCleanup func()
	if cfg.OTLPMetricsConfig != "" {
		metricsDelivery, metricsCleanup, err = newEndpointDelivery(cfg.OTLPMetricsConfig, false)
		if err != nil {
			if samplerCleanup != nil {
				samplerCleanup()
			}
			return nil, err
		}
	}
	if cfg.OTLPLogsConfig != "" {
		logsDelivery, logsCleanup, err = newEndpointDelivery(cfg.OTLPLogsConfig, true)
		if err != nil {
			if metricsCleanup != nil {
				metricsCleanup()
			}
			if metricsDelivery != nil {
				_ = metricsDelivery.Close(context.Background())
			}
			if samplerCleanup != nil {
				samplerCleanup()
			}
			return nil, err
		}
	}

	collectorCfg := nvtelemetry.CollectorConfig{
		SchemaVersion:      nvtelemetry.SchemaVersionV3,
		IntervalNanos:      cfg.Interval.Nanoseconds(),
		MaxRecordBudget:    cfg.Budget,
		MaxByteBudget:      128 << 20,
		MaxBufferedRecords: 1,
		DropOnFull:         true,
	}
	writerCfg := collectorCfg
	writerCfg.DropOnFull = false
	collector := nvtelemetry.NewCollector("http", sampler, nvtelemetry.RealClock{}, collectorCfg, nvtelemetry.NewWriter(io.Discard, writerCfg))

	targetCtx, targetCancel := context.WithCancel(a.ctx)
	target := &runningTarget{
		sampler:         sampler,
		samplerCleanup:  samplerCleanup,
		collector:       collector,
		metricsDelivery: metricsDelivery,
		metricsCleanup:  metricsCleanup,
		logsDelivery:    logsDelivery,
		logsCleanup:     logsCleanup,
		ctx:             targetCtx,
		cancel:          targetCancel,
		collectorDone:   make(chan struct{}),
		targetSig:       targetSig,
	}

	collector.SetSampleHook(func(s nvtelemetry.Sample) {
		if targetCtx.Err() != nil || a.shuttingDown.Load() {
			return
		}
		a.metrics.collectionAttempts.Add(1)

		numeric := make([]nvtelemetry.Reading, 0, len(s.Readings))
		for _, reading := range s.Readings {
			kind, ok := nvtelemetry.LookupMetricKind(reading.Name)
			if !ok || kind != nvtelemetry.KindEvent && kind != nvtelemetry.KindBitmask {
				numeric = append(numeric, reading)
			}
		}

		h := a.getMetricsHandler()
		var exportErr error
		if h != nil {
			exportErr = h.Update(numeric)
		}
		hasUnavailable := len(s.Readings) == 0
		for _, r := range s.Readings {
			if !r.IsAvailable() {
				hasUnavailable = true
			}
		}

		if exportErr != nil || hasUnavailable {
			a.metrics.collectionFailures.Add(1)
		} else {
			a.metrics.collectionSuccesses.Add(1)
		}

		opts := nvtelemetry.ExportOptions{
			MaxBytes:              4 << 20,
			MaxReadings:           4096,
			HasCounterStart:       cfg.CounterStartUnixNanos > 0,
			CounterStartUnixNanos: cfg.CounterStartUnixNanos,
		}

		if metricsDelivery != nil {
			available := make([]nvtelemetry.Reading, 0, len(numeric))
			for _, r := range numeric {
				if r.IsAvailable() {
					available = append(available, r)
				}
			}
			if len(available) > 0 {
				var encoded bytes.Buffer
				exportErr := nvtelemetry.WriteOTLPMetrics(&encoded, available, opts)
				if exportErr == nil {
					a.metrics.deliveryEnqueued.Add(1)
					if err := metricsDelivery.Enqueue(context.Background(), encoded.Bytes()); err != nil {
						a.metrics.deliveryFailures.Add(1)
					}
				} else {
					a.metrics.deliveryFailures.Add(1)
				}
			}
		}

		if logsDelivery != nil && len(s.Readings) > 0 {
			var encoded bytes.Buffer
			exportErr := nvtelemetry.WriteOTLPLogs(&encoded, s.Readings, opts)
			if exportErr == nil {
				a.metrics.deliveryEnqueued.Add(1)
				if err := logsDelivery.Enqueue(context.Background(), encoded.Bytes()); err != nil {
					a.metrics.deliveryFailures.Add(1)
				}
			} else {
				a.metrics.deliveryFailures.Add(1)
			}
		}
	})

	return target, nil
}

func (a *Agent) startTarget(t *runningTarget) {
	go func() {
		_, err := t.collector.Run(t.ctx)
		t.collectorErr = err
		close(t.collectorDone)
		h := a.getMetricsHandler()
		if h != nil {
			h.Invalidate()
		}
		if !t.stopping.Load() && !a.shuttingDown.Load() {
			select {
			case a.targetFatalErr <- err:
			default:
			}
		}
	}()
}

func (a *Agent) transitionTarget(ctx context.Context, newCfg *ParsedConfig) error {
	a.targetMu.Lock()
	defer a.targetMu.Unlock()

	if a.shuttingDown.Load() {
		return errors.New("agent is shutting down")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	newSig := a.computeTargetSig(newCfg)
	if a.target != nil && !a.target.stopping.Load() && a.target.ctx.Err() == nil && a.target.targetSig == newSig {
		return nil
	}

	// 1. If an active target is running, STOP AND DRAIN IT FIRST before acquiring replacement!
	if a.target != nil {
		oldTgt := a.target
		if !oldTgt.stopping.Swap(true) {
			oldTgt.cancel()
			go func() {
				_, _ = oldTgt.collector.Stop()
			}()
		}

		drainTimeout := a.cfg.ShutdownTimeout
		if drainTimeout <= 0 {
			drainTimeout = 5 * time.Second
		}
		drainTimer := time.NewTimer(drainTimeout)
		defer drainTimer.Stop()

		select {
		case <-oldTgt.collectorDone:
			// Old target drained cleanly
		case <-ctx.Done():
			return ctx.Err()
		case <-drainTimer.C:
			// A blocked source must not silently release ownership; do NOT acquire replacement!
			return fmt.Errorf("active target failed to drain within %v: blocked source retained", drainTimeout)
		}

		// Drain deliveries
		drainCtx, drainCancel := context.WithTimeout(context.Background(), drainTimeout)
		if oldTgt.metricsDelivery != nil {
			_ = oldTgt.metricsDelivery.Close(drainCtx)
			st := oldTgt.metricsDelivery.Stats()
			a.metrics.retiredDeliverySuccesses.Add(st.SucceededBatches)
			a.metrics.retiredDeliveryFailures.Add(st.FailedBatches + st.TransportFailures + st.RejectedBatches + st.AuthFailures)
			a.metrics.retiredDeliveryDropped.Add(st.DroppedBatches)
		}
		if oldTgt.logsDelivery != nil {
			_ = oldTgt.logsDelivery.Close(drainCtx)
			st := oldTgt.logsDelivery.Stats()
			a.metrics.retiredDeliverySuccesses.Add(st.SucceededBatches)
			a.metrics.retiredDeliveryFailures.Add(st.FailedBatches + st.TransportFailures + st.RejectedBatches + st.AuthFailures)
			a.metrics.retiredDeliveryDropped.Add(st.DroppedBatches)
		}
		drainCancel()
		oldTgt.cleanup()
		a.target = nil
	}

	// 2. Acquire replacement target ONLY AFTER old target drained cleanly
	newTgt, err := a.buildTarget(newCfg)
	if err != nil {
		if a.cfg != nil {
			// Roll back by re-acquiring and restarting old target with previous config
			rbTgt, rbErr := a.buildTarget(a.cfg)
			if rbErr == nil {
				a.target = rbTgt
				a.startTarget(rbTgt)
			} else {
				// Rollback failed; terminate daemon cleanly
				select {
				case a.targetFatalErr <- fmt.Errorf("reload failed (%v) and rollback failed: %w", err, rbErr):
				default:
				}
			}
		}
		return fmt.Errorf("prepare new target: %w", err)
	}

	// 3. New target takes ownership
	a.target = newTgt
	a.startTarget(newTgt)
	return nil
}

// Reload reloads the configuration from disk if configPath was set, or re-applies current configuration.
func (a *Agent) Reload(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return err
	}
	if a.shuttingDown.Load() {
		return errors.New("cannot reload: agent is shutting down")
	}

	var newCfg *ParsedConfig
	var err error
	if a.configPath != "" {
		newCfg, err = a.readConfigFile()
	} else {
		newCfg = a.cfg
	}
	if err != nil {
		a.metrics.reloadFailures.Add(1)
		return fmt.Errorf("reload validation failed: %w", err)
	}

	return a.applyReloadLocked(ctx, newCfg)
}

// ReloadConfig dynamically updates settings from an in-memory ParsedConfig.
func (a *Agent) ReloadConfig(newCfg *ParsedConfig) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.shuttingDown.Load() {
		return errors.New("cannot reload: agent is shutting down")
	}
	if newCfg == nil {
		a.metrics.reloadFailures.Add(1)
		return errors.New("nil configuration")
	}

	return a.applyReloadLocked(context.Background(), newCfg)
}

func (a *Agent) applyReloadLocked(ctx context.Context, newCfg *ParsedConfig) error {
	if _, err := a.targetConfig(newCfg); err != nil {
		a.metrics.reloadFailures.Add(1)
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if newCfg.Listen != a.cfg.Listen {
		a.metrics.reloadFailures.Add(1)
		return errors.New("reload cannot change listen address")
	}
	if (newCfg.TLSConfig == nil) != (a.cfg.TLSConfig == nil) {
		a.metrics.reloadFailures.Add(1)
		return errors.New("reload cannot toggle TLS")
	}

	var newMetricsHandler *nvtelemetry.MetricsHandler
	if newCfg.MaxAge != a.cfg.MaxAge {
		var err error
		newMetricsHandler, err = nvtelemetry.NewMetricsHandler(nvtelemetry.ExportOptions{}, newCfg.MaxAge)
		if err != nil {
			a.metrics.reloadFailures.Add(1)
			return fmt.Errorf("create reloaded metrics handler: %w", err)
		}
	}

	if err := a.transitionTarget(ctx, newCfg); err != nil {
		a.metrics.reloadFailures.Add(1)
		return fmt.Errorf("reload target transition failed: %w", err)
	}

	if newMetricsHandler != nil {
		a.handlerMu.Lock()
		a.metricsHandler = newMetricsHandler
		a.handlerMu.Unlock()
	}
	if a.ownedDelivery != nil {
		copy := *newCfg
		copy.DeviceUUIDs = append([]string(nil), newCfg.DeviceUUIDs...)
		newCfg = &copy
	}
	a.cfg = newCfg
	a.metrics.reloadSuccesses.Add(1)
	return nil
}

// Start begins listening and serving HTTP requests, runs the collector, and handles OS signals.
func (a *Agent) Start(parent context.Context) error {
	a.mu.RLock()
	cfg := a.cfg
	a.mu.RUnlock()

	a.ctx, a.cancel = context.WithCancel(parent)
	if cfg.Duration > 0 {
		var stop context.CancelFunc
		a.ctx, stop = context.WithTimeout(a.ctx, cfg.Duration)
		defer stop()
	}

	listener, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	a.listener = listener

	// Build initial target
	tgt, err := a.buildTarget(cfg)
	if err != nil {
		_ = listener.Close()
		return err
	}
	a.target = tgt
	a.startTarget(tgt)

	mux := http.NewServeMux()
	a.setupRoutes(mux)

	a.server = &http.Server{
		Handler:           a.bearerAuth(mux),
		TLSConfig:         cfg.TLSConfig,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       30 * time.Second,
		MaxHeaderBytes:    16 << 10,
	}

	tlsCert := cfg.TLSCert
	serverErr := make(chan error, 1)
	go func() {
		if tlsCert != "" {
			serverErr <- a.server.ServeTLS(listener, "", "")
		} else {
			serverErr <- a.server.Serve(listener)
		}
	}()

	scheme := "http"
	if tlsCert != "" {
		scheme = "https"
	}
	if _, err := fmt.Fprintf(a.stdout, "%s://%s/metrics\n", scheme, listener.Addr()); err != nil {
		a.cancel()
		_ = a.server.Close()
		return err
	}

	// SIGHUP signal listener
	sigHup := make(chan os.Signal, 1)
	signal.Notify(sigHup, syscall.SIGHUP)
	defer signal.Stop(sigHup)

	go func() {
		for {
			select {
			case <-a.ctx.Done():
				return
			case <-sigHup:
				_ = a.Reload(a.ctx)
			}
		}
	}()

	var terminal error
	select {
	case terminal = <-serverErr:
	case <-a.ctx.Done():
	case terminal = <-a.targetFatalErr:
	}

	if terminal == context.Canceled || terminal == context.DeadlineExceeded || errors.Is(terminal, http.ErrServerClosed) {
		terminal = nil
	}

	shutdownErr := a.Shutdown(context.Background())
	return errors.Join(terminal, shutdownErr)
}

// Addr returns the listening network address.
func (a *Agent) Addr() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.listener == nil {
		return ""
	}
	return a.listener.Addr().String()
}

// Shutdown gracefully shuts down the HTTP server, collector, and delivery queues within the timeout.
func (a *Agent) Shutdown(ctx context.Context) error {
	a.shutdownOnce.Do(func() {
		a.shuttingDown.Store(true)
		if a.cancel != nil {
			a.cancel()
		}
		if handler := a.getMetricsHandler(); handler != nil {
			handler.Invalidate()
		}

		timeout := 5 * time.Second
		if a.cfg != nil && a.cfg.ShutdownTimeout > 0 {
			timeout = a.cfg.ShutdownTimeout
		}
		shutdownCtx, stop := context.WithTimeout(ctx, timeout)
		defer stop()

		var terminal error

		// 1. Shutdown HTTP server
		if a.server != nil {
			if err := a.server.Shutdown(shutdownCtx); err != nil {
				_ = a.server.Close()
				terminal = errors.Join(terminal, err)
			}
		}

		// 2. Stop and drain collector target
		a.targetMu.Lock()
		tgt := a.target
		a.targetMu.Unlock()

		if tgt != nil {
			tgt.stopping.Store(true)
			tgt.cancel()
			go func() {
				_, _ = tgt.collector.Stop()
			}()

			drainTimer := time.NewTimer(timeout)
			defer drainTimer.Stop()

			drainedCleanly := false
			select {
			case <-tgt.collectorDone:
				drainedCleanly = true
				if tgt.collectorErr != nil && tgt.collectorErr != context.Canceled && tgt.collectorErr != context.DeadlineExceeded {
					terminal = errors.Join(terminal, tgt.collectorErr)
				}
			case <-drainTimer.C:
				terminal = errors.Join(terminal, errors.New("collector drain timed out"))
			}

			if tgt.metricsDelivery != nil {
				terminal = errors.Join(terminal, tgt.metricsDelivery.Close(shutdownCtx))
				st := tgt.metricsDelivery.Stats()
				a.metrics.retiredDeliverySuccesses.Add(st.SucceededBatches)
				a.metrics.retiredDeliveryFailures.Add(st.FailedBatches + st.TransportFailures + st.RejectedBatches + st.AuthFailures)
				a.metrics.retiredDeliveryDropped.Add(st.DroppedBatches)
				terminal = errors.Join(terminal, json.NewEncoder(a.stdout).Encode(struct {
					Signal string                    `json:"signal"`
					Stats  nvtelemetry.DeliveryStats `json:"stats"`
				}{"metrics", st}))
			}
			if tgt.logsDelivery != nil {
				terminal = errors.Join(terminal, tgt.logsDelivery.Close(shutdownCtx))
				st := tgt.logsDelivery.Stats()
				a.metrics.retiredDeliverySuccesses.Add(st.SucceededBatches)
				a.metrics.retiredDeliveryFailures.Add(st.FailedBatches + st.TransportFailures + st.RejectedBatches + st.AuthFailures)
				a.metrics.retiredDeliveryDropped.Add(st.DroppedBatches)
				terminal = errors.Join(terminal, json.NewEncoder(a.stdout).Encode(struct {
					Signal string                    `json:"signal"`
					Stats  nvtelemetry.DeliveryStats `json:"stats"`
				}{"logs", st}))
			}

			if drainedCleanly {
				tgt.cleanup()
			} else {
				// Source is blocked during shutdown; clean up only when the blocked source actually returns.
				go func() {
					<-tgt.collectorDone
					tgt.cleanup()
				}()
			}
		}

		a.shutdownErr = terminal
		close(a.closed)
	})
	<-a.closed
	return a.shutdownErr
}

func (a *Agent) setupRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/healthz", a.handleHealthz)
	mux.HandleFunc("/readyz", a.handleReadyz)
	mux.HandleFunc("/-/reload", a.handleReload)
	mux.HandleFunc("/metrics", a.handleMetrics)
}

func (a *Agent) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if a.shuttingDown.Load() {
		http.Error(w, "shutting down", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodGet {
		_, _ = io.WriteString(w, "ok\n")
	}
}

func (a *Agent) handleReadyz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	h := a.getMetricsHandler()
	if a.shuttingDown.Load() || h == nil || !h.Ready() {
		http.Error(w, "telemetry unavailable", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodGet {
		_, _ = io.WriteString(w, "ready\n")
	}
}

func (a *Agent) handleReload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := a.Reload(r.Context()); err != nil {
		http.Error(w, fmt.Sprintf("reload failed: %v", err), http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, "reloaded\n")
}

type responseRecorder struct {
	header http.Header
	code   int
	buf    bytes.Buffer
}

func (r *responseRecorder) Header() http.Header {
	if r.header == nil {
		r.header = make(http.Header)
	}
	return r.header
}
func (r *responseRecorder) Write(p []byte) (int, error) {
	if r.code == 0 {
		r.code = http.StatusOK
	}
	return r.buf.Write(p)
}
func (r *responseRecorder) WriteHeader(code int) {
	r.code = code
}

func (a *Agent) handleMetrics(w http.ResponseWriter, r *http.Request) {
	rec := &responseRecorder{}
	h := a.getMetricsHandler()
	if h != nil {
		h.ServeHTTP(rec, r)
	}

	if rec.code == http.StatusMethodNotAllowed || rec.code == http.StatusNotAcceptable {
		for k, v := range rec.header {
			w.Header()[k] = v
		}
		w.WriteHeader(rec.code)
		_, _ = w.Write(rec.buf.Bytes())
		return
	}

	// Capture active delivery stats
	a.targetMu.Lock()
	var dSuccess, dFailed, dDropped uint64
	if a.target != nil {
		if a.target.metricsDelivery != nil {
			st := a.target.metricsDelivery.Stats()
			dSuccess += st.SucceededBatches
			dFailed += st.FailedBatches + st.TransportFailures + st.RejectedBatches + st.AuthFailures
			dDropped += st.DroppedBatches
		}
		if a.target.logsDelivery != nil {
			st := a.target.logsDelivery.Stats()
			dSuccess += st.SucceededBatches
			dFailed += st.FailedBatches + st.TransportFailures + st.RejectedBatches + st.AuthFailures
			dDropped += st.DroppedBatches
		}
	}
	a.targetMu.Unlock()

	collAtt := a.metrics.collectionAttempts.Load()
	collFail := a.metrics.collectionFailures.Load()
	collSucc := a.metrics.collectionSuccesses.Load()
	delEnq := a.metrics.deliveryEnqueued.Load()
	delFail := a.metrics.deliveryFailures.Load() + a.metrics.retiredDeliveryFailures.Load() + dFailed
	delSucc := a.metrics.deliverySuccesses.Load() + a.metrics.retiredDeliverySuccesses.Load() + dSuccess
	delDrop := a.metrics.deliveryDropped.Load() + a.metrics.retiredDeliveryDropped.Load() + dDropped
	relSucc := a.metrics.reloadSuccesses.Load()
	relFail := a.metrics.reloadFailures.Load()

	var isOpenMetrics bool
	if rec.code == http.StatusOK && rec.Header().Get("Content-Type") != "" {
		isOpenMetrics = strings.HasPrefix(rec.Header().Get("Content-Type"), "application/openmetrics-text")
	} else {
		isOpenMetrics = negotiateOpenMetrics(strings.Join(r.Header.Values("Accept"), ", "))
	}

	var selfMetrics bytes.Buffer
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_collection_attempts_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_collection_attempts_total %d\n", collAtt)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_collection_failures_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_collection_failures_total %d\n", collFail)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_collection_successes_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_collection_successes_total %d\n", collSucc)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_delivery_enqueued_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_delivery_enqueued_total %d\n", delEnq)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_delivery_failures_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_delivery_failures_total %d\n", delFail)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_delivery_successes_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_delivery_successes_total %d\n", delSucc)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_delivery_dropped_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_delivery_dropped_total %d\n", delDrop)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_reloads_total counter\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_reloads_total{result=\"success\"} %d\n", relSucc)
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_reloads_total{result=\"failure\"} %d\n", relFail)
	fmt.Fprintf(&selfMetrics, "# TYPE nvtelemetry_agent_up gauge\n")
	fmt.Fprintf(&selfMetrics, "nvtelemetry_agent_up 1\n")

	var body []byte
	if rec.code == http.StatusOK {
		body = rec.buf.Bytes()
	}
	if isOpenMetrics {
		// Strip trailing # EOF\n if present, append self-metrics, and add # EOF\n
		body = bytes.TrimSuffix(body, []byte("# EOF\n"))
		body = append(body, selfMetrics.Bytes()...)
		body = append(body, []byte("# EOF\n")...)
	} else {
		body = append(body, selfMetrics.Bytes()...)
	}

	for k, v := range rec.header {
		w.Header()[k] = v
	}
	if isOpenMetrics {
		w.Header().Set("Content-Type", "application/openmetrics-text; version=1.0.0; charset=utf-8")
	} else if w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodGet {
		_, _ = w.Write(body)
	}
}

func (a *Agent) bearerAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		a.mu.RLock()
		token := ""
		if a.cfg != nil {
			token = a.cfg.BearerToken
		}
		a.mu.RUnlock()

		if token == "" {
			next.ServeHTTP(w, r)
			return
		}

		expected := []byte("Bearer " + token)
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), expected) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func negotiateOpenMetrics(accept string) bool {
	open, err := metricsFormat(accept, false)
	if err != nil {
		return false
	}
	return open
}

func metricsFormat(accept string, defaultOpen bool) (bool, error) {
	if accept == "" {
		return defaultOpen, nil
	}
	if len(accept) > 4096 {
		return false, fmt.Errorf("accept header too large")
	}
	parts := strings.Split(accept, ",")
	if len(parts) > 64 {
		return false, fmt.Errorf("too many media ranges")
	}
	quality := [2]float64{-1, -1}
	specificity := [2]int{-1, -1}
	for _, part := range parts {
		media, params, err := mime.ParseMediaType(strings.TrimSpace(part))
		if err != nil {
			return false, err
		}
		q := 1.0
		if value, ok := params["q"]; ok {
			if !metricsQuality(value) {
				return false, fmt.Errorf("invalid quality")
			}
			q, err = strconv.ParseFloat(value, 64)
			if err != nil || math.IsNaN(q) || math.IsInf(q, 0) || q < 0 || q > 1 {
				return false, fmt.Errorf("invalid quality")
			}
		}
		for i, name := range []string{"text/plain", "application/openmetrics-text"} {
			spec := -1
			switch {
			case media == name:
				spec = 2
			case media == strings.SplitN(name, "/", 2)[0]+"/*":
				spec = 1
			case media == "*/*":
				spec = 0
			}
			if spec < 0 {
				continue
			}
			version := "0.0.4"
			if i == 1 {
				version = "1.0.0"
			}
			if v, ok := params["version"]; ok && v != version {
				continue
			}
			if charset, ok := params["charset"]; ok && !strings.EqualFold(charset, "utf-8") {
				continue
			}
			if spec > specificity[i] || spec == specificity[i] && q > quality[i] {
				specificity[i], quality[i] = spec, q
			}
		}
	}
	if quality[0] <= 0 && quality[1] <= 0 {
		return false, fmt.Errorf("no supported format")
	}
	return quality[1] > quality[0] || quality[1] == quality[0] && defaultOpen, nil
}

func metricsQuality(s string) bool {
	if len(s) == 0 || len(s) > 5 || s[0] != '0' && s[0] != '1' {
		return false
	}
	if len(s) > 1 && s[1] != '.' {
		return false
	}
	for i := 2; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' || s[0] == '1' && s[i] != '0' {
			return false
		}
	}
	return true
}
