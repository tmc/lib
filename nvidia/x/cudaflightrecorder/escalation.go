package cudaflightrecorder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/internal/cuptiowner"
)

// State represents the lifecycle state of the escalation state machine.
type State string

const (
	// StateIdle indicates the coordinator is passive; continuous observation may run,
	// but escalation triggers are ignored or refused.
	StateIdle State = "idle"
	// StateArmed indicates the coordinator is actively evaluating triggers.
	StateArmed State = "armed"
	// StateCapturing indicates an incident escalation capture is in progress.
	StateCapturing State = "capturing"
	// StateCooldown indicates capture has completed and the system is cooling down
	// to prevent trigger storms and thrashing.
	StateCooldown State = "cooldown"
	// StateRefused indicates a trigger was evaluated and rejected due to a budget or conflict.
	StateRefused State = "refused"
	// StateStopped indicates permanent terminal shutdown.
	StateStopped State = "stopped"
)

// Budgets defines both enforceable hard limits and measured targets for continuous
// flight recording and incident escalation.
//
// Ring capacity, per-bundle byte count, and capture concurrency are enforced.
// Capture duration uses context cancellation; providers and writers must cooperate.
//
// Measured targets (such as CPU percent and RSS bytes) are evaluated periodically via
// sampled checks over an explicit measurement window. They represent measured target
// bounds rather than instantaneous hard caps.
type Budgets struct {
	// Measured targets (sampled):
	CPUWindow     time.Duration `json:"cpu_window"`      // Sampling window for host CPU measurement (e.g. 1s)
	MaxCPUPercent float64       `json:"max_cpu_percent"` // Target maximum CPU percent (e.g. 10.0%)
	MaxRSSBytes   uint64        `json:"max_rss_bytes"`   // Target maximum host RSS in bytes

	// Enforceable hard limits:
	MaxRingRecords           int           `json:"max_ring_records"`             // Upper bound on in-memory ring records
	MaxRetainedBytes         int64         `json:"max_retained_bytes"`           // Per-bundle byte limit; not aggregate storage retention
	MaxCaptureDuration       time.Duration `json:"max_capture_duration"`         // Context timeout; providers and writers must honor cancellation
	MaxConcurrentCaptures    int           `json:"max_concurrent_captures"`      // Max active captures across all tenants
	MaxTenantConcurrency     int           `json:"max_tenant_concurrency"`       // Max active captures for a single tenant
	MaxTenantCapturesPerHour int           `json:"max_tenant_captures_per_hour"` // Rate limit per tenant per hour

	// Temporal limits & hysteresis:
	CooldownDuration    time.Duration `json:"cooldown_duration"`     // Minimum cooldown duration after capture
	HysteresisCount     int           `json:"hysteresis_count"`      // Required consecutive trigger events before escalating
	HysteresisWindow    time.Duration `json:"hysteresis_window"`     // Time window for consecutive trigger events
	MaxRetainedReceipts int           `json:"max_retained_receipts"` // Bounded ring capacity for incident receipts
}

// DefaultBudgets returns conservative initial limits for snapshot capture.
func DefaultBudgets() Budgets {
	return Budgets{
		CPUWindow:                1 * time.Second,
		MaxCPUPercent:            15.0,
		MaxRSSBytes:              512 << 20, // 512 MiB
		MaxRingRecords:           65536,
		MaxRetainedBytes:         128 << 20, // 128 MiB
		MaxCaptureDuration:       10 * time.Second,
		MaxConcurrentCaptures:    1,
		MaxTenantConcurrency:     1,
		MaxTenantCapturesPerHour: 30,
		CooldownDuration:         5 * time.Second,
		HysteresisCount:          2,
		HysteresisWindow:         5 * time.Second,
		MaxRetainedReceipts:      100,
	}
}

// QualificationBudgets returns the frozen qualification budgets.
func QualificationBudgets() Budgets {
	return Budgets{
		CPUWindow:                1 * time.Second,
		MaxCPUPercent:            10.0,
		MaxRSSBytes:              256 << 20, // 256 MiB
		MaxRingRecords:           65536,
		MaxRetainedBytes:         64 << 20, // 64 MiB
		MaxCaptureDuration:       5 * time.Second,
		MaxConcurrentCaptures:    1,
		MaxTenantConcurrency:     1,
		MaxTenantCapturesPerHour: 10,
		CooldownDuration:         2 * time.Second,
		HysteresisCount:          2,
		HysteresisWindow:         5 * time.Second,
		MaxRetainedReceipts:      100,
	}
}

// Refusal reason codes.
const (
	RefusalCooldownActive    = "cooldown_active"
	RefusalCPUExceeded       = "cpu_budget_exceeded"
	RefusalRSSExceeded       = "rss_budget_exceeded"
	RefusalConcurrencyLimit  = "concurrency_limit_exceeded"
	RefusalTenantConcurrency = "tenant_concurrency_exceeded"
	RefusalTenantRate        = "tenant_rate_exceeded"
	RefusalCUPTIConflict     = "cupti_ownership_conflict"
	RefusalHysteresisNotMet  = "hysteresis_not_met"
	RefusalStopped           = "stopped"
	RefusalDisarmed          = "disarmed"
)

// ErrEscalationRefused indicates that an escalation trigger was examined and refused.
var ErrEscalationRefused = fmt.Errorf("cudaflightrecorder: escalation refused: %w", cudaerrors.ErrUnsupported)

// PreTriggerReport documents the availability and fidelity of pre-incident observations
// retained in memory before the trigger was fired.
type PreTriggerReport struct {
	Available                 bool          `json:"available"`
	WindowDuration            time.Duration `json:"window_duration"`
	RecordCount               int           `json:"record_count"`
	Truncated                 bool          `json:"truncated"`
	EarliestTimestampNs       int64         `json:"earliest_timestamp_ns,omitempty"`
	LatestTimestampNs         int64         `json:"latest_timestamp_ns,omitempty"`
	MissingPreTriggerFamilies []string      `json:"missing_pre_trigger_families,omitempty"`
}

// TriggerEvidence captures the factual trigger context, metric values, and pre-trigger availability.
type TriggerEvidence struct {
	TimestampMono int64             `json:"timestamp_mono"`
	TimestampUnix int64             `json:"timestamp_unix"`
	TenantID      string            `json:"tenant_id"`
	Reason        string            `json:"reason"`
	MetricName    string            `json:"metric_name"`
	MetricValue   float64           `json:"metric_value"`
	Threshold     float64           `json:"threshold"`
	PreTrigger    PreTriggerReport  `json:"pre_trigger"`
	Details       map[string]string `json:"details,omitempty"`
}

// TriggerRequest defines an incoming escalation request.
type TriggerRequest struct {
	TenantID      string            `json:"tenant_id"`
	Reason        string            `json:"reason"`
	MetricName    string            `json:"metric_name"`
	MetricValue   float64           `json:"metric_value"`
	Threshold     float64           `json:"threshold"`
	DeviceOrdinal uint32            `json:"device_ordinal"`
	Details       map[string]string `json:"details,omitempty"`
}

// PerturbationReport captures measured overhead induced by high-fidelity incident capture.
type PerturbationReport struct {
	CaptureDuration  time.Duration `json:"capture_duration"`
	BundleBytes      int64         `json:"bundle_bytes"`
	RecordsCaptured  int           `json:"records_captured"`
	CPUPercentActual float64       `json:"cpu_percent_actual"`
	RSSBytesActual   uint64        `json:"rss_bytes_actual"`
}

// IncidentReceipt is the comprehensive audit and diagnostic receipt of an incident escalation.
type IncidentReceipt struct {
	IncidentID    string             `json:"incident_id"`
	Timestamp     time.Time          `json:"timestamp"`
	TenantID      string             `json:"tenant_id"`
	State         State              `json:"state"`
	Outcome       string             `json:"outcome"` // "captured", "refused", "failed"
	RefusalReason string             `json:"refusal_reason,omitempty"`
	Trigger       TriggerEvidence    `json:"trigger"`
	Budgets       Budgets            `json:"budgets"`
	Perturbation  PerturbationReport `json:"perturbation"`
	Capture       CaptureReport      `json:"capture"`
	Error         string             `json:"error,omitempty"`
}

// Clock abstracts time for deterministic simulation in unit and acceptance tests.
type Clock interface {
	Now() time.Time
	NowMonotonic() int64
	NowUnix() int64
}

// RealClock implements Clock using the operating system clock.
type RealClock struct{}

// Now returns the current time.
func (RealClock) Now() time.Time { return time.Now() }

// NowMonotonic returns a nanosecond monotonic timestamp.
var escalationEpoch = time.Now()

func (RealClock) NowMonotonic() int64 { return time.Since(escalationEpoch).Nanoseconds() }

// NowUnix returns current Unix time in nanoseconds.
func (RealClock) NowUnix() int64 { return time.Now().UnixNano() }

// FakeClock is a thread-safe injected clock for testing.
type FakeClock struct {
	mu    sync.Mutex
	t     time.Time
	mono  int64
	epoch int64
}

// NewFakeClock creates a FakeClock anchored at start.
func NewFakeClock(start time.Time) *FakeClock {
	return &FakeClock{
		t:     start,
		mono:  1_000_000_000,
		epoch: start.UnixNano(),
	}
}

// Now returns current simulated time.
func (f *FakeClock) Now() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.t
}

// NowMonotonic returns current simulated monotonic nanoseconds.
func (f *FakeClock) NowMonotonic() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.mono
}

// NowUnix returns current simulated Unix nanoseconds.
func (f *FakeClock) NowUnix() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.epoch
}

// Advance moves simulated time forward by d.
func (f *FakeClock) Advance(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.t = f.t.Add(d)
	f.mono += d.Nanoseconds()
	f.epoch += d.Nanoseconds()
}

// ResourceSampler samples host process CPU and memory (RSS) metrics.
type ResourceSampler interface {
	SampleUsage(ctx context.Context) (cpuPercent float64, rssBytes uint64, err error)
}

// HostResourceSampler measures process CPU and resident memory over Window.
// A zero Window selects one second. CPU is expressed as percent of one CPU.
type HostResourceSampler struct{ Window time.Duration }

// SampleUsage measures usage over a cancellable window. Unsupported platforms
// return an error rather than reporting invented zero-valued measurements.
func (s HostResourceSampler) SampleUsage(ctx context.Context) (float64, uint64, error) {
	window := s.Window
	if window <= 0 {
		window = time.Second
	}
	return sampleHostUsage(ctx, window)
}

// FakeResourceSampler provides scripted resource measurements for testing.
type FakeResourceSampler struct {
	mu         sync.Mutex
	CPUPercent float64
	RSSBytes   uint64
	Err        error
}

// SetUsage updates the simulated resource usage.
func (f *FakeResourceSampler) SetUsage(cpu float64, rss uint64, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.CPUPercent = cpu
	f.RSSBytes = rss
	f.Err = err
}

// SampleUsage returns the configured values.
func (f *FakeResourceSampler) SampleUsage(ctx context.Context) (float64, uint64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.CPUPercent, f.RSSBytes, f.Err
}

// CUPTINegotiator negotiates CUPTI compatibility and device ownership without session theft.
type CUPTINegotiator interface {
	CheckCompatibility(ctx context.Context, device uint32) error
	AcquireLease(ctx context.Context, device uint32, ownerID string) (release func(), err error)
}

// DefaultCUPTINegotiator coordinates device ownership via x/internal/cuptiowner.
type DefaultCUPTINegotiator struct{}

// CheckCompatibility verifies if device is available without holding the lease.
func (DefaultCUPTINegotiator) CheckCompatibility(ctx context.Context, device uint32) error {
	lease, err := cuptiowner.AcquireDevice(device)
	if err != nil {
		return fmt.Errorf("cupti device %d conflict: %w: %v", device, cudaerrors.ErrUnsupported, err)
	}
	lease.Release()
	return nil
}

// AcquireLease reserves the device ordinal for the capture duration.
func (DefaultCUPTINegotiator) AcquireLease(ctx context.Context, device uint32, ownerID string) (func(), error) {
	lease, err := cuptiowner.AcquireDevice(device)
	if err != nil {
		return nil, fmt.Errorf("cupti device %d conflict: %w: %v", device, cudaerrors.ErrUnsupported, err)
	}
	return lease.Release, nil
}

// FakeCUPTINegotiator allows simulating conflicting owners and compatibility errors.
type FakeCUPTINegotiator struct {
	mu       sync.Mutex
	conflict bool
	err      error
}

// SetConflict sets whether compatibility check should indicate a conflict.
func (f *FakeCUPTINegotiator) SetConflict(conflict bool, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.conflict = conflict
	f.err = err
}

// CheckCompatibility checks configured conflict status.
func (f *FakeCUPTINegotiator) CheckCompatibility(ctx context.Context, device uint32) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.conflict {
		if f.err != nil {
			return f.err
		}
		return fmt.Errorf("cupti conflict on device %d: %w", device, cudaerrors.ErrUnsupported)
	}
	return nil
}

// AcquireLease acquires a mock lease.
func (f *FakeCUPTINegotiator) AcquireLease(ctx context.Context, device uint32, ownerID string) (func(), error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.conflict {
		if f.err != nil {
			return nil, f.err
		}
		return nil, fmt.Errorf("cupti conflict on device %d: %w", device, cudaerrors.ErrUnsupported)
	}
	return func() {}, nil
}

// CaptureWriterFactory creates an io.Writer for an incident capture bundle.
type CaptureWriterFactory interface {
	CreateWriter(incidentID string, tenantID string) (io.Writer, func() error, error)
}

// DirectoryWriterFactory writes incident bundles to files in a directory.
type DirectoryWriterFactory struct {
	dir string
}

// NewDirectoryWriterFactory creates a factory writing to dir.
func NewDirectoryWriterFactory(dir string) *DirectoryWriterFactory {
	return &DirectoryWriterFactory{dir: dir}
}

// CreateWriter creates a new tar file in the configured directory.
func (d *DirectoryWriterFactory) CreateWriter(incidentID string, tenantID string) (io.Writer, func() error, error) {
	if err := os.MkdirAll(d.dir, 0755); err != nil {
		return nil, nil, fmt.Errorf("create incident dir: %w", err)
	}
	name := fmt.Sprintf("incident-%x-%x.tar", sha256.Sum256([]byte(tenantID)), sha256.Sum256([]byte(incidentID)))
	path := filepath.Join(d.dir, name)
	f, err := os.Create(path)
	if err != nil {
		return nil, nil, fmt.Errorf("create incident file: %w", err)
	}
	return f, f.Close, nil
}

// BufferWriterFactory retains up to 100 incident bundles in memory.
// Each bundle is independently byte-limited by the coordinator; this factory
// does not enforce an aggregate byte limit. Access Buffers only after Trigger returns.
type BufferWriterFactory struct {
	mu       sync.Mutex
	Buffers  map[string]*bytes.Buffer
	order    []string
	capacity int
}

// NewBufferWriterFactory creates an in-memory writer factory.
func NewBufferWriterFactory() *BufferWriterFactory {
	return &BufferWriterFactory{
		Buffers:  make(map[string]*bytes.Buffer),
		capacity: 100,
	}
}

// CreateWriter allocates an in-memory buffer.
func (b *BufferWriterFactory) CreateWriter(incidentID string, tenantID string) (io.Writer, func() error, error) {
	buf := new(bytes.Buffer)
	b.mu.Lock()
	if b.Buffers == nil {
		b.Buffers = make(map[string]*bytes.Buffer)
	}
	capacity := b.capacity
	if capacity <= 0 {
		capacity = 100
	}
	if len(b.order) >= capacity {
		delete(b.Buffers, b.order[0])
		b.order = b.order[1:]
	}
	b.order = append(b.order, incidentID)
	b.Buffers[incidentID] = buf
	b.mu.Unlock()
	return buf, func() error { return nil }, nil
}

// EscalationOption configures an EscalationCoordinator.
type EscalationOption func(*EscalationCoordinator)

// WithClock sets a custom Clock.
func WithClock(c Clock) EscalationOption {
	return func(ec *EscalationCoordinator) {
		ec.clock = c
	}
}

// WithResourceSampler sets a custom ResourceSampler.
func WithResourceSampler(s ResourceSampler) EscalationOption {
	return func(ec *EscalationCoordinator) {
		ec.sampler = s
	}
}

// WithCUPTINegotiator sets a custom CUPTINegotiator.
func WithCUPTINegotiator(n CUPTINegotiator) EscalationOption {
	return func(ec *EscalationCoordinator) {
		ec.negotiator = n
	}
}

// WithCaptureWriterFactory sets a custom CaptureWriterFactory.
func WithCaptureWriterFactory(f CaptureWriterFactory) EscalationOption {
	return func(ec *EscalationCoordinator) {
		ec.writerFact = f
	}
}

// EscalationCoordinator creates bounded snapshots of an existing Recorder.
// It does not start native acquisition or change its fidelity. Snapshot storage
// retention and deletion are the writer factory owner's responsibility.
type EscalationCoordinator struct {
	recorder *Recorder
	budgets  Budgets

	clock           Clock
	sampler         ResourceSampler
	negotiator      CUPTINegotiator
	writerFact      CaptureWriterFactory
	captureProvider IncidentCapture

	mu sync.Mutex

	state State

	cooldownUntil time.Time

	// Hysteresis tracking: metric name -> list of trigger timestamps
	triggerHistory map[string][]time.Time

	// Tenant tracking:
	activeTenants        map[string]int
	tenantHourlyCaptures map[string][]time.Time

	// Global concurrency
	activeCaptures int

	// Incident receipts history (bounded ring)
	history []IncidentReceipt

	// Sequence counter for incidents
	incidentSeq uint64

	closed         bool
	shutdown       context.Context
	cancelShutdown context.CancelFunc
	jobs           chan escalationJob
	jobsDone       chan struct{}
}

// NewEscalationCoordinator creates a new coordinator wrapping recorder.
func NewEscalationCoordinator(rec *Recorder, budgets Budgets, opts ...EscalationOption) (*EscalationCoordinator, error) {
	if rec == nil {
		return nil, errors.New("cudaflightrecorder: nil recorder")
	}
	if budgets.MaxRingRecords <= 0 {
		budgets.MaxRingRecords = 65536
	}
	if budgets.MaxRetainedBytes <= 0 {
		budgets.MaxRetainedBytes = 64 << 20
	}
	if budgets.MaxCaptureDuration <= 0 {
		budgets.MaxCaptureDuration = 5 * time.Second
	}
	if budgets.MaxConcurrentCaptures <= 0 {
		budgets.MaxConcurrentCaptures = 1
	}
	if budgets.MaxConcurrentCaptures > 1 {
		return nil, fmt.Errorf("cudaflightrecorder: concurrent snapshots: %w", cudaerrors.ErrUnsupported)
	}
	if budgets.MaxTenantConcurrency <= 0 {
		budgets.MaxTenantConcurrency = 1
	}
	if budgets.MaxTenantCapturesPerHour <= 0 {
		budgets.MaxTenantCapturesPerHour = 10
	}
	if budgets.CooldownDuration <= 0 {
		budgets.CooldownDuration = 2 * time.Second
	}
	if budgets.HysteresisCount <= 0 {
		budgets.HysteresisCount = 1
	}
	if budgets.HysteresisWindow <= 0 {
		budgets.HysteresisWindow = 5 * time.Second
	}
	if budgets.MaxRetainedReceipts <= 0 {
		budgets.MaxRetainedReceipts = 100
	}

	rec.mu.Lock()
	capacity := len(rec.records)
	rec.mu.Unlock()
	if capacity > budgets.MaxRingRecords {
		return nil, fmt.Errorf("cudaflightrecorder: recorder capacity %d exceeds ring budget %d", capacity, budgets.MaxRingRecords)
	}

	ec := &EscalationCoordinator{
		recorder:             rec,
		budgets:              budgets,
		clock:                RealClock{},
		sampler:              HostResourceSampler{Window: budgets.CPUWindow},
		negotiator:           DefaultCUPTINegotiator{},
		writerFact:           NewBufferWriterFactory(),
		state:                StateIdle,
		triggerHistory:       make(map[string][]time.Time),
		activeTenants:        make(map[string]int),
		tenantHourlyCaptures: make(map[string][]time.Time),
		history:              make([]IncidentReceipt, 0, budgets.MaxRetainedReceipts),
	}

	for _, opt := range opts {
		if opt == nil {
			return nil, errors.New("cudaflightrecorder: nil escalation option")
		}
		opt(ec)
	}
	if nilEscalationDependency(ec.clock) || nilEscalationDependency(ec.sampler) || nilEscalationDependency(ec.negotiator) || nilEscalationDependency(ec.writerFact) || (ec.captureProvider != nil && nilEscalationDependency(ec.captureProvider)) {
		return nil, errors.New("cudaflightrecorder: nil escalation dependency")
	}
	ec.shutdown, ec.cancelShutdown = context.WithCancel(context.Background())
	ec.jobsDone = make(chan struct{})
	return ec, nil
}

// State reports the current state machine state.
func (ec *EscalationCoordinator) State() State {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	return ec.state
}

// Arm transitions the coordinator from StateIdle or StateCooldown into StateArmed.
func (ec *EscalationCoordinator) Arm() error {
	ec.mu.Lock()
	defer ec.mu.Unlock()

	if ec.closed {
		return errors.New("cudaflightrecorder: coordinator is stopped")
	}
	if ec.state == StateCapturing {
		return errors.New("cudaflightrecorder: cannot arm while capture is active")
	}
	ec.state = StateArmed
	return nil
}

// Disarm transitions the coordinator from StateArmed to StateIdle.
func (ec *EscalationCoordinator) Disarm() error {
	ec.mu.Lock()
	defer ec.mu.Unlock()

	if ec.closed {
		return errors.New("cudaflightrecorder: coordinator is stopped")
	}
	if ec.state == StateCapturing {
		return errors.New("cudaflightrecorder: cannot disarm while capture is active")
	}
	ec.state = StateIdle
	return nil
}

// CooldownRemaining returns remaining cooldown duration if currently in cooldown.
func (ec *EscalationCoordinator) CooldownRemaining() time.Duration {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	if ec.state != StateCooldown {
		return 0
	}
	rem := ec.cooldownUntil.Sub(ec.clock.Now())
	if rem < 0 {
		return 0
	}
	return rem
}

// Trigger evaluates an incident escalation request against declared budgets and state.
// If admitted, it snapshots the recorder, enforces byte and concurrency limits,
// and records the receipt. It does not start a new acquisition session.
// If refused, it records the refusal reason and trigger evidence, returning ErrEscalationRefused.
func (ec *EscalationCoordinator) Trigger(ctx context.Context, req TriggerRequest) (receipt *IncidentReceipt, err error) {
	if ctx == nil {
		return nil, errors.New("cudaflightrecorder: nil trigger context")
	}
	req, err = copyTriggerRequest(req)
	if err != nil {
		return nil, err
	}
	requestCtx, cancelRequest := context.WithCancel(ctx)
	stopShutdown := context.AfterFunc(ec.shutdown, cancelRequest)
	defer stopShutdown()
	defer cancelRequest()
	ctx = requestCtx
	if req.TenantID == "" {
		req.TenantID = "default"
	}

	now := ec.clock.Now()
	nowMono := ec.clock.NowMonotonic()
	nowUnix := ec.clock.NowUnix()

	ec.mu.Lock()
	ec.incidentSeq++
	incidentID := fmt.Sprintf("inc-%06d-%d", ec.incidentSeq, now.Unix())

	// 1. Check if stopped
	if ec.closed || ec.state == StateStopped {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalStopped, "coordinator is stopped")
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: coordinator is stopped", ErrEscalationRefused)
	}

	// 2. Check if context is already canceled
	if err := ctx.Err(); err != nil {
		receipt := ec.recordRefusalLocked(incidentID, now, req, "context_canceled", err.Error())
		ec.mu.Unlock()
		return receipt, err
	}

	// 3. Check if disarmed
	if ec.state == StateIdle {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalDisarmed, "coordinator is not armed")
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: coordinator is not armed", ErrEscalationRefused)
	}

	// Cooldown remains binding even after Disarm and Arm.
	if now.Before(ec.cooldownUntil) {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalCooldownActive,
			fmt.Sprintf("cooldown active (%v remaining)", ec.cooldownUntil.Sub(now)))
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: cooldown active", ErrEscalationRefused)
	}
	if ec.state == StateCooldown {
		ec.state = StateArmed
	}

	// 5. Global concurrency budget check
	if ec.activeCaptures >= ec.budgets.MaxConcurrentCaptures {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalConcurrencyLimit,
			fmt.Sprintf("active captures %d reached limit %d", ec.activeCaptures, ec.budgets.MaxConcurrentCaptures))
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: concurrent captures limit reached", ErrEscalationRefused)
	}

	// 6. Tenant concurrency budget check
	if ec.activeTenants[req.TenantID] >= ec.budgets.MaxTenantConcurrency {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalTenantConcurrency,
			fmt.Sprintf("tenant %s active captures %d reached limit %d", req.TenantID, ec.activeTenants[req.TenantID], ec.budgets.MaxTenantConcurrency))
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: tenant concurrency limit reached", ErrEscalationRefused)
	}

	// Expire inactive keys and refuse new keys beyond the bounded tracking table.
	for tenant, timestamps := range ec.tenantHourlyCaptures {
		if len(timestamps) == 0 || !timestamps[len(timestamps)-1].After(now.Add(-time.Hour)) {
			delete(ec.tenantHourlyCaptures, tenant)
		}
	}
	for key, timestamps := range ec.triggerHistory {
		if len(timestamps) == 0 || !timestamps[len(timestamps)-1].After(now.Add(-ec.budgets.HysteresisWindow)) {
			delete(ec.triggerHistory, key)
		}
	}
	if _, exists := ec.tenantHourlyCaptures[req.TenantID]; !exists && len(ec.tenantHourlyCaptures) >= 1024 {
		receipt := ec.recordRefusalLocked(incidentID, now, req, "tracking_capacity", "tenant tracking capacity reached")
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: tenant tracking capacity reached", ErrEscalationRefused)
	}

	// 7. Tenant hourly rate limit check
	hourlyCutoff := now.Add(-1 * time.Hour)
	var filteredHourly []time.Time
	for _, t := range ec.tenantHourlyCaptures[req.TenantID] {
		if t.After(hourlyCutoff) {
			filteredHourly = append(filteredHourly, t)
		}
	}
	ec.tenantHourlyCaptures[req.TenantID] = filteredHourly
	if len(filteredHourly) >= ec.budgets.MaxTenantCapturesPerHour {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalTenantRate,
			fmt.Sprintf("tenant %s hourly captures %d reached limit %d", req.TenantID, len(filteredHourly), ec.budgets.MaxTenantCapturesPerHour))
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: tenant hourly capture limit reached", ErrEscalationRefused)
	}

	// 8. Hysteresis evaluation
	metricKey := req.MetricName
	if metricKey == "" {
		metricKey = req.Reason
	}
	metricKey = fmt.Sprintf("%d:%s%s", len(req.TenantID), req.TenantID, metricKey)
	if _, exists := ec.triggerHistory[metricKey]; !exists && len(ec.triggerHistory) >= 1024 {
		receipt := ec.recordRefusalLocked(incidentID, now, req, "tracking_capacity", "trigger tracking capacity reached")
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: trigger tracking capacity reached", ErrEscalationRefused)
	}
	windowCutoff := now.Add(-ec.budgets.HysteresisWindow)
	var filteredHistory []time.Time
	for _, t := range ec.triggerHistory[metricKey] {
		if t.After(windowCutoff) {
			filteredHistory = append(filteredHistory, t)
		}
	}
	filteredHistory = append(filteredHistory, now)
	ec.triggerHistory[metricKey] = filteredHistory

	if len(filteredHistory) < ec.budgets.HysteresisCount {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalHysteresisNotMet,
			fmt.Sprintf("hysteresis count %d/%d not satisfied within %v", len(filteredHistory), ec.budgets.HysteresisCount, ec.budgets.HysteresisWindow))
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: trigger hysteresis not met", ErrEscalationRefused)
	}

	// Reserve admission before resource sampling or lease acquisition can block.
	delete(ec.triggerHistory, metricKey)
	ec.activeCaptures++
	ec.activeTenants[req.TenantID]++
	ec.state = StateCapturing
	ec.mu.Unlock()
	defer func() {
		ec.mu.Lock()
		defer ec.mu.Unlock()
		ec.activeCaptures--
		ec.activeTenants[req.TenantID]--
		if ec.activeTenants[req.TenantID] == 0 {
			delete(ec.activeTenants, req.TenantID)
		}
		if ec.closed {
			ec.state = StateStopped
		} else if ec.activeCaptures == 0 {
			ec.state = StateCooldown
			ec.cooldownUntil = ec.clock.Now().Add(ec.budgets.CooldownDuration)
		}
	}()

	// 9. Sampled target checks (CPU / RSS)
	cpuActual, rssActual, sampleErr := ec.sampler.SampleUsage(ctx)
	if sampleErr == nil && (math.IsNaN(cpuActual) || math.IsInf(cpuActual, 0) || cpuActual < 0) {
		sampleErr = errors.New("invalid CPU measurement")
	}
	if sampleErr != nil {
		ec.mu.Lock()
		receipt := ec.recordRefusalLocked(incidentID, now, req, "sampler_error", sampleErr.Error())
		ec.mu.Unlock()
		return receipt, fmt.Errorf("sample resource usage: %w", sampleErr)
	}
	if ec.budgets.MaxCPUPercent > 0 && cpuActual > ec.budgets.MaxCPUPercent {
		ec.mu.Lock()
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalCPUExceeded,
			fmt.Sprintf("CPU usage %.2f%% exceeds target %.2f%%", cpuActual, ec.budgets.MaxCPUPercent))
		receipt.Perturbation.CPUPercentActual = cpuActual
		receipt.Perturbation.RSSBytesActual = rssActual
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: CPU budget exceeded (%.2f%% > %.2f%%)", ErrEscalationRefused, cpuActual, ec.budgets.MaxCPUPercent)
	}
	if ec.budgets.MaxRSSBytes > 0 && rssActual > ec.budgets.MaxRSSBytes {
		ec.mu.Lock()
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalRSSExceeded,
			fmt.Sprintf("RSS %d bytes exceeds target %d bytes", rssActual, ec.budgets.MaxRSSBytes))
		receipt.Perturbation.CPUPercentActual = cpuActual
		receipt.Perturbation.RSSBytesActual = rssActual
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: RSS budget exceeded (%d > %d bytes)", ErrEscalationRefused, rssActual, ec.budgets.MaxRSSBytes)
	}

	// 10. CUPTI ownership & compatibility negotiation (never steal session)
	if err := ec.negotiator.CheckCompatibility(ctx, req.DeviceOrdinal); err != nil {
		ec.mu.Lock()
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalCUPTIConflict, err.Error())
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: %v", ErrEscalationRefused, err)
	}

	releaseLease, err := ec.negotiator.AcquireLease(ctx, req.DeviceOrdinal, req.TenantID)
	if err != nil {
		ec.mu.Lock()
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalCUPTIConflict, err.Error())
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: acquire cupti lease: %v", ErrEscalationRefused, err)
	}
	var captureReport CaptureReport
	defer func() {
		if !captureReport.RetainedOwnership {
			releaseLease()
		}
	}()

	// 11. Pre-trigger report
	preTrigger := ec.recorder.PreTriggerDiagnostics(nowUnix, ec.budgets.CPUWindow)

	// 12. Enter StateCapturing
	ec.mu.Lock()
	ec.tenantHourlyCaptures[req.TenantID] = append(ec.tenantHourlyCaptures[req.TenantID], now)
	if ec.closed {
		receipt := ec.recordRefusalLocked(incidentID, now, req, RefusalStopped, "stopped during admission")
		ec.mu.Unlock()
		return receipt, fmt.Errorf("%w: stopped during admission", ErrEscalationRefused)
	}
	ec.mu.Unlock()

	// Ensure cleanup transitions on return
	var captureErr error
	var bundleBytes int64
	var recordsCount int
	var lastReceipt IncidentReceipt
	startCapture := ec.clock.Now()

	defer func() {
		dur := ec.clock.Now().Sub(startCapture)
		ec.mu.Lock()
		finalState := StateCooldown
		if ec.closed {
			finalState = StateStopped
		}

		outcome := "captured"
		errStr := ""
		if captureErr != nil {
			outcome = "failed"
			errStr = captureErr.Error()
		}

		lastReceipt = IncidentReceipt{
			IncidentID: incidentID,
			Timestamp:  now,
			TenantID:   req.TenantID,
			State:      finalState,
			Outcome:    outcome,
			Budgets:    ec.budgets,
			Trigger: TriggerEvidence{
				TimestampMono: nowMono,
				TimestampUnix: nowUnix,
				TenantID:      req.TenantID,
				Reason:        req.Reason,
				MetricName:    req.MetricName,
				MetricValue:   req.MetricValue,
				Threshold:     req.Threshold,
				PreTrigger:    preTrigger,
				Details:       req.Details,
			},
			Perturbation: PerturbationReport{
				CaptureDuration:  dur,
				BundleBytes:      bundleBytes,
				RecordsCaptured:  recordsCount,
				CPUPercentActual: cpuActual,
				RSSBytesActual:   rssActual,
			},
			Error:   errStr,
			Capture: captureReport,
		}
		ec.appendHistoryLocked(lastReceipt)
		receipt = &lastReceipt
		if captureErr != nil {
			err = captureErr
		}
		ec.mu.Unlock()
	}()

	// 13. Execute bounded capture
	captureTimeout := ec.budgets.MaxCaptureDuration
	captureCtx, cancel := context.WithTimeout(ctx, captureTimeout)
	defer cancel()

	w, closeFn, wErr := ec.writerFact.CreateWriter(incidentID, req.TenantID)
	if wErr != nil {
		captureErr = fmt.Errorf("create incident writer: %w", wErr)
		return
	}
	defer func() {
		if closeErr := closeFn(); closeErr != nil {
			captureErr = errors.Join(captureErr, fmt.Errorf("close snapshot writer: %w", closeErr))
		}
	}()

	// Count bytes written to enforce hard MaxRetainedBytes limit
	bcw := &byteCountingWriter{w: w, limit: ec.budgets.MaxRetainedBytes}

	if ec.captureProvider != nil {
		captureReport, captureErr = ec.captureProvider.Capture(captureCtx, ec.recorder, req, releaseLease)
		defer ec.recorder.DetachFile("incident-activity.json")
	}

	// Prepare trigger evidence metadata for the snapshot bundle
	triggerEvidence := TriggerEvidence{
		TimestampMono: nowMono,
		TimestampUnix: nowUnix,
		TenantID:      req.TenantID,
		Reason:        req.Reason,
		MetricName:    req.MetricName,
		MetricValue:   req.MetricValue,
		Threshold:     req.Threshold,
		PreTrigger:    preTrigger,
		Details:       req.Details,
	}
	incidentJSON, err := json.MarshalIndent(triggerEvidence, "", "  ")
	if err == nil {
		ec.recorder.AttachFile("incident.json", incidentJSON)
		defer ec.recorder.DetachFile("incident.json")
	}

	manifest, snapErr := ec.recorder.Snapshot(captureCtx, bcw, "incident:"+req.Reason)
	bundleBytes = bcw.written
	if manifest != nil {
		recordsCount = manifest.GPURecords
	}
	if snapErr != nil {
		captureErr = errors.Join(captureErr, fmt.Errorf("capture snapshot bundle: %w", snapErr))
		return
	}

	return
}

func (ec *EscalationCoordinator) recordRefusalLocked(incidentID string, now time.Time, req TriggerRequest, reason, detail string) *IncidentReceipt {
	receipt := IncidentReceipt{
		IncidentID:    incidentID,
		Timestamp:     now,
		TenantID:      req.TenantID,
		State:         StateRefused,
		Outcome:       "refused",
		RefusalReason: reason,
		Budgets:       ec.budgets,
		Trigger: TriggerEvidence{
			TimestampMono: ec.clock.NowMonotonic(),
			TimestampUnix: ec.clock.NowUnix(),
			TenantID:      req.TenantID,
			Reason:        req.Reason,
			MetricName:    req.MetricName,
			MetricValue:   req.MetricValue,
			Threshold:     req.Threshold,
			PreTrigger:    ec.recorder.PreTriggerDiagnostics(now.UnixNano(), ec.budgets.CPUWindow),
			Details:       req.Details,
		},
		Error: detail,
	}
	ec.appendHistoryLocked(receipt)
	return &receipt
}

func (ec *EscalationCoordinator) appendHistoryLocked(r IncidentReceipt) {
	if ec.budgets.MaxRetainedReceipts <= 0 {
		return
	}
	if len(ec.history) >= ec.budgets.MaxRetainedReceipts {
		copy(ec.history, ec.history[1:])
		ec.history[len(ec.history)-1] = copyIncidentReceipt(r)
	} else {
		ec.history = append(ec.history, copyIncidentReceipt(r))
	}
}

// History returns a copy of retained incident receipts.
func (ec *EscalationCoordinator) History() []IncidentReceipt {
	ec.mu.Lock()
	defer ec.mu.Unlock()
	res := make([]IncidentReceipt, len(ec.history))
	for i, receipt := range ec.history {
		res[i] = copyIncidentReceipt(receipt)
	}
	return res
}

// Stop permanently disarms the coordinator, cancels active capture contexts,
// and closes asynchronous admission. It is idempotent and does not wait for
// provider cleanup. Wait drains asynchronous completion; retained native cleanup
// remains the IncidentCapture provider owner's responsibility.
func (ec *EscalationCoordinator) Stop() error {
	ec.mu.Lock()
	defer ec.mu.Unlock()

	if ec.closed {
		return nil
	}
	ec.closed = true
	ec.state = StateStopped
	ec.cancelShutdown()
	if ec.jobs != nil {
		close(ec.jobs)
	} else {
		close(ec.jobsDone)
	}
	return nil
}

// Close implements io.Closer.
func (ec *EscalationCoordinator) Close() error {
	return ec.Stop()
}

type byteCountingWriter struct {
	w       io.Writer
	limit   int64
	written int64
}

func (b *byteCountingWriter) Write(p []byte) (int, error) {
	if b.limit > 0 && int64(len(p)) > b.limit-b.written {
		return 0, fmt.Errorf("%w: capture bundle exceeded byte budget %d bytes", ErrSnapshotBudgetExceeded, b.limit)
	}
	n, err := b.w.Write(p)
	if n < 0 || n > len(p) {
		return 0, errors.New("cudaflightrecorder: invalid writer count")
	}
	b.written += int64(n)
	if n != len(p) && err == nil {
		err = io.ErrShortWrite
	}
	return n, err
}

func nilEscalationDependency(v any) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.Interface:
		return rv.IsNil()
	}
	return false
}
