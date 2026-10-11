package cudaflightrecorder

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

// Default settings for the flight recorder.
const (
	DefaultWindow   = 10 * time.Second
	DefaultCapacity = 65536
	DefaultEndpoint = "/debug/flightrecorder"
)

// Config configures in-memory flight recording.
type Config struct {
	Window                 time.Duration   // Time window to retain in memory (default 10s)
	Capacity               int             // Maximum number of GPU records in ring buffer (default 65536)
	ClockUncertaintyNs     int64           // Estimated host-to-device clock alignment error (nanoseconds)
	GoTraceEnabled         bool            // Whether to capture Go execution trace in snapshots (go.trace)
	GoTraceWindow          time.Duration   // Time window for Go execution trace (default: matches Window)
	GoTraceMaxBytes        uint64          // Upper bound on in-memory Go execution trace size in bytes (0 = runtime default)
	GoTraceStandard        bool            // If true, use standard trace.Start/Stop adapter instead of trace.FlightRecorder
	GoTracer               GoTraceRecorder // Optional custom Go execution trace flight recorder
	RecordGPUPprof         bool            // Whether to capture GPU pprof profile in snapshots (gpu.pprof)
	RecordPerfetto         bool            // Whether to capture Perfetto timeline in snapshots (timeline.pftrace)
	MaxProviderOutputBytes int64           // Max bytes allowed per provider output (default: 64 MiB)
	MaxSnapshotBytes       int64           // Max retained file payload bytes; manifest/TAR framing have a separate global limit (default: 256 MiB)
	MaxQueuedJobs          int             // Max pending async snapshot jobs (default: 16)
	MaxDiskRetentionBytes  int64           // Max total disk bytes for retained snapshots (default: 1 GiB)
}

// Record represents a single timestamped GPU execution observation in the ring buffer.
type Record struct {
	TimestampNs   int64             // Host or calibrated nanosecond timestamp
	KernelName    string            // Executed kernel symbol name
	CorrelationID uint64            // CUPTI correlation ID
	DurationNs    uint64            // Device execution duration in nanoseconds
	Labels        map[string]string // Contextual attributes (e.g. layer, expert, op)
}

// Recorder retains recent GPU observations in a host-memory ring buffer and
// combines them with configured artifact providers when taking a snapshot.
type Recorder struct {
	cfg        Config
	mu         sync.Mutex
	running    bool
	closed     bool
	cleanupErr error

	// Circular ring buffer
	records []Record
	head    int
	count   int
	dropped int64

	// Multi-modal artifact providers
	goTracer              GoTraceRecorder
	startTime             time.Time
	goTraceProvider       func(context.Context) ([]byte, error)
	gpuPprofProvider      func(context.Context) ([]byte, error)
	perfettoProvider      func(context.Context) ([]byte, error)
	extraFiles            map[string][]byte
	activityHostNs        int64
	activityDeviceNs      int64
	activityUncertaintyNs int64
	activityCalibratedAt  time.Time
	activityCalibrated    bool
}

// New creates a new flight recorder with the supplied configuration.
func New(cfg Config) (*Recorder, error) {
	if cfg.ClockUncertaintyNs < 0 {
		return nil, errors.New("cudaflightrecorder: negative clock uncertainty")
	}
	if cfg.Window <= 0 {
		cfg.Window = DefaultWindow
	}
	if cfg.Capacity <= 0 {
		cfg.Capacity = DefaultCapacity
	}
	if cfg.MaxProviderOutputBytes <= 0 {
		cfg.MaxProviderOutputBytes = 64 << 20
	}
	if cfg.MaxSnapshotBytes <= 0 {
		cfg.MaxSnapshotBytes = 256 << 20
	}
	if cfg.MaxQueuedJobs <= 0 {
		cfg.MaxQueuedJobs = 16
	}
	if cfg.MaxDiskRetentionBytes <= 0 {
		cfg.MaxDiskRetentionBytes = 1 << 30
	}

	r := &Recorder{
		cfg:        cfg,
		records:    make([]Record, cfg.Capacity),
		extraFiles: make(map[string][]byte),
	}

	if cfg.GoTraceEnabled {
		if cfg.GoTracer != nil {
			r.goTracer = cfg.GoTracer
		} else if cfg.GoTraceStandard {
			r.goTracer = NewStandardTraceAdapter(cfg.GoTraceMaxBytes)
		} else {
			window := cfg.GoTraceWindow
			if window <= 0 {
				window = cfg.Window
			}
			r.goTracer = NewAdaptiveTraceRecorder(window, cfg.GoTraceMaxBytes)
		}
		r.goTraceProvider = r.goTracer.Snapshot
	}

	return r, nil
}

// SetGoTraceProvider configures a provider that yields raw go.trace bytes.
func (r *Recorder) SetGoTraceProvider(p func(context.Context) ([]byte, error)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.goTraceProvider = p
}

// SetGPUPprofProvider configures a provider that yields raw gpu.pprof bytes.
func (r *Recorder) SetGPUPprofProvider(p func(context.Context) ([]byte, error)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.gpuPprofProvider = p
}

// SetPerfettoProvider configures a provider that yields raw timeline.pftrace bytes.
func (r *Recorder) SetPerfettoProvider(p func(context.Context) ([]byte, error)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.perfettoProvider = p
}

// AttachFile attaches a named static artifact (such as kernels.ptx or kernels.sass) to snapshots.
func (r *Recorder) AttachFile(name string, data []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.extraFiles[name] = data
}

// DetachFile removes an attached static artifact by name.
func (r *Recorder) DetachFile(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.extraFiles, name)
}

// PreTriggerDiagnostics returns diagnostic details about the currently retained records
// in the ring buffer relative to a trigger timestamp.
func (r *Recorder) PreTriggerDiagnostics(triggerNs int64, window time.Duration) PreTriggerReport {
	r.mu.Lock()
	defer r.mu.Unlock()

	capacity := len(r.records)
	if capacity == 0 || r.count == 0 {
		return PreTriggerReport{
			Available:                 false,
			WindowDuration:            window,
			RecordCount:               0,
			Truncated:                 r.dropped > 0,
			MissingPreTriggerFamilies: allKnownPreTriggerFamilies(),
		}
	}

	windowNs := window.Nanoseconds()
	cutoffNs := triggerNs - windowNs
	if windowNs <= 0 {
		cutoffNs = 0
	}

	count := 0
	earliest := int64(0)
	latest := int64(0)
	observedFamilies := make(map[string]bool)

	startIdx := (r.head - r.count + capacity) % capacity
	for i := 0; i < r.count; i++ {
		idx := (startIdx + i) % capacity
		rec := r.records[idx]
		if rec.TimestampNs <= triggerNs && (cutoffNs == 0 || rec.TimestampNs >= cutoffNs) {
			if count == 0 || rec.TimestampNs < earliest {
				earliest = rec.TimestampNs
			}
			if count == 0 || rec.TimestampNs > latest {
				latest = rec.TimestampNs
			}
			count++

			// Detect family
			if _, ok := rec.Labels["pm.sample_index"]; ok {
				observedFamilies["cupti_pm_sampling"] = true
			}
			if _, ok := rec.Labels["activity.kind"]; ok {
				observedFamilies["cupti_activity"] = true
			}
			if _, ok := rec.Labels["sass.instruction"]; ok {
				observedFamilies["sass_pc_sampling"] = true
			}
			if rec.Labels["record_kind"] == "driver_policy" {
				observedFamilies["driver_policy"] = true
			}
			if rec.KernelName != "" && rec.KernelName != "device.pm_sampling" && rec.Labels["record_kind"] != "driver_policy" {
				observedFamilies["gpu_kernel_launch"] = true
			}
		}
	}

	if r.goTracer != nil {
		observedFamilies["go_execution_trace"] = true
	}

	var missing []string
	for _, fam := range allKnownPreTriggerFamilies() {
		if !observedFamilies[fam] {
			missing = append(missing, fam)
		}
	}

	return PreTriggerReport{
		Available:                 count > 0,
		WindowDuration:            window,
		RecordCount:               count,
		Truncated:                 r.dropped > 0,
		EarliestTimestampNs:       earliest,
		LatestTimestampNs:         latest,
		MissingPreTriggerFamilies: missing,
	}
}

func allKnownPreTriggerFamilies() []string {
	return []string{
		"gpu_kernel_launch",
		"cupti_activity",
		"cupti_pm_sampling",
		"sass_pc_sampling",
		"driver_policy",
		"go_execution_trace",
	}
}

// Start begins continuous in-memory flight recording.
func (r *Recorder) Start() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return errors.New("cudaflightrecorder: recorder is closed")
	}
	if r.running {
		return errors.New("cudaflightrecorder: recorder already running")
	}

	if r.goTracer != nil {
		if err := r.goTracer.Start(); err != nil {
			return fmt.Errorf("cudaflightrecorder: start go trace: %w", err)
		}
	}

	r.startTime = time.Now()
	if hostNs, deviceNs, uncertaintyNs, err := calibrateCUPTITimestamp(); err == nil {
		r.activityHostNs, r.activityDeviceNs = hostNs, deviceNs
		r.activityUncertaintyNs, r.activityCalibrated = uncertaintyNs, true
		r.activityCalibratedAt = time.Now()
	}
	r.running = true
	return nil
}

// clockDriftPPM is the measured host-to-device oscillator drift, in parts per million.
const clockDriftPPM = 11

func uncertaintyAfter(bracketNs int64, elapsed time.Duration) int64 {
	if elapsed < 0 {
		elapsed = 0
	}
	return bracketNs + int64(elapsed)*clockDriftPPM/1e6
}

// clockUncertainty reports the host-to-device clock alignment error to record
// with a snapshot, and where that number came from. A bracket measured at Start
// takes precedence over a configured estimate, widened for the drift accrued
// since it was taken. With neither, the uncertainty is unknown and is reported
// as -1 rather than 0, so that a caller cannot mistake "not measured" for
// "perfectly aligned".
func clockUncertainty(calibrated bool, bracketNs int64, sinceCalibration time.Duration, configuredNs int64) (int64, string) {
	switch {
	case calibrated:
		return uncertaintyAfter(bracketNs, sinceCalibration), "measured"
	case configuredNs > 0:
		return configuredNs, "configured"
	}
	return -1, "unavailable"
}

// Stop halts flight recording permanently. Later starts fail and pushes are
// counted as dropped. Repeated stops are harmless; snapshots remain available.
func (r *Recorder) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.goTracer != nil {
		r.goTracer.Stop()
	}

	r.running = false
	r.closed = true
}

// Close halts flight recording and returns any cleanup or collector shutdown errors.
func (r *Recorder) Close() error {
	r.Stop()
	return r.CleanupErr()
}

// CleanupErr returns any error encountered during collector shutdown or resource cleanup.
func (r *Recorder) CleanupErr() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cleanupErr
}

// SetCleanupErr records a collector or cleanup error explicitly.
func (r *Recorder) SetCleanupErr(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cleanupErr == nil {
		r.cleanupErr = err
	}
}

// Push adds an observation to the in-memory circular buffer.
// It is non-blocking and safe for concurrent execution.
func (r *Recorder) Push(rec Record) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.running {
		r.dropped++
		return
	}

	r.records[r.head] = rec
	r.head = (r.head + 1) % len(r.records)
	if r.count < len(r.records) {
		r.count++
	} else {
		r.dropped++
	}
}

// PushSASSMetric records an instruction-level SASS metric into the flight recorder ring buffer.
func (r *Recorder) PushSASSMetric(timestampNs int64, correlationID uint64, sm cuptipcsampling.SASSMetric) {
	labels := sm.FlightRecordLabels()
	r.Push(Record{
		TimestampNs:   timestampNs,
		KernelName:    sm.KernelName,
		CorrelationID: correlationID,
		DurationNs:    0,
		Labels:        labels,
	})
}

// PushPMSample records a hardware Performance Monitor counter observation into the flight recorder.
func (r *Recorder) PushPMSample(sample cuptipmsampling.Sample) {
	labels := make(map[string]string, len(sample.Values)+2)
	labels["pm.sample_index"] = fmt.Sprintf("%d", sample.Index)
	labels["pm.duration_ns"] = fmt.Sprintf("%d", sample.DurationNs)
	for k, v := range sample.Values {
		labels["pm."+k] = fmt.Sprintf("%.4f", v)
	}
	r.Push(Record{
		TimestampNs: int64(sample.StartTimestamp),
		KernelName:  "device.pm_sampling",
		DurationNs:  sample.DurationNs,
		Labels:      labels,
	})
}

// PushActivityRecord records a CUPTI activity using the host clock as its timestamp.
// Device timestamps are not comparable with the host timestamps used by Go traces.
func (r *Recorder) PushActivityRecord(activity cuptiactivity.Record) {
	var duration uint64
	if activity.Complete {
		duration = activity.End - activity.Start
	}
	r.mu.Lock()
	calibrated := r.activityCalibrated
	hostNs, deviceNs := r.activityHostNs, r.activityDeviceNs
	r.mu.Unlock()
	timestampNs := int64(0)
	labels := map[string]string{"activity.kind": fmt.Sprintf("%d", activity.Kind)}
	if calibrated {
		timestampNs = mapCUPTITimestamp(hostNs, deviceNs, activity.Start)
	} else {
		labels["activity.uncalibrated"] = "true"
	}
	r.Push(Record{
		TimestampNs:   timestampNs,
		KernelName:    activity.Name,
		CorrelationID: uint64(activity.CorrelationID),
		DurationNs:    duration,
		Labels:        labels,
	})
}

func mapCUPTITimestamp(hostNs, deviceNs int64, activityNs uint64) int64 {
	// The unsigned difference wraps before int64 reinterprets it, preserving offsets within 2^63.
	return hostNs + int64(activityNs-uint64(deviceNs))
}

func calibrateCUPTITimestamp() (hostNs, deviceNs, uncertaintyNs int64, err error) {
	if !cupti.HasGetTimestamp() {
		return 0, 0, 0, errors.New("cudaflightrecorder: CUPTI timestamp unavailable")
	}
	return calibrateWith(cupti.GetTimestamp)
}

// calibrateWith brackets one device clock read between two host clock reads
// and returns the midpoint as the correspondence between the clocks, with half
// the bracket as the uncertainty.
//
// The first read is taken and discarded. CUPTI initializes lazily, so
// bracketing its first call times the library's cold start -- milliseconds --
// and reports that as the uncertainty, when a warm read costs microseconds.
func calibrateWith(read func() (uint64, error)) (hostNs, deviceNs, uncertaintyNs int64, err error) {
	if _, err := read(); err != nil {
		return 0, 0, 0, err
	}
	before := time.Now().UnixNano()
	device, err := read()
	if err != nil {
		return 0, 0, 0, err
	}
	after := time.Now().UnixNano()
	if after < before {
		return 0, 0, 0, errors.New("cudaflightrecorder: host clock went backwards")
	}
	return before + (after-before)/2, int64(device), (after - before) / 2, nil
}

// Snapshot writes a tar bundle to w containing recent flight records, a manifest,
// and artifacts supplied by configured providers. The reason labels the snapshot
// for later analysis.
// Record extraction reads host memory. Snapshot also calls configured providers
// and writes to w; their behavior determines whether those operations block or
// access GPU hardware.
func (r *Recorder) Snapshot(ctx context.Context, w io.Writer, reason string) (*Manifest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	r.mu.Lock()
	// Drain current valid records within the configured time window
	capacity := len(r.records)
	records := make([]Record, 0, r.count)
	now := time.Now()
	nowNs := now.UnixNano()
	windowNs := r.cfg.Window.Nanoseconds()
	cutoffNs := nowNs - windowNs

	outsideWindow := 0
	startIdx := (r.head - r.count + capacity) % capacity
	for i := 0; i < r.count; i++ {
		idx := (startIdx + i) % capacity
		rec := r.records[idx]
		if rec.TimestampNs >= cutoffNs {
			records = append(records, rec)
		} else {
			outsideWindow++
		}
	}
	dropped := r.dropped
	startTime := r.startTime
	gt := r.goTracer
	goProvider := r.goTraceProvider
	gpuProvider := r.gpuPprofProvider
	perfettoProvider := r.perfettoProvider
	cfg := r.cfg
	calibrated := r.activityCalibrated
	calibratedBracketNs := r.activityUncertaintyNs
	calibratedAt := r.activityCalibratedAt
	extraFiles := make(map[string][]byte, len(r.extraFiles))
	for k, v := range r.extraFiles {
		extraFiles[k] = v
	}
	r.mu.Unlock()

	bundle := &Bundle{
		Files: make(map[string][]byte),
	}

	// Format GPU records as a human-readable text dump inside the bundle
	var recBuf []byte
	for _, rec := range records {
		recBuf = append(recBuf, fmt.Sprintf("%d %s dur=%dns corr=%d\n",
			rec.TimestampNs, rec.KernelName, rec.DurationNs, rec.CorrelationID)...)
	}
	bundle.Files["gpu_records.txt"] = recBuf
	// Preserve labels and numeric fields for consumers of policy and profiling
	// evidence. The text file above remains available to existing readers.
	recordJSON, err := json.Marshal(records)
	if err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: marshal records: %w", err)
	}
	bundle.Files["gpu_records.json"] = recordJSON

	maxProviderBytes := cfg.MaxProviderOutputBytes
	if maxProviderBytes <= 0 {
		maxProviderBytes = 64 << 20
	}

	providerStatus := make(map[string]string)
	var lossReasons []string

	// 1. go.trace
	var goTraceBytes int64
	var goHasData bool
	if goProvider != nil {
		data, err := goProvider(ctx)
		if err != nil {
			providerStatus["go.trace"] = fmt.Sprintf("unavailable: %v", err)
			lossReasons = append(lossReasons, fmt.Sprintf("go.trace provider unavailable: %v", err))
		} else if len(data) == 0 {
			providerStatus["go.trace"] = "empty"
		} else if int64(len(data)) > maxProviderBytes {
			providerStatus["go.trace"] = fmt.Sprintf("rejected: output %d bytes exceeds budget %d bytes", len(data), maxProviderBytes)
			lossReasons = append(lossReasons, "go.trace exceeded output budget")
		} else if err := validateGoTrace(data); err != nil {
			providerStatus["go.trace"] = fmt.Sprintf("rejected: %v", err)
			lossReasons = append(lossReasons, fmt.Sprintf("go.trace invalid: %v", err))
		} else {
			bundle.Files["go.trace"] = data
			goTraceBytes = int64(len(data))
			providerStatus["go.trace"] = "unvalidated: Go execution trace decoder unavailable; raw artifact retained, timing unavailable"
			lossReasons = append(lossReasons, "go.trace format and timing unvalidated")
		}
	} else if cfg.GoTraceEnabled {
		providerStatus["go.trace"] = "unavailable: provider not configured"
		lossReasons = append(lossReasons, "go.trace provider not configured")
	}

	// 2. gpu.ppprof
	if gpuProvider != nil {
		data, err := gpuProvider(ctx)
		if err != nil {
			providerStatus["gpu.pprof"] = fmt.Sprintf("unavailable: %v", err)
			lossReasons = append(lossReasons, fmt.Sprintf("gpu.pprof provider unavailable: %v", err))
		} else if len(data) == 0 {
			providerStatus["gpu.pprof"] = "empty"
		} else if int64(len(data)) > maxProviderBytes {
			providerStatus["gpu.pprof"] = fmt.Sprintf("rejected: output %d bytes exceeds budget %d bytes", len(data), maxProviderBytes)
			lossReasons = append(lossReasons, "gpu.pprof exceeded output budget")
		} else if err := validateGPUPprof(data, maxProviderBytes); err != nil {
			providerStatus["gpu.pprof"] = fmt.Sprintf("rejected: %v", err)
			lossReasons = append(lossReasons, fmt.Sprintf("gpu.pprof invalid: %v", err))
		} else {
			bundle.Files["gpu.pprof"] = data
			providerStatus["gpu.pprof"] = fmt.Sprintf("valid (%d bytes)", len(data))
		}
	} else if cfg.RecordGPUPprof {
		providerStatus["gpu.pprof"] = "unavailable: provider not configured"
		lossReasons = append(lossReasons, "gpu.pprof provider not configured")
	}

	// 3. timeline.pftrace
	if perfettoProvider != nil {
		data, err := perfettoProvider(ctx)
		if err != nil {
			providerStatus["timeline.pftrace"] = fmt.Sprintf("unavailable: %v", err)
			lossReasons = append(lossReasons, fmt.Sprintf("timeline.pftrace provider unavailable: %v", err))
		} else if len(data) == 0 {
			providerStatus["timeline.pftrace"] = "empty"
		} else if int64(len(data)) > maxProviderBytes {
			providerStatus["timeline.pftrace"] = fmt.Sprintf("rejected: output %d bytes exceeds budget %d bytes", len(data), maxProviderBytes)
			lossReasons = append(lossReasons, "timeline.pftrace exceeded output budget")
		} else if err := validatePerfetto(data); err != nil {
			providerStatus["timeline.pftrace"] = fmt.Sprintf("rejected: %v", err)
			lossReasons = append(lossReasons, fmt.Sprintf("timeline.pftrace invalid: %v", err))
		} else {
			bundle.Files["timeline.pftrace"] = data
			providerStatus["timeline.pftrace"] = "unvalidated: Perfetto decoder unavailable; raw artifact retained"
			lossReasons = append(lossReasons, "timeline.pftrace format unvalidated")
		}
	} else if cfg.RecordPerfetto {
		providerStatus["timeline.pftrace"] = "unavailable: provider not configured"
		lossReasons = append(lossReasons, "timeline.pftrace provider not configured")
	}

	// 4. Attach extra user files
	for name, data := range extraFiles {
		if _, exists := bundle.Files[name]; exists {
			return nil, fmt.Errorf("cudaflightrecorder: attachment replaces snapshot artifact %q", name)
		}
		bundle.Files[name] = data
	}

	// Check total snapshot size budget
	if err := checkSnapshotPayload(bundle.Files, cfg.MaxSnapshotBytes); err != nil {
		return nil, err
	}

	// 5. Align host clock and device timestamps to compute spans
	var goStartNs int64
	if !startTime.IsZero() {
		goStartNs = startTime.UnixNano()
	} else if timer, ok := gt.(interface{ StartTime() time.Time }); ok && !timer.StartTime().IsZero() {
		goStartNs = timer.StartTime().UnixNano()
	}
	clockUncertaintyNs, clockUncertaintySource := clockUncertainty(calibrated, calibratedBracketNs, now.Sub(calibratedAt), cfg.ClockUncertaintyNs)
	goSpanNs, gpuSpanNs, overlapSpanNs, overlapReliable := AlignSpans(
		nowNs,
		windowNs,
		goStartNs,
		goHasData,
		records,
		clockUncertaintyNs,
	)

	manifest := Manifest{
		Timestamp:              now,
		Reason:                 reason,
		WindowRequestedNs:      windowNs,
		GoSpanNs:               goSpanNs,
		GPUSpanNs:              gpuSpanNs,
		OverlapSpanNs:          overlapSpanNs,
		GPURecords:             len(records),
		GPUDropped:             dropped,
		GPUOutsideWindow:       outsideWindow,
		GoTraceBytes:           goTraceBytes,
		ClockUncertaintyNs:     clockUncertaintyNs,
		ClockUncertaintySource: clockUncertaintySource,
		OverlapReliable:        overlapReliable,
		ProviderStatus:         providerStatus,
		LossReasons:            lossReasons,
	}
	bundle.Manifest = manifest

	if err := WriteBundle(w, bundle); err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: write snapshot bundle: %w", err)
	}

	return &manifest, nil
}

func isPlaceholderPayload(data []byte) bool {
	if bytes.Equal(data, []byte("go_trace_v2\x00")) ||
		bytes.Equal(data, []byte("\x1f\x8b\x08gpu_pprof\x00")) ||
		bytes.Equal(data, []byte("timeline_pftrace\x00")) ||
		bytes.Equal(data, []byte("fake_go_trace_v2")) {
		return true
	}
	return false
}

// validateGoTrace rejects placeholders and bad headers only. Successful header
// admission does not validate trace records or establish timing coverage.
func validateGoTrace(data []byte) error {
	if isPlaceholderPayload(data) {
		return fmt.Errorf("%w: go.trace contains placeholder string", ErrPlaceholderRejected)
	}
	if len(data) < 4 {
		return fmt.Errorf("%w: go.trace too short (%d bytes)", ErrArtifactInvalid, len(data))
	}
	if !bytes.HasPrefix(data, []byte("go 1.")) &&
		!bytes.HasPrefix(data, []byte("\x16go 1.")) &&
		!bytes.HasPrefix(data, []byte("go_trace")) {
		return fmt.Errorf("%w: invalid go.trace magic header", ErrArtifactInvalid)
	}
	return nil
}

func validateGPUPprof(data []byte, limit int64) error {
	if limit <= 0 || limit > maxBundleBytes {
		limit = maxBundleBytes
	}
	if isPlaceholderPayload(data) {
		return fmt.Errorf("%w: gpu.pprof contains placeholder string", ErrPlaceholderRejected)
	}
	if len(data) < 4 {
		return fmt.Errorf("%w: gpu.pprof too short (%d bytes)", ErrArtifactInvalid, len(data))
	}
	if !bytes.HasPrefix(data, []byte{0x1f, 0x8b}) {
		return fmt.Errorf("%w: gpu.pprof missing gzip magic header", ErrArtifactInvalid)
	}
	// Bound decompression before handing bytes to the existing profile decoder.
	z, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("%w: gpu.pprof gzip: %v", ErrArtifactInvalid, err)
	}
	defer z.Close()
	decoded, err := io.ReadAll(io.LimitReader(z, limit+1))
	if err != nil {
		return fmt.Errorf("%w: gpu.pprof gzip: %v", ErrArtifactInvalid, err)
	}
	if int64(len(decoded)) > limit {
		return fmt.Errorf("%w: gpu.pprof decompressed output exceeds %d bytes", ErrProviderBudgetExceeded, limit)
	}
	p, err := profile.ParseUncompressed(decoded)
	if err == nil {
		err = p.CheckValid()
	}
	if err != nil {
		return fmt.Errorf("%w: gpu.pprof profile: %v", ErrArtifactInvalid, err)
	}
	return nil
}

// validatePerfetto rejects placeholders and short input only. Without a full
// decoder, retained bytes remain explicitly unvalidated in ProviderStatus.
func validatePerfetto(data []byte) error {
	if isPlaceholderPayload(data) {
		return fmt.Errorf("%w: timeline.pftrace contains placeholder string", ErrPlaceholderRejected)
	}
	if len(data) < 4 {
		return fmt.Errorf("%w: timeline.pftrace too short (%d bytes)", ErrArtifactInvalid, len(data))
	}
	return nil
}

// checkSnapshotPayload applies the configured retained-file budget. Manifest
// metadata and TAR framing are subject to WriteBundle's separate global limit.
func checkSnapshotPayload(files map[string][]byte, limit int64) error {
	if limit <= 0 {
		limit = 256 << 20
	}
	var total int64
	for _, data := range files {
		if int64(len(data)) > limit-total {
			return fmt.Errorf("%w: retained file payload exceeds budget %d bytes", ErrSnapshotBudgetExceeded, limit)
		}
		total += int64(len(data))
	}
	return nil
}

// FaultSnapshot writes a zero-device crash dump TAR archive to w using exclusively
// host DRAM circular buffer records and host-side telemetry, without contacting
// the GPU or driver. It preserves fault facts and candidate identity without
// admitting broad rewrite bans from an unreproduced fault.
func (r *Recorder) FaultSnapshot(ctx context.Context, w io.Writer, fault FaultInfo) (*Manifest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	r.mu.Lock()
	capacity := len(r.records)
	records := make([]Record, 0, r.count)
	now := time.Now()
	nowNs := now.UnixNano()
	windowNs := r.cfg.Window.Nanoseconds()
	cutoffNs := nowNs - windowNs

	outsideWindow := 0
	startIdx := (r.head - r.count + capacity) % capacity
	for i := 0; i < r.count; i++ {
		idx := (startIdx + i) % capacity
		rec := r.records[idx]
		if rec.TimestampNs >= cutoffNs {
			records = append(records, rec)
		} else {
			outsideWindow++
		}
	}
	dropped := r.dropped
	maxSnapshotBytes := r.cfg.MaxSnapshotBytes
	extraFiles := make(map[string][]byte, len(r.extraFiles))
	for k, v := range r.extraFiles {
		extraFiles[k] = v
	}
	r.mu.Unlock()

	bundle := &Bundle{
		Files: make(map[string][]byte),
	}

	// Format GPU records as a human-readable text dump inside the bundle
	var recBuf []byte
	for _, rec := range records {
		recBuf = append(recBuf, fmt.Sprintf("%d %s dur=%dns corr=%d\n",
			rec.TimestampNs, rec.KernelName, rec.DurationNs, rec.CorrelationID)...)
	}
	bundle.Files["gpu_records.txt"] = recBuf
	recordJSON, err := json.Marshal(records)
	if err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: marshal records: %w", err)
	}
	bundle.Files["gpu_records.json"] = recordJSON

	// 1. Serialize fault.json
	faultBytes, err := json.MarshalIndent(fault, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: marshal fault info: %w", err)
	}
	bundle.Files["fault.json"] = faultBytes

	// 2. Preserve a candidate-only constraint. Failed rule names remain raw facts
	// in fault.json; reproduction and broader policy admission are not provided.
	digest := fault.ModuleDigest
	decodedDigest, digestErr := hex.DecodeString(digest)
	identityStatus := "candidate-only quarantine; no reproduced rule policy"
	if digestErr != nil || len(decodedDigest) != 32 {
		digest = ""
		identityStatus = "quarantine withheld: exact SHA256 candidate identity unavailable"
	}
	negConstraint := NegativeRewriteConstraint{
		CandidateDigest: digest,
		OffendingKernel: fault.OffendingKernel,
		Reason:          fmt.Sprintf("%s; gpu_fault: %s (%d): %s", identityStatus, fault.ErrorName, fault.ErrorCode, fault.ErrorMessage),
		Timestamp:       now,
	}
	negBytes, err := json.MarshalIndent(negConstraint, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: marshal negative constraints: %w", err)
	}
	bundle.Files["negative_rules.json"] = negBytes

	// 3. Attach static extra files
	for name, data := range extraFiles {
		if _, exists := bundle.Files[name]; exists {
			return nil, fmt.Errorf("cudaflightrecorder: attachment replaces fault artifact %q", name)
		}
		bundle.Files[name] = data
	}
	if err := checkSnapshotPayload(bundle.Files, maxSnapshotBytes); err != nil {
		return nil, err
	}

	reason := "fault:" + fault.ErrorName
	if fault.ErrorName == "" {
		reason = fmt.Sprintf("fault:code_%d", fault.ErrorCode)
	}

	manifest := Manifest{
		Timestamp:              now,
		Reason:                 reason,
		WindowRequestedNs:      windowNs,
		GPURecords:             len(records),
		GPUDropped:             dropped,
		GPUOutsideWindow:       outsideWindow,
		ClockUncertaintyNs:     -1,
		ClockUncertaintySource: "host_only",
		OverlapReliable:        false,
		LossReasons:            []string{fmt.Sprintf("fault_occurred: %s", fault.ErrorMessage)},
	}
	bundle.Manifest = manifest

	if err := WriteBundle(w, bundle); err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: write fault snapshot bundle: %w", err)
	}

	return &manifest, nil
}

// Register registers the flight recorder HTTP handler at DefaultEndpoint ("/debug/flightrecorder") on mux.
// If mux is nil, it registers on http.DefaultServeMux.
func (r *Recorder) Register(mux *http.ServeMux) {
	if mux == nil {
		http.Handle(DefaultEndpoint, r.Handler())
		return
	}
	mux.Handle(DefaultEndpoint, r.Handler())
}

// Handler returns an HTTP handler that triggers a snapshot and streams the resulting
// TAR bundle to the client upon GET/POST request.
func (r *Recorder) Handler() http.Handler {
	var snapMu sync.Mutex

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if !snapMu.TryLock() {
			http.Error(w, "snapshot currently in progress; retry later", http.StatusTooManyRequests)
			return
		}
		defer snapMu.Unlock()

		reason := req.URL.Query().Get("reason")
		if reason == "" {
			reason = "http_request"
		}
		if len(reason) > 64 {
			reason = reason[:64]
		}

		filename := fmt.Sprintf("flightrecord-%s.tar", time.Now().UTC().Format("20060102-150405"))
		w.Header().Set("Content-Type", "application/x-tar")
		w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, filename))

		ctx, cancel := context.WithTimeout(req.Context(), 10*time.Second)
		defer cancel()

		if _, err := r.Snapshot(ctx, w, reason); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
}
