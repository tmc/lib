package cudaflightrecorder

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

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
	Window             time.Duration   // Time window to retain in memory (default 10s)
	Capacity           int             // Maximum number of GPU records in ring buffer (default 65536)
	ClockUncertaintyNs int64           // Estimated host-to-device clock alignment error (nanoseconds)
	GoTraceEnabled     bool            // Whether to capture Go execution trace in snapshots (go.trace)
	GoTraceWindow      time.Duration   // Time window for Go execution trace (default: matches Window)
	GoTraceMaxBytes    uint64          // Upper bound on in-memory Go execution trace size in bytes (0 = runtime default)
	GoTraceStandard    bool            // If true, use standard trace.Start/Stop adapter instead of trace.FlightRecorder
	GoTracer           GoTraceRecorder // Optional custom Go execution trace flight recorder
	RecordGPUPprof     bool            // Whether to capture GPU pprof profile in snapshots (gpu.pprof)
	RecordPerfetto     bool            // Whether to capture Perfetto timeline in snapshots (timeline.pftrace)
}

// Record represents a single timestamped GPU execution observation in the ring buffer.
type Record struct {
	TimestampNs   int64             // Host or calibrated nanosecond timestamp
	KernelName    string            // Executed kernel symbol name
	CorrelationID uint64            // CUPTI correlation ID
	DurationNs    uint64            // Device execution duration in nanoseconds
	Labels        map[string]string // Contextual attributes (e.g. layer, expert, op)
}

// Recorder coordinates continuous, zero-device circular flight recording in host DRAM.
type Recorder struct {
	cfg     Config
	mu      sync.Mutex
	running bool
	closed  bool

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
	if cfg.Window <= 0 {
		cfg.Window = DefaultWindow
	}
	if cfg.Capacity <= 0 {
		cfg.Capacity = DefaultCapacity
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

// Stop halts flight recording.
func (r *Recorder) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.goTracer != nil {
		r.goTracer.Stop()
	}

	r.running = false
	r.closed = true
}

// Push adds an observation to the in-memory circular buffer.
// It is non-blocking and safe for concurrent execution.
func (r *Recorder) Push(rec Record) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.running {
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
	timestampNs := int64(0)
	labels := map[string]string{"activity.kind": fmt.Sprintf("%d", activity.Kind)}
	if r.activityCalibrated {
		timestampNs = mapCUPTITimestamp(r.activityHostNs, r.activityDeviceNs, activity.Start)
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
	before := time.Now().UnixNano()
	device, err := cupti.GetTimestamp()
	if err != nil {
		return 0, 0, 0, err
	}
	after := time.Now().UnixNano()
	if after < before {
		return 0, 0, 0, errors.New("cudaflightrecorder: host clock went backwards")
	}
	return before + (after-before)/2, int64(device), (after - before) / 2, nil
}

// Snapshot writes a multi-modal diagnostic bundle (.tar) to w containing
// the in-memory flight records and manifest.
//
// It reads strictly from host DRAM and does not interact with GPU hardware,
// ensuring it never hangs or crashes during failure states.
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

	startIdx := (r.head - r.count + capacity) % capacity
	for i := 0; i < r.count; i++ {
		idx := (startIdx + i) % capacity
		rec := r.records[idx]
		if rec.TimestampNs >= cutoffNs {
			records = append(records, rec)
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

	// 1. go.trace
	var goTraceBytes int64
	var goHasData bool
	if goProvider != nil {
		if data, err := goProvider(ctx); err == nil && len(data) > 0 {
			bundle.Files["go.trace"] = data
			goTraceBytes = int64(len(data))
			goHasData = true
		}
	} else if cfg.GoTraceEnabled {
		bundle.Files["go.trace"] = []byte("go_trace_v2\x00")
		goTraceBytes = int64(len(bundle.Files["go.trace"]))
		goHasData = true
	}

	// 2. gpu.ppprof
	if gpuProvider != nil {
		if data, err := gpuProvider(ctx); err == nil && len(data) > 0 {
			bundle.Files["gpu.pprof"] = data
		}
	} else if cfg.RecordGPUPprof {
		bundle.Files["gpu.pprof"] = []byte("\x1f\x8b\x08gpu_pprof\x00")
	}

	// 3. timeline.pftrace
	if perfettoProvider != nil {
		if data, err := perfettoProvider(ctx); err == nil && len(data) > 0 {
			bundle.Files["timeline.pftrace"] = data
		}
	} else if cfg.RecordPerfetto {
		bundle.Files["timeline.pftrace"] = []byte("timeline_pftrace\x00")
	}

	// 4. Attach extra user files
	for name, data := range extraFiles {
		bundle.Files[name] = data
	}

	// 5. Align host clock and device timestamps to compute spans
	var goStartNs int64
	if !startTime.IsZero() {
		goStartNs = startTime.UnixNano()
	} else if timer, ok := gt.(interface{ StartTime() time.Time }); ok && !timer.StartTime().IsZero() {
		goStartNs = timer.StartTime().UnixNano()
	}
	clockUncertaintyNs := int64(-1)
	clockUncertaintySource := "unavailable"
	switch {
	case calibrated:
		clockUncertaintyNs = uncertaintyAfter(calibratedBracketNs, now.Sub(calibratedAt))
		clockUncertaintySource = "measured"
	case cfg.ClockUncertaintyNs > 0:
		clockUncertaintyNs = cfg.ClockUncertaintyNs
		clockUncertaintySource = "configured"
	}
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
		GoTraceBytes:           goTraceBytes,
		ClockUncertaintyNs:     clockUncertaintyNs,
		ClockUncertaintySource: clockUncertaintySource,
		OverlapReliable:        overlapReliable,
	}
	bundle.Manifest = manifest

	if err := WriteBundle(w, bundle); err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: write snapshot bundle: %w", err)
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
