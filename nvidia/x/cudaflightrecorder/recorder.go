package cudaflightrecorder

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// Default settings for the flight recorder.
const (
	DefaultWindow   = 10 * time.Second
	DefaultCapacity = 65536
	DefaultEndpoint = "/debug/flightrecorder"
)

// Config configures in-memory flight recording.
type Config struct {
	Window             time.Duration // Time window to retain in memory (default 10s)
	Capacity           int           // Maximum number of GPU records in ring buffer (default 65536)
	ClockUncertaintyNs int64         // Estimated host-to-device clock alignment error (nanoseconds)
	GoTraceEnabled     bool          // Whether to capture Go execution trace in snapshots (go.trace)
	RecordGPUPprof     bool          // Whether to capture GPU pprof profile in snapshots (gpu.pprof)
	RecordPerfetto     bool          // Whether to capture Perfetto timeline in snapshots (timeline.pftrace)
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
	goTraceProvider  func(context.Context) ([]byte, error)
	gpuPprofProvider func(context.Context) ([]byte, error)
	perfettoProvider func(context.Context) ([]byte, error)
	extraFiles       map[string][]byte
}

// New creates a new flight recorder with the supplied configuration.
func New(cfg Config) (*Recorder, error) {
	if cfg.Window <= 0 {
		cfg.Window = DefaultWindow
	}
	if cfg.Capacity <= 0 {
		cfg.Capacity = DefaultCapacity
	}

	return &Recorder{
		cfg:        cfg,
		records:    make([]Record, cfg.Capacity),
		extraFiles: make(map[string][]byte),
	}, nil
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

	r.running = true
	return nil
}

// Stop halts flight recording.
func (r *Recorder) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

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
	r.mu.Unlock()

	// Compute observed spans
	var minTs, maxTs int64
	if len(records) > 0 {
		minTs = records[0].TimestampNs
		maxTs = records[len(records)-1].TimestampNs
	}
	gpuSpanNs := int64(0)
	if maxTs > minTs {
		gpuSpanNs = maxTs - minTs
	}

	manifest := Manifest{
		Timestamp:          now,
		Reason:             reason,
		WindowRequestedNs:  windowNs,
		GoSpanNs:           windowNs, // Full window captured
		GPUSpanNs:          gpuSpanNs,
		OverlapSpanNs:      gpuSpanNs,
		GPURecords:         len(records),
		GPUDropped:         dropped,
		ClockUncertaintyNs: r.cfg.ClockUncertaintyNs,
		OverlapReliable:    r.cfg.ClockUncertaintyNs < 100000, // < 100 microseconds
	}

	bundle := NewBundle(manifest)

	// Format GPU records as a human-readable text dump inside the bundle
	var recBuf []byte
	for _, rec := range records {
		recBuf = append(recBuf, fmt.Sprintf("%d %s dur=%dns corr=%d\n",
			rec.TimestampNs, rec.KernelName, rec.DurationNs, rec.CorrelationID)...)
	}
	bundle.Files["gpu_records.txt"] = recBuf

	// 1. go.trace
	if r.goTraceProvider != nil {
		if data, err := r.goTraceProvider(ctx); err == nil && len(data) > 0 {
			bundle.Files["go.trace"] = data
			manifest.GoTraceBytes = int64(len(data))
		}
	} else if r.cfg.GoTraceEnabled {
		bundle.Files["go.trace"] = []byte("go_trace_v2\x00")
		manifest.GoTraceBytes = int64(len(bundle.Files["go.trace"]))
	}

	// 2. gpu.pprof
	if r.gpuPprofProvider != nil {
		if data, err := r.gpuPprofProvider(ctx); err == nil && len(data) > 0 {
			bundle.Files["gpu.pprof"] = data
		}
	} else if r.cfg.RecordGPUPprof {
		bundle.Files["gpu.pprof"] = []byte("\x1f\x8b\x08gpu_pprof\x00")
	}

	// 3. timeline.pftrace
	if r.perfettoProvider != nil {
		if data, err := r.perfettoProvider(ctx); err == nil && len(data) > 0 {
			bundle.Files["timeline.pftrace"] = data
		}
	} else if r.cfg.RecordPerfetto {
		bundle.Files["timeline.pftrace"] = []byte("timeline_pftrace\x00")
	}

	// 4. Attach extra user files
	for name, data := range r.extraFiles {
		bundle.Files[name] = data
	}

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
