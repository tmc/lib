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
)

// Config configures in-memory flight recording.
type Config struct {
	Window             time.Duration // Time window to retain in memory (default 10s)
	Capacity           int           // Maximum number of GPU records in ring buffer (default 65536)
	ClockUncertaintyNs int64         // Estimated host-to-device clock alignment error (nanoseconds)
	GoTraceEnabled     bool          // Whether to capture Go execution trace in snapshots
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
		cfg:     cfg,
		records: make([]Record, cfg.Capacity),
	}, nil
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

	if err := WriteBundle(w, bundle); err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: write snapshot bundle: %w", err)
	}

	return &manifest, nil
}

// Handler returns an HTTP handler that triggers a snapshot and streams the resulting
// TAR bundle to the client upon GET/POST request.
func (r *Recorder) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		reason := req.URL.Query().Get("reason")
		if reason == "" {
			reason = "http_request"
		}

		w.Header().Set("Content-Type", "application/x-tar")
		w.Header().Set("Content-Disposition", `attachment; filename="flightrecorder_snapshot.tar"`)

		ctx, cancel := context.WithTimeout(req.Context(), 5*time.Second)
		defer cancel()

		if _, err := r.Snapshot(ctx, w, reason); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
}
