package cudaprof

import (
	"io"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

// SampleRecord represents a recorded GPU execution event stored in the flight recorder buffer.
type SampleRecord struct {
	TimestampNs uint64
	KernelName  string
	CuFile      string
	CuLine      int64
	PtxFile     string
	PtxLine     int64
	SassPC      uint64
	Values      SampleValues
	Labels      map[string]string
	Frames      []Frame
}

// FlightRecorder manages continuous in-memory ring buffer recording of GPU activities and PC samples.
type FlightRecorder struct {
	mu           sync.Mutex
	enabled      bool
	ringCap      int
	ringHead     int
	buffer       []SampleRecord
	windowDur    time.Duration
	samplePeriod uint32
	binaryPath   string
	mappingFile  string
}

// FlightRecorderOption configures a FlightRecorder.
type FlightRecorderOption func(*FlightRecorder)

// WithWindowSize sets the maximum number of sample records held in the in-memory ring buffer.
func WithWindowSize(size int) FlightRecorderOption {
	return func(fr *FlightRecorder) {
		if size > 0 {
			fr.ringCap = size
		}
	}
}

// WithWindowDuration sets the rolling time window retained by the flight recorder.
func WithWindowDuration(d time.Duration) FlightRecorderOption {
	return func(fr *FlightRecorder) {
		fr.windowDur = d
	}
}

// WithPCSamplingPeriod sets the hardware SM PC sampling frequency.
func WithPCSamplingPeriod(period uint32) FlightRecorderOption {
	return func(fr *FlightRecorder) {
		fr.samplePeriod = period
	}
}

// WithFlightRecorderBinaryPath sets the target path for synthesized ELF disassembly sidecars.
func WithFlightRecorderBinaryPath(path string) FlightRecorderOption {
	return func(fr *FlightRecorder) {
		fr.binaryPath = path
	}
}

// NewFlightRecorder initializes a new in-memory GPU flight recorder.
func NewFlightRecorder(opts ...FlightRecorderOption) *FlightRecorder {
	fr := &FlightRecorder{
		ringCap:      8192,
		windowDur:    30 * time.Second,
		samplePeriod: 1024,
	}
	for _, opt := range opts {
		opt(fr)
	}
	fr.buffer = make([]SampleRecord, 0, fr.ringCap)
	return fr
}

// Start begins continuous background recording into the in-memory ring buffer.
func (fr *FlightRecorder) Start() error {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	fr.enabled = true
	return nil
}

// Stop halts background recording.
func (fr *FlightRecorder) Stop() error {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	fr.enabled = false
	return nil
}

// Enabled reports whether the flight recorder is actively capturing records.
func (fr *FlightRecorder) Enabled() bool {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	return fr.enabled
}

// Record appends a sample record to the in-memory ring buffer.
func (fr *FlightRecorder) Record(rec SampleRecord) {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	if !fr.enabled {
		return
	}
	if rec.TimestampNs == 0 {
		rec.TimestampNs = uint64(time.Now().UnixNano())
	}
	if len(fr.buffer) < fr.ringCap {
		fr.buffer = append(fr.buffer, rec)
	} else {
		fr.buffer[fr.ringHead] = rec
		fr.ringHead = (fr.ringHead + 1) % fr.ringCap
	}
}

// WriteTo snapshots the active flight recording buffer as a standard gzipped pprof profile.
func (fr *FlightRecorder) WriteTo(w io.Writer) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	b := NewBuilder()
	if fr.binaryPath != "" {
		b.SetBinaryPath(fr.binaryPath)
	}
	if fr.mappingFile != "" {
		b.SetMappingFile(fr.mappingFile, "")
	}

	n := len(fr.buffer)
	for i := 0; i < n; i++ {
		idx := i
		if len(fr.buffer) == fr.ringCap {
			idx = (fr.ringHead + i) % fr.ringCap
		}
		rec := fr.buffer[idx]
		if len(rec.Frames) > 0 {
			b.AddStackSample(rec.Values, rec.Frames, rec.Labels)
		} else {
			b.AddTracebackSample(rec.Values, rec.KernelName, rec.CuFile, rec.CuLine, rec.PtxFile, rec.PtxLine, rec.SassPC, rec.Labels)
		}
	}

	return b.WriteGzip(w)
}

// WritePerfettoTo snapshots the flight recording buffer into a Perfetto .pftrace timeline.
func (fr *FlightRecorder) WritePerfettoTo(w io.Writer) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	pw := perfetto.NewWriter(w)
	_ = pw.WriteTrackDescriptor(1000, "GPU Flight Recorder", 0)
	_ = pw.WriteCounterDescriptor(1001, "VRAM Allocation", "bytes", 1000)
	_ = pw.WriteCounterDescriptor(1002, "DRAM Read Bandwidth", "bytes", 1000)
	_ = pw.WriteCounterDescriptor(1003, "DRAM Write Bandwidth", "bytes", 1000)
	_ = pw.WriteCounterDescriptor(1004, "Tensor Core Warps", "count", 1000)

	n := len(fr.buffer)
	for i := 0; i < n; i++ {
		idx := i
		if len(fr.buffer) == fr.ringCap {
			idx = (fr.ringHead + i) % fr.ringCap
		}
		rec := fr.buffer[idx]

		var annotations []perfetto.DebugAnnotation
		if rec.CuFile != "" {
			annotations = append(annotations, perfetto.NewStringAnnotation("source.file", rec.CuFile))
		}
		if rec.CuLine > 0 {
			annotations = append(annotations, perfetto.NewIntAnnotation("source.line", rec.CuLine))
		}
		if rec.PtxFile != "" {
			annotations = append(annotations, perfetto.NewStringAnnotation("ptx.file", rec.PtxFile))
		}
		for k, v := range rec.Labels {
			annotations = append(annotations, perfetto.NewStringAnnotation(k, v))
		}

		var loc *perfetto.SourceLocation
		if rec.CuFile != "" || rec.CuLine > 0 {
			loc = &perfetto.SourceLocation{
				FileName:     rec.CuFile,
				FunctionName: rec.KernelName,
				LineNumber:   uint32(rec.CuLine),
			}
		}

		ts := rec.TimestampNs
		dur := uint64(rec.Values.Samples * 10)
		if dur == 0 {
			dur = 1000
		}

		_ = pw.WriteSliceBeginWithAnnotations(1000, ts, rec.KernelName, "gpu_kernel", annotations, loc)
		_ = pw.WriteSliceEnd(1000, ts+dur)

		// Emit continuous hardware counter updates
		if rec.Values.VRAMAllocBytes > 0 {
			_ = pw.WriteCounterValue(1001, ts, rec.Values.VRAMAllocBytes)
		}
		if rec.Values.DRAMReadBytes > 0 {
			_ = pw.WriteCounterValue(1002, ts, rec.Values.DRAMReadBytes)
		}
		if rec.Values.DRAMWriteBytes > 0 {
			_ = pw.WriteCounterValue(1003, ts, rec.Values.DRAMWriteBytes)
		}
		if rec.Values.TensorWarps > 0 {
			_ = pw.WriteCounterValue(1004, ts, rec.Values.TensorWarps)
		}
	}

	return nil
}

// WriteBoth snapshots the flight recording buffer to both a pprof writer and a Perfetto writer.
func (fr *FlightRecorder) WriteBoth(pprofWriter io.Writer, perfettoWriter io.Writer) error {
	if pprofWriter != nil {
		if err := fr.WriteTo(pprofWriter); err != nil {
			return err
		}
	}
	if perfettoWriter != nil {
		if err := fr.WritePerfettoTo(perfettoWriter); err != nil {
			return err
		}
	}
	return nil
}

