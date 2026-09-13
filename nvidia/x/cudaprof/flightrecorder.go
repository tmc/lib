package cudaprof

import (
	"errors"
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

// WithPCSamplingPeriod sets the target SM PC sampling period configuration.
// Callers ingesting PC samples into the ring buffer must provide native samples
// via RecordSample, RecordDetailedSample, or RecordTraceback.
func WithPCSamplingPeriod(period uint32) FlightRecorderOption {
	return func(fr *FlightRecorder) {
		fr.samplePeriod = period
	}
}

// WithFlightRecorderBinaryPath requests the legacy generated ELF sidecar.
// Nonempty paths cause Start and WriteTo to fail; sample names do not supply code.
//
// Deprecated: retain native code artifacts separately instead.
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
	if fr.binaryPath != "" {
		return errors.New("cudaprof: generated ELF sidecars are not captured code")
	}
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
func (fr *FlightRecorder) WriteTo(w io.Writer) (int64, error) {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	cw := &countingWriter{w: w}

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

	err := b.WriteGzip(cw)
	return cw.n, err
}

// WriteProfile writes a snapshot and reports only the write error.
func (fr *FlightRecorder) WriteProfile(w io.Writer) error {
	_, err := fr.WriteTo(w)
	return err
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (w *countingWriter) Write(p []byte) (int, error) {
	n, err := w.w.Write(p)
	w.n += int64(n)
	return n, err
}

// WritePerfettoTo snapshots the flight recording buffer into a Perfetto .pftrace timeline.
func (fr *FlightRecorder) WritePerfettoTo(w io.Writer) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	pw := perfetto.NewWriter(w)
	var writeErr error
	write := func(err error) {
		if writeErr == nil {
			writeErr = err
		}
	}
	write(pw.WriteTrackDescriptor(1000, "GPU Flight Recorder", 0))
	write(pw.WriteCounterDescriptor(1001, "VRAM Allocation", "bytes", 1000))
	write(pw.WriteCounterDescriptor(1002, "DRAM Read Traffic", "bytes", 1000))
	write(pw.WriteCounterDescriptor(1003, "DRAM Write Traffic", "bytes", 1000))
	write(pw.WriteCounterDescriptor(1004, "Tensor Core Warps", "count", 1000))
	write(pw.WriteCounterDescriptor(1005, "L2 Sectors Read", "count", 1000))
	write(pw.WriteCounterDescriptor(1006, "L2 Sectors Miss", "count", 1000))
	write(pw.WriteCounterDescriptor(1007, "Active Warp Cycles", "count", 1000))
	write(pw.WriteCounterDescriptor(1008, "Eligible Warp Cycles", "count", 1000))
	write(pw.WriteCounterDescriptor(1009, "Issued Warps", "count", 1000))
	write(pw.WriteCounterDescriptor(1010, "Instructions Executed", "count", 1000))
	write(pw.WriteCounterDescriptor(1011, "Bank Conflicts", "count", 1000))
	write(pw.WriteCounterDescriptor(1012, "Register Spills", "bytes", 1000))
	write(pw.WriteCounterDescriptor(1013, "Interconnect TX", "bytes", 1000))
	write(pw.WriteCounterDescriptor(1014, "Interconnect RX", "bytes", 1000))
	write(pw.WriteCounterDescriptor(1015, "Stall Math Throttle", "count", 1000))
	write(pw.WriteCounterDescriptor(1016, "Stall Mem Latency", "count", 1000))
	write(pw.WriteCounterDescriptor(1017, "Launch Delay", "nanoseconds", 1000))
	write(pw.WriteCounterDescriptor(1018, "Queue Delay", "nanoseconds", 1000))

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

		write(pw.WriteSliceBeginWithAnnotations(1000, ts, rec.KernelName, "gpu_kernel", annotations, loc))
		write(pw.WriteSliceEnd(1000, ts+dur))

		// Emit continuous hardware counter updates
		if rec.Values.VRAMAllocBytes > 0 {
			write(pw.WriteCounterValue(1001, ts, rec.Values.VRAMAllocBytes))
		}
		if rec.Values.DRAMReadBytes > 0 {
			write(pw.WriteCounterValue(1002, ts, rec.Values.DRAMReadBytes))
		}
		if rec.Values.DRAMWriteBytes > 0 {
			write(pw.WriteCounterValue(1003, ts, rec.Values.DRAMWriteBytes))
		}
		if rec.Values.TensorWarps > 0 {
			write(pw.WriteCounterValue(1004, ts, rec.Values.TensorWarps))
		}
		if rec.Values.L2SectorsRead > 0 {
			write(pw.WriteCounterValue(1005, ts, rec.Values.L2SectorsRead))
		}
		if rec.Values.L2SectorsMiss > 0 {
			write(pw.WriteCounterValue(1006, ts, rec.Values.L2SectorsMiss))
		}
		if rec.Values.ActiveWarps > 0 {
			write(pw.WriteCounterValue(1007, ts, rec.Values.ActiveWarps))
		}
		if rec.Values.EligibleWarps > 0 {
			write(pw.WriteCounterValue(1008, ts, rec.Values.EligibleWarps))
		}
		if rec.Values.IssuedWarps > 0 {
			write(pw.WriteCounterValue(1009, ts, rec.Values.IssuedWarps))
		}
		if rec.Values.InstExecuted > 0 {
			write(pw.WriteCounterValue(1010, ts, rec.Values.InstExecuted))
		}
		if rec.Values.BankConflicts > 0 {
			write(pw.WriteCounterValue(1011, ts, rec.Values.BankConflicts))
		}
		if rec.Values.RegSpillBytes > 0 {
			write(pw.WriteCounterValue(1012, ts, rec.Values.RegSpillBytes))
		}
		if rec.Values.InterconnectTxBytes > 0 {
			write(pw.WriteCounterValue(1013, ts, rec.Values.InterconnectTxBytes))
		}
		if rec.Values.InterconnectRxBytes > 0 {
			write(pw.WriteCounterValue(1014, ts, rec.Values.InterconnectRxBytes))
		}
		if rec.Values.StallMath > 0 {
			write(pw.WriteCounterValue(1015, ts, rec.Values.StallMath))
		}
		if rec.Values.StallMem > 0 {
			write(pw.WriteCounterValue(1016, ts, rec.Values.StallMem))
		}
		if rec.Values.LaunchDelayNanos > 0 {
			write(pw.WriteCounterValue(1017, ts, rec.Values.LaunchDelayNanos))
		}
		if rec.Values.QueueDelayNanos > 0 {
			write(pw.WriteCounterValue(1018, ts, rec.Values.QueueDelayNanos))
		}
	}

	return writeErr
}

// WriteBoth snapshots the flight recording buffer to both a pprof writer and a Perfetto writer.
func (fr *FlightRecorder) WriteBoth(pprofWriter io.Writer, perfettoWriter io.Writer) error {
	if pprofWriter != nil {
		if _, err := fr.WriteTo(pprofWriter); err != nil {
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
