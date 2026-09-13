package cudaprof

import (
	"context"
	"fmt"
	"io"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"time"
)

// SASSMetric represents hardware performance counters associated with a SASS instruction.
// Stalls and issued/executed counts must be provided from native hardware collection.
type SASSMetric struct {
	InstructionPC uint64
	Opcode        string
	StallReason   string
	StallWarps    int64 // Stalled warp count/cycles attributed to StallReason
	IssuedWarps   int64 // Warp instructions issued
	ExecutedWarps int64 // Warp instructions executed
}

// Session manages active GPU profile collection and thread-safe callstack correlation.
type Session struct {
	mu      sync.Mutex
	w       io.Writer
	builder *Builder
}

type profileConfig struct {
	binaryPath        string
	mappingFile       string
	pcSamplingPeriod  uint32
	sourceAttribution bool
}

// Option configures GPU profiling parameters.
type Option func(*profileConfig)

// WithPCSampling configures the SM PC sampling period for native collectors.
// In the current session builder, PC samples must be explicitly recorded via
// [Session.RecordDetailedSample] or [Session.RecordSASSMetric].
func WithPCSampling(samplePeriod uint32) Option {
	return func(c *profileConfig) {
		c.pcSamplingPeriod = samplePeriod
	}
}

// WithBinaryPath requests the legacy generated ELF sidecar. A nonempty path
// causes Start to return an error; inferred instructions are not captured code.
//
// Deprecated: use a retained native artifact with Session.SetMappingFile.
func WithBinaryPath(path string) Option {
	return func(c *profileConfig) {
		c.binaryPath = path
	}
}

// WithSourceAttribution configures source-line attribution for native collectors.
// The session builder expects resolved line numbers in recorded frames.
func WithSourceAttribution(enabled bool) Option {
	return func(c *profileConfig) {
		c.sourceAttribution = enabled
	}
}

var (
	globalMu      sync.Mutex
	globalSession *Session
)

// Start begins live GPU profiling and writes the resulting profile to w upon Stop.
func Start(w io.Writer, opts ...Option) error {
	globalMu.Lock()
	defer globalMu.Unlock()

	var cfg profileConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.binaryPath != "" {
		return fmt.Errorf("cudaprof: generated ELF sidecars are not captured code")
	}

	s := NewSession(w)
	if cfg.mappingFile != "" {
		s.SetMappingFile(cfg.mappingFile, "")
	}
	globalSession = s
	return nil
}

// Stop flushes and finalizes the active GPU profile, writing it to the destination provided to Start.
func Stop() error {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalSession == nil {
		return nil
	}
	err := globalSession.Close()
	globalSession = nil
	return err
}

// NewSession creates a new profiling session writing to w.
func NewSession(w io.Writer) *Session {
	return &Session{
		w:       w,
		builder: NewBuilder(),
	}
}

// SetSampleTypes selects the metrics to write; see [Builder.SetSampleTypes].
// Use it to omit metrics the caller does not measure.
func (s *Session) SetSampleTypes(names ...string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.builder.SetSampleTypes(names...)
}

// SetBinaryPath requests the legacy generated ELF sidecar. A nonempty path
// causes Close to fail without writing an inferred instruction image.
//
// Deprecated: use SetMappingFile with a retained native artifact instead.
func (s *Session) SetBinaryPath(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builder.SetBinaryPath(path)
}

// SetMappingFile configures the binary/PTX file path for instruction disassembly resolution.
func (s *Session) SetMappingFile(filename, buildID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builder.SetMappingFile(filename, buildID)
}

// AddMapping registers an additional logical image. See [Builder.AddMapping].
func (s *Session) AddMapping(filename, buildID string, start, limit uint64) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.builder.AddMapping(filename, buildID, start, limit)
}

// SetTimespan records the profile's start time and duration. See
// [Builder.SetTimespan].
func (s *Session) SetTimespan(start time.Time, d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builder.SetTimespan(start, d)
}

// SetPeriod records the sampling period and what it measures. See
// [Builder.SetPeriod].
func (s *Session) SetPeriod(name, unit string, period int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builder.SetPeriod(name, unit, period)
}

// SetDefaultSampleType names the sample type a viewer should open. See
// [Builder.SetDefaultSampleType].
func (s *Session) SetDefaultSampleType(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.builder.SetDefaultSampleType(name)
}

// AddComment records a line of free text on the profile, for a fact that holds
// for the whole run. See [Builder.AddComment].
func (s *Session) AddComment(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builder.AddComment(text)
}

// SetDocURL records a documentation link for the profile.
// See [Builder.SetDocURL].
func (s *Session) SetDocURL(url string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builder.SetDocURL(url)
}

// SetNumericLabels declares which label keys hold numbers, so that pprof's
// range filters apply to them. See [Builder.SetNumericLabels].
func (s *Session) SetNumericLabels(unitByKey map[string]string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builder.SetNumericLabels(unitByKey)
}

// RecordSample adds a GPU sample correlated with Go context pprof labels and OS thread affinity.

func (s *Session) RecordSample(ctx context.Context, kernelName, filename string, line int64, sassPC uint64, count int64) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	s.mu.Lock()
	defer s.mu.Unlock()

	labels := make(map[string]string)
	if ctx != nil {
		pprof.ForLabels(ctx, func(key, value string) bool {
			labels[key] = value
			return true
		})
	}

	s.builder.AddSample(kernelName, filename, line, sassPC, count, labels)
}

// RecordDetailedSample records caller-supplied counters and device intervals.
// It does not measure or verify them. Leave unmeasured dimensions zero.
func (s *Session) RecordDetailedSample(ctx context.Context, kernelName, filename string, line int64, sassPC uint64, vals SampleValues) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	s.mu.Lock()
	defer s.mu.Unlock()

	labels := make(map[string]string)
	if ctx != nil {
		pprof.ForLabels(ctx, func(key, value string) bool {
			labels[key] = value
			return true
		})
	}

	s.builder.AddDetailedSample(kernelName, filename, line, sassPC, vals, labels)
}

// RecordSASSMetric records modern range-profiling / SASS instruction execution metric data.
func (s *Session) RecordSASSMetric(ctx context.Context, kernelName, filename string, line int64, m SASSMetric) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	s.mu.Lock()
	defer s.mu.Unlock()

	labels := make(map[string]string)
	if ctx != nil {
		pprof.ForLabels(ctx, func(key, value string) bool {
			labels[key] = value
			return true
		})
	}
	if m.Opcode != "" {
		labels["sass.opcode"] = m.Opcode
	}
	if m.StallReason != "" {
		labels["sass.stall"] = m.StallReason
	}
	if m.IssuedWarps > 0 {
		labels["sass.issued_warps"] = fmt.Sprintf("%d", m.IssuedWarps)
	}

	vals := SampleValues{
		Samples: m.ExecutedWarps,
	}
	if isTensorOpcode(m.Opcode) {
		vals.TensorWarps = m.ExecutedWarps
	}
	// Stall reasons must only account for stalled warps/cycles, never issued warps.
	if m.StallWarps > 0 {
		switch m.StallReason {
		case "stall_scoreboard", "stall_long_scoreboard", "stall_short_scoreboard":
			vals.StallScoreboard = m.StallWarps
		case "stall_wait", "stall_barrier":
			vals.StallWait = m.StallWarps
		case "stall_math_pipe_throttle", "stall_math":
			vals.StallMath = m.StallWarps
		case "stall_mem_throttle", "stall_mem":
			vals.StallMem = m.StallWarps
		}
	}

	s.builder.AddDetailedSample(kernelName, filename, line, m.InstructionPC, vals, labels)
}

func isTensorOpcode(op string) bool {
	upper := strings.ToUpper(op)
	return strings.HasPrefix(upper, "HMMA") || strings.HasPrefix(upper, "IMMA") ||
		strings.HasPrefix(upper, "QMMA") || strings.HasPrefix(upper, "BMMA") ||
		strings.HasPrefix(upper, "MMA") || strings.HasPrefix(upper, "WGMMA")
}

// RecordStackSample records a multi-frame callstack sample with Go context labels.
func (s *Session) RecordStackSample(ctx context.Context, vals SampleValues, frames []Frame) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	s.mu.Lock()
	defer s.mu.Unlock()

	labels := make(map[string]string)
	if ctx != nil {
		pprof.ForLabels(ctx, func(key, value string) bool {
			labels[key] = value
			return true
		})
	}

	s.builder.AddStackSample(vals, frames, labels)
}

// RecordTraceback records a unified Go -> CUDA (.cu) -> PTX -> SASS multi-frame callstack sample.
func (s *Session) RecordTraceback(
	ctx context.Context,
	vals SampleValues,
	kernelName string,
	cuFile string,
	cuLine int64,
	ptxFile string,
	ptxLine int64,
	sassPC uint64,
) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	s.mu.Lock()
	defer s.mu.Unlock()

	labels := make(map[string]string)
	if ctx != nil {
		pprof.ForLabels(ctx, func(key, value string) bool {
			labels[key] = value
			return true
		})
	}

	s.builder.AddTracebackSample(vals, kernelName, cuFile, cuLine, ptxFile, ptxLine, sassPC, labels)
}

// Close finalizes and flushes the gzipped profile to the underlying writer.
func (s *Session) Close() error {

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.builder.WriteGzip(s.w)
}
