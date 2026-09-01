package cudaprof

import (
	"context"
	"io"
	"runtime"
	"runtime/pprof"
	"sync"
)

// SASSMetric represents a modern hardware performance counter associated with a SASS instruction.
type SASSMetric struct {
	InstructionPC uint64
	Opcode        string
	StallReason   string
	IssuedWarps   int64
	ExecutedWarps int64
}

// Session manages active GPU profile collection and thread-safe callstack correlation.
type Session struct {
	mu      sync.Mutex
	w       io.Writer
	builder *Builder
}

// NewSession creates a new profiling session writing to w.
func NewSession(w io.Writer) *Session {
	return &Session{
		w:       w,
		builder: NewBuilder(),
	}
}

// SetBinaryPath sets the target path for the synthesized ELF binary sidecar.
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

// RecordDetailedSample records multi-dimensional metrics (VRAM, DRAM read/write, Shared Memory, Tensor warps, Stalls).
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

	s.builder.AddDetailedSample(kernelName, filename, line, m.InstructionPC, SampleValues{
		Samples:     m.ExecutedWarps,
		TensorWarps: m.ExecutedWarps,
	}, labels)
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
