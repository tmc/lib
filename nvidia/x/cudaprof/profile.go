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

	s.builder.AddSample(kernelName, filename, line, m.InstructionPC, m.ExecutedWarps, labels)
}

// Close finalizes and flushes the gzipped profile to the underlying writer.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.builder.WriteGzip(s.w)
}
