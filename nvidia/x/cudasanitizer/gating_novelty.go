package cudasanitizer

import (
	"crypto/sha256"
	"fmt"
	"sync"
)

// SanitizerGating selects a host trace-abstraction policy. It does not enable
// instrumentation or establish that an oracle is observable in a fuzz worker.
type SanitizerGating int

const (
	GatingAllTrace SanitizerGating = iota + 1
	GatingSimplifyTrace
	GatingUniqueTrace
	GatingCoverageIncrease
)

// TraceNovelty is a bounded, concurrent host filter. AllTrace admits every
// nonzero trace; SimplifyTrace compares edge presence; UniqueTrace compares
// quantized hit counts; CoverageIncrease admits a previously unseen hit class
// at any edge. The zero value is unusable; construct it with NewTraceNovelty.
// This filter has no device provenance and must not by itself authorize skipping
// sanitizer execution. Evaluate marks admission, not successful sanitization.
type TraceNovelty struct {
	mu      sync.Mutex
	mode    SanitizerGating
	size    int
	limit   int
	seen    map[[32]byte]struct{}
	classes []byte
}

// NewTraceNovelty bounds trace size to 16 MiB and retained signatures to one
// million. Reaching maxTraces refuses further novel traces instead of eviction
// or silently treating them as duplicates.
func NewTraceNovelty(mode SanitizerGating, size, maxTraces int) (*TraceNovelty, error) {
	if mode < GatingAllTrace || mode > GatingCoverageIncrease {
		return nil, fmt.Errorf("trace novelty: invalid mode")
	}
	if size < 2 || size > 16<<20 || size%2 != 0 {
		return nil, fmt.Errorf("trace novelty: invalid map size")
	}
	if maxTraces <= 0 || maxTraces > 1<<20 {
		return nil, fmt.Errorf("trace novelty: invalid trace limit")
	}
	return &TraceNovelty{mode: mode, size: size, limit: maxTraces, seen: make(map[[32]byte]struct{}), classes: make([]byte, size)}, nil
}

// Evaluate returns whether a complete raw counter vector is admitted. Empty,
// zero and mismatched vectors are errors and do not alter novelty state. The
// caller owns the vector and must not mutate it concurrently with this call.
func (g *TraceNovelty) Evaluate(raw []byte) (bool, error) {
	if g == nil {
		return false, fmt.Errorf("trace novelty: uninitialized filter")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.size == 0 || len(raw) != g.size {
		return false, fmt.Errorf("trace novelty: mismatched map size")
	}
	any := false
	for _, n := range raw {
		any = any || n != 0
	}
	if !any {
		return false, fmt.Errorf("trace novelty: zero coverage")
	}
	if g.mode == GatingAllTrace {
		return true, nil
	}
	trace := make([]byte, len(raw))
	for i, n := range raw {
		if g.mode == GatingSimplifyTrace {
			trace[i] = 1
			if n != 0 {
				trace[i] = 128
			}
		} else {
			trace[i] = traceCountClass(n)
		}
	}
	if g.mode == GatingCoverageIncrease {
		novel := false
		for i, n := range trace {
			novel = novel || n&^g.classes[i] != 0
		}
		if !novel {
			return false, nil
		}
		for i, n := range trace {
			g.classes[i] |= n
		}
		return true, nil
	}
	h := sha256.Sum256(trace)
	if _, ok := g.seen[h]; ok {
		return false, nil
	}
	if len(g.seen) >= g.limit {
		return false, fmt.Errorf("trace novelty: retained trace limit reached")
	}
	g.seen[h] = struct{}{}
	return true, nil
}

// MustEvaluate is like Evaluate, but panics if raw has a mismatched map size,
// contains zero coverage, or if the filter is uninitialized.
func (g *TraceNovelty) MustEvaluate(raw []byte) bool {
	ok, err := g.Evaluate(raw)
	if err != nil {
		panic(err)
	}
	return ok
}

func traceCountClass(n byte) byte {
	switch {
	case n <= 1:
		return n
	case n <= 3:
		return 2
	case n <= 7:
		return 4
	case n <= 15:
		return 8
	case n <= 31:
		return 16
	case n <= 63:
		return 32
	case n <= 127:
		return 64
	default:
		return 128
	}
}
