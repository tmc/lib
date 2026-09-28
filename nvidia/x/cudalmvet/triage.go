package cudalmvet

import (
	"fmt"
)

// TriageReport details the first differing intermediate value between two runs.
type TriageReport struct {
	DiffIndex       int
	NodeName        string
	ValueA          float64
	ValueB          float64
	BitsA           uint64
	BitsB           uint64
	HashA           uint64
	HashB           uint64
	ULPDistance     uint64
	ULPKnown        bool
	NumericalPolicy string
	Admissible      bool
	Message         string
}

func float32ULPDistance(a, b uint32) (uint64, bool) {
	if a&0x7f800000 == 0x7f800000 || b&0x7f800000 == 0x7f800000 {
		return 0, false
	}
	ordered := func(bits uint32) uint32 {
		if bits&0x80000000 != 0 {
			return ^bits
		}
		return bits | 0x80000000
	}
	x, y := uint64(ordered(a)), uint64(ordered(b))
	if x > y {
		return x - y, true
	}
	return y - x, true
}

func (r *TriageReport) String() string {
	if r == nil {
		return "triage: no difference observed"
	}
	return r.Message
}

// Triage compares the recorded traces of two accumulators to identify the exact
// first divergence point.
func Triage(a, b *Accumulator) *TriageReport {
	if a == nil || b == nil {
		return &TriageReport{
			DiffIndex: -1,
			Message:   "triage: one or both accumulators are nil",
		}
	}

	eventsA := a.Events()
	eventsB := b.Events()

	minLen := len(eventsA)
	if len(eventsB) < minLen {
		minLen = len(eventsB)
	}

	for i := 0; i < minLen; i++ {
		ea := eventsA[i]
		eb := eventsB[i]

		if ea.Bits != eb.Bits || ea.Hash != eb.Hash {
			node := ea.NodeName
			if node == "" {
				node = eb.NodeName
			}
			ulp, known := float32ULPDistance(uint32(ea.Bits), uint32(eb.Bits))
			msg := fmt.Sprintf("first divergence at fold #%d (node %q): trace A has value %v (0x%08x, hash 0x%016x), trace B has value %v (0x%08x, hash 0x%016x); numerical policy: bit-exact, ULP distance: %s, admissible: no",
				i, node, ea.Value, uint32(ea.Bits), ea.Hash, eb.Value, uint32(eb.Bits), eb.Hash, formatULP(ulp, known))
			return &TriageReport{
				DiffIndex:       i,
				NodeName:        node,
				ValueA:          ea.Value,
				ValueB:          eb.Value,
				BitsA:           ea.Bits,
				BitsB:           eb.Bits,
				HashA:           ea.Hash,
				HashB:           eb.Hash,
				ULPDistance:     ulp,
				ULPKnown:        known,
				NumericalPolicy: "bit-exact",
				Admissible:      false,
				Message:         msg,
			}
		}
	}

	if len(eventsA) != len(eventsB) {
		msg := fmt.Sprintf("length mismatch: trace A recorded %d events, trace B recorded %d events (agreed on first %d)",
			len(eventsA), len(eventsB), minLen)
		return &TriageReport{
			DiffIndex: minLen,
			Message:   msg,
		}
	}

	if a.Hash() != b.Hash() {
		return &TriageReport{
			DiffIndex: -1,
			HashA:     a.Hash(),
			HashB:     b.Hash(),
			Message: fmt.Sprintf("trace hashes differ (0x%016x vs 0x%016x) but event logs are identical or empty",
				a.Hash(), b.Hash()),
		}
	}

	return nil
}

func formatULP(distance uint64, known bool) string {
	if !known {
		return "unknown"
	}
	return fmt.Sprint(distance)
}
