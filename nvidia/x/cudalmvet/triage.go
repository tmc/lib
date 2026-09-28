package cudalmvet

import (
	"fmt"
)

// TriageReport details the first differing intermediate value between two runs.
type TriageReport struct {
	DiffIndex int
	NodeName  string
	ValueA    float64
	ValueB    float64
	BitsA     uint64
	BitsB     uint64
	HashA     uint64
	HashB     uint64
	Message   string
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
			msg := fmt.Sprintf("first divergence at fold #%d (node %q): trace A has value %v (0x%016x, hash 0x%016x), trace B has value %v (0x%016x, hash 0x%016x)",
				i, node, ea.Value, ea.Bits, ea.Hash, eb.Value, eb.Bits, eb.Hash)
			return &TriageReport{
				DiffIndex: i,
				NodeName:  node,
				ValueA:    ea.Value,
				ValueB:    eb.Value,
				BitsA:     ea.Bits,
				BitsB:     eb.Bits,
				HashA:     ea.Hash,
				HashB:     eb.Hash,
				Message:   msg,
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
