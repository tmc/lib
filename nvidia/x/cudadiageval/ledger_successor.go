package cudadiageval

import (
	"fmt"
	"reflect"
	"slices"
)

// CheckQualificationLedgerSuccessor reports whether next preserves the
// evidence history recorded in prev. Both ledgers must validate. Every cell
// in prev must remain in next, and its attempts must be an exact ordered
// prefix of the attempts in next.
//
// References are a set: every earlier reference must appear, unchanged, in
// next, while order and duplicates are ignored. A reference recorded with an
// empty SHA256 (an unknown artifact) cannot gain a digest in place; a
// recovered artifact is added as a new reference beside it.
//
// A cell with attempts may not change its closure, gate, capability ID,
// source paths, consumers or dependencies. Changed scope requires a new cell
// ID, which leaves earlier failures intact. Status, capability, owner and
// next action may change.
//
// It compares declarations only. It does not read receipts; use
// VerifyQualificationLedger to check retained evidence.
func CheckQualificationLedgerSuccessor(prev, next QualificationLedger) error {
	if err := prev.Validate(); err != nil {
		return fmt.Errorf("ledger successor: previous: %w", err)
	}
	if err := next.Validate(); err != nil {
		return fmt.Errorf("ledger successor: next: %w", err)
	}
	cells := make(map[string]QualificationCell, len(next.Cells))
	for _, c := range next.Cells {
		cells[c.ID] = c
	}
	for _, p := range prev.Cells {
		n, ok := cells[p.ID]
		if !ok {
			return fmt.Errorf("ledger successor: cell %q removed", p.ID)
		}
		if len(n.Attempts) < len(p.Attempts) {
			return fmt.Errorf("ledger successor: cell %q attempts truncated from %d to %d", p.ID, len(p.Attempts), len(n.Attempts))
		}
		for i, a := range p.Attempts {
			if !reflect.DeepEqual(a, n.Attempts[i]) {
				return fmt.Errorf("ledger successor: cell %q attempt %d changed", p.ID, i)
			}
		}
		for _, ref := range p.References {
			if !hasReference(n.References, ref) {
				return fmt.Errorf("ledger successor: cell %q reference %q removed or changed", p.ID, ref.Path)
			}
		}
		if len(p.Attempts) > 0 && !sameScope(p, n) {
			return fmt.Errorf("ledger successor: cell %q with attempts changed scope; register a new cell", p.ID)
		}
	}
	return nil
}

// sameScope reports whether two cells claim the same target and evidence chain.
func sameScope(a, b QualificationCell) bool {
	return a.Gate == b.Gate &&
		a.CapabilityID == b.CapabilityID &&
		reflect.DeepEqual(a.Closure, b.Closure) &&
		slices.Equal(a.SourcePaths, b.SourcePaths) &&
		slices.Equal(a.Consumers, b.Consumers) &&
		slices.Equal(a.DependsOn, b.DependsOn)
}

func hasReference(refs []QualificationReference, ref QualificationReference) bool {
	for _, r := range refs {
		if r == ref {
			return true
		}
	}
	return false
}
