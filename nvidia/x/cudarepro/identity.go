package cudarepro

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// Identity identifies an exact uint64 mismatch at a logical output element.
// Other numerical policies and access-violation identities are unsupported.
// Its zero value is invalid.
type Identity struct {
	Index  uint64
	Policy string
}

// Identify validates the campaign's execution evidence and returns the
// runner-produced comparison identity. It never derives identity from prose.
func Identify(run cudaschedule.RunRecord) (Identity, error) {
	if verdict := cudaschedule.Classify(run); verdict != cudaschedule.VerdictMismatch {
		return Identity{}, fmt.Errorf("identify failure: unsupported verdict %s", verdict)
	}
	if run.Mismatch == nil {
		return Identity{}, fmt.Errorf("identify failure: missing comparison identity")
	}
	if run.Mismatch.Policy != "exact-uint64" {
		return Identity{}, fmt.Errorf("identify failure: unsupported comparison policy %q", run.Mismatch.Policy)
	}
	return Identity{Index: run.Mismatch.Index, Policy: run.Mismatch.Policy}, nil
}

// Matches reports whether a classified run reproduces i. A different verdict
// or output element does not reproduce it. Missing or invalid execution or
// comparison evidence is an error, not an accepted or healthy candidate.
func (i Identity) Matches(run cudaschedule.RunRecord) (bool, error) {
	if i.Policy != "exact-uint64" {
		return false, fmt.Errorf("compare failure: invalid identity policy %q", i.Policy)
	}
	verdict := cudaschedule.Classify(run)
	if verdict == cudaschedule.VerdictHarnessError {
		return false, fmt.Errorf("compare failure: invalid execution evidence")
	}
	if verdict != cudaschedule.VerdictMismatch {
		return false, nil
	}
	got, err := Identify(run)
	if err != nil {
		return false, err
	}
	return got == i, nil
}
