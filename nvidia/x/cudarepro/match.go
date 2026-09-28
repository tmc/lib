package cudarepro

import "github.com/tmc/lib/nvidia/x/cudasanitizer"

// Match is a candidate argument associated with a reported memory access.
// Multiple candidates are retained for aliased arguments and reused addresses.
// No candidate establishes launch identity by itself; callers must restrict
// records to the failing process and execution scope using independent evidence.
type Match struct {
	LaunchID          string
	Argument          string
	CallSite          string
	ArtifactDigest    string
	OutsideSlice      bool
	NearestAllocation bool
}

// Attribute returns candidate arguments whose captured allocation contains the
// access or equals the nearest allocation identified by the sanitizer. It does
// not guess using kernel names or choose one of several concurrent launches.
func Attribute(f cudasanitizer.Finding, launches ...*Launch) []Match {
	access, ok := f.MemoryAccess()
	if !ok {
		return nil
	}
	var matches []Match
	for _, l := range launches {
		if l == nil || l.Validate() != nil {
			continue
		}
		for _, a := range l.Arguments {
			if !a.Buffer || a.Address == 0 {
				continue
			}
			i := l.allocation(a.Address, a.Size)
			if i < 0 {
				continue
			}
			b := l.Allocations[i]
			inside := access.Address >= b.Address && access.Address-b.Address < uint64(len(b.Data))
			nearest := access.Allocation == b.Address && access.AllocationBytes == uint64(len(b.Data))
			if !inside && !nearest {
				continue
			}
			outside := access.Address < a.Address || access.Address-a.Address >= a.Size
			if !outside {
				outside = access.Bytes > a.Size-(access.Address-a.Address)
			}
			matches = append(matches, Match{l.ID, a.Name, l.CallSite, l.Artifact.Digest, outside, !inside})
		}
	}
	return matches
}
