package cudatrial

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashared"
)

func TestEvaluateAccessesFollowsRecordsAcrossContradictoryLabels(t *testing.T) {
	base := cudashared.MemoryAccess{
		Scope: cudashared.ScopeCTA, AgentID: 1, Space: cudashared.MemoryShared,
		Offset: 0x1000, SizeBytes: 1, Stage: 0,
	}
	other := base
	other.ThreadID = 1
	other.IsWrite = true
	tests := []struct {
		name    string
		label   RaceCategory
		fixed   bool
		bOffset uint64
		want    bool
	}{
		{"race label with disjoint accesses", CategoryMemcpy, false, 0x1001, false},
		{"false-positive label with conflicting accesses", CategoryFalsePositives, false, 0x1000, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			accesses := []cudashared.MemoryAccess{base, other}
			accesses[1].Offset = tt.bOffset
			hasRace, pairs, err := evaluateAccesses("sm_90", accesses)
			if err != nil {
				t.Fatal(err)
			}
			if hasRace != tt.want || pairs != 1 {
				t.Fatalf("label=%s fixed=%t records=%d race=%t, want one record and race=%t", tt.label, tt.fixed, pairs, hasRace, tt.want)
			}
		})
	}
}
