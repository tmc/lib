package warpsynth

import (
	"errors"
	"testing"
)

func schedules(t *testing.T) map[string]Schedule {
	t.Helper()
	out := map[string]Schedule{}
	early, err := Synthesize(gemmProblem())
	if err != nil {
		t.Fatal(err)
	}
	out["early"] = early
	late := early
	late.UseEarlySMEMRelease = false
	out["late"] = late
	p := gemmProblem()
	p.Roles = append(p.Roles, RoleDemand{Kind: RoleFabricStorer, MinWarps: 4, MinRegs: 32, WorkCyclesTile: 300})
	p.Totals = []int{16, 20}
	four, err := Synthesize(p)
	if err != nil {
		t.Fatal(err)
	}
	out["four roles"] = four
	two := early
	two.Roles = two.Roles[:2]
	two.RingDepths = two.RingDepths[:1]
	two.TotalWarps = two.Roles[0].Warps + two.Roles[1].Warps
	two.TotalRegsUsed = 32 * (two.Roles[0].Warps*two.Roles[0].Regs + two.Roles[1].Warps*two.Roles[1].Regs)
	two.LaunchRegs = 256
	out["two roles"] = two
	return out
}

func TestVerifyDeadlockFreedom(t *testing.T) {
	for name, s := range schedules(t) {
		for _, d := range []int{2, 3, 8} {
			s.PipelineStagesD, s.RingDepths = d, append([]int{d}, s.RingDepths[1:]...)
			if err := VerifyDeadlockFreedom(s, 1000); err != nil {
				t.Errorf("%s D=%d: %v", name, d, err)
			}
		}
	}
}

func TestVerifyMutants(t *testing.T) {
	tests := []struct {
		name string
		p    protocol
		want []ProtocolErrorKind
	}{
		{"parity k&1", protocol{parityByIter: true}, []ProtocolErrorKind{ParityAlias, StaleRead, Race}},
		{"no empty wait", protocol{skipEmptyWait: true}, []ProtocolErrorKind{Race}},
	}
	for name, s := range schedules(t) {
		for _, tt := range tests {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				err := verify(s, 1000, tt.p)
				var pe *ProtocolError
				if !errors.As(err, &pe) {
					t.Fatalf("err = %v, want a ProtocolError", err)
				}
				ok := false
				for _, k := range tt.want {
					ok = ok || pe.Kind == k
				}
				if !ok {
					t.Fatalf("kind %v, want one of %v: %v", pe.Kind, tt.want, pe)
				}
			})
		}
	}
}

// TestVerifyDeadlockDetected removes a release so the producer waits for a
// stage nobody frees.
func TestVerifyDeadlockDetected(t *testing.T) {
	s := schedules(t)["early"]
	m := &model{s: s, rings: rings(s), tiles: 20}
	m.rings[0].releaser = 99
	err := m.exhaustive()
	var pe *ProtocolError
	if !errors.As(err, &pe) || pe.Kind != Deadlock {
		t.Fatalf("err = %v, want deadlock", err)
	}
	if pe.Role != RoleTMAProducer || pe.Tile != s.PipelineStagesD {
		t.Errorf("deadlock at %s tile %d, want producer at tile %d", pe.Role, pe.Tile, s.PipelineStagesD)
	}
}
