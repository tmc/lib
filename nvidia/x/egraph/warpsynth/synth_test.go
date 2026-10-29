package warpsynth

import (
	"errors"
	"math/rand/v2"
	"strings"
	"testing"
)

func gemmProblem() Problem {
	return Problem{
		Arch:          "sm_100a",
		SMEMCapBytes:  228 * 1024,
		TileSMEMBytes: 32 * 1024,
		TMALatency:    1000,
		Roles: []RoleDemand{
			{Kind: RoleEpilogueMath, MinWarps: 4, MinRegs: 200, WorkCyclesTile: 2400, ScalesWithWarps: true},
			{Kind: RoleTMAProducer, MinWarps: 4, MinRegs: 40, WorkCyclesTile: 400},
			{Kind: RoleMMAIssuer, MinWarps: 4, MinRegs: 64, WorkCyclesTile: 1000},
		},
		TMEM: []TMEMBufferReq{{Name: "acc", WidthCols: 256, DoubleBuf: true}},
	}
}

func TestSynthesizeGEMM(t *testing.T) {
	s, err := Synthesize(gemmProblem())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Check(); err != nil {
		t.Fatal(err)
	}
	// Within 16 warps the epilogue gets at most two warpgroups, which halves
	// its 2400 cycles; it stays the bottleneck.
	epi, _ := s.Role(RoleEpilogueMath)
	if s.BottleneckCycTile != 1200 || epi.Warps != 8 || s.TotalWarps != 16 {
		t.Errorf("bottleneck %d with %d epilogue warps of %d, want 1200 with 8 of 16", s.BottleneckCycTile, epi.Warps, s.TotalWarps)
	}
	if !s.UseEarlySMEMRelease {
		t.Error("early release not chosen")
	}
	if s.TotalTMEMCols != 512 || s.TMEM[0].Cols != [2]uint16{0, 256} {
		t.Errorf("TMEM %v in %d columns, want [0 256] in 512", s.TMEM[0].Cols, s.TotalTMEMCols)
	}
	if s.TotalRegsUsed > 65536 {
		t.Errorf("TotalRegsUsed = %d", s.TotalRegsUsed)
	}
	if err := VerifyDeadlockFreedom(s, 1000); err != nil {
		t.Fatal(err)
	}
}

func TestSynthesizeErrors(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Problem)
		want string
	}{
		{"unknown arch", func(p *Problem) { p.Arch = "sm_99" }, "unknown target"},
		{"tmem on sm_121a", func(p *Problem) { p.Arch = "sm_121a" }, "no tcgen05 TMEM"},
		{"tmem overflow", func(p *Problem) { p.TMEM = append(p.TMEM, TMEMBufferReq{Name: "x", WidthCols: 64}) }, "more than 512"},
		{"bad width", func(p *Problem) { p.TMEM[0].WidthCols = 96 }, "power of two"},
		{"no producer", func(p *Problem) {
			p.Roles = p.Roles[:1:1]
			p.Roles = append(p.Roles, RoleDemand{Kind: RoleMMAIssuer, MinWarps: 4, MinRegs: 32})
		}, "must start"},
		{"registers", func(p *Problem) {
			for i := range p.Roles {
				p.Roles[i].MinRegs = 248
			}
		}, "no feasible"},
		{"smem", func(p *Problem) { p.TileSMEMBytes = 120 * 1024 }, "no feasible"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := gemmProblem()
			tt.edit(&p)
			_, err := Synthesize(p)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
	p := gemmProblem()
	p.TileSMEMBytes = 120 * 1024
	if _, err := Synthesize(p); !errors.Is(err, ErrInfeasible) {
		t.Errorf("err = %v, want ErrInfeasible", err)
	}
}

func TestSynthesizeRubin576(t *testing.T) {
	p := gemmProblem()
	p.Arch = "sm_107a"
	p.TMEM = append(p.TMEM, TMEMBufferReq{Name: "lut", WidthCols: 64})
	s, err := Synthesize(p)
	if err != nil {
		t.Fatal(err)
	}
	if s.TotalTMEMCols != 576 {
		t.Errorf("TotalTMEMCols = %d, want 576", s.TotalTMEMCols)
	}
}

// TestHardwareInvariantOracle checks every schedule synthesized for random
// demands against the invariants, independently of Check.
func TestHardwareInvariantOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	kinds := []WarpRoleKind{RoleTMAProducer, RoleMMAIssuer, RoleEpilogueMath, RoleFabricStorer}
	found := 0
	for range 300 {
		n := 2 + rng.IntN(3)
		var p Problem
		p.Arch = []string{"sm_100a", "sm_103a", "sm_107a", "sm_121a", "sm_90a"}[rng.IntN(5)]
		p.SMEMCapBytes = 228 * 1024
		p.TileSMEMBytes = 8*1024 + rng.IntN(64*1024)
		p.TMALatency = uint64(rng.IntN(2000))
		for i := range n {
			p.Roles = append(p.Roles, RoleDemand{
				Kind: kinds[i], MinWarps: 1 + rng.IntN(8), MinRegs: 24 + rng.IntN(220),
				WorkCyclesTile: uint64(100 + rng.IntN(4000)), ScalesWithWarps: rng.IntN(2) == 0,
			})
		}
		if TMEMColumns(p.Arch) > 0 {
			for i := range rng.IntN(3) {
				p.TMEM = append(p.TMEM, TMEMBufferReq{Name: string(rune('a' + i)), WidthCols: uint16(32 << rng.IntN(4)), DoubleBuf: rng.IntN(2) == 0})
			}
		}
		s, err := Synthesize(p)
		if err != nil {
			if !errors.Is(err, ErrInfeasible) && !strings.Contains(err.Error(), "TMEM") {
				t.Fatalf("%+v: %v", p, err)
			}
			continue
		}
		found++
		regs := 0
		for _, r := range s.Roles {
			if r.Warps%4 != 0 || r.Regs%8 != 0 || r.Regs < 24 || r.Regs > 256 {
				t.Fatalf("role %+v breaks warpgroup or setmaxnreg granularity", r)
			}
			regs += 32 * r.Warps * r.Regs
		}
		if regs > 65536 || regs > 32*s.TotalWarps*s.LaunchRegs {
			t.Fatalf("registers %d exceed pool %d", regs, 32*s.TotalWarps*s.LaunchRegs)
		}
		if s.TotalTMEMCols > TMEMColumns(p.Arch) {
			t.Fatalf("TMEM %d > %d", s.TotalTMEMCols, TMEMColumns(p.Arch))
		}
		used := make([]bool, s.TotalTMEMCols)
		for _, b := range s.TMEM {
			for h, c := range b.Cols {
				if h == 1 && b.Cols[1] == b.Cols[0] {
					continue
				}
				for x := int(c); x < int(c)+int(b.Width); x++ {
					if used[x] {
						t.Fatalf("TMEM column %d used twice in %+v", x, s.TMEM)
					}
					used[x] = true
				}
			}
		}
		if s.SMEMBytes > p.SMEMCapBytes {
			t.Fatalf("SMEM %d > %d", s.SMEMBytes, p.SMEMCapBytes)
		}
		if err := VerifyDeadlockFreedom(s, 40); err != nil {
			t.Fatal(err)
		}
	}
	if found < 50 {
		t.Errorf("only %d of 300 random problems were feasible", found)
	}
}

// TestSynthesizeOptimal compares the search with a brute force that also
// enumerates every register assignment, on a small problem.
func TestSynthesizeOptimal(t *testing.T) {
	p := gemmProblem()
	p.TMEM = nil
	got, err := Synthesize(p)
	if err != nil {
		t.Fatal(err)
	}
	roles, _ := normalize(p.Roles)
	best := uint64(1 << 62)
	for a := 4; a <= 16; a += 4 {
		for b := 4; a+b <= 16; b += 4 {
			for c := 4; a+b+c <= 16; c += 4 {
				if a+b+c < 8 {
					continue
				}
				for d := 2; d <= 8; d++ {
					for _, early := range []bool{false, true} {
						s, ok := build(p, roles, []int{a, b, c}, a+b+c, d, 1, early)
						if ok && s.SteadyStateCycTile < best {
							best = s.SteadyStateCycTile
						}
					}
				}
			}
		}
	}
	if got.SteadyStateCycTile != best {
		t.Errorf("synthesized period %d, brute force %d", got.SteadyStateCycTile, best)
	}
}

func TestCheckRejects(t *testing.T) {
	good, err := Synthesize(gemmProblem())
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*Schedule)
	}{
		{"regs not multiple of 8", func(s *Schedule) { s.Roles[0].Regs += 4; s.TotalRegsUsed += 32 * 4 * s.Roles[0].Warps }},
		{"warps not multiple of 4", func(s *Schedule) { s.Roles[2].Warps-- }},
		{"register pool", func(s *Schedule) {
			s.Roles[2].Regs += 64
			s.TotalRegsUsed += 32 * 64 * s.Roles[2].Warps
		}},
		{"tmem overlap", func(s *Schedule) { s.TMEM[0].Cols[1] = 0 + 32 }},
		{"tmem past limit", func(s *Schedule) { s.TotalTMEMCols = 1024 }},
		{"smem", func(s *Schedule) { s.SMEMCapBytes = s.SMEMBytes - 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := good
			s.Roles = append([]RoleAssignment(nil), good.Roles...)
			s.TMEM = append([]TMEMPlacement(nil), good.TMEM...)
			tt.edit(&s)
			if err := s.Check(); err == nil {
				t.Fatal("Check accepted a broken schedule")
			}
		})
	}
}
