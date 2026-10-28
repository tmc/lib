package train

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func unitConfig(p, m int, comm float64) ScheduleConfig {
	return ScheduleConfig{NumStages: p, NumMicrobatch: m, FwdUs: 1, BwdInputUs: 1, BwdWeightUs: 1, CommAll2AllUs: comm}
}

func TestPipelineSchedulesVerify(t *testing.T) {
	plans := []struct {
		name string
		plan func(ScheduleConfig) (PipelineSchedule, error)
	}{
		{"1f1b", PlanOneFOneB},
		{"zb", PlanZeroBubble},
		{"dualpipe", PlanDualPipe},
	}
	for _, p := range []int{4, 8, 16} {
		for _, m := range []int{2 * p, 3 * p, 4 * p, 8 * p} {
			for _, comm := range []float64{0, 0.5} {
				cfg := unitConfig(p, m, comm)
				cfg.BwdInputUs = 1.25 // unequal phases exercise tie-breaking
				for _, pl := range plans {
					s, err := pl.plan(cfg)
					if err != nil {
						t.Fatalf("%s P=%d M=%d: %v", pl.name, p, m, err)
					}
					if err := VerifyScheduleDAG(s, cfg); err != nil {
						t.Errorf("%s P=%d M=%d comm=%v: %v", pl.name, p, m, comm, err)
					}
					limit := cfg.inflightLimit()
					for st, peak := range s.PeakInflight {
						if pl.name == "1f1b" {
							limit = min(p-st, m)
							if peak != limit {
								t.Errorf("1f1b P=%d M=%d stage %d peak in flight %d, want %d", p, m, st, peak, limit)
							}
						}
						if peak > limit {
							t.Errorf("%s P=%d M=%d stage %d peak in flight %d > %d", pl.name, p, m, st, peak, limit)
						}
					}
				}
			}
		}
	}
}

func TestPipelineBubbles(t *testing.T) {
	for _, p := range []int{4, 8, 16} {
		m := 4 * p
		t.Run(fmt.Sprintf("P%d", p), func(t *testing.T) {
			cfg := unitConfig(p, m, 0)
			base, err := PlanOneFOneB(cfg)
			if err != nil {
				t.Fatal(err)
			}
			zb, err := PlanZeroBubble(cfg)
			if err != nil {
				t.Fatal(err)
			}
			dp, err := PlanDualPipe(cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("bubble 1f1b=%.3f zb=%.3f (interior %.3f) dualpipe=%.3f", base.BubbleFraction, zb.BubbleFraction, zb.InteriorBubbleFraction, dp.BubbleFraction)

			// 1F1B with B = B_X+B_W = 2F: bubble (P-1)/(M+P-1).
			want := float64(p-1) / float64(m+p-1)
			if math.Abs(base.BubbleFraction-want) > 1e-9 {
				t.Errorf("1f1b bubble = %v, want %v", base.BubbleFraction, want)
			}
			// Stage s cannot start before s·T_F, so with a synchronous
			// optimizer step the bubble is at least (P-1)/(P-1+3M).
			lb := float64(p-1) / float64(p-1+3*m)
			if math.Abs(zb.BubbleFraction-lb) > 1e-9 {
				t.Errorf("zero bubble = %v, want lower bound %v", zb.BubbleFraction, lb)
			}
			if zb.InteriorBubbleFraction >= 0.02 {
				t.Errorf("zero bubble interior = %v, want < 0.02", zb.InteriorBubbleFraction)
			}
			if dp.MakespanUs > zb.MakespanUs {
				t.Errorf("dualpipe makespan %v > zero bubble %v", dp.MakespanUs, zb.MakespanUs)
			}
		})
	}
}

func TestDualPipeHidesAllToAll(t *testing.T) {
	cfg := unitConfig(8, 32, 0.5)
	zb, err := PlanZeroBubble(cfg)
	if err != nil {
		t.Fatal(err)
	}
	dp, err := PlanDualPipe(cfg)
	if err != nil {
		t.Fatal(err)
	}
	overlapped := 0
	for _, slots := range dp.Slots {
		for _, sl := range slots {
			if sl.Phase == PhaseOverlappedFwdBwd {
				overlapped++
				if sl.EndUs-sl.StartUs != cfg.FwdUs+cfg.BwdInputUs {
					t.Errorf("overlapped slot lasts %v, want %v", sl.EndUs-sl.StartUs, cfg.FwdUs+cfg.BwdInputUs)
				}
			}
		}
	}
	if overlapped == 0 {
		t.Fatal("dualpipe produced no overlapped slots")
	}
	if dp.MakespanUs >= zb.MakespanUs {
		t.Errorf("dualpipe makespan %v, zero bubble %v: all-to-all not hidden", dp.MakespanUs, zb.MakespanUs)
	}
}

func TestVerifyScheduleDAGMutants(t *testing.T) {
	cfg := unitConfig(4, 8, 0)
	find := func(s PipelineSchedule, ph OpPhase, stage, m int) *PipelineSlot {
		for i := range s.Slots[stage] {
			sl := &s.Slots[stage][i]
			if sl.Phase == ph && sl.MicrobatchID == m {
				return sl
			}
		}
		panic(fmt.Sprintf("no %v slot for microbatch %d on stage %d", ph, m, stage))
	}
	tests := []struct {
		name   string
		mutate func(PipelineSchedule)
	}{
		{"bw_before_bx", func(s PipelineSchedule) {
			// Run B_W(2, 5) in the slot of B_X(2, 5) and vice versa.
			x, w := find(s, PhaseBwdInput, 2, 5), find(s, PhaseBwdWeight, 2, 5)
			x.Phase, w.Phase = w.Phase, x.Phase
		}},
		{"fwd_out_of_order", func(s PipelineSchedule) {
			a, b := find(s, PhaseForward, 0, 2), find(s, PhaseForward, 1, 2)
			a.MicrobatchID, b.MicrobatchID = 7, 7
		}},
		{"overlap", func(s PipelineSchedule) {
			s.Slots[3][1].StartUs -= 0.5
		}},
		{"short_slot", func(s PipelineSchedule) {
			s.Slots[0][0].EndUs = s.Slots[0][0].StartUs + 0.5
		}},
		{"missing", func(s PipelineSchedule) {
			s.Slots[2] = s.Slots[2][:len(s.Slots[2])-1]
		}},
		{"wrong_stage", func(s PipelineSchedule) {
			s.Slots[1][0].Stage = 0
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, err := PlanZeroBubble(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := VerifyScheduleDAG(s, cfg); err != nil {
				t.Fatalf("unmutated: %v", err)
			}
			tc.mutate(s)
			if err := VerifyScheduleDAG(s, cfg); err == nil {
				t.Fatal("mutated schedule verified")
			} else {
				t.Log(err)
			}
		})
	}
}

// TestVerifyScheduleDAGRules checks each precedence rule on a hand-written
// two-stage, one-microbatch schedule that violates only that rule.
func TestVerifyScheduleDAGRules(t *testing.T) {
	cfg := unitConfig(2, 1, 0)
	slot := func(stage int, ph OpPhase, start float64) PipelineSlot {
		return PipelineSlot{Stage: stage, Phase: ph, PairMicrobatchID: -1, PairDirection: -1, StartUs: start, EndUs: start + 1}
	}
	F, B, W := PhaseForward, PhaseBwdInput, PhaseBwdWeight
	tests := []struct {
		name   string
		s0, s1 []PipelineSlot
		ok     bool
	}{
		{"valid", []PipelineSlot{slot(0, F, 0), slot(0, B, 3), slot(0, W, 4)}, []PipelineSlot{slot(1, F, 1), slot(1, B, 2), slot(1, W, 3)}, true},
		{"rule1_f_order", []PipelineSlot{slot(0, F, 1), slot(0, B, 3), slot(0, W, 4)}, []PipelineSlot{slot(1, F, 1), slot(1, B, 2), slot(1, W, 3)}, false},
		{"rule2_b_before_last_f", []PipelineSlot{slot(0, F, 0), slot(0, B, 3), slot(0, W, 4)}, []PipelineSlot{slot(1, B, 1), slot(1, F, 2), slot(1, W, 3)}, false},
		{"rule3_upstream_b_first", []PipelineSlot{slot(0, F, 0), slot(0, B, 2), slot(0, W, 4)}, []PipelineSlot{slot(1, F, 1), slot(1, B, 2), slot(1, W, 3)}, false},
		{"rule4_w_before_b", []PipelineSlot{slot(0, F, 0), slot(0, W, 3), slot(0, B, 4)}, []PipelineSlot{slot(1, F, 1), slot(1, B, 2), slot(1, W, 3)}, false},
	}
	for _, tc := range tests {
		err := VerifyScheduleDAG(PipelineSchedule{Slots: [][]PipelineSlot{tc.s0, tc.s1}}, cfg)
		if (err == nil) != tc.ok {
			t.Errorf("%s: err = %v, want ok=%v", tc.name, err, tc.ok)
		}
	}
}

func TestDualPipeVerifierDirections(t *testing.T) {
	cfg := unitConfig(4, 8, 0.25)
	s, err := PlanDualPipe(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Flip one direction-1 slot to direction 0: its position changes, so
	// either a phase repeats or a dependency breaks.
	for st := range s.Slots {
		i := slices.IndexFunc(s.Slots[st], func(sl PipelineSlot) bool { return sl.Direction == 1 && sl.Phase == PhaseForward })
		if i >= 0 && st != 0 && st != cfg.NumStages-1 {
			s.Slots[st][i].Direction = 0
			break
		}
	}
	if err := VerifyScheduleDAG(s, cfg); err == nil {
		t.Fatal("direction flip verified")
	}
}

func TestScheduleConfigInvalid(t *testing.T) {
	for _, cfg := range []ScheduleConfig{
		{},
		{NumStages: 4, NumMicrobatch: 8},
		{NumStages: 4, NumMicrobatch: 8, FwdUs: 1, BwdInputUs: 1, BwdWeightUs: 1, CommAll2AllUs: -1},
	} {
		if _, err := PlanZeroBubble(cfg); err == nil {
			t.Errorf("PlanZeroBubble(%+v): no error", cfg)
		}
	}
	if _, err := PlanDualPipe(unitConfig(3, 8, 0)); err == nil {
		t.Error("PlanDualPipe with odd stages: no error")
	}
}

func TestFSDPOverlap(t *testing.T) {
	cfg := unitConfig(4, 16, 0)
	cfg.FSDPShardDeg = 8
	for _, plan := range []func(ScheduleConfig) (PipelineSchedule, error){PlanOneFOneB, PlanZeroBubble} {
		s, err := plan(cfg)
		if err != nil {
			t.Fatal(err)
		}
		ops, exposed, err := s.FSDPOverlap(cfg, 2, 3)
		if err != nil {
			t.Fatal(err)
		}
		if len(ops) != 2*cfg.NumStages {
			t.Fatalf("%d ops, want %d", len(ops), 2*cfg.NumStages)
		}
		for _, op := range ops {
			first := s.Slots[op.Stage][0].StartUs
			switch op.Kind {
			case "all_gather":
				if op.EndUs != first {
					t.Errorf("stage %d all-gather ends at %v, first slot starts at %v", op.Stage, op.EndUs, first)
				}
			case "reduce_scatter":
				if op.EndUs-op.StartUs != 3 {
					t.Errorf("reduce-scatter lasts %v", op.EndUs-op.StartUs)
				}
			}
		}
		if exposed < 0 || exposed > 3 {
			t.Errorf("exposed reduce-scatter %v outside [0, 3]", exposed)
		}
	}
}

func ExamplePlanZeroBubble() {
	cfg := ScheduleConfig{NumStages: 4, NumMicrobatch: 16, FwdUs: 1, BwdInputUs: 1, BwdWeightUs: 1}
	base, _ := PlanOneFOneB(cfg)
	zb, err := PlanZeroBubble(cfg)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := VerifyScheduleDAG(zb, cfg); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("1F1B makespan %.0f us, bubble %.3f\n", base.MakespanUs, base.BubbleFraction)
	fmt.Printf("ZB   makespan %.0f us, bubble %.3f, interior %.3f\n", zb.MakespanUs, zb.BubbleFraction, zb.InteriorBubbleFraction)
	// Output:
	// 1F1B makespan 57 us, bubble 0.158
	// ZB   makespan 51 us, bubble 0.059, interior 0.000
}
