package contextparallel

import (
	"fmt"
	"strings"
	"testing"
)

func TestParallel3DZeroBubbleStrictlyReducesBubble(t *testing.T) {
	for _, pp := range []int{4, 8} {
		for _, mb := range []int{2 * pp, 4 * pp} {
			t.Run(fmt.Sprintf("PP%d_MB%d", pp, mb), func(t *testing.T) {
				spec1F1B := Parallel3DSpec{
					PPStages:      pp,
					TPDegree:      8,
					CPDegree:      4,
					NumMicroBatch: mb,
					ZeroBubbleZBV: false,
					MicroStepUs:   100.0,
					P2PLatencyUs:  2.0,
				}
				sched1F1B, err := Compile3DParallelSchedule(spec1F1B)
				if err != nil {
					t.Fatalf("1F1B Compile3DParallelSchedule failed: %v", err)
				}

				specZBV := spec1F1B
				specZBV.ZeroBubbleZBV = true
				schedZBV, err := Compile3DParallelSchedule(specZBV)
				if err != nil {
					t.Fatalf("ZB-V Compile3DParallelSchedule failed: %v", err)
				}

				t.Logf("PP=%d MB=%d: 1F1B makespan=%.1f us bubble=%.4f | ZB-V makespan=%.1f us bubble=%.4f (reduction=%.2f%%)",
					pp, mb,
					sched1F1B.MakespanUs, sched1F1B.BubbleFraction,
					schedZBV.MakespanUs, schedZBV.BubbleFraction,
					schedZBV.BubbleReductionRatio*100.0)

				if schedZBV.BubbleFraction >= sched1F1B.BubbleFraction {
					t.Errorf("expected ZB-V BubbleFraction (%v) < 1F1B BubbleFraction (%v)",
						schedZBV.BubbleFraction, sched1F1B.BubbleFraction)
				}
				if schedZBV.MakespanUs >= sched1F1B.MakespanUs {
					t.Errorf("expected ZB-V MakespanUs (%v) < 1F1B MakespanUs (%v)",
						schedZBV.MakespanUs, sched1F1B.MakespanUs)
				}
				if schedZBV.BubbleReductionRatio <= 0 {
					t.Errorf("expected positive BubbleReductionRatio, got %v", schedZBV.BubbleReductionRatio)
				}
			})
		}
	}
}

func TestParallel3DTMPAndCPFusionAndBarriers(t *testing.T) {
	spec := Parallel3DSpec{
		PPStages:      4,
		TPDegree:      8,
		CPDegree:      4,
		NumMicroBatch: 8,
		ZeroBubbleZBV: true,
		MicroStepUs:   80.0,
		P2PLatencyUs:  1.5,
	}
	sched, err := Compile3DParallelSchedule(spec)
	if err != nil {
		t.Fatalf("Compile3DParallelSchedule failed: %v", err)
	}
	if sched.WorldSize != 4*8*4 {
		t.Errorf("WorldSize = %d, want 128", sched.WorldSize)
	}
	if len(sched.Stages) != 4 {
		t.Fatalf("len(Stages) = %d, want 4", len(sched.Stages))
	}

	// Each stage in ZB-V has 2 virtual chunks * 3 phases (F, B, W) * 8 microbatches = 48 micro-steps.
	wantStepsPerStage := 2 * 3 * spec.NumMicroBatch
	for st, steps := range sched.Stages {
		if len(steps) != wantStepsPerStage {
			t.Errorf("stage %d has %d steps, want %d", st, len(steps), wantStepsPerStage)
		}
		var countF, countB, countW int
		var seenChunk0, seenChunk1 bool
		for i, step := range steps {
			if i > 0 && step.StartUs < steps[i-1].EndUs-1e-9 {
				t.Errorf("stage %d step %d overlaps previous step: [%v, %v] vs [%v, %v]",
					st, i, steps[i-1].StartUs, steps[i-1].EndUs, step.StartUs, step.EndUs)
			}
			switch step.Phase {
			case PhaseF:
				countF++
			case PhaseB:
				countB++
			case PhaseW:
				countW++
			default:
				t.Errorf("unexpected phase %q", step.Phase)
			}
			if step.VirtualChunk == 0 {
				seenChunk0 = true
			}
			if step.VirtualChunk == 1 {
				seenChunk1 = true
			}
			if step.RingCPSteps != spec.CPDegree || !step.RingCPOverlapped {
				t.Errorf("stage %d step %d: expected RingCPSteps=%d overlapped=true, got %d/%v",
					st, i, spec.CPDegree, step.RingCPSteps, step.RingCPOverlapped)
			}
			if step.TPMultimemAllReduces != 2 || !strings.Contains(step.TPMultimemOp, "multimem") {
				t.Errorf("stage %d step %d: expected 2 TP multimem all-reduces, got %d (%q)",
					st, i, step.TPMultimemAllReduces, step.TPMultimemOp)
			}
		}
		if countF != 2*spec.NumMicroBatch || countB != 2*spec.NumMicroBatch || countW != 2*spec.NumMicroBatch {
			t.Errorf("stage %d F/B/W counts = %d/%d/%d, want %d each",
				st, countF, countB, countW, 2*spec.NumMicroBatch)
		}
		if !seenChunk0 || !seenChunk1 {
			t.Errorf("stage %d did not execute both V-shape virtual chunks (chunk0=%v, chunk1=%v)",
				st, seenChunk0, seenChunk1)
		}
	}

	if len(sched.Barriers) == 0 {
		t.Fatal("expected non-empty P2P .sys ready barriers")
	}
	for _, bar := range sched.Barriers {
		if bar.Scope != ".sys" {
			t.Errorf("barrier %d Scope = %q, want .sys", bar.BarrierID, bar.Scope)
		}
		if !strings.Contains(bar.ReleasePTX, "st.release.sys.global.u32") {
			t.Errorf("barrier %d ReleasePTX missing st.release.sys.global.u32: %q", bar.BarrierID, bar.ReleasePTX)
		}
		if !strings.Contains(bar.AcquireWaitPTX, "ld.acquire.sys.global.u32") {
			t.Errorf("barrier %d AcquireWaitPTX missing ld.acquire.sys.global.u32: %q", bar.BarrierID, bar.AcquireWaitPTX)
		}
		if bar.FromPPStage == bar.ToPPStage {
			t.Errorf("barrier %d should cross distinct PP stages, got %d -> %d",
				bar.BarrierID, bar.FromPPStage, bar.ToPPStage)
		}
	}

	if !strings.Contains(sched.SysBarrierPreludeCUDA, "st.release.sys.global.u32") ||
		!strings.Contains(sched.SysBarrierPreludeCUDA, "ld.acquire.sys.global.u32") {
		t.Errorf("SysBarrierPreludeCUDA missing .sys PTX primitives:\n%s", sched.SysBarrierPreludeCUDA)
	}
}

func TestParallel3DValidationErrors(t *testing.T) {
	badSpecs := []Parallel3DSpec{
		{PPStages: 0, TPDegree: 8, CPDegree: 2, NumMicroBatch: 8, MicroStepUs: 10},
		{PPStages: 4, TPDegree: 0, CPDegree: 2, NumMicroBatch: 8, MicroStepUs: 10},
		{PPStages: 4, TPDegree: 8, CPDegree: 0, NumMicroBatch: 8, MicroStepUs: 10},
		{PPStages: 4, TPDegree: 8, CPDegree: 2, NumMicroBatch: 0, MicroStepUs: 10},
		{PPStages: 4, TPDegree: 8, CPDegree: 2, NumMicroBatch: 8, MicroStepUs: 0},
		{PPStages: 4, TPDegree: 8, CPDegree: 2, NumMicroBatch: 8, MicroStepUs: 10, P2PLatencyUs: -1},
	}
	for i, s := range badSpecs {
		if _, err := Compile3DParallelSchedule(s); err == nil {
			t.Errorf("case %d: expected error for invalid spec %+v", i, s)
		}
	}
}
