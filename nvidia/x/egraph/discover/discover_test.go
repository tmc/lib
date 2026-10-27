package discover

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDiscoverSubByteIdiomAllTargets(t *testing.T) {
	targets := []string{
		"sm_90a",
		"sm_100a",
		"sm_103a",
		"sm_107a",
		"sm_107f",
		"sm_120a",
		"sm_121a",
	}
	kinds := []IdiomKind{
		IdiomE2M1Dequant,
		IdiomDynamicScaleQuant,
		IdiomResidualActEpilogue,
	}

	for _, target := range targets {
		for _, kind := range kinds {
			cands, winner, err := DiscoverSubByteIdiom(target, kind)
			if err != nil {
				t.Fatalf("DiscoverSubByteIdiom(%q, %q) failed: %v", target, kind, err)
			}
			if len(cands) == 0 {
				t.Fatalf("DiscoverSubByteIdiom(%q, %q) returned 0 candidates", target, kind)
			}
			if !winner.VerifiedExhaustive || !winner.MutantRejected {
				t.Fatalf("DiscoverSubByteIdiom(%q, %q) winner %q not exhaustively verified/mutant-rejected: %+v",
					target, kind, winner.Name, winner)
			}
			if winner.EvalFunc == nil {
				t.Fatalf("DiscoverSubByteIdiom(%q, %q) winner %q has nil EvalFunc", target, kind, winner.Name)
			}

			for _, c := range cands {
				if !c.VerifiedExhaustive || !c.MutantRejected {
					t.Errorf("candidate %s on %s missing verification flags: %+v", c.Name, target, c)
				}
				if target == "sm_90a" || target == "sm_121a" {
					for _, inst := range c.PTXSequence {
						if strings.Contains(inst, "cvt.rn.f16x2.e2m1x2") {
							t.Errorf("target %s illegally admitted cvt.rn.f16x2.e2m1x2 in candidate %s", target, c.Name)
						}
					}
				}
			}

			if kind == IdiomE2M1Dequant {
				switch target {
				case "sm_90a", "sm_121a":
					if winner.InstructionCount != 3 || !strings.Contains(strings.Join(winner.PTXSequence, "\n"), "prmt.b32") {
						t.Errorf("target %s E2M1 winner = %+v, want 3-instruction prmt.b32 LUT idiom", target, winner)
					}
				case "sm_100a", "sm_103a", "sm_120a":
					if !strings.Contains(strings.Join(winner.PTXSequence, "\n"), "cvt.rn.f16x2.e2m1x2") {
						t.Errorf("target %s E2M1 winner = %+v, want hardware cvt.rn.f16x2.e2m1x2", target, winner)
					}
				case "sm_107a", "sm_107f":
					if !strings.Contains(strings.Join(winner.PTXSequence, "\n"), "e2m1p4x4") {
						t.Errorf("target %s E2M1 winner = %+v, want PTX 9.4 quad-packed e2m1p4x4 idiom", target, winner)
					}
				}
			}

			if kind == IdiomDynamicScaleQuant && (target == "sm_107a" || target == "sm_107f") {
				if !strings.Contains(strings.Join(winner.PTXSequence, "\n"), "ue5m3x2") {
					t.Errorf("target %s DynamicScaleQuant winner = %+v, want PTX 9.4 scaled ue5m3x2 instruction", target, winner)
				}
			}
		}
	}
}

func TestDiscoverMacroScheduleWithOnlineFeedback(t *testing.T) {
	targets := []string{"sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_121a"}
	prefill := WorkloadSpec{
		Name:            "llama3_prefill_4096",
		M:               4096,
		N:               4096,
		K:               4096,
		BatchSize:       1,
		IsDecode:        false,
		AllowSub4BitLUT: true,
		AllowSparse2to4: true,
	}
	decode := WorkloadSpec{
		Name:            "llama3_decode_bs16",
		M:               16,
		N:               4096,
		K:               4096,
		BatchSize:       16,
		IsDecode:        true,
		AllowSub4BitLUT: true,
		AllowSparse2to4: false,
	}

	for _, target := range targets {
		winP, listP, err := DiscoverMacroSchedule(target, prefill, nil)
		if err != nil || len(listP) == 0 {
			t.Fatalf("DiscoverMacroSchedule(%q, prefill, nil) failed: %v", target, err)
		}
		if winP.Target != target || winP.EstimatedLatencyUS <= 0 {
			t.Errorf("unexpected prefill winner for %s: %+v", target, winP)
		}

		winD, listD, err := DiscoverMacroSchedule(target, decode, nil)
		if err != nil || len(listD) == 0 {
			t.Fatalf("DiscoverMacroSchedule(%q, decode, nil) failed: %v", target, err)
		}
		if winD.Target != target || winD.EstimatedLatencyUS <= 0 {
			t.Errorf("unexpected decode winner for %s: %+v", target, winD)
		}
	}

	// 1. Register spill feedback on sm_90a must enforce ProducerMaxRegs=40, ConsumerMaxRegs=168
	// and shift away from the high-register 232-reg schedule.
	base90, _, err := DiscoverMacroSchedule("sm_90a", prefill, nil)
	if err != nil {
		t.Fatal(err)
	}
	spillFB := &OnlineStallFeedback{
		DominantStall: "register_spill",
		SpillBytes:    64,
	}
	spill90, _, err := DiscoverMacroSchedule("sm_90a", prefill, spillFB)
	if err != nil {
		t.Fatal(err)
	}
	if spill90.ProducerMaxRegs != 40 || spill90.ConsumerMaxRegs != 168 {
		t.Errorf("sm_90a spill feedback regs = (%d, %d), want (40, 168)",
			spill90.ProducerMaxRegs, spill90.ConsumerMaxRegs)
	}
	if spill90.Name == base90.Name && base90.ConsumerMaxRegs > 168 {
		t.Errorf("expected sm_90a spill feedback to shift schedule away from %s", base90.Name)
	}

	// 2. Memory-bound (long scoreboard) feedback on sm_107a must promote deeper pipeline,
	// .sync_restrict, and LUT decompression on decode/sub-4-bit workloads.
	memFB := &OnlineStallFeedback{
		DominantStall:       "long_scoreboard",
		MeasuredLongSBRatio: 0.52,
	}
	mem107, _, err := DiscoverMacroSchedule("sm_107a", prefill, memFB)
	if err != nil {
		t.Fatal(err)
	}
	if mem107.PipelineStages < 4 || !mem107.UseEarlyReleaseSyncRestrict {
		t.Errorf("sm_107a memory feedback schedule = %+v, want PipelineStages >= 4 and UseEarlyReleaseSyncRestrict", mem107)
	}
	if !strings.Contains(mem107.MMAInstruction, "decompress::lut::b") {
		t.Errorf("sm_107a memory feedback MMA = %q, want decompress::lut::b", mem107.MMAInstruction)
	}

	// 3. Math-pipe throttle feedback on sm_107a with AllowSparse2to4 must promote
	// tcgen05.mma.sp + UseLdRedSpCompress.
	mathFB := &OnlineStallFeedback{
		DominantStall:         "math_pipe_throttle",
		MeasuredMathPipeRatio: 0.60,
	}
	math107, _, err := DiscoverMacroSchedule("sm_107a", prefill, mathFB)
	if err != nil {
		t.Fatal(err)
	}
	if !math107.UseLdRedSpCompress || !strings.Contains(math107.MMAInstruction, "tcgen05.mma.sp") {
		t.Errorf("sm_107a math feedback schedule = %+v, want UseLdRedSpCompress and tcgen05.mma.sp", math107)
	}
}

func TestRulebookJSONRoundTrip(t *testing.T) {
	_, idiom, err := DiscoverSubByteIdiom("sm_121a", IdiomE2M1Dequant)
	if err != nil {
		t.Fatal(err)
	}
	sched, _, err := DiscoverMacroSchedule("sm_121a", WorkloadSpec{
		Name:      "gb10_decode",
		M:         1,
		N:         4096,
		K:         4096,
		BatchSize: 8,
		IsDecode:  true,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}

	var rb Rulebook
	rb.Record("sm_121a:gb10_decode", sched, idiom)

	data, err := json.Marshal(&rb)
	if err != nil {
		t.Fatalf("json.Marshal(Rulebook) failed: %v", err)
	}

	var roundTrip Rulebook
	if err := json.Unmarshal(data, &roundTrip); err != nil {
		t.Fatalf("json.Unmarshal(Rulebook) failed: %v", err)
	}

	gotSched, ok := roundTrip.Entries["sm_121a:gb10_decode"]
	if !ok || gotSched.Name != sched.Name || gotSched.MMAInstruction != sched.MMAInstruction {
		t.Fatalf("round-trip schedule mismatch: got %+v, want %+v", gotSched, sched)
	}
	gotIdiom, ok := roundTrip.Idioms["sm_121a:gb10_decode"]
	if !ok || gotIdiom.Name != idiom.Name || !gotIdiom.VerifiedExhaustive || !gotIdiom.MutantRejected {
		t.Fatalf("round-trip idiom mismatch: got %+v, want %+v", gotIdiom, idiom)
	}
	if gotIdiom.EvalFunc == nil || gotIdiom.EvalFunc(0x21) != idiom.EvalFunc(0x21) {
		t.Fatalf("restored EvalFunc mismatch on input 0x21")
	}
}
