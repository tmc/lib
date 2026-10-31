package main

import (
	"fmt"
	"strings"
	"testing"
)

func newTestSimpleTrajectory(t *testing.T) []KernelStreamIteration {
	t.Helper()
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations(simple, 1): %v", err)
	}
	if len(iters) != 24 {
		t.Fatalf("len(iters) = %d, want 24", len(iters))
	}
	return iters
}

func TestLineInfoMapTUToCandidateLine(t *testing.T) {
	iters := newTestSimpleTrajectory(t)
	tu := BuildCompilableSimpleTrajectoryCUDA(iters)
	tuLines := strings.Split(strings.TrimSuffix(tu, "\n"), "\n")

	// Header lines 1..5 and candidate comment lines must not map to a local CUDA line.
	for line := 1; line <= 6; line++ {
		if c, l, ok := MapTUToCandidateLine(line, len(iters)); ok {
			t.Fatalf("MapTUToCandidateLine(%d) = (%d, %d, true), want ok=false", line, c, l)
		}
	}

	for candIdx, it := range iters {
		startTU, endTU := CandidateTULineRange(candIdx)
		if wantStart := 7 + candIdx*32; startTU != wantStart {
			t.Fatalf("CandidateTULineRange(%d) start = %d, want %d", candIdx, startTU, wantStart)
		}
		if wantEnd := 36 + candIdx*32; endTU != wantEnd {
			t.Fatalf("CandidateTULineRange(%d) end = %d, want %d", candIdx, endTU, wantEnd)
		}

		// Verify comment line right before startTU does not map.
		commentTU := startTU - 1
		if _, _, ok := MapTUToCandidateLine(commentTU, len(iters)); ok {
			t.Fatalf("MapTUToCandidateLine(comment line %d) = ok, want false", commentTU)
		}
		if !strings.HasPrefix(tuLines[commentTU-1], fmt.Sprintf("// === Candidate #%02d:", candIdx)) {
			t.Fatalf("TU line %d = %q, want Candidate #%02d header", commentTU, tuLines[commentTU-1], candIdx)
		}

		// Verify all 30 CUDA lines of candidate candIdx map back to (candIdx, 1..30).
		for localLine := 1; localLine <= 30; localLine++ {
			tuLine := startTU + localLine - 1
			gotCand, gotLocal, ok := MapTUToCandidateLine(tuLine, len(iters))
			if !ok || gotCand != candIdx || gotLocal != localLine {
				t.Fatalf("MapTUToCandidateLine(%d) = (%d, %d, %v), want (%d, %d, true)",
					tuLine, gotCand, gotLocal, ok, candIdx, localLine)
			}
			// Verify the TU line corresponds to the candidate's local line text.
			origLine := it.CUDALines[localLine-1]
			if localLine == 1 && tuLines[tuLine-1] != origLine {
				t.Fatalf("cand %d line 1 mismatch: TU=%q vs orig=%q", candIdx, tuLines[tuLine-1], origLine)
			}
		}

		// Blank separator line after endTU must not map.
		if _, _, ok := MapTUToCandidateLine(endTU+1, len(iters)); ok {
			t.Fatalf("MapTUToCandidateLine(blank separator %d) = ok, want false", endTU+1)
		}
	}
}

func TestLineInfoParsePTXWithAndWithoutLoc(t *testing.T) {
	const numCands = 24
	cand23Start, _ := CandidateTULineRange(23)
	tuLine := func(local int) int {
		return cand23Start + local - 1
	}

	annotatedPTX := fmt.Sprintf(`
.version 8.5
.target sm_121a
.address_size 64
.file 1 "/tmp/egserve/trajectory.cu"

.visible .entry cand_23_fused_rmsnorm_swiglu_gemv(
	.param .u64 cand_23_param_0,
	.param .u64 cand_23_param_1
)
{
	.reg .f32 %%f<32>;
	.reg .b32 %%r<16>;
	.reg .b64 %%rd<16>;
	.loc 1 %d 9
	createpolicy.fractional.L2::evict_last.b64 %%rd10, 1.0;
	ld.global.ca.L2::cache_hint.v4.f32 {%%f1,%%f2,%%f3,%%f4}, [%%rd1], %%rd10;
	.loc 1 %d 5
	shfl.sync.bfly.b32 %%f5, %%f1, 16, 31, 0xffffffff;
	.loc 1 %d 5
	st.shared.f32 [%%r1], %%f5;
	bar.sync 0;
	.loc 1 %d 5
	rsqrt.approx.f32 %%f6, %%f5;
	.loc 1 %d 5
	lop3.b32 %%r2, %%r1, %%r3, 0x0, 0x3c;
	st.shared.v4.f32 [%%r2], {%%f1,%%f2,%%f3,%%f4};
	.loc 1 %d 5
	bar.sync 0;
	.loc 1 %d 9
	ld.shared.v4.f32 {%%f8,%%f9,%%f10,%%f11}, [%%r2];
	.loc 1 %d 9
	ld.global.cs.nc.v4.f32 {%%f12,%%f13,%%f14,%%f15}, [%%rd2];
	.loc 1 %d 9
	ex2.approx.ftz.f32 %%f16, %%f12;
	rcp.approx.ftz.f32 %%f17, %%f16;
	fma.rn.f32 %%f18, %%f8, %%f17, %%f18;
	.loc 1 %d 5
	st.global.f32 [%%rd3], %%f18;
	ret;
}
`, tuLine(6), tuLine(9), tuLine(10), tuLine(11), tuLine(13), tuLine(17), tuLine(21), tuLine(22), tuLine(23), tuLine(25))

	parsedLoc := ParsePTXWithLineInfo(annotatedPTX, numCands)
	c23 := parsedLoc[23]
	if len(c23) == 0 {
		t.Fatalf("ParsePTXWithLineInfo(.loc): missing candidate 23")
	}
	wantSubstrByLine := map[int]string{
		6:  "ld.global.ca.L2::cache_hint.v4.f32",
		9:  "shfl.sync.bfly.b32",
		10: "bar.sync 0",
		11: "rsqrt.approx.f32",
		13: "st.shared.v4.f32",
		17: "bar.sync 0",
		21: "ld.shared.v4.f32",
		22: "ld.global.cs.nc.v4.f32",
		23: "fma.rn.f32",
		25: "st.global.f32",
	}
	for line, wantSub := range wantSubstrByLine {
		joined := strings.Join(c23[line], "\n")
		if !strings.Contains(joined, wantSub) {
			t.Fatalf(".loc PTX line %d = %q, want substring %q", line, joined, wantSub)
		}
	}

	// Test fallback heuristic PTX line attribution when .loc directives are absent.
	unannotatedPTX := `
.visible .entry cand_07_fused_rmsnorm_swiglu_gemv(
	.param .u64 p0
)
{
	ld.global.nc.v4.f32 {%f1,%f2,%f3,%f4}, [%rd1];
	shfl.sync.bfly.b32 %f5, %f1, 16, 31, 0xffffffff;
	bar.sync 0;
	rsqrt.approx.f32 %f6, %f5;
	st.shared.v4.f32 [%r2], {%f1,%f2,%f3,%f4};
	bar.sync 0;
	ld.shared.f32 %f8, [%r2];
	ld.global.nc.f32 %f9, [%rd2];
	ex2.approx.f32 %f10, %f9;
	rcp.approx.f32 %f11, %f10;
	fma.rn.f32 %f12, %f8, %f11, %f12;
	st.global.f32 [%rd3], %f12;
	ret;
}
`
	parsedFallback := ParsePTXWithLineInfo(unannotatedPTX, numCands)
	c07 := parsedFallback[7]
	if len(c07) == 0 {
		t.Fatalf("ParsePTXWithLineInfo(fallback): missing candidate 7")
	}
	for _, ln := range []int{6, 9, 10, 11, 13, 17, 21, 22, 23, 25} {
		if len(c07[ln]) == 0 {
			t.Fatalf("fallback PTX missing line %d attribution: %+v", ln, c07)
		}
	}
}

func TestLineInfoParseSASSWithAndWithoutLineMarkers(t *testing.T) {
	const numCands = 24
	cand23Start, _ := CandidateTULineRange(23)
	tuLine := func(local int) int {
		return cand23Start + local - 1
	}

	annotatedSASS := fmt.Sprintf(`
	.section .text.cand_23_fused_rmsnorm_swiglu_gemv,"ax",@progbits
cand_23_fused_rmsnorm_swiglu_gemv:
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0010*/                   LDG.E.EF.128 R4, [R2.64] ;               /* 0x0000000002047981 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0020*/                   SHFL.BFY PT, R8, R4, 0x10, 0x1f ;        /* 0x0000001f10047389 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0030*/                   BAR.SYNC 0x0 ;                           /* 0x0000000000007b1d */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0040*/                   MUFU.RSQ R9, R8 ;                        /* 0x0000000508097308 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0050*/                   LDG.E.128 R10, [R6.64] ;                 /* 0x00000000060a7981 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0060*/                   LOP3.LUT R11, R1, R9, RZ, 0x3c, !PT ;    /* 0x00000000010b7212 */
        /*0070*/                   STS.128 [R11.X16], R4 ;                  /* 0x000000040b007388 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0080*/                   BAR.SYNC 0x0 ;                           /* 0x0000000000007b1d */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*0090*/                   LDS.128 R12, [R11.X16] ;                 /* 0x000000000b0c7984 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*00a0*/                   LDG.E.CS.128 R16, [R14.64] ;             /* 0x000000000e107981 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*00b0*/                   MUFU.EX2 R20, -R16 ;                     /* 0x0000000410147308 */
        /*00c0*/                   MUFU.RCP R21, R20 ;                      /* 0x0000000414157308 */
        /*00d0*/                   FFMA R24, R12, R21, R24 ;                /* 0x000000150c187223 */
//## File "/tmp/egserve/trajectory.cu", line %d
        /*00e0*/                   STG.E [R28.64], R24 ;                    /* 0x000000181c007386 */
`, tuLine(6), tuLine(9), tuLine(10), tuLine(11), tuLine(12), tuLine(13), tuLine(17), tuLine(21), tuLine(22), tuLine(23), tuLine(25))

	parsedSASS := ParseSASSWithLineInfo(annotatedSASS, numCands)
	c23 := parsedSASS[23]
	if len(c23) == 0 {
		t.Fatalf("ParseSASSWithLineInfo(annotated): missing candidate 23")
	}
	wantMnemonicByLine := map[int]string{
		6:  "LDG.E.EF.128",
		9:  "SHFL.BFY",
		10: "BAR.SYNC",
		11: "MUFU.RSQ",
		12: "LDG.E.128",
		13: "STS.128",
		17: "BAR.SYNC",
		21: "LDS.128",
		22: "LDG.E.CS.128",
		23: "FFMA",
		25: "STG.E",
	}
	for line, wantMnem := range wantMnemonicByLine {
		insts := c23[line]
		if len(insts) == 0 {
			t.Fatalf("annotated SASS line %d has no instructions", line)
		}
		var found bool
		for _, ins := range insts {
			if ins.SourceLine != line {
				t.Fatalf("annotated SASS line %d instruction %+v has SourceLine=%d", line, ins, ins.SourceLine)
			}
			if ins.Mnemonic == wantMnem {
				found = true
			}
		}
		if !found {
			t.Fatalf("annotated SASS line %d instructions %+v missing mnemonic %q", line, insts, wantMnem)
		}
	}

	// Test fallback opcode attribution when //## File markers are absent.
	unannotatedSASS := `
	.section .text.cand_12_fused_rmsnorm_swiglu_gemv,"ax",@progbits
cand_12_fused_rmsnorm_swiglu_gemv:
        /*0010*/                   LDG.E.128 R4, [R2.64] ;
        /*0020*/                   SHFL.BFY PT, R8, R4, 0x10, 0x1f ;
        /*0030*/                   BAR.SYNC 0x0 ;
        /*0040*/                   MUFU.RSQ R9, R8 ;
        /*0050*/                   LDG.E.128 R10, [R6.64] ;
        /*0060*/                   STS.128 [R11.X16], R4 ;
        /*0070*/                   BAR.SYNC 0x0 ;
        /*0080*/                   LDS.128 R12, [R11.X16] ;
        /*0090*/                   LDG.E.128 R16, [R14.64] ;
        /*00a0*/                   MUFU.EX2 R20, -R16 ;
        /*00b0*/                   MUFU.RCP R21, R20 ;
        /*00c0*/                   FFMA R24, R12, R21, R24 ;
        /*00d0*/                   STG.E [R28.64], R24 ;
`
	parsedFallback := ParseSASSWithLineInfo(unannotatedSASS, numCands)
	c12 := parsedFallback[12]
	if len(c12) == 0 {
		t.Fatalf("ParseSASSWithLineInfo(fallback): missing candidate 12")
	}
	for _, ln := range []int{6, 9, 10, 11, 12, 13, 17, 21, 22, 23, 25} {
		insts := c12[ln]
		if len(insts) == 0 {
			t.Fatalf("fallback SASS missing line %d attribution: %+v", ln, c12)
		}
		for _, ins := range insts {
			if ins.SourceLine != ln {
				t.Fatalf("fallback SASS line %d instruction has SourceLine=%d", ln, ins.SourceLine)
			}
		}
	}
}

func TestLineInfoEnrichTrajectoryCandidate23AndAll30Lines(t *testing.T) {
	iters := newTestSimpleTrajectory(t)

	cand23Start, _ := CandidateTULineRange(23)
	tu23 := func(local int) int {
		return cand23Start + local - 1
	}

	ptxText := fmt.Sprintf(`
.visible .entry cand_23_fused_rmsnorm_swiglu_gemv(.param .u64 p0)
{
	.loc 1 %d 9
	createpolicy.fractional.L2::evict_last.b64 %%p, 1.0;
	ld.global.ca.L2::cache_hint.v4.f32 {%%f1,%%f2,%%f3,%%f4}, [%%rd1], %%p;
	.loc 1 %d 5
	shfl.sync.bfly.b32 %%f5, %%f1, 16, 31, 0xffffffff;
	.loc 1 %d 5
	bar.sync 0;
	.loc 1 %d 5
	rsqrt.approx.f32 %%f6, %%f5;
	.loc 1 %d 5
	st.shared.v4.f32 [s_norm + %%r_swiz*16], {%%f1,%%f2,%%f3,%%f4};
	.loc 1 %d 5
	bar.sync 0;
	.loc 1 %d 9
	ld.shared.v4.f32 {%%f8,%%f9,%%f10,%%f11}, [s_norm + %%r_swiz_k*16];
	.loc 1 %d 9
	ld.global.cs.nc.v4.f32 {%%f12,%%f13,%%f14,%%f15}, [%%rd2];
	.loc 1 %d 9
	ex2.approx.ftz.f32 %%f16, %%f12;
	rcp.approx.ftz.f32 %%f17, %%f16;
	fma.rn.f32 %%f18, %%f8, %%f17, %%f18;
	.loc 1 %d 5
	st.global.f32 [%%rd3], %%f18;
}
`, tu23(6), tu23(9), tu23(10), tu23(11), tu23(13), tu23(17), tu23(21), tu23(22), tu23(23), tu23(25))

	sassText := fmt.Sprintf(`
	.section .text.cand_23_fused_rmsnorm_swiglu_gemv,"ax",@progbits
cand_23_fused_rmsnorm_swiglu_gemv:
//## File "trajectory.cu", line %d
        /*0010*/                   LDG.E.EF.128 R4, [R2.64] ;
//## File "trajectory.cu", line %d
        /*0020*/                   SHFL.BFY PT, R8, R4, 0x10, 0x1f ;
//## File "trajectory.cu", line %d
        /*0030*/                   BAR.SYNC 0x0 ;
//## File "trajectory.cu", line %d
        /*0040*/                   MUFU.RSQ R9, R8 ;
//## File "trajectory.cu", line %d
        /*0050*/                   LOP3.LUT R11, R1, R9, RZ, 0x3c, !PT ;
        /*0060*/                   STS.128 [R11.X16], R4 ;
//## File "trajectory.cu", line %d
        /*0070*/                   BAR.SYNC 0x0 ;
//## File "trajectory.cu", line %d
        /*0080*/                   LDS.128 R12, [R11.X16] ;
//## File "trajectory.cu", line %d
        /*0090*/                   LDG.E.CS.128 R16, [R14.64] ;
//## File "trajectory.cu", line %d
        /*00a0*/                   MUFU.EX2 R20, -R16 ;
        /*00b0*/                   MUFU.RCP R21, R20 ;
        /*00c0*/                   FFMA R24, R12, R21, R24 ;
//## File "trajectory.cu", line %d
        /*00d0*/                   STG.E [R28.64], R24 ;
`, tu23(6), tu23(9), tu23(10), tu23(11), tu23(13), tu23(17), tu23(21), tu23(22), tu23(23), tu23(25))

	enriched := EnrichTrajectoryWithLineInfo(iters, ptxText, sassText)
	if len(enriched) != 24 {
		t.Fatalf("len(enriched) = %d, want 24", len(enriched))
	}

	requiredLines := []int{1, 3, 6, 7, 9, 10, 11, 12, 13, 15, 17, 19, 21, 22, 23, 25, 27, 28, 29, 30}
	cand23 := enriched[23]
	for _, ln := range requiredLines {
		ann, ok := cand23.LineAnnotations[ln]
		if !ok {
			t.Fatalf("cand #23 missing LineAnnotations[%d]", ln)
		}
		if ann.Line != ln {
			t.Fatalf("cand #23 LineAnnotations[%d].Line = %d", ln, ann.Line)
		}
		wantStage := 0
		if ln >= 15 {
			wantStage = 1
		}
		if ann.StageIndex != wantStage {
			t.Fatalf("cand #23 line %d StageIndex = %d, want %d", ln, ann.StageIndex, wantStage)
		}
		if strings.TrimSpace(ann.PTXSnippet) == "" {
			t.Fatalf("cand #23 line %d has empty PTXSnippet", ln)
		}
		if len(ann.SASSInsts) == 0 {
			t.Fatalf("cand #23 line %d has empty SASSInsts", ln)
		}
		for _, ins := range ann.SASSInsts {
			if ins.SourceLine != ln {
				t.Fatalf("cand #23 line %d SASSInst %+v has SourceLine=%d, want %d", ln, ins, ins.SourceLine, ln)
			}
			if ins.Mnemonic == "" || ins.WaitMask == "" || ins.ScoreboardWeight <= 0 {
				t.Fatalf("cand #23 line %d invalid SASSInst: %+v", ln, ins)
			}
		}
		if ann.CUPTIStallSamples == 0 || ann.CUPTIDominantStall == "" || ann.CUPTIStallPct <= 0 {
			t.Fatalf("cand #23 line %d invalid CUPTI fields: %+v", ln, ann)
		}
		if ann.EGraphCostNs <= 0 || ann.CalibratedMultiplier <= 0 || ann.L2Policy == "" || ann.BarrierProof == "" {
			t.Fatalf("cand #23 line %d invalid cost/policy/proof fields: %+v", ln, ann)
		}
	}

	// Verify all 24 candidates have all 30 lines populated with matching SourceLine.
	for i, it := range enriched {
		if len(it.LineAnnotations) < 30 {
			t.Fatalf("cand #%02d len(LineAnnotations) = %d, want 30", i, len(it.LineAnnotations))
		}
		for ln := 1; ln <= 30; ln++ {
			ann, ok := it.LineAnnotations[ln]
			if !ok || ann.PTXSnippet == "" || len(ann.SASSInsts) == 0 {
				t.Fatalf("cand #%02d line %d incomplete annotation: %+v", i, ln, ann)
			}
			if ann.SASSInsts[0].SourceLine != ln {
				t.Fatalf("cand #%02d line %d SourceLine = %d, want %d", i, ln, ann.SASSInsts[0].SourceLine, ln)
			}
		}
	}
}
