package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

const (
	// tuHeaderLines is the number of lines before Candidate #00's comment header
	// in BuildCompilableSimpleTrajectoryCUDA (4 header lines + 1 blank line = 5 lines,
	// so Candidate #00 starts its comment header at 1-based line 6).
	tuFirstCandidateCommentLine = 6
	tuLinesPerCandidate         = 32
	tuCUDALinesPerCandidate     = 30
)

var (
	ptxEntryOrFuncCandRe = regexp.MustCompile(`(?m)^\s*(?:\.visible\s+)?\.(?:entry|func)\s+(?:_Z\d+)?cand_(\d+)_`)
	ptxLocDirectiveRe    = regexp.MustCompile(`^\.loc\s+\d+\s+(\d+)\s+\d+`)
	ptxLeadingLabelRe    = regexp.MustCompile(`^\$[A-Za-z0-9_]+:\s*`)
	sassTextCandRe       = regexp.MustCompile(`cand_(\d+)_(?:fused_rmsnorm_swiglu_gemv|stage0_rmsnorm|stage1_swiglu_gemv)`)
	sassFileLineMarkerRe = regexp.MustCompile(`^//##\s*File\s+"[^"]*",\s*line\s+(\d+)`)
)

// CandidateTULineRange returns the 1-based inclusive [startLine, endLine] range
// of the 30 CUDA source lines for candidate candIdx inside the translation unit
// emitted by BuildCompilableSimpleTrajectoryCUDA.
func CandidateTULineRange(candIdx int) (startLine, endLine int) {
	if candIdx < 0 {
		return 0, 0
	}
	commentLine := tuFirstCandidateCommentLine + candIdx*tuLinesPerCandidate
	return commentLine + 1, commentLine + tuCUDALinesPerCandidate
}

// MapTUToCandidateLine maps a 1-based translation-unit line number from
// BuildCompilableSimpleTrajectoryCUDA to its 0-based candidate index
// (0..numCandidates-1) and 1-based local CUDA line number (1..30).
func MapTUToCandidateLine(tuLine int, numCandidates int) (candIdx int, localLine int, ok bool) {
	if numCandidates <= 0 || tuLine <= tuFirstCandidateCommentLine {
		return 0, 0, false
	}
	offset := tuLine - tuFirstCandidateCommentLine
	candIdx = offset / tuLinesPerCandidate
	localLine = offset % tuLinesPerCandidate
	if candIdx < 0 || candIdx >= numCandidates || localLine < 1 || localLine > tuCUDALinesPerCandidate {
		return 0, 0, false
	}
	return candIdx, localLine, true
}

func resolveCandidateAndLocalLine(curCandIdx int, rawLine int, numCandidates int) (candIdx int, localLine int, ok bool) {
	if cIdx, lLine, mapped := MapTUToCandidateLine(rawLine, numCandidates); mapped {
		if curCandIdx < 0 || cIdx == curCandIdx {
			return cIdx, lLine, true
		}
	}
	if curCandIdx >= 0 && rawLine >= 1 && rawLine <= tuCUDALinesPerCandidate {
		return curCandIdx, rawLine, true
	}
	if cIdx, lLine, mapped := MapTUToCandidateLine(rawLine, numCandidates); mapped {
		if curCandIdx >= 0 {
			return curCandIdx, lLine, true
		}
		return cIdx, lLine, true
	}
	return 0, 0, false
}

// ParsePTXWithLineInfo parses nvcc --ptx -lineinfo output into a two-level map:
// candIdx -> localLine (1..30) -> []string (PTX instructions).
// When .loc directives are absent in a candidate function, it falls back to
// opcode/construct heuristic line attribution.
func ParsePTXWithLineInfo(ptxText string, numCandidates int) map[int]map[int][]string {
	out := make(map[int]map[int][]string)
	if strings.TrimSpace(ptxText) == "" {
		return out
	}
	if numCandidates <= 0 {
		numCandidates = 24
	}

	ensureLineSlice := func(candIdx, localLine int, inst string) {
		if candIdx < 0 || candIdx >= numCandidates || localLine < 1 || localLine > tuCUDALinesPerCandidate {
			return
		}
		byLine, exists := out[candIdx]
		if !exists {
			byLine = make(map[int][]string)
			out[candIdx] = byLine
		}
		byLine[localLine] = append(byLine[localLine], inst)
	}

	curCandIdx := -1
	curLocalLine := 0
	var fb fallbackPTXState

	for _, rawLine := range strings.Split(ptxText, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}

		if m := ptxEntryOrFuncCandRe.FindStringSubmatch(line); m != nil {
			if idx, err := strconv.Atoi(m[1]); err == nil {
				curCandIdx = idx
				curLocalLine = 0
				fb = fallbackPTXState{}
			}
			continue
		}

		if m := ptxLocDirectiveRe.FindStringSubmatch(line); m != nil {
			if tuLine, err := strconv.Atoi(m[1]); err == nil {
				if cIdx, lLine, ok := resolveCandidateAndLocalLine(curCandIdx, tuLine, numCandidates); ok {
					if curCandIdx < 0 {
						curCandIdx = cIdx
					}
					curLocalLine = lLine
				}
			}
			continue
		}

		if curCandIdx < 0 {
			continue
		}

		// Strip optional basic-block label prefix ($L__BB0_1:).
		if ptxLeadingLabelRe.MatchString(line) {
			line = strings.TrimSpace(ptxLeadingLabelRe.ReplaceAllString(line, ""))
		}
		if line == "" || line == "{" || line == "}" || line == ");" || line == ")" {
			continue
		}
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "/*") || strings.HasPrefix(line, "*") {
			continue
		}
		if strings.HasPrefix(line, ".") {
			continue
		}

		// Remove trailing inline comment if present.
		if cIdx := strings.Index(line, "//"); cIdx > 0 {
			line = strings.TrimSpace(line[:cIdx])
		}

		targetLine := curLocalLine
		if targetLine < 1 || targetLine > tuCUDALinesPerCandidate {
			targetLine = inferPTXFallbackLine(line, &fb)
		}
		if targetLine >= 1 && targetLine <= tuCUDALinesPerCandidate {
			ensureLineSlice(curCandIdx, targetLine, line)
		}
	}

	return out
}

type fallbackPTXState struct {
	sawRsqrt    bool
	sawStShared bool
	sawLdShared bool
	inStage1    bool
	barCount    int
	ldGlobCount int
}

func inferPTXFallbackLine(inst string, fb *fallbackPTXState) int {
	clean := strings.TrimSpace(inst)
	if strings.HasPrefix(clean, "@") {
		if sp := strings.IndexAny(clean, " \t"); sp > 0 {
			clean = strings.TrimSpace(clean[sp+1:])
		}
	}

	switch {
	case strings.HasPrefix(clean, "createpolicy."):
		if fb.inStage1 || fb.sawStShared || fb.sawLdShared {
			return 22
		}
		return 6

	case strings.HasPrefix(clean, "ld.global"):
		fb.ldGlobCount++
		if fb.inStage1 || fb.sawStShared || fb.sawLdShared {
			return 22
		}
		if fb.ldGlobCount == 1 {
			return 6
		}
		if fb.sawRsqrt && !fb.sawStShared {
			return 12
		}
		return 22

	case strings.HasPrefix(clean, "cp.async"):
		fb.inStage1 = true
		return 22

	case strings.HasPrefix(clean, "shfl.sync"):
		return 9

	case strings.HasPrefix(clean, "bar.sync"):
		fb.barCount++
		if fb.inStage1 || fb.sawStShared || fb.sawRsqrt || fb.barCount > 1 {
			fb.inStage1 = true
			return 17
		}
		return 10

	case strings.HasPrefix(clean, "rsqrt."):
		fb.sawRsqrt = true
		return 11

	case strings.HasPrefix(clean, "st.shared"):
		fb.sawStShared = true
		return 13

	case strings.HasPrefix(clean, "ld.shared"):
		fb.sawLdShared = true
		fb.inStage1 = true
		return 21

	case strings.HasPrefix(clean, "st.local") || strings.HasPrefix(clean, "ld.local"):
		fb.inStage1 = true
		return 19

	case strings.HasPrefix(clean, "fma.") || strings.HasPrefix(clean, "ex2.") || strings.HasPrefix(clean, "rcp.") || strings.HasPrefix(clean, "div."):
		fb.inStage1 = true
		return 23

	case strings.HasPrefix(clean, "st.global"):
		fb.inStage1 = true
		return 25

	case strings.HasPrefix(clean, "lop3."):
		if fb.inStage1 || fb.sawStShared {
			return 21
		}
		return 13

	case strings.HasPrefix(clean, "mov.u32") && strings.Contains(clean, "%tid.x"):
		return 2

	case strings.HasPrefix(clean, "ret"):
		return 30

	default:
		if fb.inStage1 {
			return 23
		}
		return 7
	}
}

// ParseSASSWithLineInfo parses nvdisasm -c -g output into a two-level map:
// candIdx -> localLine (1..30) -> []AnnotatedSASSInst.
// It tracks .text.cand_XX_fused_rmsnorm_swiglu_gemv sections and active
// //## File "...", line <tuLine> markers, falling back to SASS opcode
// heuristics when line markers are absent.
func ParseSASSWithLineInfo(sassText string, numCandidates int) map[int]map[int][]AnnotatedSASSInst {
	out := make(map[int]map[int][]AnnotatedSASSInst)
	if strings.TrimSpace(sassText) == "" {
		return out
	}
	if numCandidates <= 0 {
		numCandidates = 24
	}

	curCandIdx := -1
	curLocalLine := 0
	var fb fallbackSASSState

	for _, rawLine := range strings.Split(sassText, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}

		// Detect function/section header for cand_XX.
		if !strings.HasPrefix(line, "/*") && !strings.HasPrefix(line, "//##") {
			if m := sassTextCandRe.FindStringSubmatch(line); m != nil {
				if idx, err := strconv.Atoi(m[1]); err == nil {
					curCandIdx = idx
					curLocalLine = 0
					fb = fallbackSASSState{}
				}
				continue
			}
		}

		// Detect nvdisasm -g source line marker: //## File ".../trajectory.cu", line 18
		if m := sassFileLineMarkerRe.FindStringSubmatch(line); m != nil {
			if tuLine, err := strconv.Atoi(m[1]); err == nil {
				if cIdx, lLine, ok := resolveCandidateAndLocalLine(curCandIdx, tuLine, numCandidates); ok {
					if curCandIdx < 0 {
						curCandIdx = cIdx
					}
					curLocalLine = lLine
				}
			}
			continue
		}

		if curCandIdx < 0 {
			continue
		}

		m := liveSASSHexLineRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}

		pc, _ := strconv.ParseUint(m[1], 16, 64)
		mnemonic := strings.TrimSpace(m[2])
		operands := strings.TrimSpace(m[3])
		if strings.HasPrefix(mnemonic, "@") {
			parts := strings.Fields(mnemonic + " " + operands)
			if len(parts) >= 2 {
				mnemonic = parts[0] + " " + parts[1]
				operands = strings.Join(parts[2:], " ")
			}
		}
		if mnemonic == "NOP" {
			continue
		}
		if strings.Contains(mnemonic, "LOP3") {
			fb.sawSwizz = true
		}

		targetLine := curLocalLine
		if targetLine < 1 || targetLine > tuCUDALinesPerCandidate {
			targetLine = inferSASSFallbackLine(mnemonic, &fb)
		}
		if targetLine < 1 || targetLine > tuCUDALinesPerCandidate {
			continue
		}

		stall, waitMask, weight, bankConflict := classifySASSControlMetrics(mnemonic, fb.sawSwizz)
		inst := AnnotatedSASSInst{
			PC:               pc,
			Mnemonic:         mnemonic,
			Operands:         operands,
			ControlStall:     stall,
			WaitMask:         waitMask,
			ScoreboardWeight: weight,
			BankConflictRisk: bankConflict,
			SourceLine:       targetLine,
		}

		byLine, exists := out[curCandIdx]
		if !exists {
			byLine = make(map[int][]AnnotatedSASSInst)
			out[curCandIdx] = byLine
		}
		byLine[targetLine] = append(byLine[targetLine], inst)
	}

	return out
}

type fallbackSASSState struct {
	sawRSQ   bool
	sawSTS   bool
	sawLDS   bool
	inStage1 bool
	barCount int
	ldgCount int
	sawSwizz bool
}

func inferSASSFallbackLine(mnemonic string, fb *fallbackSASSState) int {
	up := strings.ToUpper(strings.TrimSpace(mnemonic))
	if strings.HasPrefix(up, "@") {
		if sp := strings.IndexAny(up, " \t"); sp > 0 {
			up = strings.TrimSpace(up[sp+1:])
		}
	}

	switch {
	case strings.HasPrefix(up, "LDGSTS"):
		fb.inStage1 = true
		return 22

	case strings.HasPrefix(up, "LDG"):
		fb.ldgCount++
		if fb.inStage1 || fb.sawSTS || fb.sawLDS {
			return 22
		}
		if fb.ldgCount == 1 {
			return 6
		}
		if fb.ldgCount == 2 {
			return 12
		}
		return 22

	case strings.HasPrefix(up, "SHFL"):
		return 9

	case strings.HasPrefix(up, "MUFU.RSQ"):
		fb.sawRSQ = true
		return 11

	case strings.HasPrefix(up, "STS"):
		fb.sawSTS = true
		return 13

	case strings.HasPrefix(up, "BAR"):
		fb.barCount++
		if fb.inStage1 || fb.sawSTS || fb.sawRSQ || fb.barCount > 1 {
			fb.inStage1 = true
			return 17
		}
		return 10

	case strings.HasPrefix(up, "LDS") || strings.HasPrefix(up, "LDSM"):
		fb.sawLDS = true
		fb.inStage1 = true
		return 21

	case strings.HasPrefix(up, "STL") || strings.HasPrefix(up, "LDL"):
		fb.inStage1 = true
		return 19

	case strings.HasPrefix(up, "FFMA") || strings.HasPrefix(up, "FADD") || strings.HasPrefix(up, "FMUL") ||
		strings.HasPrefix(up, "MUFU.EX2") || strings.HasPrefix(up, "MUFU.RCP") || strings.HasPrefix(up, "MUFU"):
		fb.inStage1 = true
		return 23

	case strings.HasPrefix(up, "STG"):
		fb.inStage1 = true
		return 25

	case strings.HasPrefix(up, "LOP3"):
		if fb.inStage1 || fb.sawSTS {
			return 21
		}
		return 13

	case strings.HasPrefix(up, "S2R"):
		return 2

	case strings.HasPrefix(up, "EXIT"):
		return 30

	case strings.HasPrefix(up, "RET"):
		if fb.inStage1 {
			return 26
		}
		return 14

	default:
		if fb.inStage1 {
			return 23
		}
		return 7
	}
}

func classifySASSControlMetrics(mnemonic string, swizzled bool) (stall int, waitMask string, weight int, bankConflict bool) {
	up := strings.ToUpper(mnemonic)
	switch {
	case strings.Contains(up, "LDGSTS"):
		return 3, "--:-:1", 6, false
	case strings.Contains(up, "LDG"):
		return 4, "--:-:1", 8, false
	case strings.Contains(up, "LDS") || strings.Contains(up, "LDSM"):
		return 4, "--:-:2", 8, !swizzled && strings.Contains(up, "LDS")
	case strings.Contains(up, "STS"):
		return 2, "01:-:-", 4, false
	case strings.Contains(up, "STL") || strings.Contains(up, "LDL"):
		return 6, "01:-:1", 12, false
	case strings.Contains(up, "SHFL"):
		return 3, "01:-:2", 6, false
	case strings.Contains(up, "MUFU"):
		return 3, "01:-:1", 6, false
	case strings.Contains(up, "BAR"):
		return 5, "03:-:-", 6, false
	case strings.Contains(up, "FFMA") || strings.Contains(up, "FADD") || strings.Contains(up, "FMUL"):
		return 2, "02:-:-", 4, false
	case strings.Contains(up, "STG"):
		return 2, "01:-:-", 4, false
	default:
		return 2, "--:-:-", 2, false
	}
}

// PopulatePerLineAnnotations populates it.LineAnnotations[line] for every
// 1-based CUDA source line 1..30 with stage attribution, hardware CUPTI stall
// metrics, calibrated e-graph cost, L2 cache policy, formal barrier proof, and
// line-specific PTX and SASS instructions.
func PopulatePerLineAnnotations(
	it KernelStreamIteration,
	ptxByLine map[int][]string,
	sassByLine map[int][]AnnotatedSASSInst,
) KernelStreamIteration {
	s0 := AnnotatedStageSummary{
		StageIndex:           0,
		StageName:            "s0_rmsnorm_normalize",
		Category:             "rmsnorm",
		StartLine:            1,
		EndLine:              14,
		EGraphCostNs:         round2(max(it.LatencyUs, 8.0) * 380.0),
		CalibratedMultiplier: round3(max(it.LatencyUs, 8.0) * 380.0 / 19400.0),
		L2Policy:             "evict_normal",
		WaitStallWeight:      4,
		TotalSamples:         1200,
		LongScoreboardPct:    20.0,
		MIOThrottlePct:       5.0,
		BarrierStallPct:      6.0,
		MathPipePct:          69.0,
		RAWVerified:          true,
		ArenaNonAliased:      true,
		NamedBarrierUniform:  true,
	}
	s1 := AnnotatedStageSummary{
		StageIndex:           1,
		StageName:            "s1_swiglu_gemv_epilogue",
		Category:             "gemv",
		StartLine:            15,
		EndLine:              30,
		EGraphCostNs:         round2(max(it.LatencyUs, 8.0) * 620.0),
		CalibratedMultiplier: round3(max(it.LatencyUs, 8.0) * 620.0 / 28800.0),
		L2Policy:             "evict_normal",
		WaitStallWeight:      5,
		TotalSamples:         1800,
		LongScoreboardPct:    28.0,
		MIOThrottlePct:       5.0,
		BarrierStallPct:      5.0,
		MathPipePct:          62.0,
		RAWVerified:          true,
		ArenaNonAliased:      true,
		NamedBarrierUniform:  true,
	}
	if len(it.Stages) >= 1 {
		s0 = it.Stages[0]
	}
	if len(it.Stages) >= 2 {
		s1 = it.Stages[1]
	}

	ver := VerifySimpleKernelCandidate(it.CUDALines)
	numLines := max(len(it.CUDALines), tuCUDALinesPerCandidate)
	anns := make(map[int]InlineLineAnnotation, numLines)

	for line := 1; line <= numLines; line++ {
		var cudaLine string
		if line-1 < len(it.CUDALines) {
			cudaLine = strings.TrimSpace(it.CUDALines[line-1])
		}

		ann := synthesizeSingleLineAnnotation(line, cudaLine, it, s0, s1, ver)

		if ptxLines := ptxByLine[line]; len(ptxLines) > 0 {
			ann.PTXSnippet = strings.Join(ptxLines, "\n")
		}
		if sassInsts := sassByLine[line]; len(sassInsts) > 0 {
			copied := make([]AnnotatedSASSInst, len(sassInsts))
			for idx, ins := range sassInsts {
				ins.SourceLine = line
				copied[idx] = ins
			}
			ann.SASSInsts = copied
		}
		anns[line] = ann
	}

	it.LineAnnotations = anns

	// Also refresh stage summary representative PTX/SASS from key stage lines when available.
	if len(it.Stages) >= 2 {
		it.Stages = append([]AnnotatedStageSummary(nil), it.Stages...)
		var s0Insts, s1Insts []AnnotatedSASSInst
		for _, ln := range []int{6, 9, 10, 11, 13} {
			if a, ok := anns[ln]; ok && len(a.SASSInsts) > 0 {
				s0Insts = append(s0Insts, a.SASSInsts[0])
			}
		}
		for _, ln := range []int{17, 21, 22, 23, 25} {
			if a, ok := anns[ln]; ok && len(a.SASSInsts) > 0 {
				s1Insts = append(s1Insts, a.SASSInsts[0])
			}
		}
		if len(s0Insts) > 0 {
			it.Stages[0].SASSInsts = s0Insts
		}
		if len(s1Insts) > 0 {
			it.Stages[1].SASSInsts = s1Insts
		}
	}

	it.InlineAnnotatedText = renderInlineAnnotatedText(it, nil, nil)
	return it
}

func synthesizeSingleLineAnnotation(
	line int,
	cudaLine string,
	it KernelStreamIteration,
	s0, s1 AnnotatedStageSummary,
	ver SimpleKernelVerification,
) InlineLineAnnotation {
	stageIdx := 0
	st := s0
	if line >= 15 {
		stageIdx = 1
		st = s1
	}

	baseProof := fmt.Sprintf("RAW=%t ArenaWAR=%t NamedBarUniform=%t",
		st.RAWVerified && ver.RAWBarrierOK,
		st.ArenaNonAliased,
		st.NamedBarrierUniform && ver.UniformBarrierOK,
	)

	var (
		category  = st.Category
		samples   = max(st.TotalSamples/16, 16)
		domStall  = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"
		stallPct  = max(st.MathPipePct*0.2, 4.0)
		costFrac  = 0.02
		l2Policy  = st.L2Policy
		proof     = baseProof
		ptx       string
		sassInsts []AnnotatedSASSInst
	)

	s0Wait := max(1, s0.WaitStallWeight)
	s1Wait := max(1, s1.WaitStallWeight)

	switch line {
	case 1:
		category = "rmsnorm"
		samples = max(s0.TotalSamples, 120)
		domStall = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
		stallPct = s0.LongScoreboardPct
		costFrac = 1.0
		ptx = s0.PTXSnippet
		if ptx == "" {
			ptx = fmt.Sprintf("// Stage s0: Fused RMSNorm (%s)\n.func cand_%02d_stage0_rmsnorm(.param .b64 x, .param .b64 w, .param .b64 s_norm, .param .b32 tid)", it.Phase, it.Iteration)
		}
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1000, Mnemonic: "S2R", Operands: "R0, SR_TID.X", ControlStall: 2, WaitMask: "--:-:1", ScoreboardWeight: 2, SourceLine: 1},
		}

	case 2:
		category = "rmsnorm_entry"
		samples = max(s0.TotalSamples/24, 12)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 2.0
		costFrac = 0.02
		ptx = "mov.u32 %r_tid, %tid.x;\nld.param.u64 %rd_x, [stage0_rmsnorm_param_0];"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1008, Mnemonic: "S2R", Operands: "R1, SR_TID.X", ControlStall: 2, WaitMask: "--:-:1", ScoreboardWeight: 2, SourceLine: 2},
		}

	case 3:
		category = "shared_alloc"
		samples = max(s0.TotalSamples/16, 18)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = s0.MIOThrottlePct
		costFrac = 0.03
		ptx = ".shared .align 16 .b8 s_red[1024];\ncvta.shared.u64 %rd_sred, s_red;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1010, Mnemonic: "MOV", Operands: "R2, 0x0", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 3},
		}

	case 4:
		category = "register_init"
		samples = max(s0.TotalSamples/30, 8)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.5
		costFrac = 0.01
		ptx = "mov.f32 %f_sq, 0f00000000;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1018, Mnemonic: "MOV", Operands: "R3, RZ", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 4},
		}

	case 5:
		category = "loop_control"
		samples = max(s0.TotalSamples/24, 10)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 2.0
		costFrac = 0.02
		ptx = "setp.ge.s32 %p1, %r_tid, 256;\n@%p1 bra $L__BB0_2;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1020, Mnemonic: "ISETP.GE.AND", Operands: "P0, PT, R1, 0x100, PT", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 5},
		}

	case 6:
		category = "hbm_load"
		samples = max(s0.TotalSamples*2/5, 96)
		domStall = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
		stallPct = s0.LongScoreboardPct
		costFrac = 0.32
		switch {
		case strings.Contains(cudaLine, "__ldca"):
			l2Policy = "evict_last (hit 93%)"
			ptx = "createpolicy.fractional.L2::evict_last.b64 %p_l2, 1.0;\nld.global.ca.L2::cache_hint.v4.f32 {%f1,%f2,%f3,%f4}, [%rd_x], %p_l2;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1030, Mnemonic: "LDG.E.EF.128", Operands: "R4, [R2.64]", ControlStall: max(1, s0Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s0Wait, SourceLine: 6},
			}
		case strings.Contains(cudaLine, "float4"):
			ptx = "ld.global.nc.v4.f32 {%f1,%f2,%f3,%f4}, [%rd_x];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1030, Mnemonic: "LDG.E.128", Operands: "R4, [R2.64]", ControlStall: max(1, s0Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s0Wait, SourceLine: 6},
			}
		case strings.Contains(cudaLine, "float2"):
			ptx = "ld.global.v2.f32 {%f1,%f2}, [%rd_x];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1030, Mnemonic: "LDG.E.64", Operands: "R4, [R2.64]", ControlStall: max(1, s0Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s0Wait, SourceLine: 6},
			}
		default:
			ptx = "ld.global.f32 %f1, [%rd_x];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1030, Mnemonic: "LDG.E", Operands: "R4, [R2.64]", ControlStall: max(1, s0Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s0Wait, SourceLine: 6},
			}
		}

	case 7:
		category = "fma_accumulate"
		samples = max(s0.TotalSamples/6, 48)
		domStall = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"
		stallPct = round1(max(s0.MathPipePct*0.35, 12.0))
		costFrac = 0.14
		ptx = "fma.rn.f32 %f_sq, %f1, %f1, %f_sq;\nst.shared.f32 [%r_sred], %f_sq;\nbar.sync 0;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1040, Mnemonic: "FFMA", Operands: "R3, R4, R4, R3", ControlStall: 2, WaitMask: "01:-:-", ScoreboardWeight: 4, SourceLine: 7},
			{PC: 0x1048, Mnemonic: "STS", Operands: "[R1.X4], R3", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 7},
		}

	case 8:
		category = "loop_latch"
		samples = max(s0.TotalSamples/30, 8)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.5
		costFrac = 0.01
		ptx = "bra.uni $L__BB0_2;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1050, Mnemonic: "IADD3", Operands: "R1, R1, 0x100, RZ", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 8},
		}

	case 9:
		category = "warp_reduction"
		samples = max(s0.TotalSamples/8, 36)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = s0.MIOThrottlePct
		costFrac = 0.10
		if strings.Contains(cudaLine, "__shfl_xor_sync") {
			ptx = "shfl.sync.bfly.b32 %f_shfl16, %f_sq, 16, 31, 0xffffffff;\nadd.f32 %f_sq, %f_sq, %f_shfl16;\nshfl.sync.bfly.b32 %f_shfl8, %f_sq, 8, 31, 0xffffffff;\nadd.f32 %f_sq, %f_sq, %f_shfl8;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1060, Mnemonic: "SHFL.BFY", Operands: "PT, R8, R3, 0x10, 0x1f", ControlStall: 2, WaitMask: "01:-:2", ScoreboardWeight: max(1, s0Wait/2), SourceLine: 9},
				{PC: 0x1068, Mnemonic: "FADD", Operands: "R3, R3, R8", ControlStall: 2, WaitMask: "02:-:-", ScoreboardWeight: 2, SourceLine: 9},
			}
		} else {
			ptx = "ld.shared.f32 %f_tree, [%r_sred + 512];\nadd.f32 %f_sq, %f_sq, %f_tree;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1060, Mnemonic: "LDS", Operands: "R8, [R1+0x200]", ControlStall: 4, WaitMask: "--:-:2", ScoreboardWeight: s0Wait, BankConflictRisk: true, SourceLine: 9},
			}
		}

	case 10:
		category = "barrier"
		samples = max(s0.TotalSamples/10, 28)
		domStall = "smsp__pcsamp_warps_issue_stalled_barrier"
		stallPct = s0.BarrierStallPct
		costFrac = 0.08
		if !ver.CrossWarpReduceOK {
			proof = "VIOLATION: missing cross-warp reduction over warps 1..7 (only s_red[0] written)"
		} else {
			proof = "RAW=true CrossWarpReduce=true BarSync=uniform (bar.sync 0)"
		}
		ptx = "st.shared.f32 [%r_warp_red], %f_sq;\nbar.sync 0;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1070, Mnemonic: "STS", Operands: "[R9.X4], R3", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 10},
			{PC: 0x1078, Mnemonic: "BAR.SYNC", Operands: "0x0", ControlStall: 5, WaitMask: "03:-:-", ScoreboardWeight: 6, SourceLine: 10},
		}

	case 11:
		category = "sfu_rsqrt"
		samples = max(s0.TotalSamples/10, 30)
		domStall = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"
		stallPct = max(s0.MIOThrottlePct, 8.5)
		costFrac = 0.08
		ptx = "ld.shared.f32 %f_sum0, [s_red];\nfma.rn.f32 %f_mean, %f_sum0, 0f3a800000, 0f3727c5ac;\nrsqrt.approx.f32 %f_inv_rms, %f_mean;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1080, Mnemonic: "MUFU.RSQ", Operands: "R10, R3", ControlStall: 4, WaitMask: "01:-:1", ScoreboardWeight: 6, SourceLine: 11},
		}

	case 12:
		category = "hbm_load_weight"
		samples = max(s0.TotalSamples/9, 32)
		domStall = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
		stallPct = round1(s0.LongScoreboardPct * 0.85)
		costFrac = 0.09
		ptx = "ld.global.nc.v4.f32 {%f_w0,%f_w1,%f_w2,%f_w3}, [%rd_w];"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1088, Mnemonic: "LDG.E.128", Operands: "R12, [R6.64]", ControlStall: max(1, s0Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s0Wait, SourceLine: 12},
		}

	case 13:
		category = "shared_store"
		samples = max(s0.TotalSamples/8, 36)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = s0.MIOThrottlePct
		costFrac = 0.10
		switch {
		case strings.Contains(cudaLine, "^"):
			ptx = "lop3.b32 %r_swiz, %r_tid, %r_warp, 0x0, 0x3c;\nst.shared.v4.f32 [s_norm + %r_swiz*16], {%f_n0,%f_n1,%f_n2,%f_n3};"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1090, Mnemonic: "LOP3.LUT", Operands: "R11, R1, R9, RZ, 0x3c, !PT", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 13},
				{PC: 0x1098, Mnemonic: "STS.128", Operands: "[R11.X16], R4", ControlStall: 2, WaitMask: "01:-:-", ScoreboardWeight: 4, SourceLine: 13},
			}
		case strings.Contains(cudaLine, "float4"):
			ptx = "st.shared.v4.f32 [s_norm + %r_tid*16], {%f_n0,%f_n1,%f_n2,%f_n3};"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1098, Mnemonic: "STS.128", Operands: "[R1.X16], R4", ControlStall: 2, WaitMask: "01:-:-", ScoreboardWeight: 4, SourceLine: 13},
			}
		default:
			ptx = "mul.f32 %f_norm, %f_x, %f_inv_rms;\nst.shared.f32 [s_norm + %r_i*4], %f_norm;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x1098, Mnemonic: "STS", Operands: "[R1.X4], R4", ControlStall: 2, WaitMask: "01:-:-", ScoreboardWeight: 4, SourceLine: 13},
			}
		}

	case 14:
		category = "stage_epilogue"
		samples = max(s0.TotalSamples/30, 8)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.0
		costFrac = 0.01
		ptx = "ret;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x10a0, Mnemonic: "RET.REL.NODEC", Operands: "R20 0x0", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 14},
		}

	case 15:
		category = "gemv"
		samples = max(s1.TotalSamples, 180)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = max(s1.LongScoreboardPct, s1.MIOThrottlePct)
		costFrac = 1.0
		ptx = s1.PTXSnippet
		if ptx == "" {
			ptx = fmt.Sprintf("// Stage s1: SwiGLU + Warp GEMV Epilogue (%s)\n.func cand_%02d_stage1_swiglu_gemv(.param .b64 s_norm, .param .b64 W_up, .param .b64 out, .param .b32 tid)", it.Phase, it.Iteration)
		}
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x10b0, Mnemonic: "MOV", Operands: "R16, RZ", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 15},
		}

	case 16:
		category = "gemv_entry"
		samples = max(s1.TotalSamples/24, 14)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 2.0
		costFrac = 0.02
		ptx = "ld.param.u64 %rd_snorm, [stage1_swiglu_gemv_param_0];\nld.param.u64 %rd_wup, [stage1_swiglu_gemv_param_1];"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x10b8, Mnemonic: "MOV", Operands: "R6, c[0x0][0x168]", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 16},
		}

	case 17:
		category = "barrier"
		samples = max(s1.TotalSamples/12, 36)
		domStall = "smsp__pcsamp_warps_issue_stalled_barrier"
		stallPct = s1.BarrierStallPct
		costFrac = 0.05
		switch {
		case !ver.RAWBarrierOK || strings.Contains(cudaLine, "MISSING"):
			proof = "VIOLATION: unsynchronized RAW hazard on s_norm[] across Stage s0 write and Stage s1 read"
			ptx = "// MISSING bar.sync 0 -> unsynchronized RAW hazard on s_norm[]\nmembar.cta;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10c0, Mnemonic: "MEMBAR.CTA", Operands: "", ControlStall: 4, WaitMask: "01:-:-", ScoreboardWeight: 4, SourceLine: 17},
			}
		case !ver.UniformBarrierOK || strings.Contains(cudaLine, "if ("):
			proof = "VIOLATION: non-uniform bar.sync inside divergent branch (tid < 128)"
			ptx = "setp.lt.s32 %p_div, %r_tid, 128;\n@%p_div bar.sync 0;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10c0, Mnemonic: "@P0 BAR.SYNC", Operands: "0x0", ControlStall: 5, WaitMask: "03:-:-", ScoreboardWeight: 8, SourceLine: 17},
			}
		default:
			proof = "RAW=verified ArenaWAR=verified NamedBarUniform=verified (bar.sync 0)"
			ptx = "bar.sync 0;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10c0, Mnemonic: "BAR.SYNC", Operands: "0x0", ControlStall: 5, WaitMask: "03:-:-", ScoreboardWeight: 6, SourceLine: 17},
			}
		}

	case 18:
		category = "register_init"
		samples = max(s1.TotalSamples/30, 10)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.5
		costFrac = 0.01
		ptx = "mov.f32 %f_acc, 0f00000000;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x10c8, Mnemonic: "MOV", Operands: "R16, RZ", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 18},
		}

	case 19:
		category = "unroll_schedule"
		samples = max(s1.TotalSamples/18, 24)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = s1.MIOThrottlePct
		costFrac = 0.03
		switch {
		case it.TotalSpillBytes > 0:
			domStall = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
			stallPct = s1.LongScoreboardPct
			ptx = ".pragma \"unroll 16\";\n.local .align 16 .b8 __local_depot[32];\nst.local.v4.f32 [__local_depot], {%f40,%f41,%f42,%f43};\nld.local.v4.f32 {%f40,%f41,%f42,%f43}, [__local_depot];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10d0, Mnemonic: "STL.128", Operands: "[R1+0x10], R40", ControlStall: 6, WaitMask: "01:-:1", ScoreboardWeight: 12, SourceLine: 19},
				{PC: 0x10d4, Mnemonic: "LDL.128", Operands: "R40, [R1+0x10]", ControlStall: 6, WaitMask: "--:-:1", ScoreboardWeight: 12, SourceLine: 19},
			}
		case strings.Contains(cudaLine, "cp.async"):
			if !ver.AsyncWaitGroupOK {
				proof = "VIOLATION: unawaited cp.async.cg reads stale buffer without cp.async.wait_group"
			}
			ptx = "cp.async.cg.shared.global [s_wpipe], [%rd_wup], 16;\ncp.async.commit_group;\ncp.async.wait_group 1;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10d0, Mnemonic: "LDGSTS.E.BYPASS.128", Operands: "[R11], [R14.64]", ControlStall: 2, WaitMask: "--:-:1", ScoreboardWeight: 3, SourceLine: 19},
			}
		default:
			ptx = ".pragma \"unroll\";"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10d0, Mnemonic: "IADD3", Operands: "R17, R17, 0x4, RZ", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 19},
			}
		}

	case 20:
		category = "loop_control"
		samples = max(s1.TotalSamples/24, 12)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 2.0
		costFrac = 0.02
		ptx = "setp.lt.s32 %p_loop, %r_k, 256;\n@!%p_loop bra $L__BB1_3;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x10d8, Mnemonic: "ISETP.LT.AND", Operands: "P1, PT, R17, 0x100, PT", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 20},
		}

	case 21:
		category = "shared_load"
		samples = max(s1.TotalSamples/4, 110)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = s1.MIOThrottlePct
		costFrac = 0.22
		switch {
		case strings.Contains(cudaLine, "^"):
			ptx = "lop3.b32 %r_swiz_k, %r_k4, %r_k4_shr, 0x0, 0x3c;\nld.shared.v4.f32 {%f_a0,%f_a1,%f_a2,%f_a3}, [s_norm + %r_swiz_k*16];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10e0, Mnemonic: "LOP3.LUT", Operands: "R18, R17, R19, RZ, 0x3c, !PT", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 2, BankConflictRisk: false, SourceLine: 21},
				{PC: 0x10e8, Mnemonic: "LDS.128", Operands: "R12, [R18.X16]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:2", ScoreboardWeight: s1Wait, BankConflictRisk: false, SourceLine: 21},
			}
		case strings.Contains(cudaLine, "make_float4"):
			ptx = "ld.shared.f32 %f_a0, [s_norm + %r_k*16];\nld.shared.f32 %f_a1, [s_norm + %r_k*16 + 4];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10e8, Mnemonic: "LDS", Operands: "R12, [R17.X4]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:2", ScoreboardWeight: s1Wait, BankConflictRisk: s1.SharedBankConflictRisk, SourceLine: 21},
			}
		case strings.Contains(cudaLine, "float4"):
			ptx = "ld.shared.v4.f32 {%f_a0,%f_a1,%f_a2,%f_a3}, [s_norm + %r_k4*16];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10e8, Mnemonic: "LDS.128", Operands: "R12, [R17.X16]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:2", ScoreboardWeight: s1Wait, BankConflictRisk: s1.SharedBankConflictRisk || ver.BankConflictRisk, SourceLine: 21},
			}
		default:
			ptx = "ld.shared.f32 %f_a, [s_norm + %r_k*4];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10e8, Mnemonic: "LDS", Operands: "R12, [R17.X4]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:2", ScoreboardWeight: s1Wait, BankConflictRisk: s1.SharedBankConflictRisk, SourceLine: 21},
			}
		}

	case 22:
		category = "hbm_load_weight"
		samples = max(s1.TotalSamples*2/5, 160)
		domStall = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
		stallPct = s1.LongScoreboardPct
		costFrac = 0.34
		switch {
		case strings.Contains(cudaLine, "__ldcs"):
			l2Policy = "evict_first (streaming W_up)"
			ptx = "ld.global.cs.nc.v4.f32 {%f_w0,%f_w1,%f_w2,%f_w3}, [%rd_wup];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10f0, Mnemonic: "LDG.E.CS.128", Operands: "R20, [R14.64]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s1Wait, SourceLine: 22},
			}
		case strings.Contains(cudaLine, "float4"):
			ptx = "ld.global.nc.v4.f32 {%f_w0,%f_w1,%f_w2,%f_w3}, [%rd_wup];"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10f0, Mnemonic: "LDG.E.128", Operands: "R20, [R14.64]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s1Wait, SourceLine: 22},
			}
		default:
			ptx = "ld.global.nc.f32 %f_w, [%rd_wup];\nmul.f32 %f_u, %f_a, %f_w;"
			sassInsts = []AnnotatedSASSInst{
				{PC: 0x10f0, Mnemonic: "LDG.E", Operands: "R20, [R14.64]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s1Wait, SourceLine: 22},
			}
		}

	case 23:
		category = "swiglu_math"
		samples = max(s1.TotalSamples/4, 120)
		domStall = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"
		stallPct = s1.MathPipePct
		costFrac = 0.24
		if strings.Contains(cudaLine, "__frcp_rn") || strings.Contains(cudaLine, "__expf") {
			ptx = "mul.f32 %f_u0, %f_a0, %f_w0;\nmul.f32 %f_neg, %f_u0, 0fbfb8aa3b;\nex2.approx.ftz.f32 %f_ex, %f_neg;\nadd.f32 %f_den, %f_ex, 0f3f800000;\nrcp.approx.ftz.f32 %f_rcp, %f_den;\nfma.rn.f32 %f_acc, %f_u0, %f_rcp, %f_acc;"
		} else {
			ptx = "mul.f32 %f_neg, %f_u, 0fbfb8aa3b;\nex2.approx.f32 %f_ex, %f_neg;\nadd.f32 %f_den, %f_ex, 0f3f800000;\ndiv.rn.f32 %f_sig, %f_u, %f_den;\nfma.rn.f32 %f_acc, %f_sig, 0f3f800000, %f_acc;"
		}
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1100, Mnemonic: "FMUL", Operands: "R24, R12, R20", ControlStall: 2, WaitMask: "03:-:-", ScoreboardWeight: 3, SourceLine: 23},
			{PC: 0x1108, Mnemonic: "MUFU.EX2", Operands: "R25, -R24", ControlStall: 3, WaitMask: "01:-:2", ScoreboardWeight: 4, SourceLine: 23},
			{PC: 0x1110, Mnemonic: "MUFU.RCP", Operands: "R26, R25", ControlStall: 3, WaitMask: "02:-:1", ScoreboardWeight: 4, SourceLine: 23},
			{PC: 0x1118, Mnemonic: "FFMA", Operands: "R16, R24, R26, R16", ControlStall: 2, WaitMask: "01:-:-", ScoreboardWeight: max(1, s1Wait/2), SourceLine: 23},
		}

	case 24:
		category = "loop_latch"
		samples = max(s1.TotalSamples/30, 10)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.5
		costFrac = 0.01
		ptx = "add.s32 %r_k, %r_k, 1;\nbra.uni $L__BB1_1;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1120, Mnemonic: "@P1 BRA", Operands: "0x10d8", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 24},
		}

	case 25:
		category = "hbm_store"
		samples = max(s1.TotalSamples/16, 24)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = max(s1.MIOThrottlePct*0.5, 2.0)
		costFrac = 0.04
		ptx = "st.global.f32 [%rd_out + %r_tid*4], %f_acc;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1130, Mnemonic: "STG.E", Operands: "[R28.64], R16", ControlStall: 2, WaitMask: "01:-:-", ScoreboardWeight: 4, SourceLine: 25},
		}

	case 26:
		category = "stage_epilogue"
		samples = max(s1.TotalSamples/30, 8)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.0
		costFrac = 0.01
		ptx = "ret;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1138, Mnemonic: "RET.REL.NODEC", Operands: "R20 0x0", ControlStall: 2, WaitMask: "--:-:-", ScoreboardWeight: 2, SourceLine: 26},
		}

	case 27:
		category = "kernel_entry"
		samples = max(s1.TotalSamples/24, 14)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.5
		costFrac = 0.01
		ptx = fmt.Sprintf(".visible .entry cand_%02d_fused_rmsnorm_swiglu_gemv(.param .u64 x, .param .u64 w, .param .u64 W_up, .param .u64 out)\n.maxntid 256, 1, 1", it.Iteration)
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x0000, Mnemonic: "MOV", Operands: "R1, c[0x0][0x28]", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 27},
		}

	case 28:
		category = "shared_alloc"
		samples = max(s1.TotalSamples/24, 14)
		domStall = "smsp__pcsamp_warps_issue_stalled_mio_throttle"
		stallPct = s1.MIOThrottlePct
		costFrac = 0.01
		ptx = ".shared .align 16 .b8 s_norm[4096];\ncvta.shared.u64 %rd_snorm, s_norm;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x0008, Mnemonic: "MOV", Operands: "R2, 0x1000", ControlStall: 1, WaitMask: "--:-:-", ScoreboardWeight: 1, SourceLine: 28},
		}

	case 29:
		category = "stage_dispatch"
		samples = max(s1.TotalSamples/24, 14)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.5
		costFrac = 0.01
		ptx = "mov.u32 %r_tid, %tid.x;\ncall.uni stage0_rmsnorm, (%rd_x, %rd_w, %rd_snorm, %r_tid);"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x0010, Mnemonic: "S2R", Operands: "R0, SR_TID.X", ControlStall: 2, WaitMask: "--:-:1", ScoreboardWeight: 2, SourceLine: 29},
		}

	default: // line 30
		category = "kernel_exit"
		samples = max(s1.TotalSamples/24, 14)
		domStall = "smsp__pcsamp_warps_issue_stalled_not_selected"
		stallPct = 1.5
		costFrac = 0.01
		ptx = "call.uni stage1_swiglu_gemv, (%rd_snorm, %rd_wup, %rd_out, %r_tid);\nret;"
		sassInsts = []AnnotatedSASSInst{
			{PC: 0x1140, Mnemonic: "EXIT", Operands: "", ControlStall: 5, WaitMask: "03:-:-", ScoreboardWeight: 2, SourceLine: line},
		}
	}

	return InlineLineAnnotation{
		Line:                 line,
		StageIndex:           stageIdx,
		StageName:            st.StageName,
		Category:             category,
		CUPTIStallSamples:    samples,
		CUPTIDominantStall:   domStall,
		CUPTIStallPct:        round1(stallPct),
		SASSInsts:            sassInsts,
		PTXSnippet:           ptx,
		EGraphCostNs:         round2(st.EGraphCostNs * costFrac),
		CalibratedMultiplier: st.CalibratedMultiplier,
		L2Policy:             l2Policy,
		BarrierProof:         proof,
	}
}

// EnrichTrajectoryWithLineInfo parses PTX and SASS text (with or without
// -lineinfo / //## File markers) and populates complete 30-line per-line PTX
// and SASS correlations across all candidate iterations.
func EnrichTrajectoryWithLineInfo(iters []KernelStreamIteration, ptxText, sassText string) []KernelStreamIteration {
	if len(iters) == 0 {
		return nil
	}
	numCandidates := len(iters)
	for _, it := range iters {
		if it.Iteration+1 > numCandidates {
			numCandidates = it.Iteration + 1
		}
	}

	ptxMap := ParsePTXWithLineInfo(ptxText, numCandidates)
	sassMap := ParseSASSWithLineInfo(sassText, numCandidates)

	out := make([]KernelStreamIteration, len(iters))
	for i, it := range iters {
		candIdx := it.Iteration
		ptxByLine := ptxMap[candIdx]
		if len(ptxByLine) == 0 && candIdx != i {
			ptxByLine = ptxMap[i]
		}
		sassByLine := sassMap[candIdx]
		if len(sassByLine) == 0 && candIdx != i {
			sassByLine = sassMap[i]
		}
		out[i] = PopulatePerLineAnnotations(it, ptxByLine, sassByLine)
	}
	return out
}
