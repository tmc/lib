package qualify

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math/bits"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// StageSASSMetrics summarizes static SASS instruction mix, scoreboard wait
// stall weight, memory coalescing, shared-memory bank conflict risk, local
// memory spill traffic, and tensor-pipe utilization for one kernel or stage.
type StageSASSMetrics struct {
	StageName              string `json:"stage_name"`
	TotalInsts             int    `json:"total_insts"`
	WaitStallWeight        int    `json:"wait_stall_weight"`
	UncoalescedGlobalLoads int    `json:"uncoalesced_global_loads"`
	SharedBankConflictRisk bool   `json:"shared_bank_conflict_risk"`
	LocalMemInsts          int    `json:"local_mem_insts"`
	TensorPipeInsts        int    `json:"tensor_pipe_insts"`
}

// SASSAnalyzer parses static SASS disassembly and optionally disassembles CUDA
// sources or cubin binaries via nvdisasm.
type SASSAnalyzer interface {
	ParseSASSDisassembly(sassText string) ([]StageSASSMetrics, error)
}

// SASSDisassembler is an optional extension interface implemented by
// [SASSAnalyzer] values that compile and disassemble CUDA source with nvcc and
// nvdisasm prior to calling [SASSAnalyzer.ParseSASSDisassembly].
type SASSDisassembler interface {
	DisassembleSASS(ctx context.Context, cudaSource, arch string) (string, error)
}

// NVDISASMSASSAnalyzer implements [SASSAnalyzer] and [SASSDisassembler]. When
// NVCC or NVDISASM are invoked via [NVDISASMSASSAnalyzer.DisassembleSASS], it
// compiles CUDA C++ to a temporary cubin and disassembles it with nvdisasm.
type NVDISASMSASSAnalyzer struct {
	NVCC       string
	NVDISASM   string
	ExtraFlags []string
}

// ParseSASSDisassembly implements [SASSAnalyzer] by delegating to
// [ParseSASSDisassembly].
func (a NVDISASMSASSAnalyzer) ParseSASSDisassembly(sassText string) ([]StageSASSMetrics, error) {
	return ParseSASSDisassembly(sassText)
}

// DisassembleSASS compiles cudaSource to a cubin for arch using nvcc and runs
// nvdisasm to obtain raw SASS disassembly text.
func (a NVDISASMSASSAnalyzer) DisassembleSASS(ctx context.Context, cudaSource, arch string) (string, error) {
	if strings.TrimSpace(cudaSource) == "" {
		return "", errors.New("disassemble sass: empty CUDA source")
	}
	arch = strings.TrimSpace(arch)
	if arch == "" {
		arch = "sm_90a"
	}
	nvcc := strings.TrimSpace(a.NVCC)
	if nvcc == "" {
		nvcc = "nvcc"
	}
	dir, err := os.MkdirTemp("", "qualify-sass-*")
	if err != nil {
		return "", fmt.Errorf("disassemble sass: mkdir temp: %w", err)
	}
	defer os.RemoveAll(dir)

	cuPath := filepath.Join(dir, "plan.cu")
	cubinPath := filepath.Join(dir, "plan.cubin")
	if err := os.WriteFile(cuPath, []byte(cudaSource), 0o600); err != nil {
		return "", fmt.Errorf("disassemble sass: write source: %w", err)
	}

	args := []string{"--cubin", "-arch=" + arch}
	args = append(args, a.ExtraFlags...)
	args = append(args, cuPath, "-o", cubinPath)

	cmd := exec.CommandContext(ctx, nvcc, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("disassemble sass (%s): nvcc: %w: %s", arch, err, strings.TrimSpace(stderr.String()))
	}

	cubinBytes, err := os.ReadFile(cubinPath)
	if err != nil {
		return "", fmt.Errorf("disassemble sass (%s): read cubin: %w", arch, err)
	}
	d := cudasass.Disassembler{Path: strings.TrimSpace(a.NVDISASM), TempDir: dir}
	out, err := d.Disassemble(ctx, cubinBytes, cudasass.Options{PrintCode: true, PrintHex: true})
	if err != nil {
		return "", fmt.Errorf("disassemble sass (%s): nvdisasm: %w", arch, err)
	}
	return string(out.Stdout), nil
}

const (
	// DefaultWaitStallThreshold is the scoreboard/barrier stall weight above
	// which RefinePlanByPTXASAndSASSLoop triggers pipeline stage reduction or
	// SplitK adjustment.
	DefaultWaitStallThreshold = 24
)

var (
	sassFuncHeaderRe = regexp.MustCompile(`(?i)^(?:\s*\.text\.([A-Za-z0-9_$]+)\s*:?|\s*Function\s*:\s*([A-Za-z0-9_$]+)|\s*//[-+\s]*\.text\.([A-Za-z0-9_$]+)\b|\s*\.global\s+([A-Za-z0-9_$]+)\s*$)`)
	sassStallBraceRe = regexp.MustCompile(`\{\s*!(\d+)\s*\}`)
	sassStallKVRe    = regexp.MustCompile(`(?i)\b(?:stall|wait)\s*=\s*(\d+)\b`)
	sassHexPairRe    = regexp.MustCompile(`/\*\s*(0x[0-9a-fA-F]+)\s*\*/\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/`)
	sassAddrPrefixRe = regexp.MustCompile(`^\s*/\*[0-9a-fA-F]+\*/\s*(.*)$`)
)

type rawSASSInst struct {
	opcode   string
	operands string
	rawLine  string
	loWord   uint64
	hiWord   uint64
	hasWords bool
}

// ParseSASSDisassembly parses nvdisasm or synthetic SASS disassembly text and
// returns per-stage [StageSASSMetrics].
func ParseSASSDisassembly(sassText string) ([]StageSASSMetrics, error) {
	if strings.TrimSpace(sassText) == "" {
		return nil, nil
	}

	type stageAccum struct {
		name         string
		insts        []rawSASSInst
		explicitSwiz bool
		explicitBank bool
	}

	var stages []*stageAccum
	var cur *stageAccum
	ensureStage := func(name string) *stageAccum {
		name = strings.TrimSpace(name)
		if name == "" {
			name = "stage_0"
		}
		for _, st := range stages {
			if st.name == name {
				return st
			}
		}
		st := &stageAccum{name: name}
		stages = append(stages, st)
		return st
	}

	lines := strings.Split(sassText, "\n")
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}

		if m := sassFuncHeaderRe.FindStringSubmatch(line); m != nil {
			fnName := firstNonEmpty(m[1], m[2], m[3], m[4])
			if fnName != "" {
				cur = ensureStage(fnName)
				continue
			}
		}

		lowLine := strings.ToLower(line)
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "#") {
			if cur != nil {
				if strings.Contains(lowLine, "bank conflict") || strings.Contains(lowLine, "bank_conflict") {
					cur.explicitBank = true
				}
				if strings.Contains(lowLine, "swizzle128b") || strings.Contains(lowLine, "swizzled") {
					cur.explicitSwiz = true
				}
			}
			continue
		}
		if strings.HasPrefix(line, ".") {
			continue
		}

		inst, ok := parseSingleSASSLine(rawLine)
		if !ok {
			continue
		}
		if cur == nil {
			cur = ensureStage("stage_0")
		}
		if strings.Contains(lowLine, "bank conflict") || strings.Contains(lowLine, "bank_conflict") {
			cur.explicitBank = true
		}
		if strings.Contains(lowLine, "swizzle") {
			cur.explicitSwiz = true
		}
		cur.insts = append(cur.insts, inst)
	}

	if len(stages) == 0 {
		// Fall back to cudasass.ParseString if structured differently.
		dis, err := cudasass.ParseString(sassText)
		if err != nil {
			return nil, err
		}
		for _, fn := range dis.Functions {
			st := ensureStage(fn.Name)
			for _, ins := range fn.Instructions {
				st.insts = append(st.insts, rawSASSInst{
					opcode:   ins.Opcode,
					operands: ins.Operands,
					rawLine:  ins.Opcode + " " + ins.Operands,
					loWord:   ins.Encoding.Lo,
					hiWord:   ins.Encoding.Hi,
					hasWords: ins.Encoding.Lo != 0 || ins.Encoding.Hi != 0,
				})
			}
		}
	}

	out := make([]StageSASSMetrics, 0, len(stages))
	for _, st := range stages {
		if len(st.insts) == 0 && len(stages) > 1 {
			continue
		}
		out = append(out, analyzeStageSASS(st.name, st.insts, st.explicitSwiz, st.explicitBank))
	}
	return out, nil
}

func parseSingleSASSLine(rawLine string) (rawSASSInst, bool) {
	var loWord, hiWord uint64
	var hasWords bool
	if m := sassHexPairRe.FindStringSubmatch(rawLine); m != nil {
		if lo, err1 := strconv.ParseUint(m[1], 0, 64); err1 == nil {
			if hi, err2 := strconv.ParseUint(m[2], 0, 64); err2 == nil {
				loWord, hiWord, hasWords = lo, hi, true
			}
		}
	}

	body := rawLine
	if m := sassAddrPrefixRe.FindStringSubmatch(rawLine); m != nil {
		body = m[1]
	}
	// Strip /* ... */ blocks from body while keeping rawLine for stall comment tags.
	for {
		s := strings.Index(body, "/*")
		if s < 0 {
			break
		}
		e := strings.Index(body[s+2:], "*/")
		if e < 0 {
			body = body[:s]
			break
		}
		body = body[:s] + " " + body[s+2+e+2:]
	}
	if semi := strings.IndexByte(body, ';'); semi >= 0 {
		body = body[:semi]
	}
	if sl := strings.Index(body, "//"); sl >= 0 {
		body = body[:sl]
	}
	body = strings.TrimSpace(body)
	if body == "" {
		return rawSASSInst{}, false
	}

	// Strip optional leading predicate guard such as @P0 or @!P1.
	if strings.HasPrefix(body, "@") {
		fields := strings.Fields(body)
		if len(fields) < 2 {
			return rawSASSInst{}, false
		}
		body = strings.TrimSpace(strings.TrimPrefix(body, fields[0]))
	}

	fields := strings.Fields(body)
	if len(fields) == 0 {
		return rawSASSInst{}, false
	}
	opcode := fields[0]
	if !isPlausibleSASSOpcode(opcode) {
		return rawSASSInst{}, false
	}
	operands := strings.TrimSpace(strings.TrimPrefix(body, opcode))
	return rawSASSInst{
		opcode:   opcode,
		operands: operands,
		rawLine:  rawLine,
		loWord:   loWord,
		hiWord:   hiWord,
		hasWords: hasWords,
	}, true
}

func isPlausibleSASSOpcode(tok string) bool {
	if tok == "" {
		return false
	}
	c0 := tok[0]
	if !((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z')) {
		return false
	}
	if strings.HasSuffix(tok, ":") || strings.ContainsAny(tok, "(){}=,;") {
		return false
	}
	up := strings.ToUpper(tok)
	switch up {
	case "FUNCTION", "SECTION", "GLOBAL", "TEXT", "VOID", "EXTERN", "STRUCT", "CONST", "RETURN", "IF", "FOR", "WHILE":
		return false
	}
	for i := 0; i < len(tok); i++ {
		ch := tok[i]
		if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' {
			continue
		}
		return false
	}
	return true
}

func analyzeStageSASS(name string, insts []rawSASSInst, explicitSwiz, explicitBank bool) StageSASSMetrics {
	m := StageSASSMetrics{
		StageName:  name,
		TotalInsts: len(insts),
	}

	var (
		hasSwizzleSeq   = explicitSwiz
		wideSharedOps   int
		totalSharedOps  int
		recentBitwiseOp int
	)

	for _, ins := range insts {
		upOp := strings.ToUpper(ins.opcode)
		baseOp := upOp
		if dot := strings.IndexByte(baseOp, '.'); dot >= 0 {
			baseOp = baseOp[:dot]
		}
		upOperands := strings.ToUpper(ins.operands)
		upRaw := strings.ToUpper(ins.rawLine)

		// 1. Track bitwise XOR / LOP3 / SHF address swizzling sequences.
		if baseOp == "LOP3" || baseOp == "XOR" || baseOp == "SHF" ||
			(strings.Contains(upOp, "LOP3") && strings.Contains(upOperands, "0X3C")) {
			recentBitwiseOp = 6
			if strings.Contains(upRaw, "SWIZZLE") || baseOp == "LOP3" || baseOp == "XOR" {
				hasSwizzleSeq = true
			}
		} else if recentBitwiseOp > 0 {
			recentBitwiseOp--
		}

		// 2. Local memory spill instructions (STL / LDL).
		if baseOp == "STL" || baseOp == "LDL" {
			m.LocalMemInsts++
		}

		// 3. Tensor-pipe instructions.
		if isTensorPipeOpcode(baseOp, upOp) {
			m.TensorPipeInsts++
		}

		// 4. Global memory coalescing check.
		if isGlobalLoadOpcode(baseOp, upOp) {
			if isUncoalescedGlobalLoad(upOp, upOperands, upRaw) {
				m.UncoalescedGlobalLoads++
			}
		}

		// 5. Shared memory bank-conflict check.
		if isSharedMemoryOpcode(baseOp) {
			totalSharedOps++
			if strings.Contains(upOp, ".128") || strings.Contains(upOp, ".64") ||
				baseOp == "LDSM" || baseOp == "STSM" {
				wideSharedOps++
			}
			if strings.Contains(upRaw, "BANK_CONFLICT") || strings.Contains(upRaw, "NO_SWIZZLE") || strings.Contains(upRaw, "UNSWIZZLED") {
				explicitBank = true
			}
		}

		// 6. Scoreboard and barrier wait stall weight.
		m.WaitStallWeight += instructionWaitStallWeight(ins, baseOp, upOp, upOperands)
	}

	if explicitBank {
		m.SharedBankConflictRisk = true
	} else if !hasSwizzleSeq && (wideSharedOps > 0 || totalSharedOps >= 2) {
		m.SharedBankConflictRisk = true
	}

	return m
}

func isTensorPipeOpcode(baseOp, fullOp string) bool {
	switch baseOp {
	case "HMMA", "IMMA", "QMMA", "BMMA", "DMMA", "HGMMA", "IGMMA", "QGMMA", "BGMMA", "WGMMA", "UTCMMA", "OMMA":
		return true
	}
	return strings.Contains(fullOp, "MMA")
}

func isGlobalLoadOpcode(baseOp, fullOp string) bool {
	if baseOp == "LDGSTS" || baseOp == "LDGMC" {
		return false
	}
	return baseOp == "LDG" || strings.HasPrefix(fullOp, "LD.E")
}

func isUncoalescedGlobalLoad(upOp, upOperands, upRaw string) bool {
	if strings.Contains(upRaw, "UNCOALESCED") || strings.Contains(upRaw, "STRIDED") {
		return true
	}
	// Narrow 8-bit/16-bit/32-bit global loads or .LU/.CV modifiers without
	// wide 64/128-bit vector transactions risk uncoalesced memory sectors.
	if strings.Contains(upOp, ".U8") || strings.Contains(upOp, ".S8") ||
		strings.Contains(upOp, ".U16") || strings.Contains(upOp, ".S16") ||
		strings.Contains(upOp, ".32") {
		return true
	}
	if !strings.Contains(upOp, ".64") && !strings.Contains(upOp, ".128") {
		return true
	}
	return false
}

func isSharedMemoryOpcode(baseOp string) bool {
	switch baseOp {
	case "LDS", "STS", "LDSM", "STSM", "ATOMS":
		return true
	default:
		return false
	}
}

func instructionWaitStallWeight(ins rawSASSInst, baseOp, upOp, upOperands string) int {
	weight := 0

	if ins.hasWords {
		if ctrl, err := cudasass.DecodeControl(cudasass.Encoding{Lo: ins.loWord, Hi: ins.hiWord}, cudasass.SM121a); err == nil {
			weight += int(ctrl.Stall)
			weight += 4 * bits.OnesCount8(ctrl.WaitMask)
		}
	}

	for _, m := range sassStallBraceRe.FindAllStringSubmatch(ins.rawLine, -1) {
		if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
			weight += n
		}
	}
	for _, m := range sassStallKVRe.FindAllStringSubmatch(ins.rawLine, -1) {
		if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
			weight += n
		}
	}

	switch {
	case strings.HasPrefix(upOp, "WARPGROUP.DEPBAR") || strings.HasPrefix(upOp, "WARPGROUP.ARRIVE") || baseOp == "UTCBAR":
		if strings.Contains(upOperands, "0X0") || strings.HasSuffix(strings.TrimSpace(upOperands), "0") {
			weight += 16
		} else {
			weight += 12
		}
	case baseOp == "DEPBAR":
		if strings.Contains(upOperands, "0X0") || strings.HasSuffix(strings.TrimSpace(upOperands), "0") {
			weight += 14
		} else {
			weight += 10
		}
	case baseOp == "BAR" || baseOp == "BSYNC" || baseOp == "MEMBAR" || baseOp == "ERRBAR" || baseOp == "CGAERRBAR":
		weight += 10
	case baseOp == "NANOSLEEP" || baseOp == "YIELD":
		weight += 6
	}

	return weight
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// RefinePlanByPTXASAndSASSLoop runs a closed-loop static PTXAS + SASS
// scoreboard/stall and resource refinement pass over p for up to maxIters
// iterations:
//  1. Compiles p with ptxCompiler and evaluates [StaticGate] and PTXAS
//     resource/spill bottlenecks via [RefinePlanByPTXASLoop].
//  2. Disassembles and parses SASS metrics for the current iteration using
//     sassAnalyzer.
//  3. When SASS disassembly reports shared-memory bank conflict risk on a
//     [hopper.TileGEMM] stage without Swizzle128B, local memory spill
//     instructions (STL/LDL), or excessive scoreboard wait stall weight,
//     automatically mutates Swizzle128B = true, reduces Stages, or adjusts
//     SplitK across iterations until clean or maxIters is reached.
func RefinePlanByPTXASAndSASSLoop(
	ctx context.Context,
	p *hopper.Plan,
	ptxCompiler PTXASCompiler,
	sassAnalyzer SASSAnalyzer,
	gate StaticGate,
	maxIters int,
) (*PTXASLoopResult, []StageSASSMetrics, error) {
	if p == nil || len(p.Kernels) == 0 {
		return nil, nil, errors.New("refine plan by ptxas and sass loop: empty plan")
	}
	if ptxCompiler == nil {
		return nil, nil, errors.New("refine plan by ptxas and sass loop: nil ptxCompiler")
	}
	if sassAnalyzer == nil {
		return nil, nil, errors.New("refine plan by ptxas and sass loop: nil sassAnalyzer")
	}
	if maxIters <= 0 {
		maxIters = 4
	}

	initialPlan := clonePlan(p)
	currentPlan := clonePlan(p)
	res := &PTXASLoopResult{
		InitialPlan: initialPlan,
		RefinedPlan: currentPlan,
	}

	var latestMetrics []StageSASSMetrics

	for iter := 1; iter <= maxIters; iter++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}

		// 1. Run 1-step PTXAS compile & analysis pass on currentPlan.
		ptxRes, err := RefinePlanByPTXASLoop(ctx, currentPlan, ptxCompiler, PTXASLoopConfig{
			MaxIterations: 1,
			Gate:          gate,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("refine plan by ptxas and sass loop (iteration %d): %w", iter, err)
		}

		var rec PTXASIterationRecord
		if len(ptxRes.Iterations) > 0 {
			rec = ptxRes.Iterations[0]
			rec.Iteration = iter
		} else {
			rec = PTXASIterationRecord{
				Iteration: iter,
				Clean:     ptxRes.Converged,
			}
		}

		// 2. Obtain and parse SASS disassembly for currentPlan.
		src, err := currentPlan.Source()
		if err != nil {
			return nil, nil, fmt.Errorf("refine plan by ptxas and sass loop (iteration %d): emit source: %w", iter, err)
		}
		sassInput := src
		if dis, ok := sassAnalyzer.(SASSDisassembler); ok {
			disText, err := dis.DisassembleSASS(ctx, src, "sm_90a")
			if err != nil {
				return nil, nil, fmt.Errorf("refine plan by ptxas and sass loop (iteration %d): disassemble sass: %w", iter, err)
			}
			sassInput = disText
		}
		stageMetrics, err := sassAnalyzer.ParseSASSDisassembly(sassInput)
		if err != nil {
			return nil, nil, fmt.Errorf("refine plan by ptxas and sass loop (iteration %d): parse sass: %w", iter, err)
		}
		latestMetrics = alignStageSASSMetrics(currentPlan, stageMetrics)

		// 3. Apply SASS-driven schedule mutations on top of ptxRes.RefinedPlan.
		nextPlan := clonePlan(ptxRes.RefinedPlan)
		sassActions, sassDirty := applySASSScheduleMutations(currentPlan, nextPlan, latestMetrics, &rec)
		rec.Actions = append(rec.Actions, sassActions...)
		if sassDirty {
			rec.Clean = false
		}

		res.Iterations = append(res.Iterations, rec)
		if ptxRes.Megakernel != nil {
			res.Megakernel = ptxRes.Megakernel
		}

		if rec.Clean && !sassDirty && ptxRes.Converged {
			res.RefinedPlan = currentPlan
			res.Converged = true
			return res, latestMetrics, nil
		}

		res.RefinedPlan = nextPlan
		if len(rec.Actions) == 0 && plansSameSchedule(currentPlan, nextPlan) {
			break
		}
		currentPlan = nextPlan
	}

	return res, latestMetrics, nil
}

func alignStageSASSMetrics(plan *hopper.Plan, metrics []StageSASSMetrics) []StageSASSMetrics {
	if plan == nil || len(plan.Kernels) == 0 {
		return metrics
	}
	if len(metrics) == 0 {
		out := make([]StageSASSMetrics, len(plan.Kernels))
		for i, k := range plan.Kernels {
			out[i] = StageSASSMetrics{StageName: stepKernelName(k)}
		}
		return out
	}

	used := make([]bool, len(metrics))
	aligned := make([]StageSASSMetrics, len(plan.Kernels))
	for i, k := range plan.Kernels {
		stepName := stepKernelName(k)
		matchedIdx := -1
		if stepName != "" {
			for j, m := range metrics {
				if used[j] {
					continue
				}
				if m.StageName == stepName || MatchesVariantFunction(m.StageName, stepName) ||
					strings.Contains(m.StageName, stepName) || strings.Contains(stepName, m.StageName) {
					matchedIdx = j
					break
				}
			}
		}
		if matchedIdx < 0 && i < len(metrics) && !used[i] {
			matchedIdx = i
		}
		if matchedIdx < 0 && len(metrics) == 1 {
			matchedIdx = 0
		}
		if matchedIdx >= 0 {
			used[matchedIdx] = true
			m := metrics[matchedIdx]
			if stepName != "" && (m.StageName == "" || m.StageName == "stage_0") {
				m.StageName = stepName
			}
			aligned[i] = m
		} else {
			aligned[i] = StageSASSMetrics{StageName: stepName}
		}
	}
	return aligned
}

func applySASSScheduleMutations(
	prevPlan *hopper.Plan,
	nextPlan *hopper.Plan,
	metrics []StageSASSMetrics,
	rec *PTXASIterationRecord,
) ([]string, bool) {
	if nextPlan == nil {
		return nil, false
	}
	var (
		actions   []string
		anyDirty  bool
		addAction = func(msg string) {
			if msg != "" {
				actions = append(actions, msg)
			}
		}
	)
	if rec.Bottlenecks == nil {
		rec.Bottlenecks = make(map[string][]BottleneckKind)
	}

	for i := range nextPlan.Kernels {
		if i >= len(metrics) {
			break
		}
		m := metrics[i]
		step := &nextPlan.Kernels[i]
		var prevStep hopper.PlanKernel
		if prevPlan != nil && i < len(prevPlan.Kernels) {
			prevStep = prevPlan.Kernels[i]
		} else {
			prevStep = *step
		}

		name := stepKernelName(*step)
		if name == "" {
			name = m.StageName
		}
		if name == "" {
			name = fmt.Sprintf("step_%d", i)
		}

		hasSpills := m.LocalMemInsts > 0
		hasExcessiveStalls := m.WaitStallWeight >= DefaultWaitStallThreshold
		hasBankConflict := m.SharedBankConflictRisk

		if hasSpills {
			rec.TotalSpillBytes += m.LocalMemInsts * 4
			if i < len(rec.Observations) && rec.Observations[i].SpillStoreBytes == 0 && rec.Observations[i].SpillLoadBytes == 0 {
				rec.Observations[i].SpillStoreBytes = m.LocalMemInsts * 4
			}
		}

		switch {
		case step.Tile != nil:
			t := *step.Tile
			var prevTile hopper.TileGEMM
			if prevStep.Tile != nil {
				prevTile = *prevStep.Tile
			} else {
				prevTile = t
			}
			mutated := false

			// 1. Bank conflict risk on TileGEMM without Swizzle128B -> enable Swizzle128B.
			if hasBankConflict && (!prevTile.Swizzle128B || !t.Swizzle128B) {
				anyDirty = true
				if !t.Swizzle128B {
					cand := t
					cand.Swizzle128B = true
					if _, err := cand.Norm(); err == nil {
						t = cand
						mutated = true
						addAction(fmt.Sprintf("%s: enable Swizzle128B to eliminate SASS shared-memory bank conflict risk", name))
					}
				}
			}

			// 2. Local memory spill instructions (STL/LDL) or excessive wait stall weight -> reduce Stages.
			if hasSpills || hasExcessiveStalls {
				anyDirty = true
				prevStages := prevTile.Stages
				if prevStages <= 0 {
					if normed, err := prevTile.Norm(); err == nil && normed.Stages > 0 {
						prevStages = normed.Stages
					} else {
						prevStages = 2
					}
				}
				curStages := t.Stages
				if curStages <= 0 {
					if normed, err := t.Norm(); err == nil && normed.Stages > 0 {
						curStages = normed.Stages
					} else {
						curStages = prevStages
					}
				}

				steppedDownStages := false
				if curStages > 1 && curStages >= prevStages {
					targetStages := curStages - 1
					cand := t
					cand.Stages = targetStages
					if _, err := cand.Norm(); err == nil {
						t = cand
						mutated = true
						steppedDownStages = true
						if hasSpills {
							addAction(fmt.Sprintf("%s: reduce Stages %d -> %d due to %d SASS STL/LDL local-memory instructions",
								name, curStages, targetStages, m.LocalMemInsts))
						} else {
							addAction(fmt.Sprintf("%s: reduce Stages %d -> %d due to excessive SASS wait stall weight (%d)",
								name, curStages, targetStages, m.WaitStallWeight))
						}
					}
				} else if curStages > 1 && curStages < prevStages {
					// PTXAS RefinePlan already stepped down Stages in this iteration.
					mutated = true
					steppedDownStages = true
				}

				if !steppedDownStages {
					if (hasSpills || hasExcessiveStalls) && t.BN > 64 {
						cand := t
						cand.BN = 64
						if _, err := cand.Norm(); err == nil {
							t = cand
							mutated = true
							addAction(fmt.Sprintf("%s: step down BN -> 64 due to persistent SASS spills/stalls", name))
						}
					} else if hasExcessiveStalls && t.WGMMA && t.Weight != hopper.WeightE4M3 {
						cand := t
						cand.WGMMA = false
						cand.WarpSpecialized = false
						if _, err := cand.Norm(); err == nil {
							t = cand
							mutated = true
							addAction(fmt.Sprintf("%s: disable WGMMA due to persistent SASS wait stall weight (%d)", name, m.WaitStallWeight))
						}
					}
				}
			}

			if mutated {
				step.Tile = &t
			}

		case step.GEMM != nil:
			g := *step.GEMM
			var prevGEMM hopper.SkinnyGEMM
			if prevStep.GEMM != nil {
				prevGEMM = *prevStep.GEMM
			} else {
				prevGEMM = g
			}
			mutated := false

			if hasSpills {
				anyDirty = true
				if g.Prefetch > 0 {
					cand := g
					cand.Prefetch = 0
					if _, err := cand.Norm(); err == nil {
						g = cand
					} else {
						g.Prefetch = 0
					}
					mutated = true
					addAction(fmt.Sprintf("%s: disable Prefetch due to %d SASS STL/LDL local-memory instructions", name, m.LocalMemInsts))
				} else if g.RowsPerBlock > 1 {
					nextRPB := max(g.RowsPerBlock/2, 1)
					cand := g
					cand.RowsPerBlock = nextRPB
					if _, err := cand.Norm(); err == nil {
						g = cand
					} else {
						g.RowsPerBlock = nextRPB
					}
					mutated = true
					addAction(fmt.Sprintf("%s: reduce RowsPerBlock -> %d due to %d SASS STL/LDL local-memory instructions", name, nextRPB, m.LocalMemInsts))
				}
			}

			if hasExcessiveStalls {
				anyDirty = true
				prevSplitK := max(prevGEMM.SplitK, 1)
				curSplitK := max(g.SplitK, 1)
				if curSplitK == prevSplitK {
					nextSplitK := resolveNextSplitK(g, curSplitK)
					if nextSplitK != curSplitK {
						cand := g
						cand.SplitK = nextSplitK
						if _, err := cand.Norm(); err == nil {
							g = cand
						} else {
							g.SplitK = nextSplitK
						}
						mutated = true
						addAction(fmt.Sprintf("%s: adjust SplitK %d -> %d due to excessive SASS wait stall weight (%d)",
							name, curSplitK, nextSplitK, m.WaitStallWeight))
					}
				}
			}

			if mutated {
				step.GEMM = &g
			}

		case step.Attention != nil:
			a := *step.Attention
			mutated := false
			if hasBankConflict && !a.Swizzle {
				anyDirty = true
				cand := a
				cand.Swizzle = true
				if _, err := cand.Norm(); err == nil {
					a = cand
				} else {
					a.Swizzle = true
				}
				mutated = true
				addAction(fmt.Sprintf("%s: enable Swizzle to eliminate SASS shared-memory bank conflict risk", name))
			}
			if (hasSpills || hasExcessiveStalls) && a.Stages > 1 {
				anyDirty = true
				nextStages := a.Stages - 1
				cand := a
				cand.Stages = nextStages
				if _, err := cand.Norm(); err == nil {
					a = cand
				} else {
					a.Stages = nextStages
				}
				mutated = true
				addAction(fmt.Sprintf("%s: reduce Stages -> %d due to SASS spills/stalls", name, nextStages))
			}
			if mutated {
				step.Attention = &a
			}
		}
	}

	return actions, anyDirty
}

func resolveNextSplitK(g hopper.SkinnyGEMM, curSplitK int) int {
	candidates := []int{curSplitK * 2, curSplitK + 1, 2, 4, 8, 1}
	for _, sk := range candidates {
		if sk <= 0 || sk == curSplitK || sk > 16 {
			continue
		}
		trial := g
		trial.SplitK = sk
		if normed, err := trial.Norm(); err == nil {
			if _, srcErr := normed.Source(); srcErr == nil {
				return sk
			}
		}
	}
	return curSplitK
}
