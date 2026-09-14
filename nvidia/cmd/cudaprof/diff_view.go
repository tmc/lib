package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// DiffRowType specifies the relationship between baseline and candidate instructions in a diff row.
type DiffRowType string

const (
	DiffRowMatch  DiffRowType = "match"
	DiffRowAdd    DiffRowType = "add"
	DiffRowDelete DiffRowType = "delete"
	DiffRowModify DiffRowType = "modify"
)

// DiffInstruction represents an annotated SASS instruction in the differential viewer.
type DiffInstruction struct {
	Address         uint64             `json:"address,omitempty"`
	Offset          string             `json:"offset,omitempty"`
	Predicate       string             `json:"predicate,omitempty"`
	Opcode          string             `json:"opcode,omitempty"`
	Operands        string             `json:"operands,omitempty"`
	Raw             string             `json:"raw,omitempty"`
	Text            string             `json:"text,omitempty"`
	SourceLine      int                `json:"source_line,omitempty"`
	IsSpill         bool               `json:"is_spill,omitempty"`
	SpillType       string             `json:"spill_type,omitempty"`
	Pressure        string             `json:"pressure,omitempty"`
	PressureTooltip string             `json:"pressure_tooltip,omitempty"`
	LiveGPR         int                `json:"live_gpr,omitempty"`
	LivePred        int                `json:"live_pred,omitempty"`
	LiveUGPR        int                `json:"live_ugpr,omitempty"`
	LiveUPred       int                `json:"live_upred,omitempty"`
	StallPct        float64            `json:"stall_pct,omitempty"`
	StallBreakdown  map[string]float64 `json:"stall_breakdown,omitempty"`
	StallTooltip    string             `json:"stall_tooltip,omitempty"`
	SemanticTag     string             `json:"semantic_tag,omitempty"`
}

// DiffRow represents a single aligned comparison row between baseline and candidate.
type DiffRow struct {
	Num          int              `json:"num"`
	Type         DiffRowType      `json:"type"`
	Base         *DiffInstruction `json:"base,omitempty"`
	Candidate    *DiffInstruction `json:"candidate,omitempty"`
	StallDelta   float64          `json:"stall_delta,omitempty"`
	StallReason  string           `json:"stall_reason,omitempty"`
	StallSummary string           `json:"stall_summary,omitempty"`
	HasSpill     bool             `json:"has_spill,omitempty"`
}

// DiffSummary contains high-level differential statistics between two captures.
type DiffSummary struct {
	BaseInstructions   int                           `json:"base_instructions"`
	TargetInstructions int                           `json:"target_instructions"`
	InstructionDelta   int                           `json:"instruction_delta"`
	BaseSizeBytes      int                           `json:"base_size_bytes"`
	TargetSizeBytes    int                           `json:"target_size_bytes"`
	TotalSizeDelta     int                           `json:"total_size_delta"`
	BaseBranches       int                           `json:"base_branches"`
	TargetBranches     int                           `json:"target_branches"`
	BranchDelta        int                           `json:"branch_delta"`
	SpillsAdded        int                           `json:"spills_added"`
	SpillsRemoved      int                           `json:"spills_removed"`
	SpillsDelta        int                           `json:"spills_delta"`
	BaseRuntimeUs      float64                       `json:"base_runtime_us,omitempty"`
	TargetRuntimeUs    float64                       `json:"target_runtime_us,omitempty"`
	DurationDeltaUs    float64                       `json:"duration_delta_us,omitempty"`
	Speedup            float64                       `json:"speedup,omitempty"`
	Substitutions      []cudasass.OpcodeSubstitution `json:"substitutions,omitempty"`
	TopStallShifts     []StallShiftSummary           `json:"top_stall_shifts,omitempty"`
}

// StallShiftSummary records the aggregate shift of a PC stall reason across two captures.
type StallShiftSummary struct {
	Reason      string  `json:"reason"`
	BaseCount   int64   `json:"base_count"`
	TargetCount int64   `json:"target_count"`
	Delta       int64   `json:"delta"`
	ShiftPct    float64 `json:"shift_pct"`
	ShiftStr    string  `json:"shift_str"`
}

// DiffViewData is the data model passed to the differential web template and HTTP endpoints.
type DiffViewData struct {
	Title           string                 `json:"title"`
	BaseCapture     string                 `json:"base_capture"`
	TargetCapture   string                 `json:"target_capture"`
	Kernels         []string               `json:"kernels"`
	ActiveKernel    string                 `json:"active_kernel"`
	KernelRows      map[string][]DiffRow   `json:"kernel_rows"`
	ActiveRows      []DiffRow              `json:"active_rows"`
	Summary         DiffSummary            `json:"summary"`
	KernelSummaries map[string]DiffSummary `json:"kernel_summaries"`
}

// ToJSON serializes DiffViewData to JavaScript-safe JSON for offline viewing.
func (d *DiffViewData) ToJSON() template.JS {
	b, err := json.Marshal(d)
	if err != nil {
		return template.JS("{}")
	}
	return template.JS(b)
}

// DiffWebOptions configures the differential web viewer server or HTML exporter.
type DiffWebOptions struct {
	BaseFile     string
	TargetFile   string
	KernelFilter string
	HTTPAddr     string
	HTMLOut      string
	NoBrowser    bool
	Limits       cudacapture.Limits
	Writer       io.Writer
}

func runDiffWeb(opts DiffWebOptions) error {
	if opts.Writer == nil {
		opts.Writer = os.Stdout
	}
	data, err := buildDiffViewData(opts.BaseFile, opts.TargetFile, opts.KernelFilter, opts.Limits)
	if err != nil {
		return fmt.Errorf("diff view: %w", err)
	}

	// Static HTML export mode
	if opts.HTMLOut != "" {
		var buf bytes.Buffer
		if err := renderDiffHTML(&buf, data); err != nil {
			return fmt.Errorf("diff view: render HTML: %w", err)
		}
		if opts.HTMLOut == "-" || opts.HTMLOut == "stdout" {
			_, err := opts.Writer.Write(buf.Bytes())
			return err
		}
		if err := os.WriteFile(opts.HTMLOut, buf.Bytes(), 0644); err != nil {
			return fmt.Errorf("diff view: write HTML output: %w", err)
		}
		fmt.Fprintf(opts.Writer, "Wrote differential report to %s (%d bytes)\n", opts.HTMLOut, buf.Len())
		return nil
	}

	// Interactive HTTP Server mode
	if opts.HTTPAddr == "" {
		opts.HTTPAddr = ":8088"
	}
	handler := newDiffServerHandler(data)
	listener, err := net.Listen("tcp", opts.HTTPAddr)
	if err != nil {
		return fmt.Errorf("diff view: listen on %s: %w", opts.HTTPAddr, err)
	}
	defer listener.Close()

	_, port, _ := net.SplitHostPort(listener.Addr().String())
	if port == "" {
		port = listener.Addr().String()
	}
	url := fmt.Sprintf("http://127.0.0.1:%s", port)
	fmt.Fprintf(opts.Writer, "Serving differential web view at %s\n", url)
	fmt.Fprintf(opts.Writer, "Comparing: %s vs %s\n", opts.BaseFile, opts.TargetFile)
	fmt.Fprintf(opts.Writer, "Press Ctrl+C to stop.\n")

	if !opts.NoBrowser {
		openBrowser(url)
	}

	server := &http.Server{
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}
	return server.Serve(listener)
}

func buildDiffViewData(baseFile, targetFile, kernelFilter string, limits cudacapture.Limits) (*DiffViewData, error) {
	baseAbs, err := filepath.Abs(baseFile)
	if err != nil {
		return nil, fmt.Errorf("resolve base path: %w", err)
	}
	targetAbs, err := filepath.Abs(targetFile)
	if err != nil {
		return nil, fmt.Errorf("resolve target path: %w", err)
	}

	baseCap, _ := readCapturePathLimits(baseAbs, limits)
	targetCap, _ := readCapturePathLimits(targetAbs, limits)

	var baseDisasm *cudasass.Disassembly
	if baseCap != nil {
		baseDisasm, _ = loadDisassemblyWithPressure(baseCap)
	}
	if baseDisasm == nil {
		baseDisasm, err = loadDisassembly(baseAbs, limits)
		if err != nil {
			return nil, fmt.Errorf("load base disassembly %s: %w", baseFile, err)
		}
	}

	var targetDisasm *cudasass.Disassembly
	if targetCap != nil {
		targetDisasm, _ = loadDisassemblyWithPressure(targetCap)
	}
	if targetDisasm == nil {
		targetDisasm, err = loadDisassembly(targetAbs, limits)
		if err != nil {
			return nil, fmt.Errorf("load target disassembly %s: %w", targetFile, err)
		}
	}

	diff, err := cudasass.Diff(baseDisasm, targetDisasm)
	if err != nil {
		return nil, fmt.Errorf("cudasass diff: %w", err)
	}

	baseStalls := extractSASSStallMap(baseCap, nil)
	targetStalls := extractSASSStallMap(targetCap, nil)

	basePressure := extractPressureMap(baseDisasm)
	targetPressure := extractPressureMap(targetDisasm)

	data := &DiffViewData{
		Title:           fmt.Sprintf("Differential SASS Profile Viewer: %s vs %s", filepath.Base(baseFile), filepath.Base(targetFile)),
		BaseCapture:     filepath.Base(baseFile),
		TargetCapture:   filepath.Base(targetFile),
		KernelRows:      make(map[string][]DiffRow),
		KernelSummaries: make(map[string]DiffSummary),
	}

	for _, fd := range diff.Functions {
		data.Kernels = append(data.Kernels, fd.Name)
		rows := alignInstructions(fd, baseStalls, targetStalls, basePressure, targetPressure)
		data.KernelRows[fd.Name] = rows

		kSum := buildKernelSummary(fd, baseCap, targetCap)
		data.KernelSummaries[fd.Name] = kSum
	}

	if len(data.Kernels) == 0 {
		return nil, errors.New("no GPU kernels found in disassembly diff")
	}

	if kernelFilter != "" {
		found := false
		for _, k := range data.Kernels {
			if k == kernelFilter {
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("kernel %q not found in disassembly", kernelFilter)
		}
		data.ActiveKernel = kernelFilter
	} else {
		data.ActiveKernel = data.Kernels[0]
	}

	data.ActiveRows = data.KernelRows[data.ActiveKernel]
	data.Summary = buildOverallSummary(diff, baseCap, targetCap, data.ActiveKernel)

	return data, nil
}

func alignInstructions(fd cudasass.FunctionDiff, baseStalls, targetStalls *sassStallMap, basePressure, targetPressure map[string]map[uint64]cudasass.RegisterPressure) []DiffRow {
	var rows []DiffRow
	rowNum := 1

	var hunkDeletes []*cudasass.Instruction
	var hunkAdds []*cudasass.Instruction

	flush := func() {
		if len(hunkDeletes) == 0 && len(hunkAdds) == 0 {
			return
		}
		dLen := len(hunkDeletes)
		aLen := len(hunkAdds)
		common := min(dLen, aLen)

		// Pair 1:1 as modifications
		for i := 0; i < common; i++ {
			bInst := convertInstruction(hunkDeletes[i], fd.Name, baseStalls, basePressure)
			cInst := convertInstruction(hunkAdds[i], fd.Name, targetStalls, targetPressure)
			sDelta, sReason, sSummary := computeStallDelta(bInst, cInst)
			hasSpill := (bInst != nil && bInst.IsSpill) || (cInst != nil && cInst.IsSpill)

			rows = append(rows, DiffRow{
				Num:          rowNum,
				Type:         DiffRowModify,
				Base:         bInst,
				Candidate:    cInst,
				StallDelta:   sDelta,
				StallReason:  sReason,
				StallSummary: sSummary,
				HasSpill:     hasSpill,
			})
			rowNum++
		}

		// Remaining deletions
		for i := common; i < dLen; i++ {
			bInst := convertInstruction(hunkDeletes[i], fd.Name, baseStalls, basePressure)
			sDelta, sReason, sSummary := computeStallDelta(bInst, nil)
			hasSpill := bInst != nil && bInst.IsSpill

			rows = append(rows, DiffRow{
				Num:          rowNum,
				Type:         DiffRowDelete,
				Base:         bInst,
				StallDelta:   sDelta,
				StallReason:  sReason,
				StallSummary: sSummary,
				HasSpill:     hasSpill,
			})
			rowNum++
		}

		// Remaining additions
		for i := common; i < aLen; i++ {
			cInst := convertInstruction(hunkAdds[i], fd.Name, targetStalls, targetPressure)
			sDelta, sReason, sSummary := computeStallDelta(nil, cInst)
			hasSpill := cInst != nil && cInst.IsSpill

			rows = append(rows, DiffRow{
				Num:          rowNum,
				Type:         DiffRowAdd,
				Candidate:    cInst,
				StallDelta:   sDelta,
				StallReason:  sReason,
				StallSummary: sSummary,
				HasSpill:     hasSpill,
			})
			rowNum++
		}

		hunkDeletes = hunkDeletes[:0]
		hunkAdds = hunkAdds[:0]
	}

	for _, c := range fd.Changes {
		switch c.Op {
		case cudasass.DiffMatch:
			flush()
			bInst := convertInstruction(c.Base, fd.Name, baseStalls, basePressure)
			cInst := convertInstruction(c.Target, fd.Name, targetStalls, targetPressure)
			sDelta, sReason, sSummary := computeStallDelta(bInst, cInst)
			hasSpill := (bInst != nil && bInst.IsSpill) || (cInst != nil && cInst.IsSpill)

			rows = append(rows, DiffRow{
				Num:          rowNum,
				Type:         DiffRowMatch,
				Base:         bInst,
				Candidate:    cInst,
				StallDelta:   sDelta,
				StallReason:  sReason,
				StallSummary: sSummary,
				HasSpill:     hasSpill,
			})
			rowNum++
		case cudasass.DiffDelete:
			hunkDeletes = append(hunkDeletes, c.Base)
		case cudasass.DiffAdd:
			hunkAdds = append(hunkAdds, c.Target)
		}
	}
	flush()

	return rows
}

func convertInstruction(inst *cudasass.Instruction, kernelName string, stalls *sassStallMap, pressureMap map[string]map[uint64]cudasass.RegisterPressure) *DiffInstruction {
	if inst == nil {
		return nil
	}

	d := &DiffInstruction{
		Address:    inst.Address,
		Offset:     fmt.Sprintf("%04x", inst.Address),
		Predicate:  inst.Predicate,
		Opcode:     inst.Opcode,
		Operands:   inst.Operands,
		Raw:        inst.Raw,
		SourceLine: inst.SourceLine,
		LiveGPR:    inst.LiveGPR,
		LivePred:   inst.LivePred,
		LiveUGPR:   inst.LiveUGPR,
		LiveUPred:  inst.LiveUPred,
	}

	// Format text
	if inst.Raw != "" {
		d.Text = strings.TrimSpace(inst.Raw)
	} else {
		var sb strings.Builder
		if inst.Predicate != "" {
			sb.WriteString(inst.Predicate)
			sb.WriteString(" ")
		}
		sb.WriteString(inst.Opcode)
		if inst.Operands != "" {
			sb.WriteString(" ")
			sb.WriteString(inst.Operands)
		}
		d.Text = sb.String()
	}

	// Identify register spill
	isSpill, spillType := isSpillOpcode(inst.Opcode)
	if !isSpill && inst.Raw != "" {
		isSpill, spillType = isSpillOpcode(inst.Raw)
	}
	d.IsSpill = isSpill
	d.SpillType = spillType

	// Live register pressure
	liveGPR := inst.LiveGPR
	livePred := inst.LivePred
	liveUGPR := inst.LiveUGPR
	liveUPred := inst.LiveUPred
	if liveGPR == 0 && pressureMap != nil {
		if pKernel, ok := pressureMap[kernelName]; ok {
			if p, ok := pKernel[inst.Address]; ok {
				liveGPR = p.GPR
				livePred = p.Pred
				liveUGPR = p.UGPR
				liveUPred = p.UPred
			}
		}
	}

	if liveGPR > 0 {
		d.LiveGPR = liveGPR
		d.LivePred = livePred
		d.LiveUGPR = liveUGPR
		d.LiveUPred = liveUPred
		d.Pressure = fmt.Sprintf("R:%d", liveGPR)
		if liveUGPR > 0 {
			d.Pressure += fmt.Sprintf(" UR:%d", liveUGPR)
		}
		d.PressureTooltip = fmt.Sprintf("Live Registers: %d GPR, %d PRED, %d UGPR, %d UPRED",
			liveGPR, livePred, liveUGPR, liveUPred)
	}

	// PC stall percentage and breakdown
	if stalls != nil && stalls.hasStalls {
		if info, total := stalls.lookup(kernelName, inst.Address); info != nil && info.totalStalls > 0 && total > 0 {
			pct := float64(info.totalStalls) / float64(total) * 100.0
			d.StallPct = pct
			d.StallBreakdown = make(map[string]float64, len(info.breakdown))
			for r, c := range info.breakdown {
				d.StallBreakdown[r] = float64(c) / float64(total) * 100.0
			}
			d.StallTooltip = formatStallTooltip(pct, d.StallBreakdown, info.totalStalls)
		}
	}

	// Semantic classification
	opBase := strings.ToUpper(strings.Split(inst.Opcode, ".")[0])
	switch opBase {
	case "UTMA", "TMA":
		d.SemanticTag = "tma"
	case "HMMA", "HGMMA", "MMA":
		d.SemanticTag = "mma"
	case "BAR", "DEPBAR":
		d.SemanticTag = "barrier"
	case "IDP4A", "HFMA2":
		d.SemanticTag = "simd"
	case "EXIT", "RET":
		d.SemanticTag = "return"
	case "LDG", "STG", "LDS", "STS", "STL", "LDL", "LDC":
		d.SemanticTag = "memory"
	}

	return d
}

func isSpillOpcode(op string) (bool, string) {
	trimmed := strings.TrimSpace(op)
	fields := strings.Fields(trimmed)
	for _, f := range fields {
		clean := strings.TrimPrefix(f, "@")
		clean = strings.TrimSuffix(clean, ";")
		clean = strings.TrimPrefix(clean, "/*")
		if strings.HasSuffix(clean, "*/") {
			continue
		}
		base := cudasass.BaseOpcode(clean)
		switch base {
		case "STS", "STL":
			return true, "STS"
		case "LDS", "LDL":
			return true, "LDS"
		}
	}
	return false, ""
}

func shortStallReason(reason string) string {
	r := strings.ToLower(reason)
	r = strings.TrimPrefix(r, "stall_")
	switch {
	case strings.Contains(r, "scoreboard"):
		return "scoreboard"
	case strings.Contains(r, "mem") || strings.Contains(r, "memory"):
		return "mem"
	case strings.Contains(r, "barrier") || strings.Contains(r, "sync"):
		return "barrier"
	case strings.Contains(r, "drain"):
		return "drain"
	case strings.Contains(r, "math") || strings.Contains(r, "pipe"):
		return "math"
	case strings.Contains(r, "not_selected"):
		return "not_selected"
	case strings.Contains(r, "tex"):
		return "tex"
	default:
		if len(r) > 12 {
			r = r[:12]
		}
		return r
	}
}

func computeStallDelta(base, cand *DiffInstruction) (float64, string, string) {
	var basePct, candPct float64
	if base != nil {
		basePct = base.StallPct
	}
	if cand != nil {
		candPct = cand.StallPct
	}

	if basePct == 0 && candPct == 0 {
		return 0, "", ""
	}

	delta := candPct - basePct

	// Determine dominant shifted reason
	allReasons := make(map[string]bool)
	if base != nil {
		for r := range base.StallBreakdown {
			allReasons[r] = true
		}
	}
	if cand != nil {
		for r := range cand.StallBreakdown {
			allReasons[r] = true
		}
	}

	bestReason := ""
	maxShift := -1.0
	for r := range allReasons {
		var bVal, cVal float64
		if base != nil {
			bVal = base.StallBreakdown[r]
		}
		if cand != nil {
			cVal = cand.StallBreakdown[r]
		}
		diff := math.Abs(cVal - bVal)
		if diff > maxShift || (diff == maxShift && (bestReason == "" || r < bestReason)) {
			maxShift = diff
			bestReason = r
		}
	}

	shortReason := "stall"
	if bestReason != "" {
		shortReason = shortStallReason(bestReason)
	}

	summary := fmt.Sprintf("%s %+.1f%%", shortReason, delta)
	return delta, shortReason, summary
}

func buildKernelSummary(fd cudasass.FunctionDiff, baseCap, targetCap *cudacapture.Capture) DiffSummary {
	sum := DiffSummary{
		BaseInstructions:   fd.Architecture.BaseInstructions,
		TargetInstructions: fd.Architecture.TargetInstructions,
		InstructionDelta:   fd.Architecture.InstructionDelta,
		BaseSizeBytes:      fd.Architecture.BaseSizeBytes,
		TargetSizeBytes:    fd.Architecture.TargetSizeBytes,
		TotalSizeDelta:     fd.Architecture.TotalSizeDelta,
		BaseBranches:       fd.Architecture.BaseBranches,
		TargetBranches:     fd.Architecture.TargetBranches,
		BranchDelta:        fd.Architecture.BranchDelta,
		SpillsAdded:        fd.Spills.TotalAdded,
		SpillsRemoved:      fd.Spills.TotalRemoved,
		SpillsDelta:        fd.Spills.Delta(),
		Substitutions:      fd.Architecture.Substitutions,
	}

	// Execution duration delta if profile samples exist
	var bNs, tNs int64
	if baseCap != nil && baseCap.Profile != nil {
		bIdx := findTimeIndex(baseCap.Profile)
		if bIdx >= 0 {
			for _, s := range baseCap.Profile.Sample {
				if sampleKernelName(s) == fd.Name {
					bNs += s.Value[bIdx]
				}
			}
		}
	}
	if targetCap != nil && targetCap.Profile != nil {
		tIdx := findTimeIndex(targetCap.Profile)
		if tIdx >= 0 {
			for _, s := range targetCap.Profile.Sample {
				if sampleKernelName(s) == fd.Name {
					tNs += s.Value[tIdx]
				}
			}
		}
	}

	if bNs > 0 || tNs > 0 {
		sum.BaseRuntimeUs = float64(bNs) / 1000.0
		sum.TargetRuntimeUs = float64(tNs) / 1000.0
		sum.DurationDeltaUs = sum.TargetRuntimeUs - sum.BaseRuntimeUs
		if tNs > 0 {
			sum.Speedup = float64(bNs) / float64(tNs)
		}
	}

	// Top stall shifts for this kernel
	if baseCap != nil || targetCap != nil {
		bStalls := collectStallCounts(profileOrNil(baseCap), fd.Name)
		tStalls := collectStallCounts(profileOrNil(targetCap), fd.Name)
		sum.TopStallShifts = computeTopStallShifts(bStalls, tStalls)
	}

	return sum
}

func buildOverallSummary(diff *cudasass.DisassemblyDiff, baseCap, targetCap *cudacapture.Capture, activeKernel string) DiffSummary {
	sum := DiffSummary{
		BaseInstructions:   diff.Architecture.BaseInstructions,
		TargetInstructions: diff.Architecture.TargetInstructions,
		InstructionDelta:   diff.Architecture.InstructionDelta,
		BaseSizeBytes:      diff.Architecture.BaseSizeBytes,
		TargetSizeBytes:    diff.Architecture.TargetSizeBytes,
		TotalSizeDelta:     diff.Architecture.TotalSizeDelta,
		BaseBranches:       diff.Architecture.BaseBranches,
		TargetBranches:     diff.Architecture.TargetBranches,
		BranchDelta:        diff.Architecture.BranchDelta,
		SpillsAdded:        diff.Spills.TotalAdded,
		SpillsRemoved:      diff.Spills.TotalRemoved,
		SpillsDelta:        diff.Spills.Delta(),
		Substitutions:      diff.Architecture.Substitutions,
	}

	var bNs, tNs int64
	if baseCap != nil && baseCap.Profile != nil {
		bIdx := findTimeIndex(baseCap.Profile)
		if bIdx >= 0 {
			for _, s := range baseCap.Profile.Sample {
				bNs += s.Value[bIdx]
			}
		}
	}
	if targetCap != nil && targetCap.Profile != nil {
		tIdx := findTimeIndex(targetCap.Profile)
		if tIdx >= 0 {
			for _, s := range targetCap.Profile.Sample {
				tNs += s.Value[tIdx]
			}
		}
	}

	if bNs > 0 || tNs > 0 {
		sum.BaseRuntimeUs = float64(bNs) / 1000.0
		sum.TargetRuntimeUs = float64(tNs) / 1000.0
		sum.DurationDeltaUs = sum.TargetRuntimeUs - sum.BaseRuntimeUs
		if tNs > 0 {
			sum.Speedup = float64(bNs) / float64(tNs)
		}
	}

	bStalls := collectStallCounts(profileOrNil(baseCap), "")
	tStalls := collectStallCounts(profileOrNil(targetCap), "")
	sum.TopStallShifts = computeTopStallShifts(bStalls, tStalls)

	return sum
}

func profileOrNil(cap *cudacapture.Capture) *profile.Profile {
	if cap != nil {
		return cap.Profile
	}
	return nil
}

func computeTopStallShifts(bStalls, tStalls map[string]int64) []StallShiftSummary {
	allReasons := make(map[string]bool)
	for r := range bStalls {
		allReasons[r] = true
	}
	for r := range tStalls {
		allReasons[r] = true
	}

	var shifts []StallShiftSummary
	for r := range allReasons {
		b := bStalls[r]
		t := tStalls[r]
		d := t - b
		shiftPct := 0.0
		shiftStr := "N/A"
		if b > 0 {
			shiftPct = float64(d) / float64(b) * 100.0
			shiftStr = fmt.Sprintf("%+.1f%%", shiftPct)
		} else if t > 0 {
			shiftStr = "+inf"
		}
		shifts = append(shifts, StallShiftSummary{
			Reason:      r,
			BaseCount:   b,
			TargetCount: t,
			Delta:       d,
			ShiftPct:    shiftPct,
			ShiftStr:    shiftStr,
		})
	}

	// shifts came out of a map, so equal deltas must be broken by name or
	// the top five are a different five on every run and two diffs of the
	// same pair of profiles disagree.
	sort.Slice(shifts, func(i, j int) bool {
		di, dj := math.Abs(float64(shifts[i].Delta)), math.Abs(float64(shifts[j].Delta))
		if di != dj {
			return di > dj
		}
		return shifts[i].Reason < shifts[j].Reason
	})

	if len(shifts) > 5 {
		shifts = shifts[:5]
	}
	return shifts
}

func newDiffServerHandler(data *DiffViewData) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := renderDiffHTML(w, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	mux.HandleFunc("/api/diff", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(data)
	})

	mux.HandleFunc("/api/kernel", func(w http.ResponseWriter, r *http.Request) {
		kName := r.URL.Query().Get("name")
		w.Header().Set("Content-Type", "application/json")
		if rows, ok := data.KernelRows[kName]; ok {
			json.NewEncoder(w).Encode(rows)
		} else {
			json.NewEncoder(w).Encode(data.ActiveRows)
		}
	})

	return mux
}

func renderDiffHTML(w io.Writer, data *DiffViewData) error {
	tmpl, err := template.New("diff").Funcs(template.FuncMap{
		"stallWidth": func(pct float64) string {
			if pct <= 0 {
				return "0"
			}
			w := pct
			if w < 4 {
				w = 4
			}
			if w > 100 {
				w = 100
			}
			return fmt.Sprintf("%.1f", w)
		},
		"deltaWidth": func(delta float64) string {
			w := math.Abs(delta)
			if w <= 0 {
				return "0"
			}
			if w < 4 {
				w = 4
			}
			if w > 100 {
				w = 100
			}
			return fmt.Sprintf("%.1f", w)
		},
		"abs": func(v float64) float64 {
			return math.Abs(v)
		},
	}).Parse(diffHTMLTemplate)
	if err != nil {
		return fmt.Errorf("parse diff template: %w", err)
	}
	return tmpl.Execute(w, data)
}

const diffHTMLTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{.Title}}</title>
<style>
:root {
  --bg-primary: #0f141c;
  --bg-secondary: #18202c;
  --bg-panel: #131b26;
  --border-color: #2b384e;
  --text-primary: #e6edf3;
  --text-muted: #8b949e;
  --accent-green: #3fb950;
  --accent-green-bg: rgba(46, 160, 67, 0.15);
  --accent-red: #f85149;
  --accent-red-bg: rgba(218, 54, 51, 0.15);
  --accent-modify: #d29922;
  --accent-modify-bg: rgba(210, 153, 34, 0.12);
  --badge-spill-bg: #7f1d1d;
  --badge-spill-border: #ef4444;
  --badge-spill-text: #fca5a5;
  --badge-reg-bg: #312e81;
  --badge-reg-border: rgba(165, 180, 252, 0.3);
  --badge-reg-text: #c7d2fe;
  --badge-reg-spike-bg: #581c87;
  --badge-reg-spike-border: #a855f7;
  --badge-reg-spike-text: #f3e8ff;
  --hover-bg: #1f2d3f;
  --line-num-color: #48566a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.4;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
header {
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  padding: 10px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.header-title {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-title h1 {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.badge-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 12px;
  background-color: rgba(56, 189, 248, 0.15);
  color: #38bdf8;
  border: 1px solid rgba(56, 189, 248, 0.3);
  font-weight: 500;
}
.controls {
  display: flex;
  align-items: center;
  gap: 10px;
}
select, input[type="text"] {
  background-color: var(--bg-panel);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 5px 10px;
  border-radius: 6px;
  font-size: 12px;
  outline: none;
}
select:focus, input[type="text"]:focus {
  border-color: #38bdf8;
}
.btn {
  background-color: var(--bg-panel);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 5px 12px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.btn:hover {
  background-color: var(--hover-bg);
}
.btn.active {
  background-color: #38bdf8;
  color: #000;
  font-weight: 600;
  border-color: #38bdf8;
}
.summary-bar {
  background-color: #121824;
  border-bottom: 1px solid var(--border-color);
  padding: 8px 16px;
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  align-items: center;
  font-size: 12px;
}
.summary-item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.summary-label {
  color: var(--text-muted);
}
.summary-val {
  font-weight: 600;
}
.val-green { color: var(--accent-green); }
.val-red { color: var(--accent-red); }
.val-amber { color: var(--accent-modify); }

.diff-container {
  display: flex;
  flex: 1;
  overflow: hidden;
  position: relative;
}
.panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
}
.panel-header {
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  padding: 8px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  user-select: none;
}
.panel-tag {
  font-size: 10px;
  font-weight: 700;
  padding: 2px 6px;
  border-radius: 4px;
  letter-spacing: 0.5px;
  margin-right: 8px;
}
.tag-base {
  background: rgba(248, 81, 73, 0.2);
  color: var(--accent-red);
  border: 1px solid rgba(248, 81, 73, 0.4);
}
.tag-candidate {
  background: rgba(63, 185, 80, 0.2);
  color: var(--accent-green);
  border: 1px solid rgba(63, 185, 80, 0.4);
}
.panel-title {
  font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
.panel-stats {
  color: var(--text-muted);
  font-size: 11px;
}
.panel-divider {
  width: 3px;
  background-color: var(--border-color);
  flex-shrink: 0;
}

.code-viewport {
  flex: 1;
  overflow-y: scroll;
  overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 22px;
  background-color: var(--bg-panel);
}
.diff-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}
.diff-row {
  height: 24px;
  transition: background-color 0.1s ease;
}
.diff-row td {
  padding: 1px 8px;
  white-space: pre;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}
.line-num {
  width: 44px;
  text-align: right;
  color: var(--line-num-color);
  user-select: none;
  font-size: 11px;
  border-right: 1px solid rgba(43, 56, 78, 0.5);
  padding-right: 8px !important;
}
.line-sign {
  width: 22px;
  text-align: center;
  font-weight: 700;
  user-select: none;
  font-size: 13px;
}
.line-content {
  overflow: hidden;
  text-overflow: ellipsis;
}
.row-add {
  background-color: var(--accent-green-bg);
}
.row-add .line-sign {
  color: var(--accent-green);
}
.row-delete {
  background-color: var(--accent-red-bg);
}
.row-delete .line-sign {
  color: var(--accent-red);
}
.row-modify {
  background-color: var(--accent-modify-bg);
}
.row-modify .line-sign {
  color: var(--accent-modify);
}
.row-hover {
  background-color: rgba(255, 255, 255, 0.08) !important;
}
.empty-line {
  background-color: rgba(0, 0, 0, 0.22);
  height: 100%;
  border-radius: 2px;
}

.badge-offset {
  color: #64748b;
  margin-right: 6px;
  font-size: 11px;
}
.badge-spill {
  display: inline-block;
  font-size: 9px;
  font-weight: 800;
  padding: 1px 5px;
  border-radius: 3px;
  margin-right: 5px;
  cursor: help;
  vertical-align: middle;
  letter-spacing: 0.3px;
}
.badge-spill-sts {
  background: var(--badge-spill-bg);
  border: 1px solid var(--badge-spill-border);
  color: var(--badge-spill-text);
}
.badge-spill-lds {
  background: #831843;
  border: 1px solid #f43f5e;
  color: #fecdd3;
}
.badge-reg {
  display: inline-block;
  font-size: 9px;
  font-weight: 700;
  color: var(--badge-reg-text);
  background: var(--badge-reg-bg);
  border: 1px solid var(--badge-reg-border);
  padding: 1px 4px;
  border-radius: 3px;
  margin-right: 5px;
  cursor: help;
  vertical-align: middle;
}
.badge-reg-spike {
  background: var(--badge-reg-spike-bg);
  border: 1px solid var(--badge-reg-spike-border);
  color: var(--badge-reg-spike-text);
  box-shadow: 0 0 5px rgba(168, 85, 247, 0.5);
}

.stall-heat {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-right: 6px;
  vertical-align: middle;
  cursor: help;
}
.stall-bar-bg {
  width: 32px;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
}
.stall-bar {
  height: 100%;
  border-radius: 3px;
}
.stall-low { background: #3fb950; }
.stall-med { background: #d29922; }
.stall-high { background: #f85149; }
.stall-text {
  font-size: 10px;
  font-weight: 600;
}

.stall-delta {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 10px;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 3px;
  margin-right: 6px;
  vertical-align: middle;
  cursor: help;
}
.stall-improved {
  background: rgba(46, 160, 67, 0.22);
  color: var(--accent-green);
  border: 1px solid rgba(46, 160, 67, 0.45);
}
.stall-regressed {
  background: rgba(218, 54, 51, 0.22);
  color: var(--accent-red);
  border: 1px solid rgba(218, 54, 51, 0.45);
}

.sem-tma { color: #38bdf8; font-weight: 600; }
.sem-mma { color: #f472b6; font-weight: 600; }
.sem-barrier { color: #fbbf24; font-weight: 600; }
.sem-simd { color: #34d399; font-weight: 600; }
.sem-memory { color: #93c5fd; }
.sem-return { color: #f87171; }
</style>
</head>
<body>

<header>
  <div class="header-title">
    <h1>Differential CUDA SASS Viewer</h1>
    <span class="badge-tag">Myers LCS Aligned</span>
  </div>
  <div class="controls">
    <label for="kernel-select" style="font-size: 12px; color: var(--text-muted);">Kernel:</label>
    <select id="kernel-select" onchange="onKernelChange(this.value)">
      {{$active := .ActiveKernel}}
      {{range .Kernels}}
      <option value="{{.}}" {{if eq . $active}}selected{{end}}>{{.}}</option>
      {{end}}
    </select>
    <input type="text" id="filter-input" placeholder="Filter opcode..." oninput="onFilterChange(this.value)" />
    <button class="btn" id="btn-changes" onclick="toggleChangesOnly()">Changes Only</button>
  </div>
</header>

<div class="summary-bar">
  {{$s := .Summary}}
  <div class="summary-item">
    <span class="summary-label">Instructions:</span>
    <span class="summary-val">{{$s.BaseInstructions}} → {{$s.TargetInstructions}} ({{if lt $s.InstructionDelta 0}}<span class="val-green">{{$s.InstructionDelta}}</span>{{else if gt $s.InstructionDelta 0}}<span class="val-red">+{{$s.InstructionDelta}}</span>{{else}}0{{end}})</span>
  </div>
  <div class="summary-item">
    <span class="summary-label">Size:</span>
    <span class="summary-val">{{$s.BaseSizeBytes}}B → {{$s.TargetSizeBytes}}B ({{if lt $s.TotalSizeDelta 0}}<span class="val-green">{{$s.TotalSizeDelta}}B</span>{{else if gt $s.TotalSizeDelta 0}}<span class="val-red">+{{$s.TotalSizeDelta}}B</span>{{else}}0B{{end}})</span>
  </div>
  {{if or (gt $s.SpillsAdded 0) (gt $s.SpillsRemoved 0)}}
  <div class="summary-item">
    <span class="summary-label">Spills:</span>
    <span class="summary-val">+{{$s.SpillsAdded}} / -{{$s.SpillsRemoved}} ({{if lt $s.SpillsDelta 0}}<span class="val-green">{{$s.SpillsDelta}}</span>{{else if gt $s.SpillsDelta 0}}<span class="val-red">+{{$s.SpillsDelta}}</span>{{else}}0{{end}})</span>
  </div>
  {{end}}
  {{if gt $s.Speedup 0.0}}
  <div class="summary-item">
    <span class="summary-label">Runtime:</span>
    <span class="summary-val">{{printf "%.2f" $s.BaseRuntimeUs}}µs → {{printf "%.2f" $s.TargetRuntimeUs}}µs ({{if lt $s.DurationDeltaUs 0.0}}<span class="val-green">{{printf "%.2f" $s.DurationDeltaUs}}µs, {{printf "%.2f" $s.Speedup}}× speedup</span>{{else}}<span class="val-red">+{{printf "%.2f" $s.DurationDeltaUs}}µs</span>{{end}})</span>
  </div>
  {{end}}
  {{if gt (len $s.TopStallShifts) 0}}
  <div class="summary-item">
    <span class="summary-label">Top Stall Shifts:</span>
    <span class="summary-val">
      {{range $i, $ts := $s.TopStallShifts}}
        {{if $i}}, {{end}}
        <span class="{{if lt $ts.Delta 0}}val-green{{else}}val-red{{end}}">{{$ts.Reason}} {{$ts.ShiftStr}}</span>
      {{end}}
    </span>
  </div>
  {{end}}
</div>

<div class="diff-container">
  <!-- Baseline Panel -->
  <div class="panel" id="panel-base">
    <div class="panel-header">
      <div>
        <span class="panel-tag tag-base">BASELINE</span>
        <span class="panel-title">{{.BaseCapture}}</span>
      </div>
      <span class="panel-stats" id="stats-base">{{$s.BaseInstructions}} instructions</span>
    </div>
    <div class="code-viewport" id="viewport-base">
      <table class="diff-table" id="table-base">
        <tbody>
          {{range .ActiveRows}}
          <tr class="diff-row row-{{.Type}} {{if .HasSpill}}row-spill{{end}}" data-row="{{.Num}}" data-type="{{.Type}}">
            <td class="line-num">{{if .Base}}{{.Base.Offset}}{{end}}</td>
            <td class="line-sign">{{if eq .Type "delete"}}-{{else if eq .Type "modify"}}~{{else if eq .Type "match"}} {{end}}</td>
            <td class="line-content">
              {{if .Base}}
                {{if .Base.IsSpill}}<span class="badge-spill badge-spill-{{.Base.SpillType}}" title="Register Spill to local memory">SPILL: {{.Base.SpillType}}</span>{{end}}
                {{if .Base.Pressure}}<span class="badge-reg {{if .Base.IsSpill}}badge-reg-spike{{end}}" title="{{.Base.PressureTooltip}}">{{.Base.Pressure}}</span>{{end}}
                {{if gt .Base.StallPct 0.0}}
                  <span class="stall-heat" title="{{.Base.StallTooltip}}">
                    <span class="stall-bar-bg"><span class="stall-bar {{if ge .Base.StallPct 30.0}}stall-high{{else if ge .Base.StallPct 10.0}}stall-med{{else}}stall-low{{end}}" style="width: {{stallWidth .Base.StallPct}}%;"></span></span>
                    <span class="stall-text {{if ge .Base.StallPct 30.0}}stall-high{{else if ge .Base.StallPct 10.0}}stall-med{{else}}stall-low{{end}}">{{printf "%.1f" .Base.StallPct}}%</span>
                  </span>
                {{end}}
                {{if .StallSummary}}
                  <span class="stall-delta {{if lt .StallDelta 0.0}}stall-improved{{else if gt .StallDelta 0.0}}stall-regressed{{end}}" title="Baseline Stall Shift">{{.StallSummary}}</span>
                {{end}}
                <span class="{{if .Base.SemanticTag}}sem-{{.Base.SemanticTag}}{{end}}">{{.Base.Text}}</span>
              {{else}}
                <div class="empty-line">&nbsp;</div>
              {{end}}
            </td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
  </div>

  <div class="panel-divider"></div>

  <!-- Candidate Panel -->
  <div class="panel" id="panel-candidate">
    <div class="panel-header">
      <div>
        <span class="panel-tag tag-candidate">CANDIDATE</span>
        <span class="panel-title">{{.TargetCapture}}</span>
      </div>
      <span class="panel-stats" id="stats-candidate">{{$s.TargetInstructions}} instructions</span>
    </div>
    <div class="code-viewport" id="viewport-candidate">
      <table class="diff-table" id="table-candidate">
        <tbody>
          {{range .ActiveRows}}
          <tr class="diff-row row-{{.Type}} {{if .HasSpill}}row-spill{{end}}" data-row="{{.Num}}" data-type="{{.Type}}">
            <td class="line-num">{{if .Candidate}}{{.Candidate.Offset}}{{end}}</td>
            <td class="line-sign">{{if eq .Type "add"}}+{{else if eq .Type "modify"}}~{{else if eq .Type "match"}} {{end}}</td>
            <td class="line-content">
              {{if .Candidate}}
                {{if .Candidate.IsSpill}}<span class="badge-spill badge-spill-{{.Candidate.SpillType}}" title="Register Spill to local memory">SPILL: {{.Candidate.SpillType}}</span>{{end}}
                {{if .Candidate.Pressure}}<span class="badge-reg {{if .Candidate.IsSpill}}badge-reg-spike{{end}}" title="{{.Candidate.PressureTooltip}}">{{.Candidate.Pressure}}</span>{{end}}
                {{if gt .Candidate.StallPct 0.0}}
                  <span class="stall-heat" title="{{.Candidate.StallTooltip}}">
                    <span class="stall-bar-bg"><span class="stall-bar {{if ge .Candidate.StallPct 30.0}}stall-high{{else if ge .Candidate.StallPct 10.0}}stall-med{{else}}stall-low{{end}}" style="width: {{stallWidth .Candidate.StallPct}}%;"></span></span>
                    <span class="stall-text {{if ge .Candidate.StallPct 30.0}}stall-high{{else if ge .Candidate.StallPct 10.0}}stall-med{{else}}stall-low{{end}}">{{printf "%.1f" .Candidate.StallPct}}%</span>
                  </span>
                {{end}}
                {{if .StallSummary}}
                  <span class="stall-delta {{if lt .StallDelta 0.0}}stall-improved{{else if gt .StallDelta 0.0}}stall-regressed{{end}}" title="Candidate Stall Shift">{{.StallSummary}}</span>
                {{end}}
                <span class="{{if .Candidate.SemanticTag}}sem-{{.Candidate.SemanticTag}}{{end}}">{{.Candidate.Text}}</span>
              {{else}}
                <div class="empty-line">&nbsp;</div>
              {{end}}
            </td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
  </div>
</div>

<script id="diff-data" type="application/json">{{.ToJSON}}</script>

<script>
const diffData = JSON.parse(document.getElementById('diff-data').textContent);
const vpBase = document.getElementById('viewport-base');
const vpCand = document.getElementById('viewport-candidate');

let isSyncing = false;
function syncScroll(source, target) {
  if (isSyncing) return;
  isSyncing = true;
  target.scrollTop = source.scrollTop;
  target.scrollLeft = source.scrollLeft;
  requestAnimationFrame(() => { isSyncing = false; });
}

vpBase.addEventListener('scroll', () => syncScroll(vpBase, vpCand));
vpCand.addEventListener('scroll', () => syncScroll(vpCand, vpBase));

function setupRowHover() {
  document.querySelectorAll('.diff-row').forEach(row => {
    row.addEventListener('mouseenter', () => {
      const num = row.getAttribute('data-row');
      document.querySelectorAll('.diff-row[data-row="' + num + '"]').forEach(r => r.classList.add('row-hover'));
    });
    row.addEventListener('mouseleave', () => {
      const num = row.getAttribute('data-row');
      document.querySelectorAll('.diff-row[data-row="' + num + '"]').forEach(r => r.classList.remove('row-hover'));
    });
  });
}
setupRowHover();

let changesOnly = false;
function toggleChangesOnly() {
  changesOnly = !changesOnly;
  const btn = document.getElementById('btn-changes');
  btn.classList.toggle('active', changesOnly);
  applyRowFilters();
}

let filterQuery = '';
function onFilterChange(q) {
  filterQuery = q.toLowerCase().trim();
  applyRowFilters();
}

function applyRowFilters() {
  const rowsBase = document.querySelectorAll('#table-base .diff-row');
  const rowsCand = document.querySelectorAll('#table-candidate .diff-row');

  for (let i = 0; i < rowsBase.length; i++) {
    const rb = rowsBase[i];
    const rc = rowsCand[i];
    const rType = rb.getAttribute('data-type');
    const tb = rb.textContent.toLowerCase();
    const tc = rc.textContent.toLowerCase();

    let show = true;
    if (changesOnly && rType === 'match') {
      show = false;
    }
    if (show && filterQuery !== '') {
      if (!tb.includes(filterQuery) && !tc.includes(filterQuery)) {
        show = false;
      }
    }

    rb.style.display = show ? '' : 'none';
    rc.style.display = show ? '' : 'none';
  }
}

function onKernelChange(kName) {
  if (!diffData.kernel_rows || !diffData.kernel_rows[kName]) {
    return;
  }
  const rows = diffData.kernel_rows[kName];
  renderKernelRows(rows);
  if (diffData.kernel_summaries && diffData.kernel_summaries[kName]) {
    const s = diffData.kernel_summaries[kName];
    document.getElementById('stats-base').textContent = s.base_instructions + ' instructions';
    document.getElementById('stats-candidate').textContent = s.target_instructions + ' instructions';
  }
}

function escapeHtml(s) {
  if (!s) return '';
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderKernelRows(rows) {
  const tbodyBase = document.querySelector('#table-base tbody');
  const tbodyCand = document.querySelector('#table-candidate tbody');
  let htmlBase = '';
  let htmlCand = '';

  for (const r of rows) {
    const spillClass = r.has_spill ? 'row-spill' : '';
    let signBase = ' ';
    if (r.type === 'delete') signBase = '-';
    else if (r.type === 'modify') signBase = '~';

    let signCand = ' ';
    if (r.type === 'add') signCand = '+';
    else if (r.type === 'modify') signCand = '~';

    // Base cell
    let baseContent = '<div class="empty-line">&nbsp;</div>';
    let baseNum = '';
    if (r.base) {
      baseNum = r.base.offset || '';
      let parts = '';
      if (r.base.is_spill) {
        parts += '<span class="badge-spill badge-spill-' + (r.base.spill_type ? r.base.spill_type.toLowerCase() : 'sts') + '">SPILL: ' + (r.base.spill_type || 'STS') + '</span>';
      }
      if (r.base.pressure) {
        parts += '<span class="badge-reg ' + (r.base.is_spill ? 'badge-reg-spike' : '') + '" title="' + escapeHtml(r.base.pressure_tooltip) + '">' + escapeHtml(r.base.pressure) + '</span>';
      }
      if (r.base.stall_pct > 0) {
        const cls = r.base.stall_pct >= 30 ? 'stall-high' : (r.base.stall_pct >= 10 ? 'stall-med' : 'stall-low');
        const w = Math.max(4, Math.min(100, r.base.stall_pct)).toFixed(1);
        parts += '<span class="stall-heat" title="' + escapeHtml(r.base.stall_tooltip) + '"><span class="stall-bar-bg"><span class="stall-bar ' + cls + '" style="width: ' + w + '%;"></span></span><span class="stall-text ' + cls + '">' + r.base.stall_pct.toFixed(1) + '%</span></span>';
      }
      if (r.stall_summary) {
        const dCls = r.stall_delta < 0 ? 'stall-improved' : 'stall-regressed';
        parts += '<span class="stall-delta ' + dCls + '">' + escapeHtml(r.stall_summary) + '</span>';
      }
      const semCls = r.base.semantic_tag ? 'sem-' + r.base.semantic_tag : '';
      parts += '<span class="' + semCls + '">' + escapeHtml(r.base.text) + '</span>';
      baseContent = parts;
    }

    htmlBase += '<tr class="diff-row row-' + r.type + ' ' + spillClass + '" data-row="' + r.num + '" data-type="' + r.type + '">' +
      '<td class="line-num">' + baseNum + '</td>' +
      '<td class="line-sign">' + signBase + '</td>' +
      '<td class="line-content">' + baseContent + '</td>' +
      '</tr>';

    // Candidate cell
    let candContent = '<div class="empty-line">&nbsp;</div>';
    let candNum = '';
    if (r.candidate) {
      candNum = r.candidate.offset || '';
      let parts = '';
      if (r.candidate.is_spill) {
        parts += '<span class="badge-spill badge-spill-' + (r.candidate.spill_type ? r.candidate.spill_type.toLowerCase() : 'sts') + '">SPILL: ' + (r.candidate.spill_type || 'STS') + '</span>';
      }
      if (r.candidate.pressure) {
        parts += '<span class="badge-reg ' + (r.candidate.is_spill ? 'badge-reg-spike' : '') + '" title="' + escapeHtml(r.candidate.pressure_tooltip) + '">' + escapeHtml(r.candidate.pressure) + '</span>';
      }
      if (r.candidate.stall_pct > 0) {
        const cls = r.candidate.stall_pct >= 30 ? 'stall-high' : (r.candidate.stall_pct >= 10 ? 'stall-med' : 'stall-low');
        const w = Math.max(4, Math.min(100, r.candidate.stall_pct)).toFixed(1);
        parts += '<span class="stall-heat" title="' + escapeHtml(r.candidate.stall_tooltip) + '"><span class="stall-bar-bg"><span class="stall-bar ' + cls + '" style="width: ' + w + '%;"></span></span><span class="stall-text ' + cls + '">' + r.candidate.stall_pct.toFixed(1) + '%</span></span>';
      }
      if (r.stall_summary) {
        const dCls = r.stall_delta < 0 ? 'stall-improved' : 'stall-regressed';
        parts += '<span class="stall-delta ' + dCls + '">' + escapeHtml(r.stall_summary) + '</span>';
      }
      const semCls = r.candidate.semantic_tag ? 'sem-' + r.candidate.semantic_tag : '';
      parts += '<span class="' + semCls + '">' + escapeHtml(r.candidate.text) + '</span>';
      candContent = parts;
    }

    htmlCand += '<tr class="diff-row row-' + r.type + ' ' + spillClass + '" data-row="' + r.num + '" data-type="' + r.type + '">' +
      '<td class="line-num">' + candNum + '</td>' +
      '<td class="line-sign">' + signCand + '</td>' +
      '<td class="line-content">' + candContent + '</td>' +
      '</tr>';
  }

  tbodyBase.innerHTML = htmlBase;
  tbodyCand.innerHTML = htmlCand;
  setupRowHover();
  applyRowFilters();
}
</script>

</body>
</html>
`
