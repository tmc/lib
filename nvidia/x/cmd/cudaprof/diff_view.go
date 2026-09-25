package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudaview"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

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
		if err := cudaview.RenderDiff(&buf, data); err != nil {
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

func buildDiffViewData(baseFile, targetFile, kernelFilter string, limits cudacapture.Limits) (*cudaview.DiffViewData, error) {
	baseAbs, err := filepath.Abs(baseFile)
	if err != nil {
		return nil, fmt.Errorf("resolve base path: %w", err)
	}
	targetAbs, err := filepath.Abs(targetFile)
	if err != nil {
		return nil, fmt.Errorf("resolve target path: %w", err)
	}

	baseCap, baseCapErr := readCapturePathLimits(baseAbs, limits)
	targetCap, targetCapErr := readCapturePathLimits(targetAbs, limits)
	if strings.HasSuffix(baseAbs, ".pprof") && baseCapErr != nil {
		return nil, fmt.Errorf("read base capture %s: %w", baseFile, baseCapErr)
	}
	if strings.HasSuffix(targetAbs, ".pprof") && targetCapErr != nil {
		return nil, fmt.Errorf("read target capture %s: %w", targetFile, targetCapErr)
	}

	var baseDisasm *cudasass.Disassembly
	if baseCap != nil {
		baseDisasm, _ = capturefiles.Pressure(baseCap)
	}
	if baseDisasm == nil {
		baseDisasm, err = loadRetainedDisassembly(baseCap, baseAbs, limits)
		if err != nil {
			return nil, fmt.Errorf("load base disassembly %s: %w", baseFile, err)
		}
	}

	var targetDisasm *cudasass.Disassembly
	if targetCap != nil {
		targetDisasm, _ = capturefiles.Pressure(targetCap)
	}
	if targetDisasm == nil {
		targetDisasm, err = loadRetainedDisassembly(targetCap, targetAbs, limits)
		if err != nil {
			return nil, fmt.Errorf("load target disassembly %s: %w", targetFile, err)
		}
	}

	diff, err := cudasass.Diff(baseDisasm, targetDisasm)
	if err != nil {
		return nil, fmt.Errorf("cudasass diff: %w", err)
	}

	// A source-fusion failure is discarded here deliberately: this view diffs
	// SASS against SASS, and neither the instruction text nor the PC offsets it
	// compares depend on the source binding that failed. The pipeline view,
	// which does render the invalidated tiers, reports it.
	baseStalls, _ := extractSASSStallMap(baseCap, nil)
	targetStalls, _ := extractSASSStallMap(targetCap, nil)

	basePressure := extractPressureMap(baseDisasm)
	targetPressure := extractPressureMap(targetDisasm)

	data := &cudaview.DiffViewData{
		Title:           fmt.Sprintf("Differential SASS Profile Viewer: %s vs %s", filepath.Base(baseFile), filepath.Base(targetFile)),
		BaseCapture:     filepath.Base(baseFile),
		TargetCapture:   filepath.Base(targetFile),
		KernelRows:      make(map[string][]cudaview.DiffRow),
		KernelSummaries: make(map[string]cudaview.DiffSummary),
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

func alignInstructions(fd cudasass.FunctionDiff, baseStalls, targetStalls *sassStallMap, basePressure, targetPressure map[string]map[uint64]cudasass.RegisterPressure) []cudaview.DiffRow {
	var rows []cudaview.DiffRow
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
			hasSpill := (bInst != nil && bInst.IsLocalMemory) || (cInst != nil && cInst.IsLocalMemory)

			rows = append(rows, cudaview.DiffRow{
				Num:            rowNum,
				Type:           cudaview.DiffRowModify,
				Base:           bInst,
				Candidate:      cInst,
				StallDelta:     sDelta,
				StallReason:    sReason,
				StallSummary:   sSummary,
				HasLocalMemory: hasSpill,
			})
			rowNum++
		}

		// Remaining deletions
		for i := common; i < dLen; i++ {
			bInst := convertInstruction(hunkDeletes[i], fd.Name, baseStalls, basePressure)
			sDelta, sReason, sSummary := computeStallDelta(bInst, nil)
			hasSpill := bInst != nil && bInst.IsLocalMemory

			rows = append(rows, cudaview.DiffRow{
				Num:            rowNum,
				Type:           cudaview.DiffRowDelete,
				Base:           bInst,
				StallDelta:     sDelta,
				StallReason:    sReason,
				StallSummary:   sSummary,
				HasLocalMemory: hasSpill,
			})
			rowNum++
		}

		// Remaining additions
		for i := common; i < aLen; i++ {
			cInst := convertInstruction(hunkAdds[i], fd.Name, targetStalls, targetPressure)
			sDelta, sReason, sSummary := computeStallDelta(nil, cInst)
			hasSpill := cInst != nil && cInst.IsLocalMemory

			rows = append(rows, cudaview.DiffRow{
				Num:            rowNum,
				Type:           cudaview.DiffRowAdd,
				Candidate:      cInst,
				StallDelta:     sDelta,
				StallReason:    sReason,
				StallSummary:   sSummary,
				HasLocalMemory: hasSpill,
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
			hasSpill := (bInst != nil && bInst.IsLocalMemory) || (cInst != nil && cInst.IsLocalMemory)

			rows = append(rows, cudaview.DiffRow{
				Num:            rowNum,
				Type:           cudaview.DiffRowMatch,
				Base:           bInst,
				Candidate:      cInst,
				StallDelta:     sDelta,
				StallReason:    sReason,
				StallSummary:   sSummary,
				HasLocalMemory: hasSpill,
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

func convertInstruction(inst *cudasass.Instruction, kernelName string, stalls *sassStallMap, pressureMap map[string]map[uint64]cudasass.RegisterPressure) *cudaview.DiffInstruction {
	if inst == nil {
		return nil
	}

	d := &cudaview.DiffInstruction{
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
	isSpill, spillType := isLocalMemoryOpcode(inst.Opcode)
	if !isSpill && inst.Raw != "" {
		isSpill, spillType = isLocalMemoryOpcode(inst.Raw)
	}
	d.IsLocalMemory = isSpill
	d.LocalMemoryType = spillType

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

func isLocalMemoryOpcode(op string) (bool, string) {
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
		case "STL":
			return true, "STL"
		case "LDL":
			return true, "LDL"
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

func computeStallDelta(base, cand *cudaview.DiffInstruction) (float64, string, string) {
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

func buildKernelSummary(fd cudasass.FunctionDiff, baseCap, targetCap *cudacapture.Capture) cudaview.DiffSummary {
	sum := cudaview.DiffSummary{
		BaseInstructions:   fd.Architecture.BaseInstructions,
		TargetInstructions: fd.Architecture.TargetInstructions,
		InstructionDelta:   fd.Architecture.InstructionDelta,
		BaseSizeBytes:      fd.Architecture.BaseSizeBytes,
		TargetSizeBytes:    fd.Architecture.TargetSizeBytes,
		TotalSizeDelta:     fd.Architecture.TotalSizeDelta,
		BaseBranches:       fd.Architecture.BaseBranches,
		TargetBranches:     fd.Architecture.TargetBranches,
		BranchDelta:        fd.Architecture.BranchDelta,
		LocalMemoryAdded:   fd.Spills.TotalAdded,
		LocalMemoryRemoved: fd.Spills.TotalRemoved,
		LocalMemoryDelta:   fd.Spills.Delta(),
		Substitutions:      fd.Architecture.Substitutions,
	}

	baseProf := profileOrNil(baseCap)
	targetProf := profileOrNil(targetCap)

	if baseProf != nil && targetProf != nil {
		bTimeIdx := findTimeIndex(baseProf)
		tTimeIdx := findTimeIndex(targetProf)
		if bTimeIdx >= 0 && tTimeIdx >= 0 {
			var bNs, tNs int64
			for _, s := range baseProf.Sample {
				if sampleKernelName(s) == fd.Name {
					bNs += s.Value[bTimeIdx]
				}
			}
			for _, s := range targetProf.Sample {
				if sampleKernelName(s) == fd.Name {
					tNs += s.Value[tTimeIdx]
				}
			}
			if bNs > 0 || tNs > 0 {
				sum.HasDurationMetric = true
				sum.BaseRuntimeUs = float64(bNs) / 1000.0
				sum.TargetRuntimeUs = float64(tNs) / 1000.0
				sum.DurationDeltaUs = sum.TargetRuntimeUs - sum.BaseRuntimeUs
				if tNs > 0 {
					sum.Speedup = float64(bNs) / float64(tNs)
				}
			}
		}

		bSampleIdx := findSampleIndex(baseProf)
		tSampleIdx := findSampleIndex(targetProf)
		if bSampleIdx >= 0 && tSampleIdx >= 0 {
			var bSamples, tSamples, bTotal, tTotal int64
			for _, s := range baseProf.Sample {
				bTotal += s.Value[bSampleIdx]
				if sampleKernelName(s) == fd.Name {
					bSamples += s.Value[bSampleIdx]
				}
			}
			for _, s := range targetProf.Sample {
				tTotal += s.Value[tSampleIdx]
				if sampleKernelName(s) == fd.Name {
					tSamples += s.Value[tSampleIdx]
				}
			}
			sum.HasSampleMetric = true
			sum.BaseSamples = bSamples
			sum.TargetSamples = tSamples
			sum.SampleDelta = tSamples - bSamples
			if bTotal > 0 {
				sum.BaseSampleShare = float64(bSamples) / float64(bTotal) * 100.0
			}
			if tTotal > 0 {
				sum.TargetSampleShare = float64(tSamples) / float64(tTotal) * 100.0
			}
			sum.ShareShift = sum.TargetSampleShare - sum.BaseSampleShare
			if sum.SampleDelta > 0 {
				sum.SampleShiftSummary = fmt.Sprintf("+%d (%d more PC samples)", sum.SampleDelta, sum.SampleDelta)
			} else if sum.SampleDelta < 0 {
				sum.SampleShiftSummary = fmt.Sprintf("%d (%d fewer PC samples)", sum.SampleDelta, -sum.SampleDelta)
			} else {
				sum.SampleShiftSummary = "0 (no shift)"
			}
		}
	}

	// Top stall shifts for this kernel
	if baseCap != nil || targetCap != nil {
		bStalls := collectStallCounts(baseProf, fd.Name)
		tStalls := collectStallCounts(targetProf, fd.Name)
		sum.TopStallShifts = computeTopStallShifts(bStalls, tStalls)
	}

	return sum
}

func buildOverallSummary(diff *cudasass.DisassemblyDiff, baseCap, targetCap *cudacapture.Capture, activeKernel string) cudaview.DiffSummary {
	sum := cudaview.DiffSummary{
		BaseInstructions:   diff.Architecture.BaseInstructions,
		TargetInstructions: diff.Architecture.TargetInstructions,
		InstructionDelta:   diff.Architecture.InstructionDelta,
		BaseSizeBytes:      diff.Architecture.BaseSizeBytes,
		TargetSizeBytes:    diff.Architecture.TargetSizeBytes,
		TotalSizeDelta:     diff.Architecture.TotalSizeDelta,
		BaseBranches:       diff.Architecture.BaseBranches,
		TargetBranches:     diff.Architecture.TargetBranches,
		BranchDelta:        diff.Architecture.BranchDelta,
		LocalMemoryAdded:   diff.Spills.TotalAdded,
		LocalMemoryRemoved: diff.Spills.TotalRemoved,
		LocalMemoryDelta:   diff.Spills.Delta(),
		Substitutions:      diff.Architecture.Substitutions,
	}

	baseProf := profileOrNil(baseCap)
	targetProf := profileOrNil(targetCap)

	if baseProf != nil && targetProf != nil {
		bTimeIdx := findTimeIndex(baseProf)
		tTimeIdx := findTimeIndex(targetProf)
		if bTimeIdx >= 0 && tTimeIdx >= 0 {
			var bNs, tNs int64
			for _, s := range baseProf.Sample {
				bNs += s.Value[bTimeIdx]
			}
			for _, s := range targetProf.Sample {
				tNs += s.Value[tTimeIdx]
			}
			if bNs > 0 || tNs > 0 {
				sum.HasDurationMetric = true
				sum.BaseRuntimeUs = float64(bNs) / 1000.0
				sum.TargetRuntimeUs = float64(tNs) / 1000.0
				sum.DurationDeltaUs = sum.TargetRuntimeUs - sum.BaseRuntimeUs
				if tNs > 0 {
					sum.Speedup = float64(bNs) / float64(tNs)
				}
			}
		}

		bSampleIdx := findSampleIndex(baseProf)
		tSampleIdx := findSampleIndex(targetProf)
		if bSampleIdx >= 0 && tSampleIdx >= 0 {
			var bSamples, tSamples int64
			for _, s := range baseProf.Sample {
				bSamples += s.Value[bSampleIdx]
			}
			for _, s := range targetProf.Sample {
				tSamples += s.Value[tSampleIdx]
			}
			sum.HasSampleMetric = true
			sum.BaseSamples = bSamples
			sum.TargetSamples = tSamples
			sum.SampleDelta = tSamples - bSamples
			if sum.SampleDelta > 0 {
				sum.SampleShiftSummary = fmt.Sprintf("+%d (%d more PC samples)", sum.SampleDelta, sum.SampleDelta)
			} else if sum.SampleDelta < 0 {
				sum.SampleShiftSummary = fmt.Sprintf("%d (%d fewer PC samples)", sum.SampleDelta, -sum.SampleDelta)
			} else {
				sum.SampleShiftSummary = "0 (no shift)"
			}
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

func computeTopStallShifts(bStalls, tStalls map[string]int64) []cudaview.StallShiftSummary {
	allReasons := make(map[string]bool)
	for r := range bStalls {
		allReasons[r] = true
	}
	for r := range tStalls {
		allReasons[r] = true
	}

	var shifts []cudaview.StallShiftSummary
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
		shifts = append(shifts, cudaview.StallShiftSummary{
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

func newDiffServerHandler(data *cudaview.DiffViewData) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := cudaview.RenderDiff(w, data); err != nil {
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

// loadRetainedDisassembly does not invoke nvdisasm while serving a read.
func loadRetainedDisassembly(cap *cudacapture.Capture, path string, limits cudacapture.Limits) (*cudasass.Disassembly, error) {
	if cap == nil {
		return loadDisassembly(path, limits)
	}
	name := cap.Code.SASS
	if name == "" && len(cap.Modules) == 0 {
		name = "kernels.sass"
	}
	if data := cap.Files[name]; len(data) > 0 {
		return cudasass.Parse(bytes.NewReader(data))
	}
	return nil, fmt.Errorf("capture has no retained SASS for the selected module")
}
