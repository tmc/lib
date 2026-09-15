package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func runDiffCommand(args []string) error {
	return runDiff(os.Stdout, args)
}

func runDiff(w io.Writer, rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}

	useColor := true
	if f, ok := w.(*os.File); ok {
		useColor = isTerminal(f)
	} else if os.Getenv("NO_COLOR") != "" {
		useColor = false
	}

	var explain bool
	var web bool
	var httpAddr string
	var htmlOut string
	var noBrowser bool
	var kernelFilter string
	posArgs := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--no-color" || arg == "-no-color":
			useColor = false
		case arg == "--explain" || arg == "-explain":
			explain = true
		case arg == "--web" || arg == "-web":
			web = true
		case strings.HasPrefix(arg, "--http="):
			httpAddr = strings.TrimPrefix(arg, "--http=")
			web = true
		case strings.HasPrefix(arg, "-http="):
			httpAddr = strings.TrimPrefix(arg, "-http=")
			web = true
		case strings.HasPrefix(arg, "--html="):
			htmlOut = strings.TrimPrefix(arg, "--html=")
		case strings.HasPrefix(arg, "-html="):
			htmlOut = strings.TrimPrefix(arg, "-html=")
		case strings.HasPrefix(arg, "--output="):
			htmlOut = strings.TrimPrefix(arg, "--output=")
		case strings.HasPrefix(arg, "-output="):
			htmlOut = strings.TrimPrefix(arg, "-output=")
		case arg == "--no-browser" || arg == "-no-browser" || arg == "-no_browser":
			noBrowser = true
		case strings.HasPrefix(arg, "--kernel="):
			kernelFilter = strings.TrimPrefix(arg, "--kernel=")
		case strings.HasPrefix(arg, "-kernel="):
			kernelFilter = strings.TrimPrefix(arg, "-kernel=")
		case strings.HasPrefix(arg, "-k="):
			kernelFilter = strings.TrimPrefix(arg, "-k=")
		case arg == "--kernel" || arg == "-kernel" || arg == "-k":
			if i+1 < len(args) {
				i++
				kernelFilter = args[i]
			} else {
				return errors.New("flag needs an argument: " + arg)
			}
		default:
			posArgs = append(posArgs, arg)
		}
	}

	if len(posArgs) != 2 {
		return errors.New("usage: cudaprof diff [--web] [-http=addr] [--html=file] [--no-color] [--explain] [--kernel=name] <base.pprof|base.sass> <target.pprof|target.sass>")
	}

	if web || htmlOut != "" {
		return runDiffWeb(DiffWebOptions{
			BaseFile:     posArgs[0],
			TargetFile:   posArgs[1],
			KernelFilter: kernelFilter,
			HTTPAddr:     httpAddr,
			HTMLOut:      htmlOut,
			NoBrowser:    noBrowser,
			Limits:       limits,
			Writer:       w,
		})
	}

	if explain {
		return runDiffExplain(w, posArgs[0], posArgs[1], kernelFilter, limits)
	}

	baseDisasm, err := loadDisassembly(posArgs[0], limits)
	if err != nil {
		return fmt.Errorf("read base %s: %w", posArgs[0], err)
	}

	targetDisasm, err := loadDisassembly(posArgs[1], limits)
	if err != nil {
		return fmt.Errorf("read target %s: %w", posArgs[1], err)
	}

	diff, err := cudasass.Diff(baseDisasm, targetDisasm)
	if err != nil {
		return fmt.Errorf("diff: %w", err)
	}

	return printDiff(w, diff, kernelFilter, useColor)
}

func isTerminal(f *os.File) bool {
	if os.Getenv("TERM") == "dumb" || os.Getenv("NO_COLOR") != "" {
		return false
	}
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

func loadDisassembly(path string, limits cudacapture.Limits) (*cudasass.Disassembly, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	fi, err := os.Stat(absPath)
	if err != nil {
		return nil, err
	}

	// 1. Explicit SASS text file
	if !fi.IsDir() && strings.HasSuffix(strings.ToLower(absPath), ".sass") {
		data, err := os.ReadFile(absPath)
		if err != nil {
			return nil, err
		}
		return cudasass.ParseString(string(data))
	}

	// 2. Capture file or directory
	cap, capErr := readCapturePathLimits(absPath, limits)
	if capErr == nil && cap != nil {
		if cap.Code.SASS != "" {
			if data, ok := cap.Files[cap.Code.SASS]; ok && len(data) > 0 {
				return cudasass.ParseString(string(data))
			}
		}
		if data, ok := cap.Files["kernels.sass"]; ok && len(data) > 0 {
			return cudasass.ParseString(string(data))
		}
		for name, data := range cap.Files {
			if strings.HasSuffix(name, ".sass") && len(data) > 0 {
				return cudasass.ParseString(string(data))
			}
		}
		cubinName := cap.Code.CUBIN
		if cubinName == "" {
			cubinName = "kernels.cubin"
		}
		if cubin, ok := cap.Files[cubinName]; ok && len(cubin) > 0 {
			disasm, err := cudasass.DisassembleCUBIN(cubin)
			if err == nil && disasm != nil {
				return disasm, nil
			}
		}
		return nil, fmt.Errorf("capture %s holds no .sass member and no .cubin to disassemble: cudaprof capture embeds the journal and handshake artifacts only, never device code, so SASS reaches a capture from the workload -- write kernels.cubin and kernels.sass into the capture directory as nvidia/examples/ptx-sass-trace/capture.go does with nvdisasm, then embed that directory with \"cudaprof -pack out.pprof dir/gpu.pprof\"; to diff assembly you already have on disk, pass the files themselves: cudaprof diff base.sass target.sass", absPath)
	}

	// 3. Fallback: try reading raw file as SASS text
	if !fi.IsDir() {
		data, err := os.ReadFile(absPath)
		if err == nil {
			disasm, parseErr := cudasass.ParseString(string(data))
			if parseErr == nil && len(disasm.Functions) > 0 {
				return disasm, nil
			}
		}
	}

	return nil, fmt.Errorf("could not load SASS: %w", capErr)
}

func printDiff(w io.Writer, diff *cudasass.DisassemblyDiff, kernelFilter string, useColor bool) error {
	red := func(s string) string {
		if useColor {
			return "\033[31m" + s + "\033[0m"
		}
		return s
	}
	green := func(s string) string {
		if useColor {
			return "\033[32m" + s + "\033[0m"
		}
		return s
	}
	cyan := func(s string) string {
		if useColor {
			return "\033[1;36m" + s + "\033[0m"
		}
		return s
	}
	bold := func(s string) string {
		if useColor {
			return "\033[1m" + s + "\033[0m"
		}
		return s
	}

	printedCount := 0
	for _, fd := range diff.Functions {
		if kernelFilter != "" && fd.Name != kernelFilter {
			continue
		}
		printedCount++

		fmt.Fprintf(w, "%s\n", cyan(fmt.Sprintf("=== Kernel: %s ===", fd.Name)))
		fmt.Fprintf(w, "  Base: %d instructions, %d bytes | Target: %d instructions, %d bytes\n",
			fd.Architecture.BaseInstructions, fd.Architecture.BaseSizeBytes,
			fd.Architecture.TargetInstructions, fd.Architecture.TargetSizeBytes)
		fmt.Fprintf(w, "  Delta: %+d instructions, %+d bytes, %+d branches\n\n",
			fd.Architecture.InstructionDelta, fd.Architecture.TotalSizeDelta, fd.Architecture.BranchDelta)

		if len(fd.Changes) > 0 {
			fmt.Fprintln(w, bold("  Instruction Diff:"))
			for _, c := range fd.Changes {
				lineStr := c.String()
				switch c.Op {
				case cudasass.DiffAdd:
					fmt.Fprintf(w, "  %s\n", green(lineStr))
				case cudasass.DiffDelete:
					fmt.Fprintf(w, "  %s\n", red(lineStr))
				default:
					fmt.Fprintf(w, "  %s\n", lineStr)
				}
			}
			fmt.Fprintln(w)
		}

		fmt.Fprintln(w, bold("  Architectural Summary:"))
		hasArch := false
		if fd.Spills.TotalAdded > 0 || fd.Spills.TotalRemoved > 0 {
			hasArch = true
			fmt.Fprintf(w, "    Spill operations: %+d (+%d STS, +%d LDS, -%d STS, -%d LDS)\n",
				fd.Spills.Delta(), fd.Spills.AddedSTS, fd.Spills.AddedLDS, fd.Spills.RemovedSTS, fd.Spills.RemovedLDS)
		}
		if len(fd.Architecture.Substitutions) > 0 {
			hasArch = true
			fmt.Fprintln(w, "    Opcode substitutions:")
			for _, sub := range fd.Architecture.Substitutions {
				fmt.Fprintf(w, "      %s -> %s (%d)\n", sub.From, sub.To, sub.Count)
			}
		}
		if fd.Architecture.ScalarDelta != 0 {
			hasArch = true
			fmt.Fprintf(w, "    Scalar arithmetic: %d -> %d (%+d)\n",
				fd.Architecture.BaseScalarCount, fd.Architecture.TargetScalarCount, fd.Architecture.ScalarDelta)
		}
		if fd.Architecture.SIMDDelta != 0 {
			hasArch = true
			fmt.Fprintf(w, "    SIMD operations:   %d -> %d (%+d)\n",
				fd.Architecture.BaseSIMDCount, fd.Architecture.TargetSIMDCount, fd.Architecture.SIMDDelta)
		}
		if !hasArch {
			fmt.Fprintln(w, "    No architectural differences.")
		}
		fmt.Fprintln(w)
	}

	if printedCount == 0 && kernelFilter != "" {
		return fmt.Errorf("kernel %q not found in disassembly", kernelFilter)
	}

	if printedCount > 1 {
		fmt.Fprintf(w, "%s\n", cyan("=== Overall Summary ==="))
		fmt.Fprintf(w, "  Kernels:            %d\n", printedCount)
		fmt.Fprintf(w, "  Total instructions: %d -> %d (%+d)\n",
			diff.Architecture.BaseInstructions, diff.Architecture.TargetInstructions, diff.Architecture.InstructionDelta)
		fmt.Fprintf(w, "  Total size:         %d bytes -> %d bytes (%+d bytes)\n",
			diff.Architecture.BaseSizeBytes, diff.Architecture.TargetSizeBytes, diff.Architecture.TotalSizeDelta)
		fmt.Fprintf(w, "  Total branches:     %d -> %d (%+d)\n",
			diff.Architecture.BaseBranches, diff.Architecture.TargetBranches, diff.Architecture.BranchDelta)
		if diff.Spills.TotalAdded > 0 || diff.Spills.TotalRemoved > 0 {
			fmt.Fprintf(w, "  Spill operations:   +%d / -%d (net %+d)\n",
				diff.Spills.TotalAdded, diff.Spills.TotalRemoved, diff.Spills.Delta())
		}
		if diff.Architecture.ScalarDelta != 0 || diff.Architecture.SIMDDelta != 0 {
			fmt.Fprintf(w, "  Scalar arithmetic:  %d -> %d (%+d)\n",
				diff.Architecture.BaseScalarCount, diff.Architecture.TargetScalarCount, diff.Architecture.ScalarDelta)
			fmt.Fprintf(w, "  SIMD operations:    %d -> %d (%+d)\n",
				diff.Architecture.BaseSIMDCount, diff.Architecture.TargetSIMDCount, diff.Architecture.SIMDDelta)
		}
		if len(diff.Architecture.Substitutions) > 0 {
			fmt.Fprintln(w, "  Opcode substitutions:")
			for _, sub := range diff.Architecture.Substitutions {
				fmt.Fprintf(w, "    %s -> %s (%d)\n", sub.From, sub.To, sub.Count)
			}
		}
		fmt.Fprintln(w)
	}

	return nil
}

func runDiffExplain(w io.Writer, basePath, targetPath, kernelFilter string, limits cudacapture.Limits) error {
	baseCap, err := readCapturePathLimits(basePath, limits)
	if err != nil {
		return fmt.Errorf("read base capture %s: %w", basePath, err)
	}
	targetCap, err := readCapturePathLimits(targetPath, limits)
	if err != nil {
		return fmt.Errorf("read target capture %s: %w", targetPath, err)
	}

	var sb strings.Builder
	sb.WriteString("# CUDA Profile Diff & Optimization Explanation\n\n")

	// 1. Performance Duration Delta
	var baseTotalNs, targetTotalNs int64
	baseKernelNs := make(map[string]int64)
	targetKernelNs := make(map[string]int64)

	baseTimeIdx := findTimeIndex(baseCap.Profile)
	if baseCap.Profile != nil && baseTimeIdx >= 0 {
		for _, s := range baseCap.Profile.Sample {
			kName := sampleKernelName(s)
			if kernelFilter != "" && !strings.Contains(kName, kernelFilter) {
				continue
			}
			val := s.Value[baseTimeIdx]
			baseTotalNs += val
			baseKernelNs[kName] += val
		}
	}

	targetTimeIdx := findTimeIndex(targetCap.Profile)
	if targetCap.Profile != nil && targetTimeIdx >= 0 {
		for _, s := range targetCap.Profile.Sample {
			kName := sampleKernelName(s)
			if kernelFilter != "" && !strings.Contains(kName, kernelFilter) {
				continue
			}
			val := s.Value[targetTimeIdx]
			targetTotalNs += val
			targetKernelNs[kName] += val
		}
	}

	sb.WriteString("## Execution Duration Delta\n\n")
	if baseTotalNs > 0 || targetTotalNs > 0 {
		baseUs := float64(baseTotalNs) / 1000.0
		targetUs := float64(targetTotalNs) / 1000.0
		deltaUs := targetUs - baseUs
		speedup := 1.0
		if targetTotalNs > 0 {
			speedup = float64(baseTotalNs) / float64(targetTotalNs)
		}
		pctSaved := 0.0
		if baseTotalNs > 0 {
			pctSaved = float64(baseTotalNs-targetTotalNs) / float64(baseTotalNs) * 100.0
		}

		sb.WriteString(fmt.Sprintf("- **Base Runtime**:   %.2f µs\n", baseUs))
		sb.WriteString(fmt.Sprintf("- **Target Runtime**: %.2f µs\n", targetUs))
		sb.WriteString(fmt.Sprintf("- **Duration Delta**: %+.2f µs (%.1f%% reduction, %.2f× speedup)\n\n", deltaUs, pctSaved, speedup))

		// Per-kernel breakdown if multiple kernels
		allKernels := make(map[string]bool)
		for k := range baseKernelNs {
			allKernels[k] = true
		}
		for k := range targetKernelNs {
			allKernels[k] = true
		}
		if len(allKernels) > 1 {
			sb.WriteString("| Kernel | Base (µs) | Target (µs) | Delta (µs) | Speedup |\n")
			sb.WriteString("|---|---|---|---|---|\n")
			sortedKernels := make([]string, 0, len(allKernels))
			for k := range allKernels {
				sortedKernels = append(sortedKernels, k)
			}
			sort.Strings(sortedKernels)
			for _, k := range sortedKernels {
				bUs := float64(baseKernelNs[k]) / 1000.0
				tUs := float64(targetKernelNs[k]) / 1000.0
				dUs := tUs - bUs
				sp := 1.0
				if tUs > 0 {
					sp = bUs / tUs
				}
				sb.WriteString(fmt.Sprintf("| `%s` | %.2f | %.2f | %+.2f | %.2f× |\n", k, bUs, tUs, dUs, sp))
			}
			sb.WriteString("\n")
		}
	} else {
		sb.WriteString("No execution duration samples recorded in profiles.\n\n")
	}

	// The duration delta above is summed per-kernel device time, which is
	// blind to inter-kernel idle. Decompose the same pair of runs into span,
	// active and idle so that sum can never stand alone as the headline.
	sb.WriteString(gapDecompositionMarkdown(baseCap, targetCap, basePath, targetPath))

	// 2. PC Stall Reason Shifts
	baseStalls := collectStallCounts(baseCap.Profile, kernelFilter)
	targetStalls := collectStallCounts(targetCap.Profile, kernelFilter)

	allStallReasons := make(map[string]bool)
	for r := range baseStalls {
		allStallReasons[r] = true
	}
	for r := range targetStalls {
		allStallReasons[r] = true
	}

	var explanations []string

	if len(allStallReasons) > 0 {
		sb.WriteString("## PC Sampling Stall Reason Shifts\n\n")
		sb.WriteString("| Stall Reason | Base Samples | Target Samples | Delta | Shift (%) |\n")
		sb.WriteString("|---|---|---|---|---|\n")

		sortedReasons := make([]string, 0, len(allStallReasons))
		for r := range allStallReasons {
			sortedReasons = append(sortedReasons, r)
		}
		sort.Strings(sortedReasons)

		for _, r := range sortedReasons {
			b := baseStalls[r]
			t := targetStalls[r]
			delta := t - b
			shiftStr := "N/A"
			if b > 0 {
				shift := float64(delta) / float64(b) * 100.0
				shiftStr = fmt.Sprintf("%+.1f%%", shift)

				// No narrative here on purpose. The four names this block used
				// to test for -- stall_barrier, stall_mem_throttle,
				// stall_memory_throttle, stall_short_scoreboard -- are none of
				// the ten x/cudaprof emits, so none of them ever fired. Writing
				// the narrative against the real names is a judgement call and
				// not a rename: stall_wait merges barrier, wait and membar, so
				// a barrier claim over-reaches, and stall_mem carries latency
				// waits, so a throttle claim names the wrong mechanism. The
				// table below reports every real reason and is what works.
			} else if t > 0 {
				shiftStr = "+inf"
			}
			sb.WriteString(fmt.Sprintf("| `%s` | %d | %d | %+d | %s |\n", r, b, t, delta, shiftStr))
		}
		sb.WriteString("\n")
	}

	// 3. SASS Instruction Architecture Diff
	baseDisasm, err1 := loadDisassembly(basePath, limits)
	targetDisasm, err2 := loadDisassembly(targetPath, limits)
	if err1 == nil && err2 == nil && baseDisasm != nil && targetDisasm != nil {
		diff, err := cudasass.Diff(baseDisasm, targetDisasm)
		if err == nil && diff != nil {
			sb.WriteString("## SASS Assembly Differences\n\n")
			sb.WriteString(fmt.Sprintf("- **Instruction Count**: %d -> %d (%+d)\n",
				diff.Architecture.BaseInstructions, diff.Architecture.TargetInstructions, diff.Architecture.InstructionDelta))
			sb.WriteString(fmt.Sprintf("- **Binary Size**:        %d bytes -> %d bytes (%+d bytes)\n",
				diff.Architecture.BaseSizeBytes, diff.Architecture.TargetSizeBytes, diff.Architecture.TotalSizeDelta))
			sb.WriteString(fmt.Sprintf("- **Branch Instructions**: %d -> %d (%+d)\n",
				diff.Architecture.BaseBranches, diff.Architecture.TargetBranches, diff.Architecture.BranchDelta))
			if diff.Spills.TotalAdded > 0 || diff.Spills.TotalRemoved > 0 {
				sb.WriteString(fmt.Sprintf("- **Register Spills**:    +%d / -%d (net %+d)\n",
					diff.Spills.TotalAdded, diff.Spills.TotalRemoved, diff.Spills.Delta()))
			}
			if len(diff.Architecture.Substitutions) > 0 {
				sb.WriteString("- **Opcode Substitutions**:\n")
				for _, sub := range diff.Architecture.Substitutions {
					sb.WriteString(fmt.Sprintf("  - `%s` -> `%s` (%d occurrences)\n", sub.From, sub.To, sub.Count))
				}
			}
			sb.WriteString("\n")
		}
	}

	// 4. Architectural Synthesis
	sb.WriteString("## Architectural Synthesis\n\n")
	if len(explanations) > 0 {
		for _, exp := range explanations {
			sb.WriteString(fmt.Sprintf("- %s\n", exp))
		}
	} else if baseTotalNs > targetTotalNs {
		sb.WriteString("- Performance improvement observed across kernel execution with maintained pipeline balance.\n")
	} else {
		sb.WriteString("- No significant architectural bottlenecks or shifts identified between runs.\n")
	}

	_, err = io.WriteString(w, sb.String())
	return err
}

func sampleKernelName(s *profile.Sample) string {
	if s == nil {
		return "unknown"
	}
	if s.Label != nil {
		if k, ok := s.Label["kernel"]; ok && len(k) > 0 && k[0] != "" {
			return k[0]
		}
	}
	for _, loc := range s.Location {
		for _, line := range loc.Line {
			if line.Function != nil && line.Function.Name != "" {
				return line.Function.Name
			}
		}
	}
	return "unknown"
}

func findTimeIndex(p *profile.Profile) int {
	if p == nil {
		return -1
	}
	for i, st := range p.SampleType {
		tLower := strings.ToLower(st.Type)
		uLower := strings.ToLower(st.Unit)
		if strings.Contains(tLower, "time") || strings.Contains(uLower, "nanosecond") || strings.Contains(uLower, "ns") {
			return i
		}
	}
	if len(p.SampleType) > 0 && !strings.HasPrefix(p.SampleType[0].Type, "stall_") {
		return 0
	}
	return -1
}

func collectStallCounts(p *profile.Profile, kernelFilter string) map[string]int64 {
	counts := make(map[string]int64)
	if p == nil {
		return counts
	}
	stallIndices := make(map[int]string)
	for i, st := range p.SampleType {
		if strings.HasPrefix(st.Type, "stall_") {
			stallIndices[i] = st.Type
		}
	}
	for _, s := range p.Sample {
		kName := sampleKernelName(s)
		if kernelFilter != "" && !strings.Contains(kName, kernelFilter) {
			continue
		}
		for idx, name := range stallIndices {
			val := s.Value[idx]
			if val > 0 {
				counts[name] += val
			}
		}
	}
	return counts
}
