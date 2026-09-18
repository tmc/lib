package main

import (
	"errors"
	"fmt"
	"io"
	"math"
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
			fmt.Fprintf(w, "    Local-memory accesses: %+d (+%d STL, +%d LDL, -%d STL, -%d LDL)\n",
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
			fmt.Fprintf(w, "  Local-memory accesses:   +%d / -%d (net %+d)\n",
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

	// 1. Metric comparison: duration or PC-sampling
	durComp, pcComp, compMsg := compareProfiles(profileOrNil(baseCap), profileOrNil(targetCap), kernelFilter)
	if durComp != nil {
		durComp.WriteMarkdown(&sb)
	}
	if pcComp != nil {
		pcComp.WriteMarkdown(&sb)
	}
	if durComp == nil && pcComp == nil {
		sb.WriteString("## Execution Metric Comparison\n\n")
		sb.WriteString(compMsg + "\n\n")
	}

	// The duration delta above is summed per-kernel device time, which is
	// blind to inter-kernel idle. Decompose the same pair of runs into span,
	// active and idle so that sum can never stand alone as the headline.
	if durComp != nil || hasJournal(baseCap) || hasJournal(targetCap) {
		sb.WriteString(gapDecompositionMarkdown(baseCap, targetCap, basePath, targetPath))
	}

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
				sb.WriteString(fmt.Sprintf("- **Local-memory accesses**:    +%d / -%d (net %+d)\n",
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
	} else if durComp != nil && durComp.BaseTotalNs > durComp.TargetTotalNs {
		sb.WriteString("- Performance improvement observed across kernel execution with maintained pipeline balance.\n")
	} else if pcComp != nil {
		if pcComp.RawDelta < 0 {
			sb.WriteString(fmt.Sprintf("- Target run observed %d fewer PC samples (%+.1f pp shift).\n", -pcComp.RawDelta, pcComp.ShareShift))
		} else if pcComp.RawDelta > 0 {
			sb.WriteString(fmt.Sprintf("- Target run observed %d more PC samples (%+.1f pp shift).\n", pcComp.RawDelta, pcComp.ShareShift))
		} else {
			sb.WriteString("- Identical PC sample count between base and target runs.\n")
		}
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

// findTimeIndex returns the sample index of a genuine native duration metric in p,
// or -1 if no duration metric is present. It recognizes metrics such as gpu_time,
// kernel_time, time, duration_ns, cpu, or duration with nanosecond units.
// It never falls back to sample type 0.
func findTimeIndex(p *profile.Profile) int {
	if p == nil {
		return -1
	}
	for i, st := range p.SampleType {
		if isGenuineTimeMetric(st) {
			return i
		}
	}
	return -1
}

func isGenuineTimeMetric(st *profile.ValueType) bool {
	if st == nil {
		return false
	}
	t := strings.ToLower(st.Type)
	u := strings.ToLower(st.Unit)
	isNs := u == "nanoseconds" || u == "nanosecond" || u == "ns"

	switch t {
	case "gpu_time", "kernel_time", "time", "cpu", "duration":
		return isNs
	case "duration_ns":
		return isNs || u == ""
	}
	return false
}

// findSampleIndex returns the sample index of PC-sampling observations (such as
// pc_samples or samples) in p, or -1 if no PC-sampling metric is present.
func findSampleIndex(p *profile.Profile) int {
	if p == nil {
		return -1
	}
	for i, st := range p.SampleType {
		if strings.EqualFold(st.Type, "pc_samples") {
			return i
		}
	}
	for i, st := range p.SampleType {
		if strings.EqualFold(st.Type, "samples") {
			return i
		}
	}
	return -1
}

// DurationComparison compares genuine native elapsed duration metrics between two profiles.
type DurationComparison struct {
	BaseTotalNs   int64
	TargetTotalNs int64
	DeltaNs       int64
	PctSaved      float64
	Speedup       float64
	Kernels       []DurationKernelDiff
}

// DurationKernelDiff holds per-kernel duration comparison data.
type DurationKernelDiff struct {
	Name     string
	BaseNs   int64
	TargetNs int64
	DeltaNs  int64
	PctSaved float64
	Speedup  float64
}

// PCSampleComparison compares PC-sampling counts and shares between two profiles.
type PCSampleComparison struct {
	BaseTotalSamples   int64
	TargetTotalSamples int64
	RawDelta           int64
	BaseCaptureTotal   int64
	TargetCaptureTotal int64
	BaseShare          float64
	TargetShare        float64
	ShareShift         float64 // percentage points (TargetShare - BaseShare)
	Kernels            []PCSampleKernelShift
}

// PCSampleKernelShift holds per-kernel PC sample counts and share shifts.
type PCSampleKernelShift struct {
	Name          string
	BaseSamples   int64
	TargetSamples int64
	RawDelta      int64
	BaseShare     float64
	TargetShare   float64
	ShareShift    float64 // percentage points
}

func (c *DurationComparison) WriteMarkdown(sb *strings.Builder) {
	sb.WriteString("## Execution Duration Delta\n\n")
	if c.BaseTotalNs > 0 || c.TargetTotalNs > 0 {
		baseUs := float64(c.BaseTotalNs) / 1000.0
		targetUs := float64(c.TargetTotalNs) / 1000.0
		deltaUs := float64(c.DeltaNs) / 1000.0

		sb.WriteString(fmt.Sprintf("- **Base Runtime**:   %s\n", formatRuntime(baseUs)))
		sb.WriteString(fmt.Sprintf("- **Target Runtime**: %s\n", formatRuntime(targetUs)))
		sb.WriteString(fmt.Sprintf("- **Duration Delta**: %s (%.1f%% reduction, %.2f× speedup)\n\n",
			formatDeltaUs(deltaUs), c.PctSaved, c.Speedup))

		if len(c.Kernels) > 1 {
			sb.WriteString("| Kernel | Base (µs) | Target (µs) | Delta (µs) | Speedup |\n")
			sb.WriteString("|---|---|---|---|---|\n")
			for _, k := range c.Kernels {
				bUs := float64(k.BaseNs) / 1000.0
				tUs := float64(k.TargetNs) / 1000.0
				dUs := float64(k.DeltaNs) / 1000.0
				sb.WriteString(fmt.Sprintf("| `%s` | %.2f | %.2f | %+.2f | %.2f× |\n", k.Name, bUs, tUs, dUs, k.Speedup))
			}
			sb.WriteString("\n")
		}
	} else {
		sb.WriteString("No execution duration samples recorded in profiles.\n\n")
	}
}

func (c *PCSampleComparison) WriteMarkdown(sb *strings.Builder) {
	sb.WriteString("## Sample Count Shift\n\n")
	if c.BaseTotalSamples > 0 || c.TargetTotalSamples > 0 {
		sb.WriteString(fmt.Sprintf("- **Base Sample Count**:   %d (%.1f%% share)\n", c.BaseTotalSamples, c.BaseShare))
		sb.WriteString(fmt.Sprintf("- **Target Sample Count**: %d (%.1f%% share)\n", c.TargetTotalSamples, c.TargetShare))
		shiftDesc := "no shift"
		if c.RawDelta > 0 {
			shiftDesc = fmt.Sprintf("%d more PC samples", c.RawDelta)
		} else if c.RawDelta < 0 {
			shiftDesc = fmt.Sprintf("%d fewer PC samples", -c.RawDelta)
		}
		sb.WriteString(fmt.Sprintf("- **Sample Count Shift**:  %+d (%s, %+.1f pp)\n\n", c.RawDelta, shiftDesc, c.ShareShift))

		if len(c.Kernels) > 1 {
			sb.WriteString("| Kernel | Base Samples | Base Share | Target Samples | Target Share | Sample Count Shift | Percentage-Point Shift |\n")
			sb.WriteString("|---|---|---|---|---|---|---|\n")
			for _, k := range c.Kernels {
				kDesc := "0 (no shift)"
				if k.RawDelta > 0 {
					kDesc = fmt.Sprintf("+%d (%d more PC samples)", k.RawDelta, k.RawDelta)
				} else if k.RawDelta < 0 {
					kDesc = fmt.Sprintf("%d (%d fewer PC samples)", k.RawDelta, -k.RawDelta)
				}
				sb.WriteString(fmt.Sprintf("| `%s` | %d | %.1f%% | %d | %.1f%% | %s | %+.1f pp |\n",
					k.Name, k.BaseSamples, k.BaseShare, k.TargetSamples, k.TargetShare, kDesc, k.ShareShift))
			}
			sb.WriteString("\n")
		}
	} else {
		sb.WriteString("No PC samples recorded in profiles.\n\n")
	}
}

func formatRuntime(us float64) string {
	if math.Abs(us) >= 1000.0 {
		return fmt.Sprintf("%.2f ms (%.2f µs)", us/1000.0, us)
	}
	return fmt.Sprintf("%.2f µs", us)
}

func formatDeltaUs(us float64) string {
	if math.Abs(us) >= 1000.0 {
		return fmt.Sprintf("%+.2f ms (%+.2f µs)", us/1000.0, us)
	}
	return fmt.Sprintf("%+.2f µs", us)
}

func primarySampleTypeName(p *profile.Profile) string {
	if p == nil || len(p.SampleType) == 0 {
		return "none"
	}
	return p.SampleType[0].Type
}

func compareProfiles(baseProf, targetProf *profile.Profile, kernelFilter string) (*DurationComparison, *PCSampleComparison, string) {
	if baseProf == nil && targetProf == nil {
		return nil, nil, "No profile data available."
	}

	bTimeIdx := findTimeIndex(baseProf)
	tTimeIdx := findTimeIndex(targetProf)
	bSampleIdx := findSampleIndex(baseProf)
	tSampleIdx := findSampleIndex(targetProf)

	var dur *DurationComparison
	if bTimeIdx >= 0 && tTimeIdx >= 0 {
		dur = computeDurationComparison(baseProf, targetProf, bTimeIdx, tTimeIdx, kernelFilter)
	}

	var pc *PCSampleComparison
	if bSampleIdx >= 0 && tSampleIdx >= 0 {
		pc = computePCSampleComparison(baseProf, targetProf, bSampleIdx, tSampleIdx, kernelFilter)
	}

	if dur != nil || pc != nil {
		return dur, pc, ""
	}

	if baseProf == nil || targetProf == nil {
		return nil, nil, "Profile data is missing on one side."
	}

	if bTimeIdx >= 0 && tTimeIdx < 0 {
		tName := primarySampleTypeName(targetProf)
		return nil, nil, fmt.Sprintf("Mismatched sample types: base has duration metric %q, target has %q (cannot compare execution metrics).", baseProf.SampleType[bTimeIdx].Type, tName)
	}
	if bTimeIdx < 0 && tTimeIdx >= 0 {
		bName := primarySampleTypeName(baseProf)
		return nil, nil, fmt.Sprintf("Mismatched sample types: base has %q, target has duration metric %q (cannot compare execution metrics).", bName, targetProf.SampleType[tTimeIdx].Type)
	}
	if bSampleIdx >= 0 && tSampleIdx < 0 {
		tName := primarySampleTypeName(targetProf)
		return nil, nil, fmt.Sprintf("Mismatched sample types: base has sample metric %q, target has %q (cannot compare execution metrics).", baseProf.SampleType[bSampleIdx].Type, tName)
	}
	if bSampleIdx < 0 && tSampleIdx >= 0 {
		bName := primarySampleTypeName(baseProf)
		return nil, nil, fmt.Sprintf("Mismatched sample types: base has %q, target has sample metric %q (cannot compare execution metrics).", bName, targetProf.SampleType[tSampleIdx].Type)
	}

	return nil, nil, "No comparable execution duration or PC sampling metrics found in profiles."
}

func computeDurationComparison(baseProf, targetProf *profile.Profile, bIdx, tIdx int, kernelFilter string) *DurationComparison {
	comp := &DurationComparison{}
	baseKernelNs := make(map[string]int64)
	targetKernelNs := make(map[string]int64)

	for _, s := range baseProf.Sample {
		kName := sampleKernelName(s)
		if kernelFilter != "" && !strings.Contains(kName, kernelFilter) {
			continue
		}
		val := s.Value[bIdx]
		comp.BaseTotalNs += val
		baseKernelNs[kName] += val
	}

	for _, s := range targetProf.Sample {
		kName := sampleKernelName(s)
		if kernelFilter != "" && !strings.Contains(kName, kernelFilter) {
			continue
		}
		val := s.Value[tIdx]
		comp.TargetTotalNs += val
		targetKernelNs[kName] += val
	}

	comp.DeltaNs = comp.TargetTotalNs - comp.BaseTotalNs
	if comp.TargetTotalNs > 0 {
		comp.Speedup = float64(comp.BaseTotalNs) / float64(comp.TargetTotalNs)
	} else {
		comp.Speedup = 1.0
	}
	if comp.BaseTotalNs > 0 {
		comp.PctSaved = float64(comp.BaseTotalNs-comp.TargetTotalNs) / float64(comp.BaseTotalNs) * 100.0
	}

	allKernels := make(map[string]bool)
	for k := range baseKernelNs {
		allKernels[k] = true
	}
	for k := range targetKernelNs {
		allKernels[k] = true
	}
	if len(allKernels) > 1 {
		sortedKernels := make([]string, 0, len(allKernels))
		for k := range allKernels {
			sortedKernels = append(sortedKernels, k)
		}
		sort.Strings(sortedKernels)
		for _, k := range sortedKernels {
			bNs := baseKernelNs[k]
			tNs := targetKernelNs[k]
			dNs := tNs - bNs
			sp := 1.0
			if tNs > 0 {
				sp = float64(bNs) / float64(tNs)
			}
			pct := 0.0
			if bNs > 0 {
				pct = float64(bNs-tNs) / float64(bNs) * 100.0
			}
			comp.Kernels = append(comp.Kernels, DurationKernelDiff{
				Name:     k,
				BaseNs:   bNs,
				TargetNs: tNs,
				DeltaNs:  dNs,
				PctSaved: pct,
				Speedup:  sp,
			})
		}
	}

	return comp
}

func computePCSampleComparison(baseProf, targetProf *profile.Profile, bIdx, tIdx int, kernelFilter string) *PCSampleComparison {
	comp := &PCSampleComparison{}

	for _, s := range baseProf.Sample {
		comp.BaseCaptureTotal += s.Value[bIdx]
	}
	for _, s := range targetProf.Sample {
		comp.TargetCaptureTotal += s.Value[tIdx]
	}

	baseKernelSamples := make(map[string]int64)
	targetKernelSamples := make(map[string]int64)

	for _, s := range baseProf.Sample {
		kName := sampleKernelName(s)
		if kernelFilter != "" && !strings.Contains(kName, kernelFilter) {
			continue
		}
		val := s.Value[bIdx]
		comp.BaseTotalSamples += val
		baseKernelSamples[kName] += val
	}

	for _, s := range targetProf.Sample {
		kName := sampleKernelName(s)
		if kernelFilter != "" && !strings.Contains(kName, kernelFilter) {
			continue
		}
		val := s.Value[tIdx]
		comp.TargetTotalSamples += val
		targetKernelSamples[kName] += val
	}

	comp.RawDelta = comp.TargetTotalSamples - comp.BaseTotalSamples
	if comp.BaseCaptureTotal > 0 {
		comp.BaseShare = float64(comp.BaseTotalSamples) / float64(comp.BaseCaptureTotal) * 100.0
	}
	if comp.TargetCaptureTotal > 0 {
		comp.TargetShare = float64(comp.TargetTotalSamples) / float64(comp.TargetCaptureTotal) * 100.0
	}
	comp.ShareShift = comp.TargetShare - comp.BaseShare

	allKernels := make(map[string]bool)
	for k := range baseKernelSamples {
		allKernels[k] = true
	}
	for k := range targetKernelSamples {
		allKernels[k] = true
	}

	for k := range allKernels {
		bCount := baseKernelSamples[k]
		tCount := targetKernelSamples[k]
		dCount := tCount - bCount
		bShare := 0.0
		if comp.BaseCaptureTotal > 0 {
			bShare = float64(bCount) / float64(comp.BaseCaptureTotal) * 100.0
		}
		tShare := 0.0
		if comp.TargetCaptureTotal > 0 {
			tShare = float64(tCount) / float64(comp.TargetCaptureTotal) * 100.0
		}
		comp.Kernels = append(comp.Kernels, PCSampleKernelShift{
			Name:          k,
			BaseSamples:   bCount,
			TargetSamples: tCount,
			RawDelta:      dCount,
			BaseShare:     bShare,
			TargetShare:   tShare,
			ShareShift:    tShare - bShare,
		})
	}

	sort.Slice(comp.Kernels, func(i, j int) bool {
		ai, aj := math.Abs(float64(comp.Kernels[i].RawDelta)), math.Abs(float64(comp.Kernels[j].RawDelta))
		if ai != aj {
			return ai > aj
		}
		return comp.Kernels[i].Name < comp.Kernels[j].Name
	})

	return comp
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

func hasJournal(cap *cudacapture.Capture) bool {
	return cap != nil && len(cap.JournalNames()) > 0
}
