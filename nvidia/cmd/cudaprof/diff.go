package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

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

	var kernelFilter string
	posArgs := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--no-color" || arg == "-no-color":
			useColor = false
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
		return errors.New("usage: cudaprof diff [--no-color] [--kernel=name] <base.pprof|base.sass> <target.pprof|target.sass>")
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
		return nil, fmt.Errorf("capture does not contain SASS disassembly")
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
