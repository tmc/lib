package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// The forward kernels are compiled at run time by NVRTC and their PTX is
// retained on GPUModules, but nothing read it. These functions inspect that
// PTX: what each entry point compiled to, and whether the weight dtype the
// dispatch selected is the one the generated code actually loads.
//
// PTX is the compiler's output, not the machine's. It says what was asked of
// the hardware, never what the hardware did; a kernel that appears here has
// been compiled, not executed.

// halfConvert is the instruction forward.cu's f16_to_f32 emits as inline PTX.
// The bf16 path has no counterpart: bf16_to_f32 is an integer shift.
const halfConvert = "cvt.f32.f16"

// ptxKernelStat summarizes one entry point in a compiled module.
type ptxKernelStat struct {
	Name         string
	Instructions int
	Mnemonics    map[string]int // base mnemonic to count, e.g. "ld" to 42
}

// ptxKernelStats parses ptx and summarizes every kernel it defines.
func ptxKernelStats(ptx string) ([]ptxKernelStat, error) {
	mod, err := cudaptx.ParseString(ptx)
	if err != nil {
		return nil, fmt.Errorf("parse ptx: %w", err)
	}
	stats := make([]ptxKernelStat, 0, len(mod.Kernels))
	for _, k := range mod.Kernels {
		s := ptxKernelStat{Name: k.Name, Instructions: len(k.Instructions), Mnemonics: map[string]int{}}
		for _, inst := range k.Instructions {
			s.Mnemonics[inst.Mnemonic]++
		}
		stats = append(stats, s)
	}
	return stats, nil
}

// ptxCountOpcode reports how many instructions in the named kernel have an
// opcode containing substr. A missing kernel counts zero and reports false.
func ptxCountOpcode(ptx, kernel, substr string) (int, bool, error) {
	mod, err := cudaptx.ParseString(ptx)
	if err != nil {
		return 0, false, fmt.Errorf("parse ptx: %w", err)
	}
	k := mod.Kernel(kernel)
	if k == nil {
		return 0, false, nil
	}
	n := 0
	for _, inst := range k.Instructions {
		if strings.Contains(inst.Opcode, substr) {
			n++
		}
	}
	return n, true, nil
}

// checkWeightKernelPTX reports whether the compiled PTX for entry agrees with
// the weight dtype the dispatch selected.
//
// Binding checks that an entry point of the right name exists. This checks
// what that entry point compiled to, which is a different question: an F16
// name whose body holds no cvt.f32.f16 is not loading half weights whatever it
// is called. WeightBF16 converts by shifting, so it has no instruction of its
// own to require and only the absence of the f16 conversion is checked.
func checkWeightKernelPTX(ptx string, d WeightDType, entry string) error {
	n, found, err := ptxCountOpcode(ptx, entry, halfConvert)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("kernel %s is not defined in the compiled ptx", entry)
	}
	switch d {
	case WeightF16:
		if n == 0 {
			return fmt.Errorf("kernel %s is bound for f16 weights but its ptx holds no %s", entry, halfConvert)
		}
	default:
		if n > 0 {
			return fmt.Errorf("kernel %s is bound for %s weights but its ptx holds %d %s", entry, d, n, halfConvert)
		}
	}
	return nil
}

// writePTXReport writes one line per kernel: instruction count and the four
// most frequent mnemonics.
func writePTXReport(w io.Writer, ptx string) error {
	stats, err := ptxKernelStats(ptx)
	if err != nil {
		return err
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].Name < stats[j].Name })
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "KERNEL\tINSTRUCTIONS\tTOP MNEMONICS")
	for _, s := range stats {
		fmt.Fprintf(tw, "%s\t%d\t%s\n", s.Name, s.Instructions, topMnemonics(s.Mnemonics, 4))
	}
	return tw.Flush()
}

// topMnemonics formats the n most frequent mnemonics, ties broken by name so
// the report is stable across runs.
func topMnemonics(m map[string]int, n int) string {
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	sort.Slice(names, func(i, j int) bool {
		if m[names[i]] != m[names[j]] {
			return m[names[i]] > m[names[j]]
		}
		return names[i] < names[j]
	})
	if len(names) > n {
		names = names[:n]
	}
	parts := make([]string, len(names))
	for i, name := range names {
		parts[i] = fmt.Sprintf("%s=%d", name, m[name])
	}
	return strings.Join(parts, " ")
}
