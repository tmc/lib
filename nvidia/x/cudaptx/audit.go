package cudaptx

import (
	"fmt"
	"strings"
)

// OpcodeSummary records instruction counts and opcodes for a kernel. It
// mirrors cudasass.OpcodeSummary so that the same audit can be written against
// PTX and against the SASS it lowers to.
type OpcodeSummary struct {
	Total      int
	Counts     map[string]int
	Predicated int
}

// Count returns the number of instructions whose opcode matches the prefix.
// Matching is case-insensitive and stops at a dot, so that a prefix names whole
// opcode fields. For example, Count("ld.global") matches "ld.global.f32" and
// "ld.global.v4.f32" but not "ld.global_nc.f32".
func (s *OpcodeSummary) Count(prefix string) int {
	if s == nil || len(s.Counts) == 0 {
		return 0
	}
	prefix = strings.ToLower(strings.TrimSpace(prefix))
	var total int
	for op, count := range s.Counts {
		opLower := strings.ToLower(op)
		if opLower == prefix || strings.HasPrefix(opLower, prefix+".") {
			total += count
		}
	}
	return total
}

// Summary returns opcode frequency statistics for this kernel.
func (k *Kernel) Summary() *OpcodeSummary {
	if k == nil {
		return &OpcodeSummary{Counts: make(map[string]int)}
	}
	summary := &OpcodeSummary{
		Total:  len(k.Instructions),
		Counts: make(map[string]int, len(k.Instructions)),
	}
	for _, inst := range k.Instructions {
		if inst.Predicate != "" {
			summary.Predicated++
		}
		if inst.Opcode != "" {
			summary.Counts[inst.Opcode]++
		}
	}
	return summary
}

// Summary returns the opcode summary for the named kernel.
func (m *Module) Summary(kernelName string) (*OpcodeSummary, error) {
	if m == nil {
		return nil, fmt.Errorf("cudaptx: nil module")
	}
	k := m.Kernel(kernelName)
	if k == nil {
		return nil, fmt.Errorf("cudaptx: kernel %q not found", kernelName)
	}
	return k.Summary(), nil
}

// AssertOpcodes checks that all required opcodes are present and forbidden
// opcodes are absent. Opcode matching is case-insensitive and prefix-aware
// (e.g. "ld.global" matches "ld.global.v4.f32").
func (m *Module) AssertOpcodes(kernelName string, require, forbid []string) error {
	sum, err := m.Summary(kernelName)
	if err != nil {
		return err
	}

	for _, req := range require {
		if sum.Count(req) == 0 {
			return fmt.Errorf("cudaptx: kernel %q missing required opcode %q", kernelName, req)
		}
	}

	for _, f := range forbid {
		if n := sum.Count(f); n > 0 {
			return fmt.Errorf("cudaptx: kernel %q contains forbidden opcode %q (count %d)", kernelName, f, n)
		}
	}

	return nil
}
