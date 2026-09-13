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

// CategoryCount returns the number of instructions whose mnemonic belongs to cat.
func (s *OpcodeSummary) CategoryCount(cat Category) int {
	if s == nil || len(s.Counts) == 0 {
		return 0
	}
	var total int
	for op, count := range s.Counts {
		mnemonic := op
		if i := strings.Index(op, "."); i > 0 {
			mnemonic = op[:i]
		}
		if InstructionCategory(mnemonic) == cat {
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

// AuditArchitecture checks that instructions in kernelName are supported on the
// module's target architecture. If kernelName is empty, it checks all kernels.
func (m *Module) AuditArchitecture(kernelName string) error {
	if m == nil {
		return fmt.Errorf("cudaptx: nil module")
	}
	target := m.Target
	if target == "" {
		return nil
	}
	if kernelName != "" {
		k := m.Kernel(kernelName)
		if k == nil {
			return fmt.Errorf("cudaptx: kernel %q not found", kernelName)
		}
		return k.AuditArchitecture(target)
	}
	for _, k := range m.Kernels {
		if err := k.AuditArchitecture(target); err != nil {
			return err
		}
	}
	return nil
}

// AuditArchitecture verifies that the kernel's instructions are supported on target.
func (k *Kernel) AuditArchitecture(target string) error {
	if k == nil {
		return nil
	}
	for _, inst := range k.Instructions {
		req := InstructionMinTarget(inst.Mnemonic)
		if req == "" || req == "all" {
			continue
		}
		if !architectureCompatible(target, req) {
			return fmt.Errorf("cudaptx: kernel %q: instruction %q (%s) requires %s, target is %s",
				k.Name, inst.Opcode, inst.Mnemonic, req, target)
		}
	}
	return nil
}

// architectureCompatible reports whether target meets or exceeds req architecture.
func architectureCompatible(target, req string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	req = strings.ToLower(strings.TrimSpace(req))
	if req == "" || req == "all" || target == req {
		return true
	}
	parseSM := func(s string) int {
		s = strings.TrimPrefix(s, "sm_")
		var num int
		for _, ch := range s {
			if ch >= '0' && ch <= '9' {
				num = num*10 + int(ch-'0')
			} else {
				break
			}
		}
		return num
	}
	targetVal := parseSM(target)
	reqVal := parseSM(req)
	if targetVal > 0 && reqVal > 0 {
		return targetVal >= reqVal
	}
	return target >= req
}
