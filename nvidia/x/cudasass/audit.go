package cudasass

import (
	"fmt"
	"strings"
)

// OpcodeSummary records instruction counts and mnemonics for a kernel.
type OpcodeSummary struct {
	Total      int
	Counts     map[string]int
	Predicated int
}

// Count returns the number of instructions whose opcode matches the prefix.
// Matching is case-insensitive. For example, Count("IDP4A") matches "IDP4A",
// "IDP4A.U8.U8", "idp4a.s8.s8", and the Blackwell sm_12x variant "IDP.4A.S8.S8".
func (s *OpcodeSummary) Count(prefix string) int {
	if s == nil || len(s.Counts) == 0 {
		return 0
	}
	prefix = strings.ToUpper(strings.TrimSpace(prefix))
	var total int
	for op, count := range s.Counts {
		opUpper := strings.ToUpper(op)
		if opUpper == prefix || strings.HasPrefix(opUpper, prefix+".") {
			total += count
		} else if prefix == "IDP4A" && (opUpper == "IDP.4A" || strings.HasPrefix(opUpper, "IDP.4A.")) {
			total += count
		} else if prefix == "IDP.4A" && (opUpper == "IDP4A" || strings.HasPrefix(opUpper, "IDP4A.")) {
			total += count
		}
	}
	return total
}

// Summary returns opcode frequency statistics for this function.
func (f *Function) Summary() *OpcodeSummary {
	if f == nil {
		return &OpcodeSummary{Counts: make(map[string]int)}
	}
	s := &OpcodeSummary{
		Total:  len(f.Instructions),
		Counts: make(map[string]int),
	}
	for _, inst := range f.Instructions {
		if inst.Predicate != "" {
			s.Predicated++
		}
		if inst.Opcode != "" {
			s.Counts[inst.Opcode]++
		}
	}
	return s
}

// Summary returns the opcode summary for the named function.
func (d *Disassembly) Summary(kernelName string) (*OpcodeSummary, error) {
	if d == nil {
		return nil, fmt.Errorf("cudasass: nil disassembly")
	}
	fn := d.Function(kernelName)
	if fn == nil {
		return nil, fmt.Errorf("cudasass: kernel %q not found", kernelName)
	}
	return fn.Summary(), nil
}

// AssertOpcodes checks that all required opcodes are present and forbidden opcodes are absent.
// Mnemonic matching is case-insensitive and prefix-aware (e.g. "IDP4A" matches "IDP4A.U8.U8").
func (d *Disassembly) AssertOpcodes(kernelName string, require, forbid []string) error {
	sum, err := d.Summary(kernelName)
	if err != nil {
		return err
	}

	for _, req := range require {
		if sum.Count(req) == 0 {
			return fmt.Errorf("cudasass: kernel %q missing required opcode %q", kernelName, req)
		}
	}

	for _, f := range forbid {
		if n := sum.Count(f); n > 0 {
			return fmt.Errorf("cudasass: kernel %q contains forbidden opcode %q (count %d)", kernelName, f, n)
		}
	}

	return nil
}
