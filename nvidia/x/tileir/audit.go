package tileir

import (
	"fmt"
	"strings"
)

// OpcodeSummary records instruction counts and categories for a function or module.
// It mirrors cudaptx.OpcodeSummary to provide consistent auditing across the GPU compiler stack.
type OpcodeSummary struct {
	Total      int
	Counts     map[string]int
	Categories map[Category]int
}

// Count returns the number of operations whose opcode matches prefix.
// Matching is case-insensitive, prefix-optional ("cuda_tile." prefix can be omitted),
// and dot- or underscore-aware. For example, Count("mma") matches "cuda_tile.mma_f"
// and "cuda_tile.mma_i", and Count("load_view_tko") matches "cuda_tile.load_view_tko".
func (s *OpcodeSummary) Count(prefix string) int {
	if s == nil || len(s.Counts) == 0 {
		return 0
	}
	prefix = strings.ToLower(strings.TrimSpace(prefix))
	pClean := strings.TrimPrefix(prefix, "cuda_tile.")
	var total int
	for op, count := range s.Counts {
		opLower := strings.ToLower(op)
		opClean := strings.TrimPrefix(opLower, "cuda_tile.")

		if opLower == prefix || opClean == prefix || opClean == pClean {
			total += count
			continue
		}

		if prefix == "cuda_tile" {
			if strings.HasPrefix(opLower, "cuda_tile.") {
				total += count
			}
			continue
		}

		if strings.HasPrefix(opClean, pClean+".") || strings.HasPrefix(opClean, pClean+"_") ||
			strings.HasPrefix(opLower, prefix+".") || strings.HasPrefix(opLower, prefix+"_") {
			total += count
		}
	}
	return total
}

// CategoryCount returns the number of operations classified under cat.
func (s *OpcodeSummary) CategoryCount(cat Category) int {
	if s == nil || len(s.Categories) == 0 {
		return 0
	}
	return s.Categories[cat]
}

// Summary returns opcode frequency statistics for this function.
func (fn *Function) Summary() OpcodeSummary {
	if fn == nil {
		return OpcodeSummary{
			Counts:     make(map[string]int),
			Categories: make(map[Category]int),
		}
	}
	summary := OpcodeSummary{
		Total:      len(fn.Operations),
		Counts:     make(map[string]int, len(fn.Operations)),
		Categories: make(map[Category]int),
	}
	for _, op := range fn.Operations {
		if op.OpCode != "" {
			summary.Counts[op.OpCode]++
			cat := OpcodeCategory(op.OpCode)
			summary.Categories[cat]++
		}
	}
	return summary
}

// Summary returns aggregated opcode frequency statistics across all functions in the module.
func (m *Module) Summary() OpcodeSummary {
	if m == nil {
		return OpcodeSummary{
			Counts:     make(map[string]int),
			Categories: make(map[Category]int),
		}
	}
	summary := OpcodeSummary{
		Counts:     make(map[string]int),
		Categories: make(map[Category]int),
	}
	for _, fn := range m.Functions {
		for _, op := range fn.Operations {
			summary.Total++
			if op.OpCode != "" {
				summary.Counts[op.OpCode]++
				cat := OpcodeCategory(op.OpCode)
				summary.Categories[cat]++
			}
		}
	}
	return summary
}

// Fallback opcode forbid and requirement lists for [Module.AssertOpcodes].
//
// These lists capture unoptimized fallback sequences that compilers emit when
// falling back from specialized hardware acceleration (TMA, Tensor Cores) to scalar
// memory access or unfused arithmetic.
var (
	// ForbidScalarMemoryFallbacks catches regressions where 2D/3D tensor views
	// degrade into scalar pointer offsets and loads.
	ForbidScalarMemoryFallbacks = []string{
		"cuda_tile.load_ptr_tko",
		"cuda_tile.ptr_to_int",
		"cuda_tile.int_to_ptr",
	}

	// ForbidUnfusedFallbacks catches kernels where fused tensor core operations
	// fall back to unfused elementwise arithmetic.
	ForbidUnfusedFallbacks = []string{
		"cuda_tile.mul_f",
		"cuda_tile.add_f",
		"cuda_tile.mul_i",
		"cuda_tile.add_i",
	}

	// RequireTMAOperations requires tensor view loads and stores.
	RequireTMAOperations = []string{
		"cuda_tile.load_view_tko",
		"cuda_tile.store_view_tko",
	}

	// RequireTensorCoreMMA requires Tensor Core matrix multiply operations.
	RequireTensorCoreMMA = []string{
		"cuda_tile.mma_f",
		"cuda_tile.mma_i",
		"cuda_tile.mmaf_scaled",
	}
)

// AssertOpcodes checks that required opcodes are present and forbidden fallback
// opcodes are absent. If kernelName is non-empty, the assertion evaluates against
// the named kernel; otherwise it evaluates across the entire module.
func (m *Module) AssertOpcodes(kernelName string, require, forbid []string) error {
	if m == nil {
		return fmt.Errorf("tileir: nil module")
	}

	var sum OpcodeSummary
	targetDesc := fmt.Sprintf("kernel %q", kernelName)
	if kernelName != "" {
		fn := m.FindFunction(kernelName)
		if fn == nil {
			return fmt.Errorf("tileir: kernel %q not found", kernelName)
		}
		sum = fn.Summary()
	} else {
		targetDesc = "module"
		sum = m.Summary()
	}

	for _, req := range require {
		if sum.Count(req) == 0 {
			return fmt.Errorf("tileir: %s missing required opcode %q", targetDesc, req)
		}
	}

	for _, f := range forbid {
		if n := sum.Count(f); n > 0 {
			return fmt.Errorf("tileir: %s contains forbidden opcode %q (count %d)", targetDesc, f, n)
		}
	}

	return nil
}
