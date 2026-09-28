package nvvmx

import (
	"bytes"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// AddrSpace represents an LLVM/NVVM memory address space.
type AddrSpace int

const (
	// AddrSpaceGeneric is the generic address space (0).
	AddrSpaceGeneric AddrSpace = 0

	// AddrSpaceGlobal is global device memory (1).
	AddrSpaceGlobal AddrSpace = 1

	// AddrSpaceShared is CTA-local shared memory (3).
	AddrSpaceShared AddrSpace = 3

	// AddrSpaceConstant is device constant memory (4).
	AddrSpaceConstant AddrSpace = 4

	// AddrSpaceLocal is thread-local stack memory (5).
	AddrSpaceLocal AddrSpace = 5

	// AddrSpaceCluster is cluster-mapped shared memory (7).
	// Cluster pointers are distinct from CTA-local shared pointers and must
	// not be inferred into AddrSpaceShared.
	AddrSpaceCluster AddrSpace = 7
)

// LatticeState represents the state of a local slot in the address-space
// inference lattice. The lattice is a 3-element chain ordered by precision:
//
//	StateUnconstrained (top: no writes observed yet)
//	       |
//	       v
//	  StateShared (all writes are addrspace(3))
//	       |
//	       v
//	StateGenericRequired (bottom: absorbing, at least one write is non-shared)
//
// Because transitions are strictly monotonic and StateGenericRequired is
// absorbing, fixed-point iteration over N slots cannot oscillate and must
// terminate in at most 2*N + 1 passes.
type LatticeState int

const (
	// StateUnconstrained indicates no writes to the slot have been observed.
	StateUnconstrained LatticeState = iota

	// StateShared indicates every write observed is a CTA-local shared address (addrspace 3).
	StateShared

	// StateGenericRequired indicates at least one write is not CTA-local shared memory.
	// This state is absorbing.
	StateGenericRequired
)

func (s LatticeState) String() string {
	switch s {
	case StateUnconstrained:
		return "unconstrained"
	case StateShared:
		return "shared"
	case StateGenericRequired:
		return "generic-required"
	default:
		return fmt.Sprintf("state(%d)", int(s))
	}
}

// SlotInfo records analysis results for a local stack slot.
type SlotInfo struct {
	Name           string       // Register name, e.g. "%slot"
	Declared       AddrSpace    // Declared address space of the allocated pointer type
	Explicit       bool         // True if the source IR explicitly declared an address space
	State          LatticeState // Inferred lattice state
	Effective      AddrSpace    // Effective address space after inference
	IterationCount int          // Iterations to reach fixed point
}

// InferenceReport summarizes the findings of the address-space inference pass.
type InferenceReport struct {
	Slots          map[string]SlotInfo
	Iterations     int
	RewrittenSlots []string
}

// InferAddressSpaces pre-scans an NVVM IR module, determines for each local
// slot whether every value written into it is a shared-memory address, and
// declares that slot in addrspace(3) instead of the generic default.
//
// An explicit source declaration always wins over inference. Cluster-mapped
// pointers (addrspace 7) are never inferred to CTA-local shared memory.
func InferAddressSpaces(ir []byte) ([]byte, *InferenceReport, error) {
	if len(ir) == 0 {
		return nil, nil, fmt.Errorf("nvvmx: empty IR")
	}
	if bytes.HasPrefix(ir, []byte{'B', 'C', 0xc0, 0xde}) {
		return nil, nil, fmt.Errorf("nvvmx: bitcode input is not textual IR")
	}

	lines := splitLines(ir)
	globals := parseGlobalAddrSpaces(lines)

	report := &InferenceReport{
		Slots: make(map[string]SlotInfo),
	}

	var output strings.Builder
	inFunc := false
	var funcLines []string
	var funcStart int

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "define ") {
			inFunc = true
			funcStart = i
			funcLines = []string{line}
			continue
		}
		if inFunc {
			funcLines = append(funcLines, line)
			if trimmed == "}" {
				inFunc = false
				rewritten, funcReport, err := analyzeAndRewriteFunc(funcLines, globals)
				if err != nil {
					return nil, nil, fmt.Errorf("nvvmx: func at line %d: %w", funcStart+1, err)
				}
				funcName := functionName(funcLines[0])
				for k, v := range funcReport.Slots {
					report.Slots[funcName+"/"+k] = v
				}
				if funcReport.Iterations > report.Iterations {
					report.Iterations = funcReport.Iterations
				}
				for _, name := range funcReport.RewrittenSlots {
					report.RewrittenSlots = append(report.RewrittenSlots, funcName+"/"+name)
				}
				output.WriteString(rewritten)
				continue
			}
			continue
		}
		output.WriteString(line)
	}
	if inFunc {
		return nil, nil, fmt.Errorf("nvvmx: unterminated function at line %d", funcStart+1)
	}

	return []byte(output.String()), report, nil
}

func splitLines(data []byte) []string {
	if len(data) == 0 {
		return nil
	}
	lines := bytes.SplitAfter(data, []byte{'\n'})
	if len(lines[len(lines)-1]) == 0 {
		lines = lines[:len(lines)-1]
	}
	out := make([]string, len(lines))
	for i := range lines {
		out[i] = string(lines[i])
	}
	return out
}

var reFunctionName = regexp.MustCompile(`^\s*define\b.*?@((?:"(?:\\.|[^"])*")|[^\s(]+)\s*\(`)

func functionName(line string) string {
	if m := reFunctionName.FindStringSubmatch(line); len(m) == 2 {
		return "@" + m[1]
	}
	return "<unknown>"
}

func parseGlobalAddrSpaces(lines []string) map[string]AddrSpace {
	globals := make(map[string]AddrSpace)
	for _, line := range lines {
		if name, as, ok := parseGlobalSpace(line); ok {
			globals[name] = as
		}
	}
	return globals
}

type slotDecl struct {
	lineIdx     int
	rawLine     string
	name        string
	allocType   string
	ptrElemType string
	declaredAS  AddrSpace
	explicit    bool
}

func parseSlotDecl(lineIdx int, rawLine, name, allocType string) *slotDecl {
	// Check if allocType is a pointer type
	// Opaque pointer: "ptr" or "ptr addrspace(N)"
	// Typed pointer: "i32*", "float*", "i32 addrspace(N)*", etc.
	if allocType == "ptr" {
		return &slotDecl{
			lineIdx:     lineIdx,
			rawLine:     rawLine,
			name:        name,
			allocType:   allocType,
			declaredAS:  AddrSpaceGeneric,
			explicit:    false,
			ptrElemType: "",
		}
	}

	if strings.HasPrefix(allocType, "ptr addrspace(") {
		asStr := strings.TrimPrefix(allocType, "ptr addrspace(")
		asStr = strings.TrimSuffix(asStr, ")")
		as, err := strconv.Atoi(asStr)
		if err == nil {
			return &slotDecl{
				lineIdx:     lineIdx,
				rawLine:     rawLine,
				name:        name,
				allocType:   allocType,
				declaredAS:  AddrSpace(as),
				explicit:    true,
				ptrElemType: "",
			}
		}
	}

	if strings.HasSuffix(allocType, "*") {
		elem := strings.TrimSuffix(allocType, "*")
		explicit := false
		as := AddrSpaceGeneric
		if idx := strings.Index(elem, "addrspace("); idx >= 0 {
			asStr := elem[idx+len("addrspace("):]
			if end := strings.Index(asStr, ")"); end >= 0 {
				parsed, err := strconv.Atoi(asStr[:end])
				if err == nil {
					as = AddrSpace(parsed)
					explicit = true
				}
			}
		}
		return &slotDecl{
			lineIdx:     lineIdx,
			rawLine:     rawLine,
			name:        name,
			allocType:   allocType,
			declaredAS:  as,
			explicit:    explicit,
			ptrElemType: elem,
		}
	}

	return nil
}

func parseAddrSpaceType(typeVal string) AddrSpace {
	if idx := strings.Index(typeVal, "addrspace("); idx >= 0 {
		rest := typeVal[idx+len("addrspace("):]
		if end := strings.Index(rest, ")"); end >= 0 {
			if as, err := strconv.Atoi(rest[:end]); err == nil {
				return AddrSpace(as)
			}
		}
	}
	return AddrSpaceGeneric
}

func rewriteAllocaLine(line string, decl *slotDecl) string {
	if decl.allocType == "ptr" {
		// Change "alloca ptr" to "alloca ptr addrspace(3)"
		idx := strings.Index(line, "alloca ptr")
		if idx >= 0 {
			return line[:idx] + "alloca ptr addrspace(3)" + line[idx+len("alloca ptr"):]
		}
	}
	if strings.HasSuffix(decl.allocType, "*") && !decl.explicit {
		// e.g. "alloca float*" -> "alloca float addrspace(3)*"
		oldType := decl.allocType
		newType := decl.ptrElemType + " addrspace(3)*"
		return strings.Replace(line, "alloca "+oldType, "alloca "+newType, 1)
	}
	return line
}

func rewriteStoreLine(line string) string {
	return rewriteSharedInlineStoreCast(line)
}

func inferredPointerType(allocType string) (old, new string) {
	if allocType == "ptr" {
		return "ptr", "ptr addrspace(3)"
	}
	if strings.HasSuffix(allocType, "*") {
		return allocType, strings.TrimSuffix(allocType, "*") + " addrspace(3)*"
	}
	return allocType, allocType
}

func rewriteSlotStore(line string, decl *slotDecl) string {
	oldType, newType := inferredPointerType(decl.allocType)
	if strings.HasPrefix(strings.TrimSpace(line), "store "+newType+" ") {
		return line
	}
	return strings.Replace(line, "store "+oldType+" ", "store "+newType+" ", 1)
}

func rewriteSlotPointerOperand(line string, decl *slotDecl) string {
	if decl.allocType == "ptr" {
		return line
	}
	oldType, newType := inferredPointerType(decl.allocType)
	return strings.Replace(line, oldType+"* "+decl.name, newType+"* "+decl.name, 1)
}

func rewriteSlotLoad(line string, decl *slotDecl) string {
	oldType, newType := inferredPointerType(decl.allocType)
	return strings.Replace(line, "load "+oldType+",", "load "+newType+",", 1)
}
