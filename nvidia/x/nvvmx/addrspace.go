package nvvmx

import (
	"bufio"
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
	return inferAddressSpaces(ir, false, false)
}

// inferAddressSpaces performs address space inference with optional mutant flags
// for test controls:
//   - nonAbsorbing: if true, GenericRequired is non-absorbing (decays on shared write)
//   - useSiteInference: if true, infers address space at use site without full pre-scan
func inferAddressSpaces(ir []byte, nonAbsorbing, useSiteInference bool) ([]byte, *InferenceReport, error) {
	if len(ir) == 0 {
		return nil, nil, fmt.Errorf("nvvmx: empty IR")
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
				rewritten, funcReport, err := analyzeAndRewriteFunc(funcLines, globals, nonAbsorbing, useSiteInference)
				if err != nil {
					return nil, nil, fmt.Errorf("nvvmx: func at line %d: %w", funcStart+1, err)
				}
				for k, v := range funcReport.Slots {
					report.Slots[k] = v
				}
				if funcReport.Iterations > report.Iterations {
					report.Iterations = funcReport.Iterations
				}
				report.RewrittenSlots = append(report.RewrittenSlots, funcReport.RewrittenSlots...)
				output.WriteString(rewritten)
				continue
			}
			continue
		}
		output.WriteString(line)
		output.WriteString("\n")
	}

	return []byte(output.String()), report, nil
}

func splitLines(data []byte) []string {
	var lines []string
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines
}

func parseGlobalAddrSpaces(lines []string) map[string]AddrSpace {
	globals := make(map[string]AddrSpace)
	reGlobal := regexp.MustCompile(`^(@[a-zA-Z0-9_$.]+)\s*=.*addrspace\((\d+)\)`)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "@") {
			if m := reGlobal.FindStringSubmatch(trimmed); len(m) == 3 {
				as, err := strconv.Atoi(m[2])
				if err == nil {
					globals[m[1]] = AddrSpace(as)
				}
			}
		}
	}
	return globals
}

var (
	reAlloca = regexp.MustCompile(`^([%][a-zA-Z0-9_$.]+)\s*=\s*alloca\s+(.*?)(?:,\s*align\s+\d+)?$`)
	reStore  = regexp.MustCompile(`^store\s+(?:atomic\s+)?(.*?),\s*(.*?)(?:,\s*align\s+\d+)?$`)
	reLoad   = regexp.MustCompile(`^([%][a-zA-Z0-9_$.]+)\s*=\s*load\s+(.*?),\s*(.*?)(?:,\s*align\s+\d+)?$`)
	reCast   = regexp.MustCompile(`^([%][a-zA-Z0-9_$.]+)\s*=\s*addrspacecast\s+(.*?)\s+to\s+(.*?)$`)
	reGEP    = regexp.MustCompile(`^([%][a-zA-Z0-9_$.]+)\s*=\s*getelementptr\s+(?:inbounds\s+)?(.*?),\s*(.*?)$`)
)

type slotDecl struct {
	lineIdx     int
	rawLine     string
	name        string
	allocType   string
	ptrElemType string
	declaredAS  AddrSpace
	explicit    bool
}

func analyzeAndRewriteFunc(lines []string, globals map[string]AddrSpace, nonAbsorbing, useSiteInference bool) (string, *InferenceReport, error) {
	slots := make(map[string]*slotDecl)
	var slotOrder []string

	// Pass 1: Find alloca slot declarations in the function body.
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "%") && strings.Contains(trimmed, "alloca ") {
			if m := reAlloca.FindStringSubmatch(trimmed); len(m) >= 3 {
				name := m[1]
				allocType := strings.TrimSpace(m[2])
				// Strip trailing align or addrspace on the alloca itself
				if idx := strings.Index(allocType, ","); idx >= 0 {
					allocType = strings.TrimSpace(allocType[:idx])
				}
				decl := parseSlotDecl(i, line, name, allocType)
				if decl != nil {
					slots[name] = decl
					slotOrder = append(slotOrder, name)
				}
			}
		}
	}

	report := &InferenceReport{
		Slots: make(map[string]SlotInfo),
	}

	if len(slots) == 0 {
		return strings.Join(lines, "\n") + "\n", report, nil
	}

	// Initialize lattice state for each slot.
	state := make(map[string]LatticeState)
	for name := range slots {
		state[name] = StateUnconstrained
	}

	// Whole-body pre-scan fixed-point iteration.
	maxIterations := 2*len(slots) + 1
	iterations := 0

	for {
		iterations++
		if iterations > maxIterations {
			return "", nil, fmt.Errorf("fixed-point iteration exceeded bound %d", maxIterations)
		}

		changed := false

		// Map of value address spaces for intermediate registers in this pass.
		valAS := make(map[string]AddrSpace)
		for g, as := range globals {
			valAS[g] = as
		}

		for _, line := range lines {
			trimmed := strings.TrimSpace(line)

			// Track addrspacecast: %cast = addrspacecast ptr addrspace(3) %p to ptr
			if m := reCast.FindStringSubmatch(trimmed); len(m) == 4 {
				dst := m[1]
				srcPart := strings.TrimSpace(m[2])
				srcAS := parseAddrSpaceFromTypeVal(srcPart, valAS)
				valAS[dst] = srcAS
				continue
			}

			// Track getelementptr: inherits address space from base pointer
			if m := reGEP.FindStringSubmatch(trimmed); len(m) == 4 {
				dst := m[1]
				basePart := strings.TrimSpace(m[3])
				baseAS := parseAddrSpaceFromTypeVal(basePart, valAS)
				valAS[dst] = baseAS
				continue
			}

			// Track load: if loading from a slot, value inherits slot's current lattice state
			if m := reLoad.FindStringSubmatch(trimmed); len(m) == 4 {
				dst := m[1]
				ptrPart := strings.TrimSpace(m[3])
				ptrReg := extractRegister(ptrPart)
				if sDecl, ok := slots[ptrReg]; ok {
					if state[ptrReg] == StateShared {
						valAS[dst] = AddrSpaceShared
					} else if state[ptrReg] == StateGenericRequired {
						valAS[dst] = AddrSpaceGeneric
					}
					// Use-site mutant simulation: if use-site inference is enabled,
					// evaluate at first use without whole-body pre-scan.
					if useSiteInference && sDecl.declaredAS == AddrSpaceGeneric && !sDecl.explicit {
						if state[ptrReg] == StateShared {
							// Prematurely treat as shared at use site
							valAS[dst] = AddrSpaceShared
						}
					}
				}
				continue
			}

			// Track store into a slot: updates the slot's lattice state.
			if m := reStore.FindStringSubmatch(trimmed); len(m) == 3 {
				valPart := strings.TrimSpace(m[1])
				ptrPart := strings.TrimSpace(m[2])
				ptrReg := extractRegister(ptrPart)

				if _, ok := slots[ptrReg]; ok {
					valASResult := parseAddrSpaceFromTypeVal(valPart, valAS)
					curr := state[ptrReg]
					var next LatticeState

					if valASResult == AddrSpaceShared {
						switch curr {
						case StateUnconstrained:
							next = StateShared
						case StateShared:
							next = StateShared
						case StateGenericRequired:
							if nonAbsorbing {
								// Mutant: generic-required erroneously decays back to shared!
								next = StateShared
							} else {
								next = StateGenericRequired
							}
						}
					} else {
						// Any non-shared write (generic, global, constant, cluster as=7, local)
						// forces StateGenericRequired.
						next = StateGenericRequired
					}

					if next != curr {
						state[ptrReg] = next
						changed = true
					}
				}
				continue
			}
		}

		if !changed {
			break
		}
	}

	report.Iterations = iterations

	// Compute effective address space for each slot.
	toRewrite := make(map[string]bool)
	for _, name := range slotOrder {
		decl := slots[name]
		info := SlotInfo{
			Name:           name,
			Declared:       decl.declaredAS,
			Explicit:       decl.explicit,
			State:          state[name],
			IterationCount: iterations,
		}

		if decl.explicit {
			// Explicit source declaration always wins.
			info.Effective = decl.declaredAS
		} else if info.State == StateShared {
			info.Effective = AddrSpaceShared
			toRewrite[name] = true
			report.RewrittenSlots = append(report.RewrittenSlots, name)
		} else {
			info.Effective = AddrSpaceGeneric
		}

		report.Slots[name] = info
	}

	// Rewrite IR: update declarations and uses of inferred shared slots.
	var rewritten strings.Builder
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// 1. Rewrite entry-block slot declaration
		if strings.HasPrefix(trimmed, "%") && strings.Contains(trimmed, "alloca ") {
			if m := reAlloca.FindStringSubmatch(trimmed); len(m) >= 3 {
				name := m[1]
				if toRewrite[name] {
					decl := slots[name]
					newLine := rewriteAllocaLine(line, decl)
					rewritten.WriteString(newLine)
					rewritten.WriteString("\n")
					continue
				}
			}
		}

		// 2. Rewrite stores into inferred shared slots (bypass generic addrspacecast if present)
		if strings.HasPrefix(trimmed, "store ") {
			if m := reStore.FindStringSubmatch(trimmed); len(m) == 3 {
				ptrPart := strings.TrimSpace(m[2])
				ptrReg := extractRegister(ptrPart)
				if toRewrite[ptrReg] {
					newLine := rewriteStoreLine(line)
					rewritten.WriteString(newLine)
					rewritten.WriteString("\n")
					continue
				}
			}
		}

		// 3. Rewrite loads from inferred shared slots
		if strings.HasPrefix(trimmed, "%") && strings.Contains(trimmed, "load ") {
			if m := reLoad.FindStringSubmatch(trimmed); len(m) == 4 {
				ptrPart := strings.TrimSpace(m[3])
				ptrReg := extractRegister(ptrPart)
				if toRewrite[ptrReg] {
					newLine := rewriteLoadLine(line)
					rewritten.WriteString(newLine)
					rewritten.WriteString("\n")
					continue
				}
			}
		}

		rewritten.WriteString(line)
		rewritten.WriteString("\n")
	}

	return rewritten.String(), report, nil
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

func parseAddrSpaceFromTypeVal(typeVal string, valAS map[string]AddrSpace) AddrSpace {
	typeVal = strings.TrimSpace(typeVal)

	// Check if there is an embedded addrspacecast:
	// addrspacecast (ptr addrspace(3) @var to ptr)
	if strings.Contains(typeVal, "addrspacecast") {
		if strings.Contains(typeVal, "addrspace(3)") {
			return AddrSpaceShared
		}
		if strings.Contains(typeVal, "addrspace(7)") {
			return AddrSpaceCluster
		}
		return AddrSpaceGeneric
	}

	// Check for explicit addrspace annotation in type:
	// e.g. "ptr addrspace(3) %val", "float addrspace(3)* %val"
	if idx := strings.Index(typeVal, "addrspace("); idx >= 0 {
		rest := typeVal[idx+len("addrspace("):]
		if end := strings.Index(rest, ")"); end >= 0 {
			as, err := strconv.Atoi(rest[:end])
			if err == nil {
				return AddrSpace(as)
			}
		}
	}

	// Check register or global symbol in valAS map:
	reg := extractRegister(typeVal)
	if reg != "" {
		if as, ok := valAS[reg]; ok {
			return as
		}
	}

	return AddrSpaceGeneric
}

func extractRegister(s string) string {
	s = strings.TrimSpace(s)
	// Last token usually is the register or symbol: e.g. "ptr %slot" or "@shared_data"
	fields := strings.Fields(s)
	if len(fields) > 0 {
		last := fields[len(fields)-1]
		last = strings.TrimRight(last, ",)")
		if strings.HasPrefix(last, "%") || strings.HasPrefix(last, "@") {
			return last
		}
	}
	return ""
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
	// If the stored value was an addrspacecast from addrspace(3), bypass the cast.
	// E.g. "store ptr addrspacecast (ptr addrspace(3) @sym to ptr), ptr %slot"
	// -> "store ptr addrspace(3) @sym, ptr %slot"
	if strings.Contains(line, "addrspacecast") && strings.Contains(line, "addrspace(3)") {
		// Match: addrspacecast (ptr addrspace(3) <val> to ptr)
		reInlineCast := regexp.MustCompile(`addrspacecast\s*\(\s*(ptr\s+addrspace\(3\)\s+[@%][a-zA-Z0-9_$.]+)\s+to\s+ptr\s*\)`)
		if reInlineCast.MatchString(line) {
			line = reInlineCast.ReplaceAllString(line, "$1")
		}
	}
	return line
}

func rewriteLoadLine(line string) string {
	// If loading from a slot declared in addrspace(3), the loaded type is ptr addrspace(3).
	// E.g. "%p = load ptr, ptr %slot" -> "%p = load ptr addrspace(3), ptr %slot"
	idx := strings.Index(line, "load ptr,")
	if idx >= 0 {
		return line[:idx] + "load ptr addrspace(3)," + line[idx+len("load ptr,"):]
	}
	return line
}
