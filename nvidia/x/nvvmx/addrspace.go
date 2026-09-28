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
	reGlobal := regexp.MustCompile(`^(@[a-zA-Z0-9_$.]+)\s*=([^\n]*?)\bglobal\b`)
	reAS := regexp.MustCompile(`addrspace\((\d+)\)`)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "@") {
			if m := reGlobal.FindStringSubmatch(trimmed); len(m) == 3 {
				as := AddrSpaceGeneric
				if q := reAS.FindStringSubmatch(m[2]); len(q) == 2 {
					if n, err := strconv.Atoi(q[1]); err == nil {
						as = AddrSpace(n)
					}
				}
				globals[m[1]] = as
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

var reClusterMapCall = regexp.MustCompile(`^\s*(%[a-zA-Z0-9_.$]+)\s*=\s*call\b.*@llvm\.nvvm\.mapa\.shared\.cluster\b`)

func mentionsValue(line, name string) bool {
	re := regexp.MustCompile(`(^|[^a-zA-Z0-9_.$-])` + regexp.QuoteMeta(name) + `([^a-zA-Z0-9_.$-]|$)`)
	return re.MatchString(line)
}

func supportedSlotReference(line, name string) bool {
	if strings.HasPrefix(line, name+" = alloca ") {
		return true
	}
	if strings.HasPrefix(line, "store ") && !strings.HasPrefix(line, "store atomic ") {
		if directSlotPointer(instructionPointer(line), name) {
			return true
		}
	}
	if strings.HasPrefix(line, "%") && strings.Contains(line, " = load ") && !strings.Contains(line, " = load atomic ") {
		if directSlotPointer(instructionPointer(line), name) {
			return true
		}
	}
	return false
}

func instructionPointer(line string) string {
	line = regexp.MustCompile(`,\s*align\s+[0-9]+$`).ReplaceAllString(line, "")
	if i := strings.LastIndex(line, ","); i >= 0 {
		return strings.TrimSpace(line[i+1:])
	}
	return ""
}

func directSlotPointer(pointer, name string) bool {
	pattern := `^ptr(?: addrspace\([0-9]+\))?\s+` + regexp.QuoteMeta(name) + `(?:,\s*align\s+[0-9]+)?$`
	if regexp.MustCompile(pattern).MatchString(pointer) {
		return true
	}
	return regexp.MustCompile(`^.+\s+` + regexp.QuoteMeta(name) + `(?:,\s*align\s+[0-9]+)?$`).MatchString(pointer)
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

func analyzeAndRewriteFunc(lines []string, globals map[string]AddrSpace) (string, *InferenceReport, error) {
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
		return strings.Join(lines, ""), report, nil
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
		loadedFrom := make(map[string]string)
		for g, as := range globals {
			valAS[g] = as
		}

		for _, line := range lines {
			trimmed := strings.TrimSpace(line)

			for value, owner := range loadedFrom {
				if mentionsValue(trimmed, value) && !supportedInferredPointerUse(trimmed, value) && state[owner] != StateGenericRequired {
					state[owner] = StateGenericRequired
					changed = true
				}
			}

			for name := range slots {
				if mentionsValue(trimmed, name) && !supportedSlotReference(trimmed, name) && state[name] != StateGenericRequired {
					state[name] = StateGenericRequired
					changed = true
				}
			}

			if m := reClusterMapCall.FindStringSubmatch(trimmed); len(m) == 2 {
				valAS[m[1]] = AddrSpaceCluster
				continue
			}

			// Track addrspacecast: %cast = addrspacecast ptr addrspace(3) %p to ptr
			if m := reCast.FindStringSubmatch(trimmed); len(m) == 4 {
				dst := m[1]
				srcAS := parseAddrSpaceFromTypeVal(strings.TrimSpace(m[2]), valAS)
				if srcAS == AddrSpaceShared {
					valAS[dst] = parseAddrSpaceType(strings.TrimSpace(m[3]))
				} else {
					valAS[dst] = srcAS
				}
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
				if _, ok := slots[ptrReg]; ok {
					loadedType := strings.TrimSpace(m[2])
					if strings.HasPrefix(loadedType, "ptr") || strings.HasSuffix(loadedType, "*") {
						loadedFrom[dst] = ptrReg
					}
					if state[ptrReg] == StateShared {
						valAS[dst] = AddrSpaceShared
					} else if state[ptrReg] == StateGenericRequired {
						valAS[dst] = AddrSpaceGeneric
					} else {
						valAS[dst] = AddrSpace(-1)
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
					if valASResult < 0 {
						continue
					}
					curr := state[ptrReg]
					var next LatticeState

					if valASResult == AddrSpaceShared {
						switch curr {
						case StateUnconstrained:
							next = StateShared
						case StateShared:
							next = StateShared
						case StateGenericRequired:
							next = StateGenericRequired
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
	type pointerType struct{ old, new string }
	sharedValues := make(map[string]pointerType)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if m := reLoad.FindStringSubmatch(trimmed); len(m) == 4 && toRewrite[extractRegister(strings.TrimSpace(m[3]))] {
			loadedType := strings.TrimSpace(m[2])
			if strings.HasPrefix(loadedType, "ptr") || strings.HasSuffix(loadedType, "*") {
				oldType := slots[extractRegister(strings.TrimSpace(m[3]))].allocType
				old, new := inferredPointerType(oldType)
				sharedValues[m[1]] = pointerType{old: old, new: new}
			}
		}
	}
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
					line = rewriteStoreLine(line)
					line = rewriteSlotStore(line, slots[ptrReg])
					line = rewriteSlotPointerOperand(line, slots[ptrReg])
				}
			}
		}

		// 3. Rewrite loads from inferred shared slots
		if strings.HasPrefix(trimmed, "%") && strings.Contains(trimmed, "load ") {
			if m := reLoad.FindStringSubmatch(trimmed); len(m) == 4 {
				ptrPart := strings.TrimSpace(m[3])
				ptrReg := extractRegister(ptrPart)
				if toRewrite[ptrReg] {
					newLine := rewriteSlotLoad(line, slots[ptrReg])
					newLine = rewriteSlotPointerOperand(newLine, slots[ptrReg])
					rewritten.WriteString(newLine)
					continue
				}
			}
		}
		for value, types := range sharedValues {
			if mentionsValue(trimmed, value) && supportedInferredPointerUse(trimmed, value) {
				line = rewritePointerValueUses(line, value, types.old, types.new)
			}
		}

		rewritten.WriteString(line)
	}

	return rewritten.String(), report, nil
}

func supportedInferredPointerUse(line, value string) bool {
	if m := reLoad.FindStringSubmatch(line); len(m) == 4 && extractRegister(strings.TrimSpace(m[3])) == value {
		return true
	}
	if m := reStore.FindStringSubmatch(line); len(m) == 3 && extractRegister(strings.TrimSpace(m[1])) == value {
		return true
	}
	return false
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
		if m := regexp.MustCompile(`addrspacecast\s*\(\s*(.*?)\s+to\s+[^)]+\)`).FindStringSubmatch(typeVal); len(m) == 2 {
			return parseAddrSpaceFromTypeVal(m[1], valAS)
		}
		return AddrSpaceGeneric
	}
	reg := extractRegister(typeVal)
	if as, ok := valAS[reg]; ok && as != AddrSpaceShared {
		return as
	}
	if as := parseAddrSpaceType(typeVal); as != AddrSpaceGeneric {
		return as
	}
	if reg != "" {
		if as, ok := valAS[reg]; ok {
			return as
		}
	}
	return AddrSpaceGeneric
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
		reInlineCast := regexp.MustCompile(`store ptr\s+addrspacecast\s*\(\s*(ptr\s+addrspace\(3\)\s+[@%][a-zA-Z0-9_$.]+)\s+to\s+ptr\s*\)`)
		line = reInlineCast.ReplaceAllString(line, "store $1")
	}
	return line
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

func rewritePointerValueUses(line, value, oldType, newType string) string {
	quoted := regexp.QuoteMeta(value)
	oldType = regexp.QuoteMeta(oldType)
	line = regexp.MustCompile(oldType+`\s+`+quoted+`\b`).ReplaceAllString(line, newType+" "+value)
	return line
}
