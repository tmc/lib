package nvvmx

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// irID is one complete local or global LLVM identifier. Quoted identifiers
// are kept byte-for-byte so `%p` never aliases `%p.addr` or `%-p`.
type irID struct {
	text  string
	local bool
}

func llvmIDs(s string) []irID {
	var ids []irID
	for i := 0; i < len(s); {
		if s[i] == ';' {
			break
		}
		if s[i] != '%' && s[i] != '@' {
			if s[i] == '"' {
				i = skipLLVMQuoted(s, i)
			} else {
				i++
			}
			continue
		}
		start := i
		local := s[i] == '%'
		i++
		if i < len(s) && s[i] == '"' {
			i = skipLLVMQuoted(s, i)
			ids = append(ids, irID{text: s[start:i], local: local})
			continue
		}
		for i < len(s) && isLLVMNameByte(s[i]) {
			i++
		}
		if i == start+1 {
			continue
		}
		ids = append(ids, irID{text: s[start:i], local: local})
	}
	return ids
}

func skipLLVMQuoted(s string, i int) int {
	i++
	for i < len(s) {
		if s[i] == '\\' {
			i += 2
			continue
		}
		if s[i] == '"' {
			return i + 1
		}
		i++
	}
	return len(s)
}

func isLLVMNameByte(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || b == '_' || b == '.' || b == '$' || b == '-'
}

type parsedMemory struct {
	kind      string
	result    string
	valueType string
	value     string
	valueID   string
	ptrType   string
	ptrID     string
}

var simpleTypedPointer = regexp.MustCompile(`^(?:i[0-9]+|half|float|double)(?: addrspace\([0-9]+\))?\*+$`)
var supportedSlotPointer = regexp.MustCompile(`^(?:i[0-9]+|half|float|double)(?: addrspace\([0-9]+\))?\*$`)
var alignClause = regexp.MustCompile(`^align [0-9]+$`)
var scalarGEPElementType = regexp.MustCompile(`^(?:i[0-9]+|half|float|double)$`)

func llvmPointerType(s string) bool {
	s = strings.TrimSpace(s)
	if s == "ptr" || strings.HasPrefix(s, "ptr addrspace(") {
		return true
	}
	return simpleTypedPointer.MatchString(s)
}

func supportedSlotType(s string) bool {
	s = strings.TrimSpace(s)
	return strings.HasPrefix(s, "ptr") || supportedSlotPointer.MatchString(s)
}

func llvmTypeAS(s string) AddrSpace {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "ptr") {
		return parseAddrSpaceType(s)
	}
	if strings.HasSuffix(s, "*") {
		return parseAddrSpaceType(strings.TrimSuffix(s, "*"))
	}
	return AddrSpaceGeneric
}

func splitLLVMTopLevel(s string) []string {
	var out []string
	start, paren, bracket, brace, angle := 0, 0, 0, 0, 0
	quoted, escaped := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if quoted {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				quoted = false
			}
			continue
		}
		if c == '"' {
			quoted = true
			continue
		}
		switch c {
		case '(':
			paren++
		case ')':
			paren--
		case '[':
			bracket++
		case ']':
			bracket--
		case '{':
			brace++
		case '}':
			brace--
		case '<':
			angle++
		case '>':
			if angle > 0 {
				angle--
			}
		case ',':
			if paren == 0 && bracket == 0 && brace == 0 && angle == 0 {
				out = append(out, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	return append(out, strings.TrimSpace(s[start:]))
}

func assignment(line string) (string, string, bool) {
	line = stripLLVMComment(strings.TrimSpace(line))
	ids := llvmIDs(line)
	if len(ids) == 0 || !ids[0].local {
		return "", line, false
	}
	pos := strings.Index(line, ids[0].text)
	if pos < 0 {
		return "", line, false
	}
	rest := strings.TrimSpace(line[pos+len(ids[0].text):])
	if !strings.HasPrefix(rest, "=") {
		return "", line, false
	}
	return ids[0].text, strings.TrimSpace(rest[1:]), true
}

func stripLLVMComment(s string) string {
	quoted, escaped := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if quoted {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				quoted = false
			}
			continue
		}
		if c == '"' {
			quoted = true
			continue
		}
		if c == ';' {
			return s[:i]
		}
	}
	return s
}

func parseAlloca(line string) (name, allocType string, ok bool) {
	name, rhs, assigned := assignment(line)
	if !assigned || !strings.HasPrefix(rhs, "alloca ") {
		return "", "", false
	}
	parts := splitLLVMTopLevel(strings.TrimSpace(strings.TrimPrefix(rhs, "alloca ")))
	if len(parts) == 0 || len(parts) > 2 {
		return "", "", false
	}
	if len(parts) == 2 && !alignClause.MatchString(parts[1]) {
		return "", "", false
	}
	allocType = strings.TrimSpace(parts[0])
	if !supportedSlotType(allocType) {
		return name, allocType, false
	}
	return name, allocType, true
}

func parseSimplePointerOperand(s string) (typ, id string, ok bool) {
	s = strings.TrimSpace(s)
	ids := llvmIDs(s)
	if len(ids) != 1 || !ids[0].local {
		return "", "", false
	}
	id = ids[0].text
	pos := strings.LastIndex(s, id)
	if pos < 0 || strings.TrimSpace(s[pos+len(id):]) != "" {
		return "", "", false
	}
	typ = strings.TrimSpace(s[:pos])
	if !llvmPointerType(typ) {
		return "", "", false
	}
	return typ, id, true
}

func parseMemory(line string) (parsedMemory, bool) {
	line = stripLLVMComment(strings.TrimSpace(line))
	result, rhs, assigned := assignment(line)
	if !assigned {
		rhs = line
	}
	kind := ""
	if strings.HasPrefix(rhs, "store ") {
		kind = "store"
		rhs = strings.TrimSpace(strings.TrimPrefix(rhs, "store "))
	} else if strings.HasPrefix(rhs, "load ") {
		kind = "load"
		rhs = strings.TrimSpace(strings.TrimPrefix(rhs, "load "))
	} else {
		return parsedMemory{}, false
	}
	if strings.HasPrefix(rhs, "atomic ") || strings.HasPrefix(rhs, "volatile ") {
		return parsedMemory{}, false
	}
	parts := splitLLVMTopLevel(rhs)
	if len(parts) < 2 || len(parts) > 3 {
		return parsedMemory{}, false
	}
	if len(parts) == 3 && !alignClause.MatchString(parts[2]) {
		return parsedMemory{}, false
	}
	if kind == "store" && assigned || kind == "load" && !assigned {
		return parsedMemory{}, false
	}
	var valueType, value, ptrPart string
	if kind == "store" {
		valueType, value, _ = parseTypedValue(parts[0])
		ptrPart = parts[1]
	} else {
		valueType = strings.TrimSpace(parts[0])
		ptrPart = parts[1]
	}
	ptrType, ptrID, ok := parseSimplePointerOperand(ptrPart)
	if !ok {
		return parsedMemory{}, false
	}
	ids := llvmIDs(value)
	valueID := ""
	if len(ids) == 1 {
		valueID = ids[0].text
	}
	return parsedMemory{kind: kind, result: result, valueType: valueType, value: value, valueID: valueID, ptrType: ptrType, ptrID: ptrID}, true
}

func parseTypedValue(s string) (typ, value, castSource string) {
	s = strings.TrimSpace(s)
	if castAt := strings.Index(s, " addrspacecast ("); castAt > 0 {
		outerType := strings.TrimSpace(s[:castAt])
		castType, _, source := parseTypedValue(strings.TrimSpace(s[castAt+1:]))
		if llvmPointerType(outerType) && castType != "" {
			return outerType, strings.TrimSpace(s[castAt+1:]), source
		}
	}
	if strings.HasPrefix(s, "addrspacecast (") {
		inner := strings.TrimSuffix(strings.TrimPrefix(s, "addrspacecast ("), ")")
		parts := splitToTopLevel(inner, " to ")
		if len(parts) == 2 {
			srcType, srcValue, _ := parseTypedValue(parts[0])
			if llvmPointerType(srcType) && llvmPointerType(strings.TrimSpace(parts[1])) {
				return strings.TrimSpace(parts[1]), s, srcType + " " + srcValue
			}
		}
		return "", s, ""
	}
	if strings.HasPrefix(s, "ptr") {
		end := 3
		if len(s) > end && s[end] == ' ' && strings.HasPrefix(s[end:], " addrspace(") {
			p := strings.Index(s[end:], ")")
			if p < 0 {
				return "", s, ""
			}
			end += p + 1
		}
		if len(s) > end && s[end] != ' ' {
			return "", s, ""
		}
		remainder := strings.TrimSpace(s[end:])
		if strings.HasPrefix(remainder, "addrspacecast (") {
			_, _, source := parseTypedValue(remainder)
			return strings.TrimSpace(s[:end]), remainder, source
		}
		return strings.TrimSpace(s[:end]), remainder, ""
	}
	ids := llvmIDs(s)
	if len(ids) == 0 {
		return "", s, ""
	}
	last := ids[len(ids)-1].text
	pos := strings.LastIndex(s, last)
	if pos <= 0 {
		return "", s, ""
	}
	typ = strings.TrimSpace(s[:pos])
	if !llvmPointerType(typ) {
		return "", s, ""
	}
	return typ, strings.TrimSpace(s[pos:]), ""
}

func splitToTopLevel(s, delimiter string) []string {
	paren, bracket, brace := 0, 0, 0
	quoted, escaped := false, false
	for i := 0; i+len(delimiter) <= len(s); i++ {
		c := s[i]
		if quoted {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				quoted = false
			}
			continue
		}
		if c == '"' {
			quoted = true
			continue
		}
		switch c {
		case '(':
			paren++
		case ')':
			paren--
		case '[':
			bracket++
		case ']':
			bracket--
		case '{':
			brace++
		case '}':
			brace--
		}
		if paren == 0 && bracket == 0 && brace == 0 && strings.HasPrefix(s[i:], delimiter) {
			return []string{strings.TrimSpace(s[:i]), strings.TrimSpace(s[i+len(delimiter):])}
		}
	}
	return []string{s}
}

type parsedLine struct {
	index      int
	raw        string
	code       string
	ids        []irID
	result     string
	mem        parsedMemory
	hasMem     bool
	allocaName string
	allocType  string
	hasAlloca  bool
}

type parsedFunction struct {
	name         string
	lines        []parsedLine
	uses         map[string][]int
	defs         map[string]int
	duplicateDef map[string]bool
}

func parseFunctionLines(lines []string) parsedFunction {
	f := parsedFunction{name: functionName(lines[0]), uses: make(map[string][]int), defs: make(map[string]int), duplicateDef: make(map[string]bool)}
	for i, raw := range lines {
		code := stripLLVMComment(strings.TrimSpace(raw))
		p := parsedLine{index: i, raw: raw, code: code, ids: llvmIDs(code)}
		if name, typ, ok := parseAlloca(code); ok {
			p.allocaName, p.allocType, p.hasAlloca = name, typ, true
		}
		if result, _, ok := assignment(code); ok {
			p.result = result
			if _, exists := f.defs[result]; exists {
				// Duplicate local definitions are invalid IR. Keep any slot
				// depending on this value generic instead of choosing a definition.
				f.duplicateDef[result] = true
			}
			f.defs[result] = i
		}
		if mem, ok := parseMemory(code); ok {
			p.mem, p.hasMem = mem, true
		}
		for n, id := range p.ids {
			if n == 0 && id.local && id.text == p.result {
				continue
			}
			f.uses[id.text] = append(f.uses[id.text], i)
		}
		f.lines = append(f.lines, p)
	}
	return f
}

func genericSlotPointerType(slot *slotDecl) string {
	if slot.allocType == "ptr" {
		return "ptr"
	}
	return slot.allocType + "*"
}

func accessPointerType(slot *slotDecl) string {
	return genericSlotPointerType(slot)
}

func valueASForOperand(_ string, value string, valAS, globals map[string]AddrSpace) AddrSpace {
	value = strings.TrimSpace(value)
	if strings.HasPrefix(value, "addrspacecast (") {
		dstType, _, src := parseTypedValue(value)
		if src == "" || !llvmPointerType(dstType) || llvmTypeAS(dstType) != AddrSpaceGeneric {
			return AddrSpaceGeneric
		}
		srcType, srcValue, _ := parseTypedValue(src)
		if llvmTypeAS(srcType) == AddrSpaceShared {
			return valueASForOperand(srcType, srcValue, valAS, globals)
		}
		return AddrSpaceGeneric
	}
	ids := llvmIDs(value)
	if len(ids) == 1 {
		id := ids[0].text
		if strings.TrimSpace(value) != id {
			return AddrSpaceGeneric
		}
		if ids[0].local {
			if as, ok := valAS[id]; ok {
				return as
			}
		} else if as, ok := globals[id]; ok {
			return as
		}
	}
	return AddrSpaceGeneric
}

func parseGlobalSpace(line string) (string, AddrSpace, bool) {
	line = stripLLVMComment(strings.TrimSpace(line))
	ids := llvmIDs(line)
	if len(ids) == 0 || ids[0].local || !strings.HasPrefix(line, ids[0].text) {
		return "", 0, false
	}
	eq := strings.Index(line, "=")
	if eq < 0 || eq < len(ids[0].text) {
		return "", 0, false
	}
	globalAt := llvmKeyword(line, "global", eq+1)
	if globalAt < 0 {
		return "", 0, false
	}
	return ids[0].text, globalAddressSpace(line[eq+1 : globalAt]), true
}

func globalAddressSpace(s string) AddrSpace {
	quoted, escaped := false, false
	depth := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if quoted {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				quoted = false
			}
			continue
		}
		if c == '"' {
			quoted = true
			continue
		}
		if c == '(' {
			depth++
			continue
		}
		if c == ')' {
			if depth > 0 {
				depth--
			}
			continue
		}
		if depth != 0 || !strings.HasPrefix(s[i:], "addrspace(") || i > 0 && !isSpace(s[i-1]) {
			continue
		}
		start := i + len("addrspace(")
		end := strings.IndexByte(s[start:], ')')
		if end < 1 {
			continue
		}
		if !digits(s[start : start+end]) {
			continue
		}
		if as, err := strconv.Atoi(s[start : start+end]); err == nil {
			return AddrSpace(as)
		}
	}
	return AddrSpaceGeneric
}

func isSpace(b byte) bool { return b == ' ' || b == '\t' || b == '\n' || b == '\r' }

func llvmKeyword(s, word string, start int) int {
	quoted, escaped := false, false
	depth := 0
	for i := start; i+len(word) <= len(s); i++ {
		c := s[i]
		if quoted {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				quoted = false
			}
			continue
		}
		if c == '"' {
			quoted = true
			continue
		}
		switch c {
		case '(', '[', '{':
			depth++
			continue
		case ')', ']', '}':
			if depth > 0 {
				depth--
			}
			continue
		}
		if depth != 0 {
			continue
		}
		if !strings.HasPrefix(s[i:], word) {
			continue
		}
		beforeOK := i == 0 || !(isLLVMNameByte(s[i-1]) || s[i-1] == '%')
		after := i + len(word)
		afterOK := after == len(s) || !(isLLVMNameByte(s[after]) || s[after] == '%')
		if beforeOK && afterOK {
			return i
		}
	}
	return -1
}

func inferFunctionAndRewrite(lines []string, globals map[string]AddrSpace) (string, *InferenceReport, error) {
	f := parseFunctionLines(lines)
	slots := make(map[string]*slotDecl)
	candidates := make(map[string]bool)
	var slotOrder []string
	for _, line := range f.lines {
		if name, typ, ok := parseAlloca(line.code); ok {
			decl := parseSlotDecl(line.index, line.raw, name, typ)
			if decl != nil {
				slots[name] = decl
				slotOrder = append(slotOrder, name)
				candidates[name] = !decl.explicit
			}
		}
	}
	report := &InferenceReport{Slots: make(map[string]SlotInfo)}
	if len(slots) == 0 {
		return strings.Join(lines, ""), report, nil
	}

	// Record recognized accesses and value provenance without relying on textual
	// definition order. Any mention not accounted for below makes the slot escape.
	loadedFrom := make(map[string]string)
	depStores := make(map[string][]string) // destination slot -> source slot
	storeSources := make(map[string][]AddrSpace)
	bad := make(map[string]bool)
	lineSlotAccess := make(map[int]string)
	for name := range slots {
		if !candidates[name] {
			continue
		}
		if f.duplicateDef[name] {
			bad[name] = true
		}
		for _, useLine := range f.uses[name] {
			p := f.lines[useLine]
			if p.hasAlloca && p.allocaName == name {
				continue
			}
			if !p.hasMem || p.mem.ptrID != name || !p.mem.ptrTypeCompatible(slots[name]) {
				bad[name] = true
				continue
			}
			if p.mem.kind == "load" {
				if p.mem.valueType != slots[name].allocType {
					bad[name] = true
					continue
				}
				loadedFrom[p.mem.result] = name
				if f.duplicateDef[p.mem.result] {
					bad[name] = true
				}
				lineSlotAccess[useLine] = name
				continue
			}
			if p.mem.kind == "store" {
				lineSlotAccess[useLine] = name
				continue
			}
			bad[name] = true
		}
	}

	// Trace only the explicit forms in the package contract. Unknown definitions
	// stay generic. Function arguments retain the address space written in their
	// parameter type.
	valAS := make(map[string]AddrSpace)
	for id, as := range globals {
		valAS[id] = as
	}
	for _, id := range llvmIDs(lines[0]) {
		if id.local {
			valAS[id.text] = parameterAddressSpace(lines[0], id.text)
		}
	}
	for pass := 0; pass <= len(f.lines); pass++ {
		changed := false
		for _, p := range f.lines {
			if p.result == "" {
				continue
			}
			if as, ok := traceAddressSpace(p.code, valAS, globals); ok && valAS[p.result] != as {
				valAS[p.result] = as
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	for name, slot := range slots {
		if !candidates[name] {
			continue
		}
		for _, lineIdx := range f.uses[name] {
			p := f.lines[lineIdx]
			if p.hasAlloca && p.allocaName == name {
				continue
			}
			if lineSlotAccess[lineIdx] != name {
				bad[name] = true
				continue
			}
			if p.mem.kind == "store" {
				oldType, newType := inferredPointerType(slot.allocType)
				if !llvmPointerType(p.mem.valueType) || p.mem.valueType != oldType && p.mem.valueType != newType {
					bad[name] = true
					continue
				}
				if src, ok := loadedFrom[p.mem.valueID]; ok && p.mem.valueID != "" {
					if slots[src].allocType != slot.allocType {
						bad[name] = true
						bad[src] = true
						continue
					}
					depStores[name] = append(depStores[name], src)
					continue
				}
				as := valueASForOperand(p.mem.valueType, p.mem.value, valAS, globals)
				inlineCast := strings.HasPrefix(strings.TrimSpace(p.mem.value), "addrspacecast (")
				storeSources[name] = append(storeSources[name], as)
				if as != AddrSpaceShared || llvmTypeAS(p.mem.valueType) != AddrSpaceShared && !inlineCast {
					bad[name] = true
				}
			}
		}
	}

	// A loaded pointer may only feed a supported pointer load or a store into a
	// candidate slot. This checks all references from the def-use map, including
	// references before the load in textual order.
	for value, src := range loadedFrom {
		if f.duplicateDef[value] {
			bad[src] = true
		}
		for _, useLine := range f.uses[value] {
			p := f.lines[useLine]
			pointerUse := p.hasMem && p.mem.ptrID == value
			valueUse := p.hasMem && p.mem.kind == "store" && p.mem.valueID == value
			pointerAllowed := pointerUse && (p.mem.kind == "load" || p.mem.kind == "store") && llvmTypeAS(p.mem.ptrType) == AddrSpaceGeneric
			valueAllowed := valueUse && candidates[p.mem.ptrID] && slots[p.mem.ptrID] != nil && p.mem.valueType == slots[p.mem.ptrID].allocType && slots[src].allocType == slots[p.mem.ptrID].allocType
			if (pointerUse && !pointerAllowed) || (valueUse && !valueAllowed) || (!pointerUse && !valueUse) {
				bad[src] = true
			}
		}
	}

	state := make(map[string]LatticeState, len(slots))
	reachable := make(map[string]bool, len(slots))
	for name := range slots {
		state[name] = StateUnconstrained
		if !candidates[name] {
			continue
		}
		if bad[name] {
			state[name] = StateGenericRequired
		} else if len(storeSources[name]) != 0 {
			reachable[name] = true
		}
	}
	maxIterations := 2*len(slots) + 1
	reachIterations := 0
	for {
		reachIterations++
		if reachIterations > len(slots)+1 {
			return "", nil, fmt.Errorf("shared reachability exceeded bound %d", maxIterations)
		}
		changed := false
		for _, dst := range slotOrder {
			if !candidates[dst] || bad[dst] || reachable[dst] {
				continue
			}
			for _, src := range depStores[dst] {
				if reachable[src] {
					reachable[dst] = true
					changed = true
					break
				}
			}
		}
		if !changed {
			break
		}
	}
	for name := range slots {
		if candidates[name] && reachable[name] && !bad[name] {
			state[name] = StateShared
		}
	}
	stateIterations := 0
	for {
		stateIterations++
		if stateIterations > maxIterations {
			return "", nil, fmt.Errorf("fixed-point iteration exceeded bound %d", maxIterations)
		}
		changed := false
		for _, dst := range slotOrder {
			srcs := depStores[dst]
			for _, src := range srcs {
				if state[dst] == StateShared && state[src] == StateGenericRequired {
					state[dst] = StateGenericRequired
					changed = true
				} else if state[dst] == StateShared && state[src] != StateShared {
					// An unconstrained source may still become generic. Keep the
					// destination unconstrained until that dependency is resolved.
					state[dst] = StateUnconstrained
					changed = true
				}
				if state[src] == StateGenericRequired && state[dst] == StateUnconstrained {
					state[dst] = StateGenericRequired
					changed = true
				}
				if state[dst] == StateGenericRequired && state[src] != StateGenericRequired {
					state[src] = StateGenericRequired
					changed = true
				}
				if state[dst] == StateShared && state[src] != StateShared {
					state[dst], state[src] = StateGenericRequired, StateGenericRequired
					changed = true
				}
				if state[src] == StateShared && state[dst] != StateShared {
					state[src], state[dst] = StateGenericRequired, StateGenericRequired
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
	iterations := reachIterations
	if stateIterations > iterations {
		iterations = stateIterations
	}
	report.Iterations = iterations
	toRewrite := make(map[string]bool)
	for _, name := range slotOrder {
		d := slots[name]
		info := SlotInfo{Name: name, Declared: d.declaredAS, Explicit: d.explicit, State: state[name], Effective: d.declaredAS, IterationCount: iterations}
		if !d.explicit {
			info.Effective = AddrSpaceGeneric
			if info.State == StateShared {
				info.Effective = AddrSpaceShared
				toRewrite[name] = true
				report.RewrittenSlots = append(report.RewrittenSlots, name)
			}
		}
		report.Slots[name] = info
	}

	var out strings.Builder
	for _, p := range f.lines {
		line := p.raw
		if p.hasAlloca && toRewrite[p.allocaName] {
			line = rewriteAllocaLine(line, slots[p.allocaName])
		}
		if p.hasMem {
			if p.mem.kind == "store" && toRewrite[p.mem.ptrID] {
				line = rewriteStoreLine(line)
				line = rewriteSlotPointerOperand(line, slots[p.mem.ptrID])
			}
			if p.mem.kind == "load" && toRewrite[p.mem.ptrID] {
				line = rewriteSlotLoad(line, slots[p.mem.ptrID])
				line = rewriteSlotPointerOperand(line, slots[p.mem.ptrID])
			}
			if p.mem.kind == "load" && p.mem.ptrID != "" {
				if src, ok := loadedFrom[p.mem.ptrID]; ok && toRewrite[src] {
					oldType, newType := inferredPointerType(slots[src].allocType)
					line = rewriteMemoryPointerType(line, p.mem.ptrID, oldType, newType)
				}
			}
			if p.mem.kind == "store" && p.mem.ptrID != "" {
				if src, ok := loadedFrom[p.mem.ptrID]; ok && toRewrite[src] {
					oldType, newType := inferredPointerType(slots[src].allocType)
					line = rewriteMemoryPointerType(line, p.mem.ptrID, oldType, newType)
				}
			}
			if p.mem.kind == "store" && p.mem.valueID != "" {
				if src, ok := loadedFrom[p.mem.valueID]; ok && toRewrite[src] && toRewrite[p.mem.ptrID] {
					oldType, newType := inferredPointerType(slots[src].allocType)
					line = rewriteStoreValueType(line, p.mem.valueID, oldType, newType)
				}
			}
		}
		out.WriteString(line)
	}
	rewritten := out.String()
	if !rewrittenFunctionIsConsistent(f, splitLines([]byte(rewritten)), slots, toRewrite, lineSlotAccess, loadedFrom) {
		for name := range toRewrite {
			info := report.Slots[name]
			info.State = StateGenericRequired
			info.Effective = AddrSpaceGeneric
			report.Slots[name] = info
		}
		report.RewrittenSlots = nil
		return strings.Join(lines, ""), report, nil
	}
	return rewritten, report, nil
}

func rewrittenFunctionIsConsistent(original parsedFunction, rewrittenLines []string, slots map[string]*slotDecl, toRewrite map[string]bool, lineSlotAccess map[int]string, loadedFrom map[string]string) bool {
	if len(rewrittenLines) != len(original.lines) {
		return false
	}
	rewritten := parseFunctionLines(rewrittenLines)
	for i, before := range original.lines {
		after := rewritten.lines[i]
		if before.result != after.result || !sameLLVMIDs(before.ids, after.ids) {
			return false
		}
		if before.hasAlloca && toRewrite[before.allocaName] {
			name, allocType, ok := parseAlloca(after.code)
			_, newType := inferredPointerType(slots[before.allocaName].allocType)
			if !ok || name != before.allocaName || allocType != newType {
				return false
			}
		} else if before.hasAlloca {
			if !after.hasAlloca || before.allocaName != after.allocaName || before.allocType != after.allocType {
				return false
			}
		}
		if !before.hasMem {
			if after.hasMem {
				return false
			}
			continue
		}
		if !after.hasMem || before.mem.kind != after.mem.kind || before.mem.result != after.mem.result || before.mem.ptrID != after.mem.ptrID || before.mem.valueID != after.mem.valueID {
			return false
		}
		wantValueType, wantPtrType := before.mem.valueType, before.mem.ptrType
		if slot := lineSlotAccess[i]; slot != "" && toRewrite[slot] {
			_, wantValueType = inferredPointerType(slots[slot].allocType)
			if slots[slot].allocType != "ptr" {
				wantPtrType = wantValueType + "*"
			}
		}
		if src, ok := loadedFrom[before.mem.ptrID]; ok && toRewrite[src] {
			_, wantPtrType = inferredPointerType(slots[src].allocType)
		}
		if src, ok := loadedFrom[before.mem.valueID]; ok && toRewrite[src] {
			_, wantValueType = inferredPointerType(slots[src].allocType)
		}
		if after.mem.valueType != wantValueType || after.mem.ptrType != wantPtrType {
			return false
		}
	}
	return true
}

func sameLLVMIDs(a, b []irID) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (m parsedMemory) ptrTypeCompatible(slot *slotDecl) bool {
	return m.ptrType == accessPointerType(slot)
}

func traceAddressSpace(line string, valAS, globals map[string]AddrSpace) (AddrSpace, bool) {
	_, rhs, ok := assignment(line)
	if !ok {
		return 0, false
	}
	if strings.HasPrefix(rhs, "addrspacecast ") {
		parts := splitToTopLevel(strings.TrimSpace(strings.TrimPrefix(rhs, "addrspacecast ")), " to ")
		if len(parts) == 2 {
			typ, value, _ := parseTypedValue(parts[0])
			dstType := strings.TrimSpace(parts[1])
			if llvmPointerType(typ) && llvmTypeAS(typ) == AddrSpaceShared && valueASForOperand(typ, value, valAS, globals) == AddrSpaceShared && llvmPointerType(dstType) && llvmTypeAS(dstType) == AddrSpaceGeneric {
				return AddrSpaceShared, true
			}
		}
		return AddrSpaceGeneric, true
	}
	if strings.HasPrefix(rhs, "getelementptr ") {
		rhs = strings.TrimPrefix(rhs, "getelementptr ")
		rhs = strings.TrimPrefix(rhs, "inbounds ")
		parts := splitLLVMTopLevel(rhs)
		if len(parts) >= 3 && scalarGEPElementType.MatchString(parts[0]) {
			typ, value, ok := parseSimplePointerOperand(parts[1])
			if ok && validGEPIndices(parts[2:]) {
				return valueASForOperand(typ, value, valAS, globals), true
			}
		}
		return AddrSpaceGeneric, true
	}
	if strings.HasPrefix(rhs, "call ") {
		return AddrSpaceGeneric, true
	}
	return 0, false
}

func validGEPIndices(indices []string) bool {
	for _, index := range indices {
		fields := strings.Fields(index)
		if len(fields) != 2 || !strings.HasPrefix(fields[0], "i") || !digits(fields[0][1:]) {
			return false
		}
		if digits(strings.TrimPrefix(fields[1], "+")) || strings.HasPrefix(fields[1], "-") && digits(strings.TrimPrefix(fields[1], "-")) {
			continue
		}
		ids := llvmIDs(fields[1])
		if len(ids) != 1 || !ids[0].local || ids[0].text != fields[1] {
			return false
		}
	}
	return len(indices) != 0
}

func digits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func parameterAddressSpace(header, name string) AddrSpace {
	start := strings.Index(header, "(")
	end := strings.LastIndex(header, ")")
	if start < 0 || end <= start {
		return AddrSpaceGeneric
	}
	for _, arg := range splitLLVMTopLevel(header[start+1 : end]) {
		ids := llvmIDs(arg)
		if len(ids) > 0 && ids[len(ids)-1].text == name {
			return parameterPointerAddressSpace(strings.TrimSpace(strings.TrimSuffix(arg, name)))
		}
	}
	return AddrSpaceGeneric
}

func parameterPointerAddressSpace(arg string) AddrSpace {
	fields := strings.Fields(arg)
	if len(fields) == 0 {
		return AddrSpaceGeneric
	}
	if fields[0] == "ptr" {
		if len(fields) > 1 && strings.HasPrefix(fields[1], "addrspace(") && strings.HasSuffix(fields[1], ")") {
			return parseAddrSpaceType(fields[1])
		}
		return AddrSpaceGeneric
	}
	for end := 1; end <= len(fields); end++ {
		typ := strings.Join(fields[:end], " ")
		if supportedSlotPointer.MatchString(typ) {
			return llvmTypeAS(typ)
		}
		if strings.HasSuffix(fields[end-1], "*") {
			return AddrSpaceGeneric
		}
	}
	return AddrSpaceGeneric
}

func rewriteStoreValueType(line, value, oldType, newType string) string {
	code := stripLLVMComment(line)
	start := strings.Index(code, "store ")
	if start < 0 {
		return line
	}
	parts := splitLLVMTopLevel(strings.TrimSpace(code[start+len("store "):]))
	if len(parts) < 2 {
		return line
	}
	typ, val, _ := parseTypedValue(parts[0])
	if typ != oldType || val != value {
		return line
	}
	updated := code[:start] + "store " + newType + " " + val + code[start+len("store ")+len(parts[0]):]
	return updated + line[len(code):]
}

func rewriteMemoryPointerType(line, pointer, oldType, newType string) string {
	code := stripLLVMComment(line)
	pm, ok := parseMemory(code)
	if !ok || pm.ptrID != pointer || pm.ptrType != oldType {
		return line
	}
	needle := oldType + " " + pointer
	idx := strings.LastIndex(code, needle)
	if idx < 0 {
		return line
	}
	return code[:idx] + newType + " " + pointer + code[idx+len(needle):] + line[len(code):]
}

func rewriteSharedInlineStoreCast(line string) string {
	pm, ok := parseMemory(line)
	if !ok || pm.kind != "store" {
		return line
	}
	_, _, source := parseTypedValue(pm.value)
	if source == "" {
		return line
	}
	code := stripLLVMComment(line)
	start := strings.Index(code, "store ")
	if start < 0 {
		return line
	}
	parts := splitLLVMTopLevel(strings.TrimSpace(code[start+len("store "):]))
	if len(parts) < 2 {
		return line
	}
	updated := code[:start] + "store " + source + code[start+len("store ")+len(parts[0]):]
	return updated + line[len(code):]
}

func analyzeAndRewriteFunc(lines []string, globals map[string]AddrSpace) (string, *InferenceReport, error) {
	return inferFunctionAndRewrite(lines, globals)
}
