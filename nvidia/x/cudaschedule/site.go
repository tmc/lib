package cudaschedule

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// SiteKind classifies a point in a PTX module where another thread's
// progress can be observed.
type SiteKind int

const (
	// SiteAtomic is an atomic read-modify-write operation (atom.*).
	SiteAtomic SiteKind = iota + 1

	// SiteReduction is a memory reduction with no returned value (red.*).
	SiteReduction

	// SiteBarrier is barrier and mbarrier synchronization (bar.*, barrier.*, mbarrier.*).
	SiteBarrier

	// SiteFence is memory ordering and fences (fence.*, membar.*).
	SiteFence

	// SiteOrderedMemory is memory access with an explicit ordering qualifier.
	SiteOrderedMemory

	// SiteWarpCollective is warp-wide collective communication (shfl.*, vote.*, match.*, elect.*, activemask, redux.*).
	SiteWarpCollective

	// SiteAsyncProxy is an asynchronous proxy pipeline operation (cp.async.*, cp.reduce.async.*,
	// wgmma.*, tcgen05.*, clusterlaunchcontrol.*).
	SiteAsyncProxy

	// SiteGridDependency is a programmatic dependent launch boundary (griddepcontrol.*).
	SiteGridDependency

	// SiteTensorMapMutation is a generic-proxy tensor map descriptor mutation (tensormap.replace.*).
	SiteTensorMapMutation

	// SiteBackedge is an intraprocedural branch back to the same or an earlier basic block.
	SiteBackedge
)

// String returns the name of the site kind.
func (k SiteKind) String() string {
	switch k {
	case SiteAtomic:
		return "Atomic"
	case SiteReduction:
		return "Reduction"
	case SiteBarrier:
		return "Barrier"
	case SiteFence:
		return "Fence"
	case SiteOrderedMemory:
		return "OrderedMemory"
	case SiteWarpCollective:
		return "WarpCollective"
	case SiteAsyncProxy:
		return "AsyncProxy"
	case SiteGridDependency:
		return "GridDependency"
	case SiteTensorMapMutation:
		return "TensorMapMutation"
	case SiteBackedge:
		return "Backedge"
	default:
		return fmt.Sprintf("SiteKind(%d)", int(k))
	}
}

// Span represents a byte offset range [Start, End) in the source text.
type Span struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// Site represents a single schedule-sensitive location discovered in PTX assembly.
type Site struct {
	Ordinal    int      `json:"ordinal"`
	Callable   string   `json:"callable"`
	Kind       SiteKind `json:"kind"`
	Span       Span     `json:"span"`
	BlockIndex *int     `json:"block_index,omitempty"`
	Head       string   `json:"head"`
	Text       string   `json:"text"`
	Predicate  string   `json:"predicate,omitempty"`
}

// Analyze parses a PTX module and returns all schedule-sensitive sites in source order.
// Control-flow recovery is fail-closed: malformed branches are reported instead of
// being silently treated as non-loops.
func Analyze(src string) ([]Site, error) {
	if !strings.Contains(src, ".version") || !strings.Contains(src, ".target") {
		return nil, fmt.Errorf("cudaschedule: PTX requires .version and .target directives")
	}
	mod, err := cudaptx.ParseString(src)
	if err != nil {
		return nil, fmt.Errorf("cudaschedule: parse ptx: %w", err)
	}

	if len(mod.Kernels) == 0 {
		return nil, fmt.Errorf("cudaschedule: PTX has no callable body")
	}
	for _, k := range mod.Kernels {
		if k.Body && k.EndLine == 0 {
			return nil, fmt.Errorf("cudaschedule: callable %q has unterminated body", k.Name)
		}
	}
	lines := splitLines(src)
	lineOffsets := make([]int, len(lines)+1)
	curr := 0
	for i, l := range lines {
		lineOffsets[i] = curr
		curr += len(l)
	}
	lineOffsets[len(lines)] = curr

	var sites []Site
	seen := make(map[int]int)

	for _, k := range mod.Kernels {
		if !k.Body {
			continue
		}
		callableStartOffset := 0
		if k.StartLine > 0 && k.StartLine-1 < len(lineOffsets) {
			callableStartOffset = lineOffsets[k.StartLine-1]
		}
		callableEndOffset := len(src)
		if k.EndLine > 0 && k.EndLine <= len(lineOffsets) {
			callableEndOffset = lineOffsets[k.EndLine-1]
		}
		if callableEndOffset > len(src) {
			callableEndOffset = len(src)
		}

		instSpans, err := locateInstructionSpans(src, lineOffsets, k.Instructions)
		if err != nil {
			return nil, fmt.Errorf("cudaschedule: callable %q: %w", k.Name, err)
		}

		// 1. Classify non-backedge instructions
		for i, inst := range k.Instructions {
			span := instSpans[i]
			kind, isSite, err := classifyInstruction(inst.Opcode, inst.Mnemonic, inst.Line)
			if err != nil {
				return nil, err
			}
			if !isSite {
				continue
			}
			idx := len(sites)
			seen[span.Start] = idx
			sites = append(sites, Site{
				Ordinal:   idx,
				Callable:  k.Name,
				Kind:      kind,
				Span:      span,
				Head:      inst.Opcode,
				Text:      src[span.Start:span.End],
				Predicate: inst.Predicate,
			})
		}

		// 2. Control flow recovery for Backedge classification
		labels, err := findLabels(src, callableStartOffset, callableEndOffset)
		if err != nil {
			return nil, fmt.Errorf("cudaschedule: callable %q: %w", k.Name, err)
		}

		branchTables, err := findBranchTables(src, callableStartOffset, callableEndOffset)
		if err != nil {
			return nil, fmt.Errorf("cudaschedule: callable %q: %w", k.Name, err)
		}

		backedges, err := recoverBackedges(k.Name, k.Instructions, instSpans, labels, branchTables)
		if err != nil {
			return nil, err
		}

		for _, be := range backedges {
			if existing, ok := seen[be.Span.Start]; ok {
				sites[existing].Kind = SiteBackedge
				sites[existing].BlockIndex = be.BlockIndex
				continue
			}
			idx := len(sites)
			seen[be.Span.Start] = idx
			sites = append(sites, be)
		}
	}

	sort.Slice(sites, func(i, j int) bool {
		return sites[i].Span.Start < sites[j].Span.Start
	})
	for i := range sites {
		sites[i].Ordinal = i
	}

	return sites, nil
}

func classifyInstruction(opcode, mnemonic string, line int) (SiteKind, bool, error) {
	head := opcode

	// 1. Atomic
	if strings.HasPrefix(head, "atom.") || mnemonic == "atom" {
		return SiteAtomic, true, nil
	}

	// 2. Reduction
	if strings.HasPrefix(head, "red.") && !strings.HasPrefix(head, "redux.") {
		return SiteReduction, true, nil
	}

	// 3. AsyncProxy pipeline instructions
	for _, prefix := range []string{
		"cp.async",
		"cp.reduce.async",
		"st.async",
		"tensormap.cp_fenceproxy",
		"wgmma.",
		"tcgen05.",
		"clusterlaunchcontrol.",
	} {
		if strings.HasPrefix(head, prefix) {
			return SiteAsyncProxy, true, nil
		}
	}

	// 4. Barrier
	for _, prefix := range []string{
		"bar.sync",
		"bar.arrive",
		"bar.cta.",
		"bar.red",
		"bar.warp",
		"barrier.",
		"mbarrier.",
	} {
		if strings.HasPrefix(head, prefix) {
			return SiteBarrier, true, nil
		}
	}

	// 5. Fence
	if strings.HasPrefix(head, "membar.") || strings.HasPrefix(head, "fence.") {
		return SiteFence, true, nil
	}

	// 6. WarpCollective
	for _, prefix := range []string{
		"activemask",
		"match.any",
		"match.all",
		"vote.",
		"elect.",
		"shfl.",
		"redux.",
	} {
		if strings.HasPrefix(head, prefix) {
			return SiteWarpCollective, true, nil
		}
	}

	// 7. GridDependency
	if strings.HasPrefix(head, "griddepcontrol.") {
		return SiteGridDependency, true, nil
	}

	// 8. TensorMapMutation
	if strings.HasPrefix(head, "tensormap.replace.") {
		return SiteTensorMapMutation, true, nil
	}

	// 9. OrderedMemory
	parts := strings.Split(head, ".")
	base := parts[0]
	if base == "ld" || base == "st" {
		isOrdered := false
		for _, part := range parts[1:] {
			switch part {
			case "volatile", "acquire", "release", "relaxed", "acq_rel", "mmio":
				isOrdered = true
			}
		}
		if isOrdered && !strings.HasPrefix(head, "ld.global.nc") {
			return SiteOrderedMemory, true, nil
		}
	}

	// Closed taxonomy check: if an opcode belongs to synchronization, fences,
	// atomics or ordering categories but failed to classify, it is an unrecognized ordering opcode.
	if isOrderingOpcode(head, mnemonic) {
		return 0, false, fmt.Errorf("cudaschedule: unrecognized ordering-sensitive opcode %q at line %d", head, line)
	}

	return 0, false, nil
}

func isOrderingOpcode(head, mnemonic string) bool {
	if cudaptx.InstructionCategory(mnemonic) == cudaptx.CatParallelSync {
		return true
	}
	for _, p := range []string{
		"atom.", "red.", "bar.", "barrier.", "mbarrier.", "membar.",
		"fence.", "griddepcontrol.", "tensormap.", "multimem.",
	} {
		if strings.HasPrefix(head, p) {
			return true
		}
	}
	return false
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i+1])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

func locateInstructionSpans(src string, lineOffsets []int, instructions []cudaptx.Instruction) ([]Span, error) {
	spans := make([]Span, len(instructions))
	lastOffset := 0
	for i, inst := range instructions {
		startLineOffset := 0
		if inst.Line > 0 && inst.Line-1 < len(lineOffsets) {
			startLineOffset = lineOffsets[inst.Line-1]
		}
		searchStart := max(lastOffset, startLineOffset)
		if searchStart > len(src) {
			searchStart = len(src)
		}

		needle := inst.Opcode
		if inst.Predicate != "" {
			needle = inst.Predicate
		}

		idx := strings.Index(src[searchStart:], needle)
		if idx < 0 {
			// Fallback: search anywhere after startLineOffset
			idx = strings.Index(src[startLineOffset:], needle)
			if idx < 0 {
				return nil, fmt.Errorf("could not locate instruction statement %q at line %d", needle, inst.Line)
			}
			searchStart = startLineOffset
		}

		stmtStart := searchStart + idx
		semiIdx := strings.IndexByte(src[stmtStart:], ';')
		if semiIdx < 0 {
			return nil, fmt.Errorf("instruction statement missing terminating semicolon at line %d", inst.Line)
		}
		stmtEnd := stmtStart + semiIdx + 1

		spans[i] = Span{Start: stmtStart, End: stmtEnd}
		lastOffset = stmtEnd
	}
	return spans, nil
}

type labelInfo struct {
	name   string
	offset int
}

func findLabels(src string, startOffset, endOffset int) ([]labelInfo, error) {
	seen := make(map[string]bool)
	var labels []labelInfo
	sub := src[startOffset:endOffset]

	inComment := false
	inBlockComment := false
	inString := false

	for i := 0; i < len(sub); i++ {
		ch := sub[i]
		if inBlockComment {
			if ch == '*' && i+1 < len(sub) && sub[i+1] == '/' {
				inBlockComment = false
				i++
			}
			continue
		}
		if inComment {
			if ch == '\n' {
				inComment = false
			}
			continue
		}
		if inString {
			if ch == '\\' && i+1 < len(sub) {
				i++
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '/' && i+1 < len(sub) {
			if sub[i+1] == '/' {
				inComment = true
				i++
				continue
			} else if sub[i+1] == '*' {
				inBlockComment = true
				i++
				continue
			}
		}
		if ch == '"' {
			inString = true
			continue
		}

		if ch == ':' {
			if i+1 < len(sub) && sub[i+1] == ':' {
				i++
				continue
			}
			if i > 0 && sub[i-1] == ':' {
				continue
			}
			// Search backwards for label identifier
			endID := i
			for endID > 0 && (sub[endID-1] == ' ' || sub[endID-1] == '\t') {
				endID--
			}
			startID := endID
			for startID > 0 && isIdentChar(sub[startID-1]) {
				startID--
			}
			if startID < endID {
				// Ensure preceding character is statement boundary
				prev := startID - 1
				valid := true
				for prev >= 0 && (sub[prev] == ' ' || sub[prev] == '\t') {
					prev--
				}
				if prev >= 0 && sub[prev] != '\n' && sub[prev] != ';' && sub[prev] != '{' && sub[prev] != '}' {
					valid = false
				}
				if valid {
					name := sub[startID:endID]
					if seen[name] {
						return nil, fmt.Errorf("duplicate label %q", name)
					}
					seen[name] = true
					labels = append(labels, labelInfo{name: name, offset: startOffset + startID})
				}
			}
		}
	}
	return labels, nil
}

func isIdentChar(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_' || b == '$' || b == '%'
}

type branchTableInfo struct {
	name    string
	targets []string
}

func findBranchTables(src string, startOffset, endOffset int) (map[string][]string, error) {
	tables := make(map[string][]string)
	sub := src[startOffset:endOffset]
	const kw = ".branchtargets"
	idx := 0
	for {
		pos := strings.Index(sub[idx:], kw)
		if pos < 0 {
			break
		}
		actualPos := idx + pos
		idx = actualPos + len(kw)

		semi := strings.IndexByte(sub[actualPos:], ';')
		if semi < 0 {
			return nil, fmt.Errorf("malformed .branchtargets: missing terminating semicolon")
		}
		clause := strings.TrimSpace(sub[actualPos+len(kw) : actualPos+semi])
		// PTX writes a table as `name: .branchtargets L1, L2;`.
		lineStart := strings.LastIndex(sub[:actualPos], "\n") + 1
		prefix := strings.TrimSpace(sub[lineStart:actualPos])
		if !strings.HasSuffix(prefix, ":") {
			return nil, fmt.Errorf("malformed .branchtargets declaration")
		}
		tableName := strings.TrimSuffix(strings.TrimSpace(prefix), ":")
		targetStr := clause
		var targets []string
		for _, t := range strings.Split(targetStr, ",") {
			t = strings.TrimSpace(t)
			if t != "" {
				targets = append(targets, t)
			}
		}
		if len(targets) == 0 {
			return nil, fmt.Errorf("malformed .branchtargets %q: empty target list", tableName)
		}
		if _, ok := tables[tableName]; ok {
			return nil, fmt.Errorf("duplicate .branchtargets table %q", tableName)
		}
		tables[tableName] = targets
	}
	return tables, nil
}

func recoverBackedges(
	callableName string,
	instructions []cudaptx.Instruction,
	spans []Span,
	labels []labelInfo,
	branchTables map[string][]string,
) ([]Site, error) {
	if len(instructions) == 0 {
		return nil, nil
	}

	labelTargets := make(map[string]int)
	for _, l := range labels {
		assigned := false
		for i, sp := range spans {
			if sp.Start > l.offset {
				labelTargets[l.name] = i
				assigned = true
				break
			}
		}
		if !assigned {
			labelTargets[l.name] = len(instructions)
		}
	}

	// Leaders
	leaderSet := map[int]bool{0: true}
	for _, targetInst := range labelTargets {
		if targetInst < len(instructions) {
			leaderSet[targetInst] = true
		}
	}
	for i, inst := range instructions {
		isTerminator := false
		switch inst.Mnemonic {
		case "bra", "brx", "ret", "exit", "trap":
			isTerminator = true
		}
		if isTerminator && i+1 < len(instructions) {
			leaderSet[i+1] = true
		}
	}

	var leaders []int
	for leader := range leaderSet {
		leaders = append(leaders, leader)
	}
	sort.Ints(leaders)

	instToBlock := make([]int, len(instructions))
	for i := range instructions {
		b := 0
		for bi, l := range leaders {
			if l <= i {
				b = bi
			} else {
				break
			}
		}
		instToBlock[i] = b
	}

	labelToBlock := make(map[string]int)
	for name, instIdx := range labelTargets {
		if instIdx < len(instructions) {
			labelToBlock[name] = instToBlock[instIdx]
		} else {
			labelToBlock[name] = len(leaders)
		}
	}

	var backedges []Site

	for i, inst := range instructions {
		if inst.Mnemonic != "bra" && inst.Mnemonic != "brx" {
			continue
		}
		currentBlock := instToBlock[i]

		if inst.Mnemonic == "bra" {
			operands := strings.TrimSpace(inst.Operands)
			if operands == "" {
				return nil, fmt.Errorf("cudaschedule: callable %q: branch without target at line %d", callableName, inst.Line)
			}
			target := strings.Trim(strings.Fields(operands)[0], ",;")
			targetBlock, ok := labelToBlock[target]
			if !ok {
				return nil, fmt.Errorf("cudaschedule: callable %q: branch targets unknown label %q at line %d", callableName, target, inst.Line)
			}
			if targetBlock <= currentBlock {
				blk := currentBlock
				backedges = append(backedges, Site{
					Callable:   callableName,
					Kind:       SiteBackedge,
					Span:       spans[i],
					BlockIndex: &blk,
					Head:       inst.Opcode,
					Text:       inst.Raw,
					Predicate:  inst.Predicate,
				})
			}
		} else if inst.Mnemonic == "brx" {
			operands := strings.TrimSpace(inst.Operands)
			fields := strings.Fields(operands)
			if len(fields) < 2 {
				return nil, fmt.Errorf("cudaschedule: callable %q: brx.idx statement at line %d has no target table", callableName, inst.Line)
			}
			tableRef := strings.Trim(fields[1], ",;")
			targets, ok := branchTables[tableRef]
			if !ok {
				return nil, fmt.Errorf("cudaschedule: callable %q: brx.idx statement at line %d uses unknown table %q", callableName, inst.Line, tableRef)
			}
			hasBackedge := false
			for _, target := range targets {
				targetBlock, ok := labelToBlock[target]
				if !ok {
					return nil, fmt.Errorf("cudaschedule: callable %q: branch table %q targets unknown label %q", callableName, tableRef, target)
				}
				if targetBlock <= currentBlock {
					hasBackedge = true
					break
				}
			}
			if hasBackedge {
				blk := currentBlock
				backedges = append(backedges, Site{
					Callable:   callableName,
					Kind:       SiteBackedge,
					Span:       spans[i],
					BlockIndex: &blk,
					Head:       inst.Opcode,
					Text:       inst.Raw,
					Predicate:  inst.Predicate,
				})
			}
		}
	}

	return backedges, nil
}
