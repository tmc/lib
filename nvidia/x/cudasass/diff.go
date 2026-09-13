package cudasass

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// DiffOp represents the diff operation on an instruction.
type DiffOp int

const (
	DiffMatch  DiffOp = iota // Instruction matches between base and target
	DiffAdd                  // Instruction was added in target
	DiffDelete               // Instruction was removed from base
)

func (op DiffOp) String() string {
	switch op {
	case DiffMatch:
		return " "
	case DiffAdd:
		return "+"
	case DiffDelete:
		return "-"
	default:
		return "?"
	}
}

// InstructionDiff represents a single instruction diff entry.
type InstructionDiff struct {
	Op     DiffOp
	Inst   Instruction
	Base   *Instruction
	Target *Instruction
}

// String returns a unified-diff style formatted representation of the instruction diff.
func (d InstructionDiff) String() string {
	raw := d.Inst.Raw
	if raw == "" {
		if d.Inst.Predicate != "" {
			raw = fmt.Sprintf("/*%04x*/   %s %s %s ;", d.Inst.Address, d.Inst.Predicate, d.Inst.Opcode, d.Inst.Operands)
		} else {
			raw = fmt.Sprintf("/*%04x*/   %s %s ;", d.Inst.Address, d.Inst.Opcode, d.Inst.Operands)
		}
	}
	return d.Op.String() + " " + strings.TrimSpace(raw)
}

// SpillDiff summarizes register spill additions and removals.
type SpillDiff struct {
	AddedSTS     int // Store-to-stack instructions added
	RemovedSTS   int // Store-to-stack instructions removed
	AddedLDS     int // Load-from-stack instructions added
	RemovedLDS   int // Load-from-stack instructions removed
	TotalAdded   int // Total spill operations added (AddedSTS + AddedLDS)
	TotalRemoved int // Total spill operations removed (RemovedSTS + RemovedLDS)
}

// Delta returns the net change in register spill operations.
func (s SpillDiff) Delta() int {
	return s.TotalAdded - s.TotalRemoved
}

// OpcodeSubstitution records a substitution between instruction opcodes (e.g. IMAD -> IDP4A).
type OpcodeSubstitution struct {
	From      string
	To        string
	Count     int
	FromCount int
	ToCount   int
}

// ArchitectureDiff summarizes architectural changes between two function versions.
type ArchitectureDiff struct {
	BaseInstructions   int
	TargetInstructions int
	InstructionDelta   int

	BaseBranches   int
	TargetBranches int
	BranchDelta    int

	BaseSizeBytes   int
	TargetSizeBytes int
	TotalSizeDelta  int

	BaseScalarCount   int
	TargetScalarCount int
	ScalarDelta       int

	BaseSIMDCount   int
	TargetSIMDCount int
	SIMDDelta       int

	Spills        SpillDiff
	Substitutions []OpcodeSubstitution
}

// FunctionDiff represents the diff between two kernel functions.
type FunctionDiff struct {
	Name         string
	Base         Function
	Target       Function
	Changes      []InstructionDiff
	Architecture ArchitectureDiff
	Spills       SpillDiff
}

// AddedCount returns the number of instructions added in target.
func (f FunctionDiff) AddedCount() int {
	var n int
	for _, c := range f.Changes {
		if c.Op == DiffAdd {
			n++
		}
	}
	return n
}

// DeletedCount returns the number of instructions removed from base.
func (f FunctionDiff) DeletedCount() int {
	var n int
	for _, c := range f.Changes {
		if c.Op == DiffDelete {
			n++
		}
	}
	return n
}

// MatchedCount returns the number of matching instructions.
func (f FunctionDiff) MatchedCount() int {
	var n int
	for _, c := range f.Changes {
		if c.Op == DiffMatch {
			n++
		}
	}
	return n
}

// DisassemblyDiff represents the diff between two Disassembly objects.
type DisassemblyDiff struct {
	Functions    []FunctionDiff
	Architecture ArchitectureDiff
	Spills       SpillDiff
}

// Function returns the FunctionDiff for the named function, or nil if not found.
func (d *DisassemblyDiff) Function(name string) *FunctionDiff {
	for i := range d.Functions {
		if d.Functions[i].Name == name {
			return &d.Functions[i]
		}
	}
	return nil
}

// Diff compares two Disassembly objects and produces a DisassemblyDiff.
func Diff(base, target *Disassembly) (*DisassemblyDiff, error) {
	if base == nil || target == nil {
		return nil, fmt.Errorf("cudasass: nil disassembly")
	}

	res := &DisassemblyDiff{}
	targetFns := make(map[string]*Function, len(target.Functions))
	for _, fn := range target.Functions {
		targetFns[fn.Name] = fn
	}

	seen := make(map[string]bool)
	for _, baseFn := range base.Functions {
		seen[baseFn.Name] = true
		targetFn := targetFns[baseFn.Name]
		if targetFn == nil {
			targetFn = &Function{Name: baseFn.Name}
		}
		fnDiff := DiffFunctions(*baseFn, *targetFn)
		res.Functions = append(res.Functions, fnDiff)
	}

	for _, targetFn := range target.Functions {
		if !seen[targetFn.Name] {
			fnDiff := DiffFunctions(Function{Name: targetFn.Name}, *targetFn)
			res.Functions = append(res.Functions, fnDiff)
		}
	}

	for _, fd := range res.Functions {
		res.Architecture.BaseInstructions += fd.Architecture.BaseInstructions
		res.Architecture.TargetInstructions += fd.Architecture.TargetInstructions
		res.Architecture.InstructionDelta += fd.Architecture.InstructionDelta

		res.Architecture.BaseBranches += fd.Architecture.BaseBranches
		res.Architecture.TargetBranches += fd.Architecture.TargetBranches
		res.Architecture.BranchDelta += fd.Architecture.BranchDelta

		res.Architecture.BaseSizeBytes += fd.Architecture.BaseSizeBytes
		res.Architecture.TargetSizeBytes += fd.Architecture.TargetSizeBytes
		res.Architecture.TotalSizeDelta += fd.Architecture.TotalSizeDelta

		res.Architecture.BaseScalarCount += fd.Architecture.BaseScalarCount
		res.Architecture.TargetScalarCount += fd.Architecture.TargetScalarCount
		res.Architecture.ScalarDelta += fd.Architecture.ScalarDelta

		res.Architecture.BaseSIMDCount += fd.Architecture.BaseSIMDCount
		res.Architecture.TargetSIMDCount += fd.Architecture.TargetSIMDCount
		res.Architecture.SIMDDelta += fd.Architecture.SIMDDelta

		res.Architecture.Spills.AddedSTS += fd.Architecture.Spills.AddedSTS
		res.Architecture.Spills.RemovedSTS += fd.Architecture.Spills.RemovedSTS
		res.Architecture.Spills.AddedLDS += fd.Architecture.Spills.AddedLDS
		res.Architecture.Spills.RemovedLDS += fd.Architecture.Spills.RemovedLDS
		res.Architecture.Spills.TotalAdded += fd.Architecture.Spills.TotalAdded
		res.Architecture.Spills.TotalRemoved += fd.Architecture.Spills.TotalRemoved

		res.Architecture.Substitutions = append(res.Architecture.Substitutions, fd.Architecture.Substitutions...)
	}
	res.Architecture.Substitutions = mergeSubstitutions(res.Architecture.Substitutions)
	res.Spills = res.Architecture.Spills

	return res, nil
}

// DiffFunctions compares two functions and returns their structured diff.
func DiffFunctions(baseFn, targetFn Function) FunctionDiff {
	name := baseFn.Name
	if name == "" {
		name = targetFn.Name
	}

	baseNorm := make([]string, len(baseFn.Instructions))
	for i, inst := range baseFn.Instructions {
		baseNorm[i] = NormalizeInstruction(inst)
	}

	targetNorm := make([]string, len(targetFn.Instructions))
	for i, inst := range targetFn.Instructions {
		targetNorm[i] = NormalizeInstruction(inst)
	}

	edits := myersDiff(baseNorm, targetNorm)

	changes := make([]InstructionDiff, 0, len(edits))
	for _, e := range edits {
		switch e.op {
		case DiffMatch:
			b := baseFn.Instructions[e.baseIdx]
			t := targetFn.Instructions[e.targetIdx]
			changes = append(changes, InstructionDiff{
				Op:     DiffMatch,
				Inst:   t,
				Base:   &b,
				Target: &t,
			})
		case DiffAdd:
			t := targetFn.Instructions[e.targetIdx]
			changes = append(changes, InstructionDiff{
				Op:     DiffAdd,
				Inst:   t,
				Target: &t,
			})
		case DiffDelete:
			b := baseFn.Instructions[e.baseIdx]
			changes = append(changes, InstructionDiff{
				Op:   DiffDelete,
				Inst: b,
				Base: &b,
			})
		}
	}

	arch := computeArchitectureDiff(baseFn, targetFn, changes)

	return FunctionDiff{
		Name:         name,
		Base:         baseFn,
		Target:       targetFn,
		Changes:      changes,
		Architecture: arch,
		Spills:       arch.Spills,
	}
}

var (
	reMemRegOffset    = regexp.MustCompile(`\[([A-Za-z0-9_.]+)\s*[+-]\s*(?:0x[0-9a-fA-F]+|\d+)\s*\]`)
	reConstBankOffset = regexp.MustCompile(`c\[([A-Za-z0-9_.]+)\s*\]\[\s*(?:0x[0-9a-fA-F]+|\d+)\s*\]`)
	reBranchTarget    = regexp.MustCompile(`^` + "`?" + `(?:0x[0-9a-fA-F]+|[0-9a-fA-F]+)` + "`?" + `$`)
)

// NormalizeInstruction returns a normalized string representation of an instruction,
// stripping instruction address offsets, memory offset drift, and normalizing whitespace.
func NormalizeInstruction(inst Instruction) string {
	pred := strings.TrimSpace(inst.Predicate)
	opcode := strings.TrimSpace(inst.Opcode)
	operands := NormalizeOperands(opcode, inst.Operands)

	var sb strings.Builder
	if pred != "" {
		sb.WriteString(pred)
		sb.WriteString(" ")
	}
	sb.WriteString(opcode)
	if operands != "" {
		sb.WriteString(" ")
		sb.WriteString(operands)
	}
	return sb.String()
}

// NormalizeLine parses and normalizes a single SASS disassembly line.
func NormalizeLine(line string) string {
	if inst, ok := ParseInstructionLine(line); ok {
		return NormalizeInstruction(inst)
	}
	trimmed := strings.TrimSpace(line)
	if closeIdx := strings.Index(trimmed, "*/"); strings.HasPrefix(trimmed, "/*") && closeIdx != -1 {
		trimmed = strings.TrimSpace(trimmed[closeIdx+2:])
	}
	if commentIdx := strings.Index(trimmed, "//"); commentIdx != -1 {
		trimmed = strings.TrimSpace(trimmed[:commentIdx])
	}
	return strings.TrimSuffix(trimmed, ";")
}

// NormalizeOperands normalizes operands by stripping memory offset drift,
// constant bank drift, branch target offsets, and normalizing spacing.
func NormalizeOperands(opcode, operands string) string {
	ops := strings.TrimSpace(operands)
	if ops == "" {
		return ""
	}

	if isBranchOpcode(opcode) {
		trimmed := strings.TrimSpace(ops)
		if reBranchTarget.MatchString(trimmed) || trimmed == "." || strings.HasPrefix(trimmed, "`+") || strings.HasPrefix(trimmed, "+0x") {
			return "<target>"
		}
	}

	// Strip register-relative memory offsets: [R2.64+0x10] -> [R2.64]
	ops = reMemRegOffset.ReplaceAllString(ops, "[$1]")

	// Strip constant bank parameter offsets: c[0x0][0x160] -> c[$1]
	ops = reConstBankOffset.ReplaceAllString(ops, "c[$1]")

	// Normalize whitespace
	fields := strings.Fields(ops)
	ops = strings.Join(fields, " ")

	// Ensure consistent commas
	parts := strings.Split(ops, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return strings.Join(parts, ", ")
}

// BaseOpcode returns the root mnemonic without modifiers or architectures (e.g. IMAD.WIDE -> IMAD).
func BaseOpcode(opcode string) string {
	op := strings.ToUpper(strings.TrimSpace(opcode))
	if op == "IDP.4A" || strings.HasPrefix(op, "IDP.4A.") {
		return "IDP4A"
	}
	if op == "IDP.2A" || strings.HasPrefix(op, "IDP.2A.") {
		return "IDP2A"
	}
	if idx := strings.Index(op, "."); idx != -1 {
		return op[:idx]
	}
	return op
}

func isBranchOpcode(opcode string) bool {
	base := BaseOpcode(opcode)
	switch base {
	case "BRA", "JMP", "CALL", "RET", "EXIT", "BRX", "JMX", "BREAK", "CONT", "WARPSYNC", "BSSY", "BSYNC":
		return true
	}
	return false
}

func isSTSOpcode(opcode string) bool {
	base := BaseOpcode(opcode)
	return base == "STS" || base == "STL"
}

func isLDSOpcode(opcode string) bool {
	base := BaseOpcode(opcode)
	return base == "LDS" || base == "LDL"
}

// IsScalarArithmetic returns true if the opcode is a scalar arithmetic or logical operation.
func IsScalarArithmetic(opcode string) bool {
	base := BaseOpcode(opcode)
	switch base {
	case "IMAD", "IMUL", "IADD", "IADD3", "ISUB", "LEA",
		"LOP", "LOP3", "SHF", "SHR", "SHL",
		"FFMA", "FADD", "FMUL", "FMNMX", "FSET", "FSETP",
		"DFMA", "DADD", "DMUL", "DSET", "DSETP",
		"I2I", "I2F", "F2I", "F2F":
		return true
	}
	return false
}

// IsSIMD returns true if the opcode is a SIMD or Tensor Core instruction.
func IsSIMD(opcode string) bool {
	base := BaseOpcode(opcode)
	switch base {
	case "IDP4A", "IDP2A",
		"HFMA2", "HADD2", "HMUL2", "HSET2", "HSETP2",
		"HMMA", "IMMA", "DMMA", "BMMA", "QDMA",
		"VADD", "VMUL", "VMNMX", "VABSDIFF", "VSET":
		return true
	}
	return false
}

func functionSizeBytes(fn Function) int {
	if len(fn.Instructions) == 0 {
		return 0
	}
	first := fn.Instructions[0].Address
	last := fn.Instructions[len(fn.Instructions)-1].Address
	if last >= first && last > 0 {
		return int(last - first + 16)
	}
	return len(fn.Instructions) * 16
}

func computeArchitectureDiff(baseFn, targetFn Function, changes []InstructionDiff) ArchitectureDiff {
	arch := ArchitectureDiff{
		BaseInstructions:   len(baseFn.Instructions),
		TargetInstructions: len(targetFn.Instructions),
		InstructionDelta:   len(targetFn.Instructions) - len(baseFn.Instructions),

		BaseSizeBytes:   functionSizeBytes(baseFn),
		TargetSizeBytes: functionSizeBytes(targetFn),
	}
	arch.TotalSizeDelta = arch.TargetSizeBytes - arch.BaseSizeBytes

	for _, inst := range baseFn.Instructions {
		if isBranchOpcode(inst.Opcode) {
			arch.BaseBranches++
		}
		if IsScalarArithmetic(inst.Opcode) {
			arch.BaseScalarCount++
		}
		if IsSIMD(inst.Opcode) {
			arch.BaseSIMDCount++
		}
	}

	for _, inst := range targetFn.Instructions {
		if isBranchOpcode(inst.Opcode) {
			arch.TargetBranches++
		}
		if IsScalarArithmetic(inst.Opcode) {
			arch.TargetScalarCount++
		}
		if IsSIMD(inst.Opcode) {
			arch.TargetSIMDCount++
		}
	}

	arch.BranchDelta = arch.TargetBranches - arch.BaseBranches
	arch.ScalarDelta = arch.TargetScalarCount - arch.BaseScalarCount
	arch.SIMDDelta = arch.TargetSIMDCount - arch.BaseSIMDCount

	for _, c := range changes {
		switch c.Op {
		case DiffAdd:
			if isSTSOpcode(c.Inst.Opcode) {
				arch.Spills.AddedSTS++
			}
			if isLDSOpcode(c.Inst.Opcode) {
				arch.Spills.AddedLDS++
			}
		case DiffDelete:
			if isSTSOpcode(c.Inst.Opcode) {
				arch.Spills.RemovedSTS++
			}
			if isLDSOpcode(c.Inst.Opcode) {
				arch.Spills.RemovedLDS++
			}
		}
	}
	arch.Spills.TotalAdded = arch.Spills.AddedSTS + arch.Spills.AddedLDS
	arch.Spills.TotalRemoved = arch.Spills.RemovedSTS + arch.Spills.RemovedLDS

	arch.Substitutions = detectSubstitutions(changes)

	return arch
}

func detectSubstitutions(changes []InstructionDiff) []OpcodeSubstitution {
	var subs []OpcodeSubstitution

	var hunkDeletes []Instruction
	var hunkAdds []Instruction

	flush := func() {
		if len(hunkDeletes) == 0 || len(hunkAdds) == 0 {
			hunkDeletes = hunkDeletes[:0]
			hunkAdds = hunkAdds[:0]
			return
		}

		delOps := make(map[string]int)
		for _, inst := range hunkDeletes {
			delOps[BaseOpcode(inst.Opcode)]++
		}

		addOps := make(map[string]int)
		for _, inst := range hunkAdds {
			addOps[BaseOpcode(inst.Opcode)]++
		}

		delKeys := make([]string, 0, len(delOps))
		for k := range delOps {
			delKeys = append(delKeys, k)
		}
		sort.Strings(delKeys)

		addKeys := make([]string, 0, len(addOps))
		for k := range addOps {
			addKeys = append(addKeys, k)
		}
		sort.Strings(addKeys)

		// Priority 1: Scalar to SIMD substitutions
		for _, delOp := range delKeys {
			delCnt := delOps[delOp]
			if !IsScalarArithmetic(delOp) || delCnt == 0 {
				continue
			}
			for _, addOp := range addKeys {
				addCnt := addOps[addOp]
				if !IsSIMD(addOp) || addCnt == 0 {
					continue
				}
				count := min(delCnt, addCnt)
				subs = append(subs, OpcodeSubstitution{
					From:      delOp,
					To:        addOp,
					Count:     count,
					FromCount: delCnt,
					ToCount:   addCnt,
				})
				delOps[delOp] -= count
				addOps[addOp] -= count
				delCnt -= count
				if delCnt == 0 {
					break
				}
			}
		}

		// Priority 2: Other opcode substitutions
		for _, delOp := range delKeys {
			delCnt := delOps[delOp]
			if delCnt == 0 {
				continue
			}
			for _, addOp := range addKeys {
				addCnt := addOps[addOp]
				if addCnt == 0 || delOp == addOp {
					continue
				}
				count := min(delCnt, addCnt)
				subs = append(subs, OpcodeSubstitution{
					From:      delOp,
					To:        addOp,
					Count:     count,
					FromCount: delCnt,
					ToCount:   addCnt,
				})
				delOps[delOp] -= count
				addOps[addOp] -= count
				delCnt -= count
				if delCnt == 0 {
					break
				}
			}
		}

		hunkDeletes = hunkDeletes[:0]
		hunkAdds = hunkAdds[:0]
	}

	for _, c := range changes {
		switch c.Op {
		case DiffMatch:
			flush()
		case DiffDelete:
			hunkDeletes = append(hunkDeletes, c.Inst)
		case DiffAdd:
			hunkAdds = append(hunkAdds, c.Inst)
		}
	}
	flush()

	return mergeSubstitutions(subs)
}

func mergeSubstitutions(subs []OpcodeSubstitution) []OpcodeSubstitution {
	var res []OpcodeSubstitution
	lookup := make(map[string]int)
	for _, s := range subs {
		key := s.From + "->" + s.To
		if idx, ok := lookup[key]; ok {
			res[idx].Count += s.Count
			res[idx].FromCount += s.FromCount
			res[idx].ToCount += s.ToCount
		} else {
			lookup[key] = len(res)
			res = append(res, s)
		}
	}
	return res
}

type diffEdit struct {
	op        DiffOp
	baseIdx   int
	targetIdx int
}

func myersDiff(a, b []string) []diffEdit {
	n := len(a)
	m := len(b)

	if n == 0 && m == 0 {
		return nil
	}
	if n == 0 {
		edits := make([]diffEdit, m)
		for j := 0; j < m; j++ {
			edits[j] = diffEdit{op: DiffAdd, targetIdx: j}
		}
		return edits
	}
	if m == 0 {
		edits := make([]diffEdit, n)
		for i := 0; i < n; i++ {
			edits[i] = diffEdit{op: DiffDelete, baseIdx: i}
		}
		return edits
	}

	// Trim common prefix
	prefix := 0
	for prefix < n && prefix < m && a[prefix] == b[prefix] {
		prefix++
	}

	// Trim common suffix
	suffix := 0
	for suffix < n-prefix && suffix < m-prefix && a[n-1-suffix] == b[m-1-suffix] {
		suffix++
	}

	var edits []diffEdit
	for i := 0; i < prefix; i++ {
		edits = append(edits, diffEdit{op: DiffMatch, baseIdx: i, targetIdx: i})
	}

	midA := a[prefix : n-suffix]
	midB := b[prefix : m-suffix]

	if len(midA) > 0 || len(midB) > 0 {
		midEdits := myersSES(midA, midB)
		for _, e := range midEdits {
			edits = append(edits, diffEdit{
				op:        e.op,
				baseIdx:   e.baseIdx + prefix,
				targetIdx: e.targetIdx + prefix,
			})
		}
	}

	for i := 0; i < suffix; i++ {
		edits = append(edits, diffEdit{
			op:        DiffMatch,
			baseIdx:   n - suffix + i,
			targetIdx: m - suffix + i,
		})
	}

	return edits
}

func myersSES(a, b []string) []diffEdit {
	n := len(a)
	m := len(b)

	if n == 0 {
		edits := make([]diffEdit, m)
		for j := 0; j < m; j++ {
			edits[j] = diffEdit{op: DiffAdd, targetIdx: j}
		}
		return edits
	}
	if m == 0 {
		edits := make([]diffEdit, n)
		for i := 0; i < n; i++ {
			edits[i] = diffEdit{op: DiffDelete, baseIdx: i}
		}
		return edits
	}

	max := n + m
	v := make([]int, 2*max+1)
	offset := max

	trace := make([][]int, 0, max+1)

	var finalD int
search:
	for d := 0; d <= max; d++ {
		for k := -d; k <= d; k += 2 {
			var x int
			if k == -d || (k != d && v[k-1+offset] < v[k+1+offset]) {
				x = v[k+1+offset]
			} else {
				x = v[k-1+offset] + 1
			}
			y := x - k
			for x < n && y < m && a[x] == b[y] {
				x++
				y++
			}
			v[k+offset] = x
			if x >= n && y >= m {
				finalD = d
				vCopy := make([]int, len(v))
				copy(vCopy, v)
				trace = append(trace, vCopy)
				break search
			}
		}
		vCopy := make([]int, len(v))
		copy(vCopy, v)
		trace = append(trace, vCopy)
	}

	// Traceback
	x := n
	y := m
	var edits []diffEdit

	for d := finalD; d > 0; d-- {
		k := x - y
		var prevK int
		if k == -d || (k != d && trace[d-1][k-1+offset] < trace[d-1][k+1+offset]) {
			prevK = k + 1
		} else {
			prevK = k - 1
		}

		prevX := trace[d-1][prevK+offset]
		prevY := prevX - prevK

		var stepX, stepY int
		if prevK == k+1 {
			stepX = prevX
			stepY = prevY + 1
		} else {
			stepX = prevX + 1
			stepY = prevY
		}

		for x > stepX && y > stepY {
			x--
			y--
			edits = append(edits, diffEdit{op: DiffMatch, baseIdx: x, targetIdx: y})
		}

		if prevK == k+1 {
			edits = append(edits, diffEdit{op: DiffAdd, targetIdx: prevY})
		} else {
			edits = append(edits, diffEdit{op: DiffDelete, baseIdx: prevX})
		}

		x = prevX
		y = prevY
	}

	for x > 0 && y > 0 {
		x--
		y--
		edits = append(edits, diffEdit{op: DiffMatch, baseIdx: x, targetIdx: y})
	}

	// Reverse edits to get forward order
	for i, j := 0, len(edits)-1; i < j; i, j = i+1, j-1 {
		edits[i], edits[j] = edits[j], edits[i]
	}

	return edits
}
