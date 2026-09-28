package cudabpfdev

import (
	"fmt"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// KernelIdentity is an exact image, entry point and target allowlist key.
type KernelIdentity struct{ Digest, EntryPoint, Target string }

// InjectionPolicy selects insertion sites in an explicitly allowed kernel.
// MemoryInstructions are zero-based indices in the original parsed instruction
// sequence, and must name unpredicated global loads or stores. Instrumentation
// executes before these instructions. Entry and Exit select the first original
// instruction and final ret. Duplicate sites are coalesced.
type InjectionPolicy struct {
	Enabled            bool
	Allowlist          []KernelIdentity
	Entry, Exit        bool
	MemoryInstructions []int
	// StubFunctions opts into defined device functions with no parameters or
	// returns and an empty body. Bodiless declarations are rejected.
	StubFunctions []string
}

// Injection is a prepared PTX candidate and its original launch source. It is
// not proof of assembly, native equivalence or safe compiled resource cost.
// Loading a candidate requires linking, original/candidate resource comparison,
// and private module ownership per invocation so map state cannot be shared.
type Injection struct {
	original, candidate *ArtifactSource
	reason              string
	slots               uint32
	limits              resourceLimits
	addedRegisters      uint32
}

// Original returns an independent original launch source, including on fallback.
func (i *Injection) Original() (*ArtifactSource, error) {
	if i == nil || i.original == nil {
		return nil, fmt.Errorf("cudabpfdev: missing original source")
	}
	l, err := i.original.DescribeLaunch()
	if err != nil {
		return nil, err
	}
	return NewArtifactSource(l.Artifact, l.Geometry)
}

// Candidate returns a detached candidate for review and assembly, or a reason
// for leaving the original unchanged. A non-nil candidate has not been loaded.
// Its ABI and launch geometry are those of Original. Instrumentation counters
// live in module globals __cudabpfdev_map and __cudabpfdev_misses; a runtime must
// initialize and read them at invocation boundaries, using a private module.
func (i *Injection) Candidate() (*cudakernel.Artifact, string) {
	if i == nil {
		return nil, "missing injection"
	}
	if i.candidate == nil {
		return nil, i.reason
	}
	return cloneArtifact(i.candidate.launch.Artifact), ""
}

// PrepareInjection preserves the original and prepares an opt-in PTX candidate.
// The current admission path supports one straight-line entry with scalar ABI,
// register declarations and one instruction per line, ending in an unconditional
// ret. Branches, predicates, calls, barriers, asynchronous operations, globals,
// nested scopes and ambiguous formatting preserve the original with a reason.
// Expanded admission paths require their own safety proof.
//
// Each site executes the verified program with an explicitly synchronized warp
// membership mask and unique registers/labels. Original instructions and parameter
// declarations are retained verbatim. Program return values are ignored; only
// bounded map updates are observed. Declaration budgets include the original
// registers and all inserted scopes; final compiled resource checks remain required.
func PrepareInjection(source LaunchSource, p Program, b Budget, policy InjectionPolicy) (*Injection, error) {
	if source == nil {
		return nil, fmt.Errorf("cudabpfdev: nil launch source")
	}
	l, err := source.DescribeLaunch()
	if err != nil {
		return nil, err
	}
	original, err := NewArtifactSource(l.Artifact, l.Geometry)
	if err != nil {
		return nil, err
	}
	i := &Injection{original: original, slots: p.Slots}
	fallback := func(reason string) (*Injection, error) { i.reason = reason; return i, nil }
	if !policy.Enabled {
		return fallback("injection disabled")
	}
	a := original.launch.Artifact
	key := KernelIdentity{a.Digest, a.EntryPoint, a.Target}
	if !slices.Contains(policy.Allowlist, key) {
		return fallback("kernel not allowlisted")
	}
	if a.Kind != cudakernel.KindPTX {
		return fallback("injection requires original PTX")
	}
	p.Code = slices.Clone(p.Code)
	tr, err := Translate(p, b, a.Target)
	if err != nil {
		return fallback(err.Error())
	}
	layout, err := inspectInjection(a, policy)
	if err != nil {
		return fallback(err.Error())
	}
	sites := map[int]bool{}
	if policy.Entry {
		sites[layout.instructions[0].Line-1] = true
	}
	if policy.Exit {
		sites[layout.instructions[len(layout.instructions)-1].Line-1] = true
	}
	for _, index := range policy.MemoryInstructions {
		if index < 0 || index >= len(layout.instructions) {
			return fallback("memory site outside instruction sequence")
		}
		inst := layout.instructions[index]
		if !strings.HasPrefix(inst.Opcode, "ld.global.") && !strings.HasPrefix(inst.Opcode, "st.global.") {
			return fallback("selected instruction is not a global load or store")
		}
		sites[inst.Line-1] = true
	}
	stubCalls := map[int]string{}
	var probeFuncs []string
	probeSite := 0
	for _, inst := range layout.instructions {
		if inst.Mnemonic == "call" {
			target := stubCalleeName(inst.Operands)
			if slices.Contains(policy.StubFunctions, target) {
				probeName := fmt.Sprintf("__cudabpfdev_probe_%d", probeSite)
				stubCalls[inst.Line-1] = probeName
				probeFuncs = append(probeFuncs, emitProbeFunction(p, probeName))
				probeSite++
			}
		}
	}
	totalSites := len(sites) + len(stubCalls)
	if totalSites == 0 {
		return fallback("no insertion sites selected")
	}
	// Injected bodies need one additional b32 register for the live active mask
	// that is not part of standalone Translate's explicit mask argument.
	perSiteRegisters := uint64(tr.DeclaredRegisters) + 1
	if uint64(layout.registers)+uint64(totalSites)*perSiteRegisters > uint64(min(b.MaxRegisters, 255)) {
		return fallback("combined register declaration budget exceeded")
	}
	i.limits = tr.limits
	i.addedRegisters = uint32(totalSites) * uint32(perSiteRegisters)
	var out strings.Builder
	site := 0
	for line, text := range layout.lines {
		if line == layout.header {
			fmt.Fprintf(&out, ".visible .global .align 4 .u32 __cudabpfdev_map[%d];\n.visible .global .align 8 .u64 __cudabpfdev_misses;\n", max(p.Slots, 1))
			for _, fn := range probeFuncs {
				out.WriteString(fn)
			}
		}
		if sites[line] {
			out.WriteString(emitTrampoline(p, site))
			site++
		}
		if probeName, ok := stubCalls[line]; ok {
			target := stubCalleeName(layout.byLine[line].Operands)
			out.WriteString(strings.Replace(text, target, probeName, 1))
		} else {
			out.WriteString(text)
		}
		if line+1 < len(layout.lines) {
			out.WriteByte('\n')
		}
	}
	candidate := cloneArtifact(a)
	// NewArtifact recomputes the image digest; retain original ABI and provenance.
	candidate, err = cudakernel.NewArtifact(cudakernel.KindPTX, []byte(out.String()), a.EntryPoint, candidate.Schema, a.Target,
		cudakernel.WithJIT(a.JITOptions), cudakernel.WithToolchain(a.Toolchain), cudakernel.WithCompilerOptions(a.CompilerOpts), cudakernel.WithDependencies(a.Dependencies))
	if err != nil {
		return fallback(err.Error())
	}
	i.candidate, err = NewArtifactSource(candidate, l.Geometry)
	if err != nil {
		return fallback(err.Error())
	}
	return i, nil
}

type injectionLayout struct {
	lines        []string
	header       int
	instructions []cudaptx.Instruction
	byLine       map[int]cudaptx.Instruction
	registers    uint32
}

var registerDeclaration = regexp.MustCompile(`^\.reg \.(pred|b16|b32|b64|u16|u32|u64|s16|s32|s64|f32|f64) ([%a-zA-Z_][a-zA-Z_0-9]*)(?:<([0-9]+)>)?;$`)
var percentOperand = regexp.MustCompile(`%[A-Za-z_][A-Za-z_0-9]*(?:\.[xyz])?`)

func stubCalleeName(operands string) string {
	for _, tok := range strings.FieldsFunc(operands, func(r rune) bool {
		return r == '(' || r == ')' || r == ',' || r == ' ' || r == '\t' || r == ';'
	}) {
		if tok != "" {
			return tok
		}
	}
	return ""
}

func inspectInjection(a *cudakernel.Artifact, policy InjectionPolicy) (injectionLayout, error) {
	bad := func(reason string) (injectionLayout, error) {
		return injectionLayout{}, fmt.Errorf("cudabpfdev: %s", reason)
	}
	text := strings.TrimRight(string(a.Raw), "\x00")
	if strings.Contains(text, "__cudabpfdev_") || strings.Contains(text, "/*") || strings.ContainsAny(text, "\"\\\r") {
		return bad("ambiguous or reserved PTX text")
	}
	m, err := cudaptx.ParseString(text)
	if err != nil {
		return bad("PTX parsing failed")
	}
	if m.Version != "9.0" || m.Target != a.Target || m.AddressSize != 64 {
		return bad("module outside injection contract")
	}
	var entryKernel *cudaptx.Kernel
	for _, k := range m.Kernels {
		if k.Entry && k.Name == a.EntryPoint {
			if entryKernel != nil {
				return bad("multiple entry kernels with entry name")
			}
			entryKernel = k
		}
	}
	if entryKernel == nil {
		return bad("kernel outside injection contract")
	}
	called := map[string]bool{}
	for _, inst := range entryKernel.Instructions {
		if inst.Mnemonic == "call" {
			called[stubCalleeName(inst.Operands)] = true
		}
	}
	definedStubs := map[string]bool{}
	for _, k := range m.Kernels {
		if k == entryKernel {
			continue
		}
		if k.Entry {
			return bad("module outside injection contract")
		}
		if !slices.Contains(policy.StubFunctions, k.Name) {
			if called[k.Name] {
				return bad("instruction outside straight-line contract: call")
			}
			return bad("module outside injection contract")
		}
		if len(k.Returns) != 0 || len(k.Params) != 0 {
			return bad("stub function has parameters or returns")
		}
		if !k.Body {
			return bad("stub function has no body")
		}
		definedStubs[k.Name] = true
		if len(k.Instructions) > 1 || (len(k.Instructions) == 1 && k.Instructions[0].Mnemonic != "ret") {
			return bad("stub function is not empty")
		}
	}
	for name := range called {
		if slices.Contains(policy.StubFunctions, name) && !definedStubs[name] {
			return bad("stub function is not defined")
		}
	}
	k := *entryKernel
	if !k.Entry || !k.Body || k.Name != a.EntryPoint || k.Tuning != (cudaptx.Tuning{}) || len(k.Instructions) == 0 {
		return bad("kernel outside injection contract")
	}
	if len(k.Params) != len(a.Schema.Params) {
		return bad("parameter ABI unavailable")
	}
	var params []cudakernel.Param
	for j, param := range k.Params {
		want := a.Schema.Params[j]
		if param.Space != "param" || param.Size != 0 || param.Align != 0 || param.Name != want.Name || param.Type != want.Type {
			return bad("parameter ABI mismatch or unsupported layout")
		}
		if want.IsPointer && param.Type != "u64" && param.Type != "b64" {
			return bad("pointer ABI is not 64-bit")
		}
		params = append(params, cudakernel.Param{Name: param.Name, Type: param.Type, IsPointer: want.IsPointer})
	}
	canonical, err := cudakernel.NewSchema(params...)
	if err != nil || !reflect.DeepEqual(canonical, a.Schema) {
		return bad("parameter ABI width or offset mismatch")
	}
	l := injectionLayout{lines: strings.Split(text, "\n"), header: k.StartLine - 1, instructions: k.Instructions}
	if k.StartLine < 1 || k.EndLine > len(l.lines) {
		return bad("invalid kernel locations")
	}
	clean := func(line string) string { line, _, _ = strings.Cut(line, "//"); return strings.TrimSpace(line) }
	open := -1
	for line := l.header; line < k.EndLine; line++ {
		if clean(l.lines[line]) == "{" {
			open = line
			break
		}
	}
	if open < 0 || clean(l.lines[k.EndLine-1]) != "}" {
		return bad("kernel braces must occupy separate lines")
	}
	inFunc := func(line int) bool {
		if line >= l.header && line < k.EndLine {
			return true
		}
		for _, stub := range m.Kernels {
			if !stub.Entry && slices.Contains(policy.StubFunctions, stub.Name) {
				if line >= stub.StartLine-1 && line < stub.EndLine {
					return true
				}
			}
		}
		return false
	}
	for line, raw := range l.lines {
		if inFunc(line) {
			continue
		}
		s := clean(raw)
		if s != "" && s != ".version 9.0" && s != ".target "+a.Target && s != ".address_size 64" {
			return bad("module declarations outside injection contract")
		}
	}
	byLine := map[int]cudaptx.Instruction{}
	for index, inst := range k.Instructions {
		line := inst.Line - 1
		if line <= open || line >= k.EndLine-1 || inst.Predicate != "" || byLine[line].Opcode != "" || clean(l.lines[line]) != strings.TrimSpace(inst.Raw) {
			return bad("instruction location or predicate is unsupported")
		}
		if inst.Mnemonic == "ret" {
			if index != len(k.Instructions)-1 || clean(l.lines[line]) != "ret;" {
				return bad("only a final unconditional return is supported")
			}
		} else if inst.Mnemonic == "call" {
			target := stubCalleeName(inst.Operands)
			if target == "" || !slices.Contains(policy.StubFunctions, target) {
				return bad("instruction outside straight-line contract: " + inst.Opcode)
			}
		} else if !admittedInstruction(inst.Opcode) {
			return bad("instruction outside straight-line contract: " + inst.Opcode)
		}
		byLine[line] = inst
	}
	l.byLine = byLine
	if k.Instructions[len(k.Instructions)-1].Opcode != "ret" {
		return bad("missing final return")
	}
	registers := map[string]bool{}
	for _, name := range strings.Fields("%tid.x %tid.y %tid.z %ntid.x %ntid.y %ntid.z %ctaid.x %ctaid.y %ctaid.z %nctaid.x %nctaid.y %nctaid.z %laneid") {
		registers[name] = true
	}
	for line := open + 1; line < k.EndLine-1; line++ {
		s := clean(l.lines[line])
		if s == "" || byLine[line].Opcode != "" {
			continue
		}
		decl := registerDeclaration.FindStringSubmatch(s)
		if decl == nil || line >= k.Instructions[0].Line-1 {
			return bad("body declaration or control flow outside contract")
		}
		count := uint64(1)
		if decl[3] != "" {
			count, err = strconv.ParseUint(decl[3], 10, 32)
			if err != nil || count == 0 {
				return bad("invalid register count")
			}
		}
		if count > 255 {
			return bad("original register declarations exceed limit")
		}
		if decl[3] == "" {
			registers[decl[2]] = true
		} else {
			for n := uint64(0); n < count; n++ {
				registers[decl[2]+strconv.FormatUint(n, 10)] = true
			}
		}
		if strings.HasSuffix(decl[1], "64") {
			count *= 2
		}
		if count > 255 || uint64(l.registers)+count > 255 {
			return bad("original register declarations exceed limit")
		}
		l.registers += uint32(count)
	}
	for _, inst := range k.Instructions {
		for _, reg := range percentOperand.FindAllString(inst.Operands, -1) {
			if !registers[reg] {
				return bad("undeclared or timing-sensitive register: " + reg)
			}
		}
	}
	return l, nil
}

func admittedInstruction(op string) bool {
	base, _, _ := strings.Cut(op, ".")
	if base == "ld" || base == "st" {
		for _, space := range []string{"ld.param.", "ld.global.", "st.global."} {
			if strings.HasPrefix(op, space) {
				return slices.Contains([]string{"u32", "s32", "b32", "f32", "u64", "s64", "b64", "f64"}, strings.TrimPrefix(op, space))
			}
		}
		return false
	}
	return slices.Contains([]string{"mov", "add", "sub", "mul", "mad", "fma", "cvt", "cvta", "setp", "selp", "and", "or", "xor", "shl", "shr", "min", "max", "neg", "abs"}, base)
}

// ptxIdentifier matches a PTX identifier, capturing any dot that immediately
// precedes it. An identifier after a dot is a type or modifier suffix -- the
// .pred in ".reg .pred" and in "and.pred" -- and never names a register, so the
// trampoline's renaming must leave it alone.
var ptxIdentifier = regexp.MustCompile(`\.?[A-Za-z_][A-Za-z_0-9]*`)

func emitTrampoline(p Program, site int) string {
	prefix := fmt.Sprintf("__cudabpfdev_site%d_", site)
	return "{\n" + emitProgramBody(p, prefix, false) + "}\n"
}

func emitProbeFunction(p Program, name string) string {
	var out strings.Builder
	fmt.Fprintf(&out, ".func %s\n{\n", name)
	out.WriteString(emitProgramBody(p, name+"_", true))
	out.WriteString("}\n")
	return out.String()
}

// emitProgramBody is shared by inline trampolines and stub functions so their
// verifier-program body, register initialization and warp leader guard agree.
func emitProgramBody(p Program, prefix string, returnAtEnd bool) string {
	var out strings.Builder
	out.WriteString(".reg .b32 lane, leader, reverse_mask, mask, actual_mask, bit, discarded, block_id;\n.reg .b64 map_addr, result_addr, miss_addr, offset, addr, ignored64;\n.reg .pred pred, is_leader, in_bounds, write_map;\n")
	fmt.Fprintf(&out, ".reg .b32 r<%d>;\n", p.Registers)
	for pc, in := range p.Code {
		if in.Op == Repeat {
			fmt.Fprintf(&out, ".reg .b32 loop%d;\n", pc)
		}
	}
	out.WriteString("setp.eq.u32 pred, 0, 0;\nactivemask.b32 actual_mask;\nvote.sync.ballot.b32 mask, pred, actual_mask;\nmov.u32 lane, %laneid;\nbrev.b32 reverse_mask, mask;\nbfind.u32 leader, reverse_mask;\nsub.u32 leader, 31, leader;\nsetp.eq.u32 is_leader, lane, leader;\nmov.u64 map_addr, __cudabpfdev_map;\nmov.u64 miss_addr, __cudabpfdev_misses;\ncvta.to.global.u64 map_addr, map_addr;\ncvta.to.global.u64 miss_addr, miss_addr;\n")
	names := map[string]bool{}
	for _, name := range strings.Fields("lane leader reverse_mask mask actual_mask bit discarded block_id map_addr result_addr miss_addr offset addr ignored64 pred is_leader in_bounds write_map r done") {
		names[name] = true
	}
	for r := uint8(0); r < p.Registers; r++ {
		name := fmt.Sprintf("r%d", r)
		names[name] = true
		fmt.Fprintf(&out, "mov.u32 %s, 0;\n", name)
	}
	for pc, in := range p.Code {
		names[fmt.Sprintf("pc%d", pc)] = true
		if in.Op == Repeat {
			name := fmt.Sprintf("loop%d", pc)
			names[name] = true
			fmt.Fprintf(&out, "mov.u32 %s, 0;\n", name)
		}
	}
	for pc, in := range p.Code {
		fmt.Fprintf(&out, "pc%d:\n", pc)
		if in.Op == Exit {
			out.WriteString("bra done;\n")
		} else {
			emitters[in.Op](&out, pc, in, p.Slots)
		}
	}
	out.WriteString("done:\n")
	if returnAtEnd {
		out.WriteString("ret;\n")
	}
	return ptxIdentifier.ReplaceAllStringFunc(out.String(), func(token string) string {
		if strings.HasPrefix(token, ".") {
			return token
		}
		if names[token] {
			return prefix + token
		}
		return token
	})
}
