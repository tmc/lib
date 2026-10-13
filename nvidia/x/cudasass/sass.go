package cudasass

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

// Disassembly represents a parsed collection of disassembled SASS functions.
type Disassembly struct {
	// Target is the explicit architecture declared by .target or a code-for
	// preamble. It is empty when declarations are missing, malformed,
	// conflicting, or exceed the parser's target declaration bounds.
	// It is not inferred from instruction spellings or binary encodings.
	Target    string
	Functions []*Function
	funcMap   map[string]*Function
	sections  []Section
}

// Function represents a disassembled GPU kernel with its instruction sequence.
type Function struct {
	Name         string
	BaseAddress  uint64
	Instructions []Instruction

	// Symbols lists the function symbols nvdisasm declares inside this
	// function's section other than the function itself, in address order.
	// The compiler places the subroutines a kernel calls, such as its
	// division slowpath, after the kernel's own code in the same section,
	// and their instructions are in Instructions like any other.
	Symbols []Symbol

	// ELFSection is the ELF section the instructions were read from, as
	// the listing names it: ".text.<Name>" when a header spelled it, and ""
	// when every header was a spelling that names no section, such as
	// cuobjdump's "Function : <Name>". [Parse] refuses a listing that
	// opens any other code section, so no other value occurs.
	ELFSection string

	// Labels maps each code label in the function, such as ".L_x_0", to the
	// address of the instruction that follows it. A label with no instruction
	// after it in the function, such as the one nvdisasm uses to size it, is
	// omitted.
	Labels map[string]uint64
}

// A Symbol is a function symbol declared inside another function's section,
// at the address of its first instruction.
type Symbol struct {
	Name    string
	Address uint64
}

// SymbolAt returns the name of the symbol whose code contains the
// function-relative address: the last of Symbols at or before it, or the
// function's own name when no symbol precedes it.
func (f *Function) SymbolAt(addr uint64) string {
	name := f.Name
	for _, s := range f.Symbols {
		if s.Address > addr {
			break
		}
		name = s.Name
	}
	return name
}

// Instruction represents a single decoded hardware SASS instruction.
type Instruction struct {
	Address     uint64   // Hardware PC address (e.g. 0x7f001230)
	Predicate   string   // Instruction predicate, e.g. "@P0", "@!PT"
	Opcode      string   // Mnemonic and modifiers, e.g. "FFMA", "LDG.E.CONSTANT.F32", "STS.32"
	Operands    string   // Register and memory operands, e.g. "R0, R4, R5, R0"
	SourceFile  string   // Source file name from line-info comments, e.g. "kernels.cu"
	SourceLine  int      // Source line number from line-info comments
	InlinedFile string   // Inlined call site file name, if inlined
	InlinedLine int      // Inlined call site line number, if inlined
	PTXLine     int      // PTX line number from .nv_debug_ptx_txt comments, if present
	LiveGPR     int      // Live General Purpose Registers from register life-range analysis
	LivePred    int      // Live Predicate registers
	LiveUGPR    int      // Live Uniform GPRs (sm_70+)
	LiveUPred   int      // Live Uniform Predicate registers (sm_70+)
	Encoding    Encoding // Raw instruction encoding, when [Options.PrintHex] was requested
	Comment     string   // Optional disassembly comment
	Raw         string   // Full raw disassembly line
}

// An Encoding is the raw hardware encoding of one SASS instruction, as
// nvdisasm prints it under [Options.PrintHex].
//
// An instruction on the architectures nvdisasm still disassembles occupies two
// 64-bit words. Lo is the word nvdisasm prints on the instruction's own line;
// Hi is the word it prints alone on the line below. In instruction memory the
// two appear in that order, each little-endian, which is what [Encoding.Bytes]
// returns and what nvdisasm's --binary mode reads back.
//
// The zero Encoding means no encoding was captured, not an instruction whose
// encoding is zero: an all-zero 128-bit word is not a legal instruction, and
// nvdisasm rejects one with "Illegal instruction found".
type Encoding struct {
	Lo, Hi uint64
}

// Present reports whether an encoding was captured. See [Encoding] for why the
// zero value cannot be a real instruction.
func (e Encoding) Present() bool { return e != Encoding{} }

// Bytes returns the encoding as it occupies instruction memory: Lo then Hi,
// each little-endian. The result is accepted by nvdisasm --binary, which reads
// a bare sequence of such encodings.
func (e Encoding) Bytes() [16]byte {
	var b [16]byte
	binary.LittleEndian.PutUint64(b[0:8], e.Lo)
	binary.LittleEndian.PutUint64(b[8:16], e.Hi)
	return b
}

// String formats both words in the order nvdisasm prints them.
func (e Encoding) String() string {
	return fmt.Sprintf("0x%016x 0x%016x", e.Lo, e.Hi)
}

// Pressure returns the live register pressure at this instruction.
func (i Instruction) Pressure() RegisterPressure {
	return RegisterPressure{
		GPR:   i.LiveGPR,
		Pred:  i.LivePred,
		UGPR:  i.LiveUGPR,
		UPred: i.LiveUPred,
	}
}

// Function returns the function by name, or nil if not found.
func (d *Disassembly) Function(name string) *Function {
	if d.funcMap == nil {
		d.funcMap = make(map[string]*Function, len(d.Functions))
		for _, fn := range d.Functions {
			d.funcMap[fn.Name] = fn
		}
	}
	return d.funcMap[name]
}

var sourceLocationPattern = regexp.MustCompile(`^//##\s+File\s+"([^"]+)",\s+line\s+([0-9]+)`)
var inlineLocationPattern = regexp.MustCompile(`\s+inlined\s+at\s+"([^"]+)",\s+line\s+([0-9]+)`)

// findAddressCommentEnd finds the index of the closing "*/" for a leading
// "/*" comment in trimmed. It searches in trimmed[2:] to avoid matching
// overlapping characters (such as in "/*/"). It returns -1 if trimmed does
// not begin with "/*" or if no closing "*/" follows.
func findAddressCommentEnd(trimmed string) int {
	if !strings.HasPrefix(trimmed, "/*") {
		return -1
	}
	idx := strings.Index(trimmed[2:], "*/")
	if idx == -1 {
		return -1
	}
	return idx + 2
}

// targetAlignment returns the instruction alignment in bytes for target.
// Architectures prior to sm_70 (sm_30 through sm_62) have 8-byte instructions.
// Modern architectures (sm_70 and later) have 16-byte instructions.
// If target is unparseable or unknown, it defaults to 16 bytes.
func targetAlignment(target string) uint64 {
	target = strings.TrimPrefix(target, "sm_")
	target = strings.TrimPrefix(target, "SM")
	var num int
	for _, r := range target {
		if r >= '0' && r <= '9' {
			num = num*10 + int(r-'0')
		} else {
			break
		}
	}
	if num > 0 && num < 70 {
		return 8
	}
	return 16
}

// Parse reads SASS disassembly text from r and returns a Disassembly.
//
// A ".text.NAME:" label names the kernel NAME. Parse reads the label from
// left to right, skipping each terminated string or character literal as
// rule 1 below reads one; the first "//" outside a literal begins a comment
// that is not part of the label. The label's terminal colon is the first
// colon outside a literal and before that comment that is followed by
// nothing but blanks, "/* */" comments and a final "//" comment, where a
// blank is any Unicode white space. A ".text." line with no such colon names
// the kernel by its text before that comment. So these labels name k, k and
// f<S{": //"}>() in turn:
//
//	.text.k // note:
//	.text.k: // a: b
//	.text.f<S{": //"}>():
//
// A ".section .text.NAME" directive names the kernel NAME. Its flags field,
// if it has one, is the first comma outside a literal that is followed by
// optional spaces or tabs, a terminated double-quoted literal holding only
// LLVM's section flag bytes (a, e, o, w, x, M, S, T, G, R, ?, c, d, l, s
// and y, or none), and then the end of the line, a "//" comment, or a
// comma, optional spaces or tabs and a section type: '@' or '%' followed by
// identifier bytes or a quoted literal, or a quoted literal alone. Anything
// after the comma that follows the type is not read. A directive with no
// flags field names the kernel by its whole text. So a comma or quote inside
// a literal, a comma and quoted literal that more of the name follows, and a
// comma and quoted literal holding any other byte belong to the name. The
// first three of these directives name the kernel
// decltype(fp, "<char const [5]>") s1<int>(int), and the last names the
// kernel k, "lit":
//
//	.section .text.decltype(fp, "<char const [5]>") s1<int>(int),"ax",@progbits
//	.section .text.decltype(fp, "<char const [5]>") s1<int>(int), "ax", @progbits
//	.section .text.decltype(fp, "<char const [5]>") s1<int>(int)
//	.section .text.k, "lit"
//
// Parse returns an error, rather than guess where the section's name ends,
// for a ".section" directive whose name, the text before its flags field,
// holds a literal left open as rule 1 below reads one, or whose flags field
// is followed by a second comma that would itself begin a flags field, as in
//
//	.section .text.k"x
//	.section .text.k, "a","ax",@progbits
//
// A name that itself ends, outside every literal, in a comma and a quoted
// literal of flag bytes, with no second such field after it, is read as a
// shorter name and its flags: the directive
//
//	.section .text.k, "a"
//
// names the kernel k, never k, "a". No such name is supported.
//
// Parse returns an error, rather than guess where the kernel's name ends,
// for a ".text." line whose name, read from left to right, breaks this
// lexical rule:
//
//  1. A double or single quote begins a literal that ends at the next
//     unescaped copy of the same quote; a backslash escapes the byte after
//     it. Nothing inside a literal counts below. An unterminated literal is
//     refused.
//  2. Only '(', '[' and '{' open a bracket, and only ')', ']' and '}' close
//     one; '<' and '>' are ignored. A closing bracket that does not match
//     the innermost open bracket, or a bracket left open, is refused.
//  3. Every "::" is accepted. A run of three or more colons is refused.
//  4. A single colon is accepted only
//     (a) in "[abi:" followed by one or more letters, digits, '_' or '$'
//     and then ']', as in "h[abi:cxx11](int)";
//     (b) in "auto:N", where "auto" begins the name or follows a byte other
//     than a letter, digit, '_' or '$', and N is one or more digits
//     followed by a byte other than those, as in "{lambda(auto:1)#1}";
//     (c) as a conditional's colon, paired with the latest '?' at the same
//     (), [] or {} depth that no colon has paired, with text other than
//     white space between them, as in "decltype(fp ? 1 : 2) f<int>(int)".
//     Every other single colon is refused.
//  5. "/*", "*/" and "//" are refused.
//
// So ".text.k:junk", ".text.k: junk", ".text.k:\u00a0junk",
// ".text.k /* c */:" and ".text.k(a ? b) : c" are refused. Because '<' and
// '>' are not brackets, every "::" is accepted, a conditional pairs at any
// depth and "[abi:" and "auto:N" carry no condition on what precedes or
// encloses them, some constructed lines that are no C++ name are accepted
// and name a kernel by their whole name: ".text.k ? f<a : b>",
// ".text.f<1 >> 2, b ? 1 : 2>()", ".text.k ? junk : junk", ".text.k:: junk",
// ".text.k ::junk", ".text.f(a::)" and ".text.k [abi:cxx11]" among them.
//
// A colon, comma, quote or "//" inside a terminated literal, as in
// SL8<L8{"a//b"}> or S<5>{": //"}, belongs to the name and ends neither a
// label nor a directive's name. These rules are lexical; Parse is not a C++
// parser. It still refuses, or does not support:
//
//   - a label holding a literal left open, refused whatever follows the
//     open quote, since rule 1 cannot tell where the literal would end;
//   - a directive whose name itself ends, outside every literal, in a comma
//     and a quoted literal of flag bytes, which is read as a shorter name
//     followed by flags;
//   - a directive name breaking rules 2 through 5: a ".section" directive's
//     name is checked only for a literal left open, so it is otherwise read
//     as its text before the flags field and is not refused.
func Parse(r io.Reader) (*Disassembly, error) {
	scanner := bufio.NewScanner(r)
	d := &Disassembly{
		funcMap: make(map[string]*Function),
	}

	var curFn *Function
	var curSourceFile string
	var curSourceLine int
	var curInlinedFile string
	var curInlinedLine int
	var curPTXLine int
	// Instruction awaiting the high word of its encoding, identified by
	// position because appending to Instructions may move the slice.
	pendingFn, pendingIdx := (*Function)(nil), 0
	// pendingSymbol is a function symbol declared by a .type directive whose
	// address is that of the next instruction parsed.
	pendingSymbol := ""
	// pendingLabels are code labels whose address is that of the next
	// instruction parsed.
	var pendingLabels []string

	// curSection indexes d.sections, or is -1 between kernels. An index rather
	// than a pointer: appending to d.sections moves the backing array, and a
	// pointer into it would then write to a stale copy.
	curSection := -1
	pendingBanner := ""
	archAlign := uint64(16)
	targetDeclarations := 0
	targetUnknown := false
	curSecHasAddr := false
	curSecLastAddr := uint64(0)
	// curNonText names the code section other than .text.<kernel> that the
	// listing has open, or is empty. Data may follow its header; an
	// instruction may not.
	curNonText := ""

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if target, declared := disassemblyTarget(trimmed); declared {
			targetDeclarations++
			if target == "" || targetDeclarations > 4096 || (d.Target != "" && d.Target != target) {
				targetUnknown = true
			}
			if targetUnknown {
				d.Target = ""
			} else {
				d.Target = target
			}
		}

		if malformedTextLabel(trimmed) {
			return nil, fmt.Errorf("cudasass: malformed label %q", trimmed)
		}
		if malformedSectionDirective(trimmed) {
			return nil, fmt.Errorf("cudasass: malformed section directive %q", trimmed)
		}

		if strings.HasPrefix(trimmed, ".target") {
			fields := strings.Fields(trimmed)
			if len(fields) >= 2 {
				archAlign = targetAlignment(fields[1])
			}
		}

		if name, nonText, opens := openedSection(trimmed); opens {
			curNonText = ""
			if nonText {
				curNonText = name
			}
		} else if name, ok := nonTextLabel(trimmed); ok {
			curNonText = name
		}

		// Section accounting runs before the blank-line skip below so that a
		// section's Text is the listing region byte for byte, blank lines and
		// all, rather than a filtered rendering of it.
		if bannerName, isBanner := sectionBanner(trimmed); isBanner {
			curSection, pendingBanner = -1, ""
			curSecHasAddr, curSecLastAddr = false, 0
			if bannerName != "" {
				pendingBanner = line
			}
		} else if name := extractFunctionName(trimmed); name != "" && !(curSection >= 0 && d.sections[curSection].Name == name) {
			// The guard is for a kernel's own label: nvdisasm prints both
			// ".section .text.k4" and, further down, ".text.k4:" for one
			// kernel, and both spellings read as a header. Splitting on the
			// second would open a second region for a kernel the listing
			// contains once -- and leave the first region holding only the
			// kernel's header lines, with none of its instructions.
			d.sections = append(d.sections, Section{Name: name, ELFSection: headerELFSection(trimmed)})
			curSection = len(d.sections) - 1
			curSecHasAddr, curSecLastAddr = false, 0
			if pendingBanner != "" {
				d.sections[curSection].Text = append(d.sections[curSection].Text, pendingBanner...)
				d.sections[curSection].Text = append(d.sections[curSection].Text, '\n')
			}
			pendingBanner = ""
		} else if curSection >= 0 && d.sections[curSection].ELFSection == "" {
			d.sections[curSection].ELFSection = headerELFSection(trimmed)
		}
		if curSection >= 0 {
			d.sections[curSection].Text = append(d.sections[curSection].Text, line...)
			d.sections[curSection].Text = append(d.sections[curSection].Text, '\n')
		}

		if trimmed == "" {
			continue
		}

		// The high word of an instruction encoding is printed alone on the
		// line below the instruction it belongs to, so it attaches only to an
		// instruction parsed from the immediately preceding line.
		word, isEncoding := encodingWord(trimmed)
		fn, idx := pendingFn, pendingIdx
		pendingFn = nil
		if isEncoding {
			if fn != nil {
				fn.Instructions[idx].Encoding.Hi = word
			}
			continue
		}

		if dir, ok := ParseSourceDirective(trimmed); ok {
			// The PTX tier names the same instructions as the source tier but
			// in its own line namespace, so it updates PTXLine and leaves the
			// source location standing rather than overwriting it.
			if dir.Kind == PTXText {
				curPTXLine = dir.Line
				continue
			}
			curSourceFile = dir.File
			curSourceLine = dir.Line
			curInlinedFile = dir.InlinedFile
			curInlinedLine = dir.InlinedLine
			continue
		}

		if name, ok := functionSymbol(trimmed); ok && curFn != nil && name != curFn.Name {
			// nvdisasm declares the symbol twice, once with the section's
			// header lines and once before its label; the label's is the
			// one at the code.
			pendingSymbol = name
			continue
		}

		// Detect a function header. See the package doc for the four
		// spellings and which tools emit them.
		if fnName := extractFunctionName(trimmed); fnName != "" {
			pendingSymbol = ""
			pendingLabels = nil
			curSourceFile = ""
			curSourceLine = 0
			curInlinedFile = ""
			curInlinedLine = 0
			curPTXLine = 0
			curFn = d.Function(fnName)
			if curFn == nil {
				curFn = &Function{
					Name: fnName,
				}
				d.Functions = append(d.Functions, curFn)
				d.funcMap[fnName] = curFn
			}
			if curFn.ELFSection == "" {
				curFn.ELFSection = headerELFSection(trimmed)
			}
			continue
		}

		if name, ok := codeLabel(trimmed); ok {
			pendingLabels = append(pendingLabels, name)
			continue
		}

		// Parse instruction line
		if inst, sawEncoding, ok := parseInstructionLine(line); ok {
			if curNonText != "" {
				return nil, fmt.Errorf("cudasass: section %s: instruction at 0x%04x: %w", curNonText, inst.Address, ErrNotTextSection)
			}
			if curSection < 0 {
				return nil, fmt.Errorf("cudasass: instruction at 0x%04x outside section", inst.Address)
			}
			secName := d.sections[curSection].Name
			if inst.Address%archAlign != 0 {
				return nil, fmt.Errorf("cudasass: section %s: misaligned instruction address 0x%04x (must be a multiple of %d)", secName, inst.Address, archAlign)
			}
			if curSecHasAddr && inst.Address <= curSecLastAddr {
				return nil, fmt.Errorf("cudasass: section %s: decreasing address 0x%04x after 0x%04x", secName, inst.Address, curSecLastAddr)
			}
			curSecHasAddr = true
			curSecLastAddr = inst.Address

			inst.SourceFile = curSourceFile
			inst.SourceLine = curSourceLine
			inst.InlinedFile = curInlinedFile
			inst.InlinedLine = curInlinedLine
			inst.PTXLine = curPTXLine

			if curFn == nil {
				curFn = &Function{
					Name: "default",
				}
				d.Functions = append(d.Functions, curFn)
				d.funcMap["default"] = curFn
			}
			if len(curFn.Instructions) == 0 {
				curFn.BaseAddress = inst.Address
			}
			if pendingSymbol != "" {
				curFn.Symbols = append(curFn.Symbols, Symbol{Name: pendingSymbol, Address: inst.Address})
				pendingSymbol = ""
			}
			for _, name := range pendingLabels {
				if curFn.Labels == nil {
					curFn.Labels = make(map[string]uint64)
				}
				curFn.Labels[name] = inst.Address
			}
			pendingLabels = nil
			curFn.Instructions = append(curFn.Instructions, inst)
			if sawEncoding {
				pendingFn, pendingIdx = curFn, len(curFn.Instructions)-1
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return d, nil
}

// ParseString parses SASS disassembly text directly from a string.
func ParseString(sass string) (*Disassembly, error) {
	return Parse(strings.NewReader(sass))
}

// DisassembleCUBIN invokes nvdisasm on cubin bytes to produce a structured Disassembly.
func DisassembleCUBIN(cubin []byte) (*Disassembly, error) {
	r, err := (Disassembler{}).Disassemble(context.Background(), cubin, Options{})
	if err != nil {
		return nil, err
	}
	return ParseString(string(r.Stdout))
}

// DisassembleCFG invokes nvdisasm on cubin bytes to produce Graphviz DOT control flow graph text.
// If basicBlocks is true, basic blocks are generated instead of hyperblocks.
func DisassembleCFG(cubin []byte, basicBlocks bool) (string, error) {
	r, err := (Disassembler{}).ControlFlowGraph(context.Background(), cubin, basicBlocks)
	if err != nil {
		return "", err
	}
	return string(r.Stdout), nil
}

// extractFunctionName returns the kernel a header line names, or "" if line
// is not a kernel header. The name is the header's whole text between its
// prefix and its terminator, so a demangled name such as
// "scale(float *, int)" keeps its spaces, commas and colons. A
// ".section .text.NAME" directive's name ends at the comma before its flags
// field, as sectionFlags finds it, or at the end of the line when it has no
// flags field. A ".text.NAME:" label ends at the colon labelColon finds,
// which is followed only by blanks and comments, such as the live-register
// table nvdisasm appends in --life-range-mode count. A ".text." line without
// such a colon still names a kernel, by its text before any "//" comment
// outside a literal (see lineComment), so that the kernel it begins is never
// merged into the one before it. Parse refuses the label lines whose name
// malformedTextLabel rejects before extractFunctionName reads them.
func extractFunctionName(line string) string {
	if isSectionDirective(line) {
		rest := strings.TrimSpace(strings.TrimPrefix(line, ".section"))
		if i, _ := sectionFlags(rest); i >= 0 {
			rest = strings.TrimSpace(rest[:i])
		}
		name, ok := strings.CutPrefix(rest, ".text.")
		if !ok {
			return ""
		}
		return strings.TrimSuffix(name, ":")
	}
	if strings.HasPrefix(line, "Function : ") {
		return strings.TrimSpace(strings.TrimPrefix(line, "Function : "))
	}
	if strings.HasPrefix(line, "--- Kernel: ") && strings.HasSuffix(line, " ---") {
		return strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "--- Kernel: "), " ---"))
	}
	if rest, ok := strings.CutPrefix(line, ".text."); ok {
		if i := labelColon(rest); i >= 0 {
			return strings.TrimSpace(rest[:i])
		}
		if i := lineComment(rest); i >= 0 {
			rest = rest[:i]
		}
		return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(rest), ":"))
	}
	return ""
}

// labelColon returns the index in rest, a ".text." label with its prefix
// removed, of the colon that ends the label, or -1 if none does. It reads
// rest as lineComment does: a quote that opens a terminated literal is
// skipped through the literal's closing quote, and the first "//" outside a
// literal begins a comment, after which no colon ends the label. A demangled
// name's own colons, as in "ns::scale", are followed by more of the name; the
// label's colon is the first one outside a literal and before any comment
// that labelTail accepts what follows.
func labelColon(rest string) int {
	for i := 0; i < len(rest); i++ {
		switch rest[i] {
		case '"', '\'':
			if end := literalEnd(rest, i); end >= 0 {
				i = end
			}
		case '/':
			if i+1 < len(rest) && rest[i+1] == '/' {
				return -1
			}
		case ':':
			if labelTail(rest[i+1:]) {
				return i
			}
		}
	}
	return -1
}

// labelTail reports whether tail, the text after a colon in a ".text."
// label, is nothing but blanks, "/* */" comments and a final "//" comment,
// so that the colon ends the label.
func labelTail(tail string) bool {
	tail = strings.TrimSpace(tail)
	for strings.HasPrefix(tail, "/*") {
		end := strings.Index(tail[2:], "*/")
		if end < 0 {
			break
		}
		tail = strings.TrimSpace(tail[2+end+2:])
	}
	return tail == "" || strings.HasPrefix(tail, "//")
}

// codeLabel parses a label line such as ".L_x_0:". Function headers are
// recognized before this is consulted.
func codeLabel(line string) (string, bool) {
	name, ok := strings.CutSuffix(line, ":")
	if !ok || name == "" || strings.ContainsAny(name, " \t:,/") {
		return "", false
	}
	return name, true
}

// functionSymbol parses a ".type NAME,@function" directive.
func functionSymbol(line string) (string, bool) {
	rest, ok := strings.CutPrefix(line, ".type")
	if !ok || rest == "" || (rest[0] != ' ' && rest[0] != '\t') {
		return "", false
	}
	rest = strings.TrimSpace(rest)
	// A trailing comment cannot supply the type attribute. A quoted name
	// can contain a literal //, so find the comment outside its quotes.
	quoted := false
	for i := 0; i < len(rest); i++ {
		switch {
		case quoted && rest[i] == '\\':
			i++
		case rest[i] == '"':
			quoted = !quoted
		case !quoted && strings.HasPrefix(rest[i:], "//"):
			rest = strings.TrimSpace(rest[:i])
		}
	}
	name, ok := strings.CutSuffix(rest, ",@function")
	if !ok {
		return "", false
	}
	name = strings.TrimSpace(name)
	if strings.HasPrefix(name, "\"") && strings.HasSuffix(name, "\"") && len(name) >= 2 {
		name = name[1 : len(name)-1]
	}
	if name == "" {
		return "", false
	}
	return name, true
}

// ParseInstructionLine parses a single SASS instruction line.
// Supports both nvdisasm formats:
//
//	/* 0x7f001230 */   LDG.E.CONSTANT.F32 R4, [R2.64] ;   // comment
//	/*0030*/                   LDG.E.CONSTANT.F32 R4, [R2.64];
//
// A trailing encoding comment written by --print-instruction-encoding sets
// [Instruction.Encoding].Lo. The matching high word is on the next line, which
// is not an instruction line; only [Parse] sees it, so an instruction parsed
// through this function alone carries a half encoding.
func ParseInstructionLine(line string) (Instruction, bool) {
	inst, _, ok := parseInstructionLine(line)
	return inst, ok
}

// parseInstructionLine also reports whether the line carried a trailing encoding
// comment. [Parse] needs that to decide whether a following bare encoding word
// belongs to this instruction; the low word alone cannot answer it, because a
// legal instruction with a zero low word would be indistinguishable from a line
// that carried no encoding at all.
func parseInstructionLine(line string) (Instruction, bool, bool) {
	trimmed := strings.TrimSpace(line)
	closeIdx := findAddressCommentEnd(trimmed)
	if closeIdx == -1 {
		return Instruction{}, false, false
	}

	addrStr := strings.TrimSpace(trimmed[2:closeIdx])
	addrStr = strings.TrimPrefix(addrStr, "0x")
	addr, err := strconv.ParseUint(addrStr, 16, 64)
	if err != nil {
		return Instruction{}, false, false
	}

	rest := strings.TrimSpace(trimmed[closeIdx+2:])
	var comment string
	if commentIdx := strings.Index(rest, "//"); commentIdx != -1 {
		comment = strings.TrimSpace(rest[commentIdx+2:])
		rest = strings.TrimSpace(rest[:commentIdx])
	}
	var encoding Encoding
	var sawEncoding bool
	if encodingIdx := strings.Index(rest, "/*"); encodingIdx != -1 {
		trailing := strings.TrimSpace(rest[encodingIdx:])
		if word, ok := encodingWord(trailing); ok {
			encoding.Lo = word
			sawEncoding = true
		} else if comment == "" {
			comment = trailing
		} else {
			comment = trailing + " " + comment
		}
		rest = strings.TrimSpace(rest[:encodingIdx])
	}
	rest = strings.TrimSuffix(rest, ";")
	rest = strings.TrimSpace(rest)

	var predicate string
	if strings.HasPrefix(rest, "@") {
		fields := strings.Fields(rest)
		if len(fields) > 1 {
			predicate = fields[0]
			rest = strings.TrimSpace(strings.TrimPrefix(rest, predicate))
		}
	}

	fields := strings.Fields(rest)
	if len(fields) == 0 {
		return Instruction{}, false, false
	}

	opcode := fields[0]
	// nvdisasm also prefixes data directives with byte-offset comments.
	if strings.HasPrefix(opcode, ".") {
		return Instruction{}, false, false
	}
	operands := strings.TrimSpace(strings.TrimPrefix(rest, opcode))

	return Instruction{
		Address:   addr,
		Predicate: predicate,
		Opcode:    opcode,
		Operands:  operands,
		Encoding:  encoding,
		Comment:   comment,
		Raw:       line,
	}, sawEncoding, true
}

// encodingWord reports the 64-bit word in a bare /* 0x... */ encoding comment.
// nvdisasm writes exactly sixteen hexadecimal digits there, which distinguishes
// an encoding from the shorter instruction-offset comment that opens each line.
func encodingWord(s string) (uint64, bool) {
	s, ok := strings.CutPrefix(s, "/*")
	if !ok {
		return 0, false
	}
	s, ok = strings.CutSuffix(strings.TrimSpace(s), "*/")
	if !ok {
		return 0, false
	}
	s, ok = strings.CutPrefix(strings.TrimSpace(s), "0x")
	if !ok || len(s) != 16 {
		return 0, false
	}
	word, err := strconv.ParseUint(s, 16, 64)
	if err != nil {
		return 0, false
	}
	return word, true
}

// disassemblyTarget extracts only an explicit bounded architecture token.
// Target failures leave ordinary instruction parsing usable; consumers that
// need target identity must reject an unavailable Disassembly.Target.
func disassemblyTarget(line string) (target string, declared bool) {
	var rest string
	if line == ".target" || line == "code for" {
		return "", true
	} else if strings.HasPrefix(line, ".target ") || strings.HasPrefix(line, ".target\t") {
		rest = line[len(".target"):]
	} else if strings.HasPrefix(line, "code for ") {
		rest = line[len("code for "):]
	} else {
		return "", false
	}
	rest = strings.TrimLeft(rest, " \t")
	end := 0
	for end < len(rest) && rest[end] != ' ' && rest[end] != '\t' && rest[end] != ',' {
		if end >= 128 {
			return "", true
		}
		c := rest[end]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
			return "", true
		}
		end++
	}
	token := rest[:end]
	if !strings.HasPrefix(token, "sm_") || len(token) <= len("sm_") {
		return "", true
	}
	return strings.Clone(token), true
}
