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
	curSecHasAddr := false
	curSecLastAddr := uint64(0)

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, ".target") {
			fields := strings.Fields(trimmed)
			if len(fields) >= 2 {
				archAlign = targetAlignment(fields[1])
			}
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
			d.sections = append(d.sections, Section{Name: name})
			curSection = len(d.sections) - 1
			curSecHasAddr, curSecLastAddr = false, 0
			if pendingBanner != "" {
				d.sections[curSection].Text = append(d.sections[curSection].Text, pendingBanner...)
				d.sections[curSection].Text = append(d.sections[curSection].Text, '\n')
			}
			pendingBanner = ""
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
			continue
		}

		if name, ok := codeLabel(trimmed); ok {
			pendingLabels = append(pendingLabels, name)
			continue
		}

		// Parse instruction line
		if inst, sawEncoding, ok := parseInstructionLine(line); ok {
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
					Name: secName,
				}
				d.Functions = append(d.Functions, curFn)
				d.funcMap[secName] = curFn
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

func extractFunctionName(line string) string {
	if strings.HasPrefix(line, ".section") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && strings.HasPrefix(fields[1], ".text.") {
			return strings.TrimSuffix(strings.SplitN(strings.TrimPrefix(fields[1], ".text."), ",", 2)[0], ":")
		}
	}
	if strings.HasPrefix(line, "Function : ") {
		return strings.TrimSpace(strings.TrimPrefix(line, "Function : "))
	}
	if strings.HasPrefix(line, "--- Kernel: ") && strings.HasSuffix(line, " ---") {
		return strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "--- Kernel: "), " ---"))
	}
	if strings.HasPrefix(line, ".text.") {
		name := strings.TrimPrefix(line, ".text.")
		if idx := strings.IndexAny(name, ":\t "); idx != -1 {
			name = name[:idx]
		}
		return strings.TrimSpace(name)
	}
	return ""
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
	if !strings.HasPrefix(line, ".type") {
		return "", false
	}
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return "", false
	}
	name, typ, ok := strings.Cut(fields[len(fields)-1], ",")
	if !ok || typ != "@function" || name == "" {
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
