package cudasass

import (
	"bufio"
	"context"
	"io"
	"regexp"
	"strconv"
	"strings"
)

// Disassembly represents a parsed collection of disassembled SASS functions.
type Disassembly struct {
	Functions []*Function
	funcMap   map[string]*Function
}

// Function represents a disassembled GPU kernel with its instruction sequence.
type Function struct {
	Name         string
	BaseAddress  uint64
	Instructions []Instruction
}

// Instruction represents a single decoded hardware SASS instruction.
type Instruction struct {
	Address     uint64 // Hardware PC address (e.g. 0x7f001230)
	Predicate   string // Instruction predicate, e.g. "@P0", "@!PT"
	Opcode      string // Mnemonic and modifiers, e.g. "FFMA", "LDG.E.CONSTANT.F32", "STS.32"
	Operands    string // Register and memory operands, e.g. "R0, R4, R5, R0"
	SourceFile  string // Source file name from line-info comments, e.g. "kernels.cu"
	SourceLine  int    // Source line number from line-info comments
	InlinedFile string // Inlined call site file name, if inlined
	InlinedLine int    // Inlined call site line number, if inlined
	PTXLine     int    // PTX line number from .nv_debug_ptx_txt comments, if present
	Comment     string // Optional disassembly comment
	Raw         string // Full raw disassembly line
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

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "" {
			continue
		}

		if m := sourceLocationPattern.FindStringSubmatch(trimmed); m != nil {
			if m[1] == ".nv_debug_ptx_txt" {
				if lineNum, err := strconv.Atoi(m[2]); err == nil {
					curPTXLine = lineNum
				}
				continue
			}
			curSourceFile = m[1]
			if lineNum, err := strconv.Atoi(m[2]); err == nil {
				curSourceLine = lineNum
			}
			curInlinedFile = ""
			curInlinedLine = 0
			if mInline := inlineLocationPattern.FindStringSubmatch(trimmed); mInline != nil {
				curInlinedFile = mInline[1]
				if inlinedNum, err := strconv.Atoi(mInline[2]); err == nil {
					curInlinedLine = inlinedNum
				}
			}
			continue
		}

		// Detect function header (e.g. ".text._Z10LinearGEMV...", "Function : LinearGEMV", "--- Kernel: LinearGEMV ---")
		if fnName := extractFunctionName(trimmed); fnName != "" {
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

		// Parse instruction line
		if inst, ok := ParseInstructionLine(line); ok {
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
			curFn.Instructions = append(curFn.Instructions, inst)
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

// ParseInstructionLine parses a single SASS instruction line.
// Supports both nvdisasm formats:
//
//	/* 0x7f001230 */   LDG.E.CONSTANT.F32 R4, [R2.64] ;   // comment
//	/*0030*/                   LDG.E.CONSTANT.F32 R4, [R2.64];
func ParseInstructionLine(line string) (Instruction, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "/*") {
		return Instruction{}, false
	}

	closeIdx := strings.Index(trimmed, "*/")
	if closeIdx == -1 {
		return Instruction{}, false
	}

	addrStr := strings.TrimSpace(trimmed[2:closeIdx])
	addrStr = strings.TrimPrefix(addrStr, "0x")
	addr, err := strconv.ParseUint(addrStr, 16, 64)
	if err != nil {
		return Instruction{}, false
	}

	rest := strings.TrimSpace(trimmed[closeIdx+2:])
	var comment string
	if commentIdx := strings.Index(rest, "//"); commentIdx != -1 {
		comment = strings.TrimSpace(rest[commentIdx+2:])
		rest = strings.TrimSpace(rest[:commentIdx])
	}
	if encodingIdx := strings.Index(rest, "/*"); encodingIdx != -1 {
		encoding := strings.TrimSpace(rest[encodingIdx:])
		if comment == "" {
			comment = encoding
		} else {
			comment = encoding + " " + comment
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
		return Instruction{}, false
	}

	opcode := fields[0]
	// nvdisasm also prefixes data directives with byte-offset comments.
	if strings.HasPrefix(opcode, ".") {
		return Instruction{}, false
	}
	operands := strings.TrimSpace(strings.TrimPrefix(rest, opcode))

	return Instruction{
		Address:   addr,
		Predicate: predicate,
		Opcode:    opcode,
		Operands:  operands,
		Comment:   comment,
		Raw:       line,
	}, true
}
