package cudasass

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os/exec"
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
	Address   uint64 // Hardware PC address (e.g. 0x7f001230)
	Predicate string // Instruction predicate, e.g. "@P0", "@!PT"
	Opcode    string // Mnemonic and modifiers, e.g. "FFMA", "LDG.E.CONSTANT.F32", "STS.32"
	Operands  string // Register and memory operands, e.g. "R0, R4, R5, R0"
	Comment   string // Optional disassembly comment
	Raw       string // Full raw disassembly line
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

// Parse reads SASS disassembly text from r and returns a Disassembly.
func Parse(r io.Reader) (*Disassembly, error) {
	scanner := bufio.NewScanner(r)
	d := &Disassembly{
		funcMap: make(map[string]*Function),
	}

	var curFn *Function

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "" {
			continue
		}

		// Detect function header (e.g. ".text._Z10LinearGEMV...", "Function : LinearGEMV", "--- Kernel: LinearGEMV ---")
		if fnName := extractFunctionName(trimmed); fnName != "" {
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
			if curFn == nil {
				curFn = &Function{
					Name: "default",
				}
				d.Functions = append(d.Functions, curFn)
				d.funcMap["default"] = curFn
			}
			if curFn.BaseAddress == 0 && inst.Address != 0 {
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
	nvdisasmPath, err := exec.LookPath("nvdisasm")
	if err != nil {
		return nil, fmt.Errorf("nvdisasm not found in PATH: %w", err)
	}

	cmd := exec.Command(nvdisasmPath, "-")
	cmd.Stdin = bytes.NewReader(cubin)
	var outBuf bytes.Buffer
	var errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("nvdisasm failed: %w (stderr: %s)", err, errBuf.String())
	}

	return Parse(&outBuf)
}

func extractFunctionName(line string) string {
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
//   /* 0x7f001230 */   LDG.E.CONSTANT.F32 R4, [R2.64] ;   // comment
//   /*0030*/                   LDG.E.CONSTANT.F32 R4, [R2.64];
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
