package cudaptx

import (
	"bufio"
	"io"
	"strings"
)

// Module represents a parsed PTX module containing version information and kernels.
type Module struct {
	Version     string
	Target      string
	AddressSize int
	Kernels     []*Kernel
	kernelMap   map[string]*Kernel
}

// Kernel represents a parsed .entry or .func definition in PTX.
type Kernel struct {
	Name         string
	StartLine    int
	EndLine      int
	Instructions []Instruction
}

// Instruction represents a single PTX instruction line with its 1-based line number.
type Instruction struct {
	Line     int
	Opcode   string // Full opcode token, e.g. "fma.rn.f32", "st.global.f32", "shfl.sync.down.b32"
	Mnemonic string // Base mnemonic, e.g. "fma", "st", "shfl", "ld"
	Raw      string
}

// Kernel returns the kernel by name, or nil if not found.
func (m *Module) Kernel(name string) *Kernel {
	if m.kernelMap == nil {
		m.kernelMap = make(map[string]*Kernel, len(m.Kernels))
		for _, k := range m.Kernels {
			m.kernelMap[k.Name] = k
		}
	}
	return m.kernelMap[name]
}

// FirstOpcodeLine returns the 1-based line number of the first instruction whose opcode
// contains substr within the given kernel, or 0 if not found.
func (m *Module) FirstOpcodeLine(kernelName, substr string) int {
	k := m.Kernel(kernelName)
	if k == nil {
		return 0
	}
	for _, inst := range k.Instructions {
		if strings.Contains(inst.Opcode, substr) || strings.Contains(inst.Raw, substr) {
			return inst.Line
		}
	}
	return 0
}

// Parse reads PTX assembly from r and returns a parsed Module.
func Parse(r io.Reader) (*Module, error) {
	scanner := bufio.NewScanner(r)
	mod := &Module{
		kernelMap: make(map[string]*Kernel),
	}

	var curKernel *Kernel
	lineNum := 0
	inBraces := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "//") || trimmed == "" {
			continue
		}

		if strings.HasPrefix(trimmed, ".version ") {
			mod.Version = strings.TrimSpace(strings.TrimPrefix(trimmed, ".version "))
			continue
		}
		if strings.HasPrefix(trimmed, ".target ") {
			mod.Target = strings.TrimSpace(strings.TrimPrefix(trimmed, ".target "))
			continue
		}
		if strings.HasPrefix(trimmed, ".address_size ") {
			if strings.Contains(trimmed, "64") {
				mod.AddressSize = 64
			} else if strings.Contains(trimmed, "32") {
				mod.AddressSize = 32
			}
			continue
		}

		// Detect entry/func definition
		if strings.Contains(trimmed, ".entry ") || strings.Contains(trimmed, ".entry\t") ||
			strings.Contains(trimmed, ".func ") || strings.Contains(trimmed, ".func\t") {
			kName := extractKernelName(trimmed)
			if kName != "" {
				curKernel = &Kernel{
					Name:      kName,
					StartLine: lineNum,
				}
				mod.Kernels = append(mod.Kernels, curKernel)
				mod.kernelMap[kName] = curKernel
			}
		}

		if strings.Contains(trimmed, "{") {
			inBraces++
		}
		if strings.Contains(trimmed, "}") {
			inBraces--
			if inBraces <= 0 && curKernel != nil {
				curKernel.EndLine = lineNum
				curKernel = nil
				inBraces = 0
			}
		}

		if curKernel != nil && inBraces > 0 {
			if inst, ok := parseInstructionLine(lineNum, line); ok {
				curKernel.Instructions = append(curKernel.Instructions, inst)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return mod, nil
}

// ParseString parses PTX assembly directly from a string.
func ParseString(ptx string) (*Module, error) {
	return Parse(strings.NewReader(ptx))
}

func extractKernelName(line string) string {
	parts := strings.Fields(line)
	for i, p := range parts {
		if (p == ".entry" || p == ".func") && i+1 < len(parts) {
			namePart := parts[i+1]
			if idx := strings.IndexAny(namePart, "(;{\t "); idx != -1 {
				namePart = namePart[:idx]
			}
			return strings.TrimSpace(namePart)
		}
	}
	return ""
}

func parseInstructionLine(lineNum int, line string) (Instruction, bool) {
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, ".") ||
		strings.HasPrefix(trimmed, "$") || strings.HasSuffix(trimmed, ":") ||
		trimmed == "{" || trimmed == "}" || trimmed == "" {
		return Instruction{}, false
	}

	// Remove predicate modifier if present (e.g., "@%p1 bra ...")
	body := trimmed
	if strings.HasPrefix(body, "@") {
		fields := strings.Fields(body)
		if len(fields) > 1 {
			body = strings.Join(fields[1:], " ")
		}
	}

	fields := strings.Fields(body)
	if len(fields) == 0 {
		return Instruction{}, false
	}

	opcode := strings.TrimSuffix(fields[0], ";")
	mnemonic := opcode
	if idx := strings.Index(opcode, "."); idx != -1 {
		mnemonic = opcode[:idx]
	}

	return Instruction{
		Line:     lineNum,
		Opcode:   opcode,
		Mnemonic: mnemonic,
		Raw:      line,
	}, true
}
