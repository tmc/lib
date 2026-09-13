package cudacapture

import (
	"bytes"
	"debug/elf"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Code represents decoded CUDA kernel instructions, source mappings, and sections.
type Code struct {
	Kernels    []*Kernel
	SourceFile string
	PTXFile    string
	SASSFile   string
}

// Kernel represents a disassembled GPU kernel function with its instruction sequence.
type Kernel struct {
	Name         string
	Offset       uint64
	Size         uint64
	SourceLine   int
	PTXLine      int
	Instructions []Instruction
}

// Instruction represents a single decoded hardware SASS instruction within a Kernel.
type Instruction struct {
	Offset     uint64 // Offset within the kernel's .text section.
	Text       string
	SASSLine   int
	SourceLine int
	PTXLine    int
	InlinedAt  []cudaptx.Location
}

var sourceLocation = regexp.MustCompile(`//## File "([^"]+)", line ([0-9]+)`)
var textSection = regexp.MustCompile(`^\s*\.section\s+\.text\.([^, \t]+)`)

// LoadCode extracts and parses the CUBIN, PTX, SASS, and CUDA source files
// identified by c.Code in the Capture.
func LoadCode(c *Capture) (*Code, error) {
	if c == nil {
		return nil, errors.New("cudacapture: nil capture")
	}
	cubinBytes, ok := c.Files[c.Code.CUBIN]
	if !ok {
		if reason := c.CodeUnavailable["cubin"]; reason != "" {
			return nil, fmt.Errorf("cudacapture: cubin unavailable: %s", reason)
		}
		return nil, fmt.Errorf("cudacapture: missing CUBIN file %q", c.Code.CUBIN)
	}
	ptxBytes, ok := c.Files[c.Code.PTX]
	if !ok {
		if reason := c.CodeUnavailable["ptx"]; reason != "" {
			return nil, fmt.Errorf("cudacapture: ptx unavailable: %s", reason)
		}
		return nil, fmt.Errorf("cudacapture: missing PTX file %q", c.Code.PTX)
	}
	sassBytes, ok := c.Files[c.Code.SASS]
	if !ok {
		if reason := c.CodeUnavailable["sass"]; reason != "" {
			return nil, fmt.Errorf("cudacapture: sass unavailable: %s", reason)
		}
		return nil, fmt.Errorf("cudacapture: missing SASS file %q", c.Code.SASS)
	}
	sourceBytes, ok := c.Files[c.Code.Source]
	if !ok {
		if reason := c.CodeUnavailable["source"]; reason != "" {
			return nil, fmt.Errorf("cudacapture: source unavailable: %s", reason)
		}
		return nil, fmt.Errorf("cudacapture: missing source file %q", c.Code.Source)
	}

	code, err := parseCode(cubinBytes, string(ptxBytes), string(sassBytes), string(sourceBytes), c.Code.Source)
	if err != nil {
		return nil, err
	}
	code.SourceFile = c.Code.Source
	code.PTXFile = c.Code.PTX
	code.SASSFile = c.Code.SASS
	return code, nil
}

func parseCode(cubin []byte, ptx, sass, source, sourceFilename string) (*Code, error) {
	e, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read cubin: %w", err)
	}
	defer e.Close()
	if e.Machine != elf.EM_CUDA {
		return nil, fmt.Errorf("cudacapture: cubin architecture is %s, want CUDA", e.Machine)
	}
	p, err := cudaptx.ParseString(ptx)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: parse ptx: %w", err)
	}
	d, err := cudasass.ParseString(sass)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: parse sass: %w", err)
	}

	lineNumbers := make(map[string]int)
	cuLines := make(map[int]int)
	cuLine := 0
	currentKernel := ""
	sourceBase := filepath.Base(sourceFilename)

	for i, line := range strings.Split(sass, "\n") {
		if m := textSection.FindStringSubmatch(line); m != nil {
			currentKernel, cuLine = m[1], 0
		}
		if m := sourceLocation.FindStringSubmatch(line); m != nil {
			cuLine = 0
			if filepath.Base(m[1]) == sourceBase {
				cuLine, _ = strconv.Atoi(m[2])
			}
		}
		lineNumbers[currentKernel+"\x00"+line] = i + 1
		cuLines[i+1] = cuLine
	}

	var code Code
	for _, fn := range d.Functions {
		section := e.Section(".text." + fn.Name)
		pk := p.Kernel(fn.Name)
		if section == nil || pk == nil {
			return nil, fmt.Errorf("cudacapture: %s: missing CUBIN section or PTX entry", fn.Name)
		}
		if section.Size == 0 || section.Offset > uint64(len(cubin)) || section.Size > uint64(len(cubin))-section.Offset {
			return nil, fmt.Errorf("cudacapture: %s: invalid CUBIN section", fn.Name)
		}
		k := &Kernel{Name: fn.Name, Offset: section.Offset, Size: section.Size, PTXLine: pk.StartLine}
		if len(pk.Instructions) > 0 {
			k.PTXLine = pk.Instructions[0].Line
			for _, inst := range pk.Instructions {
				if isArithmeticPTXOpcode(inst.Opcode) {
					k.PTXLine = inst.Line
					break
				}
			}
		}
		for i, line := range strings.Split(source, "\n") {
			if strings.Contains(line, "void "+fn.Name+"(") {
				k.SourceLine = i + 1
				break
			}
		}
		if k.SourceLine == 0 {
			return nil, fmt.Errorf("cudacapture: %s: missing CUDA definition", fn.Name)
		}

		ptxByLine := make(map[int][]cudaptx.Instruction)
		for _, pInst := range pk.Instructions {
			if pInst.Loc.Line > 0 {
				ptxByLine[pInst.Loc.Line] = append(ptxByLine[pInst.Loc.Line], pInst)
			}
		}

		for _, inst := range fn.Instructions {
			if inst.Address >= section.Size || inst.Address%16 != 0 {
				return nil, fmt.Errorf("cudacapture: %s: invalid SASS offset %#x", fn.Name, inst.Address)
			}
			if len(k.Instructions) > 0 && inst.Address <= k.Instructions[len(k.Instructions)-1].Offset {
				return nil, fmt.Errorf("cudacapture: %s: unordered SASS offsets", fn.Name)
			}
			line := lineNumbers[fn.Name+"\x00"+inst.Raw]
			if line == 0 {
				return nil, fmt.Errorf("cudacapture: %s: missing SASS source line", fn.Name)
			}
			cuLine := inst.SourceLine
			if inst.SourceFile != "" && filepath.Base(inst.SourceFile) != sourceBase {
				cuLine = 0
			}
			if cuLine == 0 {
				cuLine = cuLines[line]
			}

			ptxLine := k.PTXLine
			var inlinedAt []cudaptx.Location
			if candidates := ptxByLine[cuLine]; len(candidates) > 0 {
				matched := candidates[0]
				for _, c := range candidates {
					if matchOpcodeAffinity(inst.Opcode, c.Mnemonic) {
						matched = c
						break
					}
				}
				ptxLine = matched.Line
				inlinedAt = matched.InlinedAt
			} else if strings.HasPrefix(strings.ToUpper(inst.Opcode), "RET") || strings.HasPrefix(strings.ToUpper(inst.Opcode), "EXIT") {
				for j := len(pk.Instructions) - 1; j >= 0; j-- {
					if pk.Instructions[j].Mnemonic == "ret" || pk.Instructions[j].Mnemonic == "exit" {
						ptxLine = pk.Instructions[j].Line
						inlinedAt = pk.Instructions[j].InlinedAt
						break
					}
				}
			}

			text := strings.TrimSpace(strings.TrimSpace(inst.Predicate+" "+inst.Opcode)+" "+inst.Operands) + ";"
			k.Instructions = append(k.Instructions, Instruction{
				Offset:     inst.Address,
				Text:       text,
				SASSLine:   line,
				SourceLine: cuLine,
				PTXLine:    ptxLine,
				InlinedAt:  inlinedAt,
			})
		}
		if len(k.Instructions) == 0 || k.Instructions[0].Offset != 0 {
			return nil, fmt.Errorf("cudacapture: %s: missing SASS entry instruction", fn.Name)
		}
		code.Kernels = append(code.Kernels, k)
	}

	if len(code.Kernels) == 0 {
		return nil, errors.New("cudacapture: no disassembled kernels")
	}

	slices.SortFunc(code.Kernels, func(a, b *Kernel) int {
		if a.Offset < b.Offset {
			return -1
		}
		if a.Offset > b.Offset {
			return 1
		}
		return 0
	})
	for i := 1; i < len(code.Kernels); i++ {
		if code.Kernels[i].Offset < code.Kernels[i-1].Offset+code.Kernels[i-1].Size {
			return nil, errors.New("cudacapture: overlapping CUBIN sections")
		}
	}
	return &code, nil
}

// Kernel returns the kernel by name, or nil if not present.
func (c *Code) Kernel(name string) *Kernel {
	for _, k := range c.Kernels {
		if k.Name == name {
			return k
		}
	}
	return nil
}

// KernelByOffset returns the kernel containing the given CUBIN file offset.
func (c *Code) KernelByOffset(offset uint64) *Kernel {
	if c == nil {
		return nil
	}
	for _, k := range c.Kernels {
		if offset >= k.Offset && offset < k.Offset+k.Size {
			return k
		}
	}
	return nil
}

// Instruction returns the instruction at the specified SASS offset within the kernel,
// or nil if not found. Instructions must be strictly ascending by Offset for
// binary search.
func (k *Kernel) Instruction(offset uint64) *Instruction {
	if k == nil {
		return nil
	}
	idx, ok := slices.BinarySearchFunc(k.Instructions, offset, func(inst Instruction, target uint64) int {
		if inst.Offset < target {
			return -1
		}
		if inst.Offset > target {
			return 1
		}
		return 0
	})
	if ok {
		return &k.Instructions[idx]
	}
	return nil
}

func matchOpcodeAffinity(sassOp, ptxMnemonic string) bool {
	s := strings.ToUpper(sassOp)
	p := strings.ToLower(ptxMnemonic)
	switch {
	case strings.HasPrefix(s, "LD") && p == "ld":
		return true
	case strings.HasPrefix(s, "ST") && p == "st":
		return true
	case (strings.Contains(s, "FMA") || strings.Contains(s, "MAD") || strings.Contains(s, "MUL")) && (p == "fma" || p == "mad" || p == "mul"):
		return true
	case (strings.Contains(s, "ADD") || strings.Contains(s, "SUB")) && (p == "add" || p == "sub"):
		return true
	case strings.Contains(s, "DP4A") && p == "dp4a":
		return true
	case strings.Contains(s, "SHFL") && p == "shfl":
		return true
	case strings.Contains(s, "BRA") && p == "bra":
		return true
	case (strings.Contains(s, "RET") || strings.Contains(s, "EXIT")) && (p == "ret" || p == "exit"):
		return true
	case strings.Contains(s, "BAR") && p == "bar":
		return true
	case (strings.Contains(s, "SETP") || strings.Contains(s, "SEL")) && (p == "setp" || p == "selp"):
		return true
	case strings.Contains(s, "MOV") && (p == "mov" || p == "cvt" || p == "ld"):
		return true
	default:
		return false
	}
}

func isArithmeticPTXOpcode(op string) bool {
	lower := strings.ToLower(op)
	for _, prefix := range []string{
		"add", "sub", "mul", "fma", "mad", "dp4a", "shfl", "rsqrt", "div", "cvt", "ld", "st",
	} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}
