package cudacapture

import (
	"bytes"
	"debug/elf"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudalowering"
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
	Offset   uint64 // Offset within the kernel's .text section.
	Text     string
	SASSLine int

	// SourceFile is the file the disassembler attributed this instruction to,
	// which is not always the capture's main source: an instruction inlined
	// from a header is attributed to that header. It is empty exactly when
	// SourceLine is 0. Readers that assume one file must test it; dropping a
	// header-originated line renders a measured location as no location.
	SourceFile string
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

	// An instruction is located in the disassembly text by (kernel, address).
	// Keying on the instruction's text instead happens to be unique, because
	// nvdisasm prints the offset into the line, but that is a property of the
	// printer rather than a guarantee: the two sides also have to agree on
	// the exact bytes, and they do not on a CRLF disassembly, where cudasass's
	// scanner drops the "\r" this split keeps. The address on both sides comes
	// from cudasass's own line parser, so the two cannot drift apart.
	type instKey struct {
		kernel string
		addr   uint64
	}
	lineNumbers := make(map[instKey]int)
	// cuLines carries the location a "//## File" comment establishes for every
	// SASS line until the next one, which is nvdisasm's own convention. The
	// file travels with the line: an instruction inlined from a header is
	// attributed to that header, and reducing it to a bare line number in the
	// main source would move it to an unrelated statement.
	type cuLoc struct {
		file string
		line int
	}
	cuLines := make(map[int]cuLoc)
	var cur cuLoc
	currentKernel := ""
	sourceBase := filepath.Base(sourceFilename)

	for i, line := range strings.Split(sass, "\n") {
		// cudasass scans with a bufio.Scanner, which drops a trailing "\r".
		// Without this the kernel name captured below would keep one and
		// match no function.
		line = strings.TrimSuffix(line, "\r")
		if m := textSection.FindStringSubmatch(line); m != nil {
			currentKernel, cur = m[1], cuLoc{}
		}
		if m := sourceLocation.FindStringSubmatch(line); m != nil {
			if m[1] == ".nv_debug_ptx_txt" {
				continue
			}
			n, _ := strconv.Atoi(m[2])
			cur = cuLoc{file: filepath.Base(m[1]), line: n}
		}
		if inst, ok := cudasass.ParseInstructionLine(line); ok {
			lineNumbers[instKey{currentKernel, inst.Address}] = i + 1
		}
		cuLines[i+1] = cur
	}

	if err := verifyLineBinding(p, d, sourceBase); err != nil {
		return nil, err
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
		// PTXLine is where the kernel is declared in the PTX text. It used to
		// be moved to the first instruction whose opcode looked "arithmetic",
		// which is a line derived from opcode spelling rather than from the
		// PTX's own structure.
		k := &Kernel{Name: fn.Name, Offset: section.Offset, Size: section.Size, PTXLine: pk.StartLine}
		srcLine, err := cudalowering.DefinitionLine(source, fn.Name)
		if err != nil {
			return nil, fmt.Errorf("cudacapture: %s: %w", fn.Name, err)
		}
		k.SourceLine = srcLine

		// Keyed on (file, line): two files can both have a line 12, and
		// joining on the number alone fuses them.
		ptxByLine := make(map[cuLoc][]cudaptx.Instruction)
		for _, pInst := range pk.Instructions {
			if pInst.Loc.Line > 0 {
				key := cuLoc{file: filepath.Base(pInst.Loc.File), line: pInst.Loc.Line}
				ptxByLine[key] = append(ptxByLine[key], pInst)
			}
		}

		for _, inst := range fn.Instructions {
			if inst.Address >= section.Size || inst.Address%16 != 0 {
				return nil, fmt.Errorf("cudacapture: %s: invalid SASS offset %#x", fn.Name, inst.Address)
			}
			if len(k.Instructions) > 0 && inst.Address <= k.Instructions[len(k.Instructions)-1].Offset {
				return nil, fmt.Errorf("cudacapture: %s: unordered SASS offsets", fn.Name)
			}
			line := lineNumbers[instKey{fn.Name, inst.Address}]
			if line == 0 {
				return nil, fmt.Errorf("cudacapture: %s: missing SASS source line", fn.Name)
			}
			loc := cuLoc{line: inst.SourceLine}
			if inst.SourceFile != "" {
				loc.file = filepath.Base(inst.SourceFile)
			}
			if loc.line == 0 {
				loc = cuLines[line]
			}

			// PTXLine is set only where the PTX instruction is identified.
			// The disassembler's own PTX line is direct evidence; failing
			// that, a source line that exactly one PTX instruction claims
			// identifies it too, because both sides name the same line of the
			// same file. Anything beyond that -- picking among several
			// candidates by opcode text, or walking back to the last "ret" --
			// is a guess, and a guess here is indistinguishable in the output
			// from a measurement. Zero means unknown, which every consumer
			// already tests for.
			var ptxLine int
			var inlinedAt []cudaptx.Location
			if inst.InlinedLine > 0 {
				inlinedAt = []cudaptx.Location{{
					File: filepath.Base(inst.InlinedFile),
					Line: inst.InlinedLine,
				}}
			}
			if inst.PTXLine > 0 {
				ptxLine = inst.PTXLine
			} else if candidates := ptxByLine[loc]; len(candidates) == 1 {
				ptxLine = candidates[0].Line
				if len(inlinedAt) == 0 {
					inlinedAt = candidates[0].InlinedAt
				}
			}

			text := strings.TrimSpace(strings.TrimSpace(inst.Predicate+" "+inst.Opcode)+" "+inst.Operands) + ";"
			k.Instructions = append(k.Instructions, Instruction{
				Offset:     inst.Address,
				Text:       text,
				SASSLine:   line,
				SourceFile: loc.file,
				SourceLine: loc.line,
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

func verifyLineBinding(p *cudaptx.Module, d *cudasass.Disassembly, sourceBase string) error {
	claimed := make(map[int]bool)
	for _, k := range p.Kernels {
		for _, inst := range k.Instructions {
			for _, l := range append([]cudaptx.Location{inst.Loc}, inst.InlinedAt...) {
				if l.Line > 0 && filepath.Base(l.File) == sourceBase {
					claimed[l.Line] = true
				}
			}
		}
	}
	var unclaimed []int
	seen := make(map[int]bool)
	for _, fn := range d.Functions {
		for _, inst := range fn.Instructions {
			for _, l := range []struct {
				file string
				line int
			}{{inst.SourceFile, inst.SourceLine}, {inst.InlinedFile, inst.InlinedLine}} {
				if l.line <= 0 || filepath.Base(l.file) != sourceBase || claimed[l.line] || seen[l.line] {
					continue
				}
				seen[l.line] = true
				unclaimed = append(unclaimed, l.line)
			}
		}
	}
	if len(unclaimed) == 0 {
		return nil
	}
	sort.Ints(unclaimed)
	return fmt.Errorf("cudacapture: PTX and SASS are not from one compilation of %s: the SASS attributes instructions to %d source line(s) the PTX never names (%s); the PTX names %d line(s) of that file",
		sourceBase, len(unclaimed), sampleLines(unclaimed, maxReportedLines), len(claimed))
}

// maxReportedLines bounds how many unclaimed line numbers an error names. A
// mismatch of this kind runs to tens of thousands of lines -- one observed
// failure listed 67741 of them, roughly 400 KB of error text -- and the count
// plus a sample identifies the problem as well as the set does while staying
// readable in a terminal and a log.
const maxReportedLines = 16

// sampleLines renders at most n of lines, noting how many it withheld. The
// count is already in the caller's message, so this says only what it dropped.
func sampleLines(lines []int, n int) string {
	if len(lines) <= n {
		return fmt.Sprint(lines)
	}
	return fmt.Sprintf("%v and %d more", lines[:n], len(lines)-n)
}
