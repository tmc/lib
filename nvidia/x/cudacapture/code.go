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
	Name   string
	Offset uint64
	Size   uint64

	// SourceFile is the base name of the CUDA file that defines this kernel,
	// and SourceLine the line within it. For a module compiled from several
	// files they differ from kernel to kernel, so a reader must take the file
	// from the kernel rather than from [Code.SourceFile], which names only the
	// unit's primary file.
	SourceFile string
	SourceLine int

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

var textSection = regexp.MustCompile(`^\s*\.section\s+\.text\.([^, \t]+)`)

// LoadCode extracts and parses the CUBIN, PTX, SASS, and CUDA source files
// identified by c.Code in the Capture.
//
// The CUDA files are those c.Code.Sources names, or c.Code.Source alone when it
// names none. Every kernel in the disassembly must be defined in exactly one of
// them.
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
	sources, err := loadSources(c)
	if err != nil {
		return nil, err
	}

	code, err := parseCode(cubinBytes, string(ptxBytes), string(sassBytes), sources)
	if code == nil {
		return nil, err
	}
	code.SourceFile = c.Code.Source
	code.PTXFile = c.Code.PTX
	code.SASSFile = c.Code.SASS
	// err is a *SourceFusionError here and code is the intact kernel tier; see
	// that type for what a caller may still read from it.
	return code, err
}

// loadSources reads the CUDA files of the capture's translation unit, keyed by
// the base name the PTX and the disassembly refer to them by. Those tiers carry
// the path the compiler recorded, which is a build-host path, so the base name
// is the only part that can be matched against what a capture bundles.
func loadSources(c *Capture) (map[string]string, error) {
	names := c.Code.Sources
	if len(names) == 0 {
		names = []string{c.Code.Source}
	}
	sources := make(map[string]string, len(names))
	origin := make(map[string]string, len(names))
	for _, name := range names {
		b, ok := c.Files[name]
		if !ok {
			if reason := c.CodeUnavailable["source"]; reason != "" {
				return nil, fmt.Errorf("cudacapture: source unavailable: %s", reason)
			}
			return nil, fmt.Errorf("cudacapture: missing source file %q", name)
		}
		base := filepath.Base(name)
		if prev, dup := origin[base]; dup {
			// Both would be addressed as the same file by every line
			// directive, so one would silently shadow the other and half the
			// unit's instructions would be attributed to lines of the wrong
			// file. There is no correct choice to make here.
			return nil, fmt.Errorf("cudacapture: sources %q and %q share the base name %q", prev, name, base)
		}
		origin[base] = name
		sources[base] = string(b)
	}
	return sources, nil
}

// parseCode decodes one module. sources holds the CUDA files of its translation
// unit keyed by base name; see [loadSources].
func parseCode(cubin []byte, ptx, sass string, sources map[string]string) (*Code, error) {
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

	for i, line := range strings.Split(sass, "\n") {
		// cudasass scans with a bufio.Scanner, which drops a trailing "\r".
		// Without this the kernel name captured below would keep one and
		// match no function.
		line = strings.TrimSuffix(line, "\r")
		if m := textSection.FindStringSubmatch(line); m != nil {
			currentKernel, cur = m[1], cuLoc{}
		}
		// The PTX cross-reference tier names the same instructions in its own
		// line namespace, with numbers that run past any real source's length.
		// Its exclusion is by classification rather than by comparing against
		// a literal here, so this call site cannot drift from the others.
		if dir, ok := cudasass.ParseSourceDirective(line); ok && dir.Kind == cudasass.CUDASource {
			cur = cuLoc{file: filepath.Base(dir.File), line: dir.Line}
		}
		if inst, ok := cudasass.ParseInstructionLine(line); ok {
			lineNumbers[instKey{currentKernel, inst.Address}] = i + 1
		}
		cuLines[i+1] = cur
	}

	// A fusion failure is not a reason to withhold the kernel tier: kernel
	// names, offsets, sizes and instruction text are read from the CUBIN and
	// the disassembly, neither of which the PTX participates in. Refusing here
	// made the tool unusable on profiles that pprof itself reads correctly.
	// Carry on without source attribution and report the failure to the caller.
	fused := verifyLineBinding(p, d, sources)

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
		srcFile, srcLine, err := definitionIn(sources, fn.Name)
		if err != nil {
			return nil, fmt.Errorf("cudacapture: %s: %w", fn.Name, err)
		}
		k.SourceFile, k.SourceLine = srcFile, srcLine

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
			var loc cuLoc
			if fused == nil {
				loc = cuLoc{line: inst.SourceLine}
				if inst.SourceFile != "" {
					loc.file = filepath.Base(inst.SourceFile)
				}
				if loc.line == 0 {
					loc = cuLines[line]
				}
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
			if fused == nil && inst.InlinedLine > 0 {
				inlinedAt = []cudaptx.Location{{
					File: filepath.Base(inst.InlinedFile),
					Line: inst.InlinedLine,
				}}
			}
			if fused != nil {
				// Both branches below read a binding the two tiers do not
				// share; either would report a guess in the shape of a
				// measurement.
			} else if inst.PTXLine > 0 {
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
	if fused != nil {
		return &code, fused
	}
	return &code, nil
}

// definitionIn finds the one file of the translation unit that defines sym, and
// the line within it.
//
// Every file is searched even after a match, rather than returning the first:
// two files defining the same kernel is a real condition the unit cannot be
// attributed under, and stopping early would report whichever happened to sort
// first as though it were the only one. An ambiguity inside a single file is a
// refusal for the same reason and is not resolved by another file's answer.
func definitionIn(sources map[string]string, sym string) (file string, line int, err error) {
	names := make([]string, 0, len(sources))
	for name := range sources {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		n, err := cudalowering.DefinitionLine(sources[name], sym)
		if errors.Is(err, cudalowering.ErrNoDefinition) {
			continue
		}
		if err != nil {
			return "", 0, fmt.Errorf("%s: %w", name, err)
		}
		if file != "" {
			return "", 0, fmt.Errorf("%w of %s: %s:%d and %s:%d",
				cudalowering.ErrAmbiguousDefinition, sym, file, line, name, n)
		}
		file, line = name, n
	}
	if file == "" {
		return "", 0, fmt.Errorf("%w of %s in %s", cudalowering.ErrNoDefinition, sym, strings.Join(names, ", "))
	}
	return file, line, nil
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

// SourceFusionError reports that a capture's PTX and SASS are not from one
// compilation of the same source, so no instruction in it can be attributed to
// a source line.
//
// LoadCode returns it alongside a usable Code. Kernel names, offsets, sizes and
// instruction text come from the CUBIN and the disassembly alone and do not
// depend on the two tiers agreeing, so a ranking over kernels remains correct;
// what the failure invalidates is the source and PTX attribution, and every
// Instruction in that Code has SourceFile, SourceLine, PTXLine and InlinedAt
// left at their zero values for exactly that reason. A caller that needs source
// attribution must treat this as fatal. A caller that does not must say in its
// output that attribution is missing, since a zero SourceLine is otherwise
// indistinguishable from an instruction the compiler never located.
type SourceFusionError struct {
	// SourceFusionFile is the first disagreeing file in Files, by name. It is
	// embedded so that Source, Unclaimed and Claimed read directly off the
	// error, which is the whole truth for a single-source capture.
	SourceFusionFile

	// Files is every file of the translation unit the two tiers disagree
	// about, sorted by name and never empty. Files[0] is the embedded one.
	Files []SourceFusionFile
}

// A SourceFusionFile is one source file's share of a fusion failure.
type SourceFusionFile struct {
	Source    string // base name of the source file the two tiers disagree about
	Unclaimed []int  // lines of it the SASS names and the PTX never does, sorted
	Claimed   int    // how many lines of it the PTX does name
}

func (e *SourceFusionError) Error() string {
	msg := fmt.Sprintf("cudacapture: PTX and SASS are not from one compilation of %s: the SASS attributes instructions to %d source line(s) the PTX never names (%s); the PTX names %d line(s) of that file",
		e.Source, len(e.Unclaimed), sampleLines(e.Unclaimed, maxReportedLines), e.Claimed)
	if len(e.Files) > 1 {
		msg += fmt.Sprintf(" (and %d more file(s) of the same unit disagree: %s)",
			len(e.Files)-1, strings.Join(otherSourceNames(e.Files), ", "))
	}
	return msg
}

func otherSourceNames(files []SourceFusionFile) []string {
	names := make([]string, 0, len(files)-1)
	for _, f := range files[1:] {
		names = append(names, f.Source)
	}
	return names
}

// verifyLineBinding checks, for each file of the translation unit the capture
// supplies, that the PTX names every line of it the SASS attributes an
// instruction to.
//
// The check is per file and keyed on (file, line) because two files of one unit
// both have a line 12, and pooling their numbers lets one file's PTX coverage
// vouch for another file's SASS lines. A file the SASS names but the capture
// does not supply is not checked at all: the capture holds no line table to
// check it against, so its lines are unknowable here rather than unclaimed, and
// counting them would fail every multi-file unit that bundles a subset of its
// headers.
func verifyLineBinding(p *cudaptx.Module, d *cudasass.Disassembly, sources map[string]string) *SourceFusionError {
	claimed := make(map[string]map[int]bool, len(sources))
	for base := range sources {
		claimed[base] = make(map[int]bool)
	}
	note := func(file string, line int) {
		if line <= 0 {
			return
		}
		if lines, ok := claimed[filepath.Base(file)]; ok {
			lines[line] = true
		}
	}
	for _, k := range p.Kernels {
		for _, inst := range k.Instructions {
			note(inst.Loc.File, inst.Loc.Line)
			for _, l := range inst.InlinedAt {
				note(l.File, l.Line)
			}
		}
	}

	unclaimed := make(map[string][]int, len(sources))
	seen := make(map[string]map[int]bool, len(sources))
	for base := range sources {
		seen[base] = make(map[int]bool)
	}
	for _, fn := range d.Functions {
		for _, inst := range fn.Instructions {
			for _, l := range []struct {
				file string
				line int
			}{{inst.SourceFile, inst.SourceLine}, {inst.InlinedFile, inst.InlinedLine}} {
				if l.line <= 0 {
					continue
				}
				base := filepath.Base(l.file)
				lines, supplied := claimed[base]
				if !supplied || lines[l.line] || seen[base][l.line] {
					continue
				}
				seen[base][l.line] = true
				unclaimed[base] = append(unclaimed[base], l.line)
			}
		}
	}

	var files []SourceFusionFile
	for base, lines := range unclaimed {
		sort.Ints(lines)
		files = append(files, SourceFusionFile{Source: base, Unclaimed: lines, Claimed: len(claimed[base])})
	}
	if len(files) == 0 {
		return nil
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Source < files[j].Source })
	return &SourceFusionError{SourceFusionFile: files[0], Files: files}
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
