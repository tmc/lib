package cudalineinfo

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// An Attribution joins every tier of one compiled unit back to its sites.
//
// CUDA, PTXLines, and SASS are keyed by site and hold what each tier
// attributes to it: the generated lines lowered from it, the lines of the PTX
// text holding a .loc directive naming it, and the SASS instructions naming
// it. A site absent from a map produced nothing at that tier.
type Attribution struct {
	Unit        Unit
	PTX         string
	Disassembly *cudasass.Disassembly

	// Module is the compiled module when the attribution came from
	// [Compile], so the same binary that was attributed can be run. It is nil
	// for an attribution built from artifacts.
	Module *nvrtcx.CompiledModule

	CUDA     map[Site][]int
	PTXLines map[Site][]int
	SASS     map[Site][]cudasass.Instruction

	// Instructions counts every instruction in the disassembly, in every
	// function. Attributed counts those naming a site of the program.
	// Remainder classifies the rest.
	Instructions, Attributed int
	Remainder                Remainder
}

// A Remainder classifies the instructions that name no site of the program.
// "37 of 168 attributed" invites reading 131 as unexplained; the remainder
// says where they are instead.
type Remainder struct {
	// Generated counts instructions naming the unit's own generated lines:
	// the grid-stride loop, parameter loads, and other scaffolding no
	// program line produced.
	Generated int
	// Headers counts instructions naming a header, which is library code
	// inlined into the kernel.
	Headers int
	// Unlocated counts instructions carrying no line information at all.
	Unlocated int
	// Subroutines counts, by symbol, the instructions of every function
	// that is not one of the unit's kernels, such as the compiler's
	// division slowpath, whether it has a section of its own or follows a
	// kernel in the kernel's. Line information inside them describes the
	// subroutine, so they are counted here whatever file they name.
	Subroutines map[string]int
}

// Total is the number of instructions the remainder accounts for.
func (r Remainder) Total() int {
	n := r.Generated + r.Headers + r.Unlocated
	for _, c := range r.Subroutines {
		n += c
	}
	return n
}

// Attribute joins the tiers of a unit from its PTX and disassembly. Either may
// be absent: an empty ptx leaves PTXLines empty, and a nil dis leaves SASS
// empty.
func Attribute(unit Unit, ptx string, dis *cudasass.Disassembly) *Attribution {
	a := &Attribution{
		Unit:        unit,
		PTX:         ptx,
		Disassembly: dis,
		CUDA:        map[Site][]int{},
		PTXLines:    map[Site][]int{},
		SASS:        map[Site][]cudasass.Instruction{},
		Remainder:   Remainder{Subroutines: map[string]int{}},
	}
	for _, s := range unit.Sites() {
		a.CUDA[s] = unit.Lines(s)
	}
	for s, lines := range ptxSites(ptx) {
		if unit.ProgramFile(s.File) {
			a.PTXLines[s] = lines
		}
	}
	if dis == nil {
		return a
	}
	for _, fn := range dis.Functions {
		for _, in := range fn.Instructions {
			a.Instructions++
			s := Site{File: in.SourceFile, Line: in.SourceLine}
			owner := fn.SymbolAt(in.Address)
			switch {
			case owner != fn.Name || !unit.Kernel(fn.Name):
				a.Remainder.Subroutines[owner]++
			case unit.ProgramFile(s.File):
				a.Attributed++
				a.SASS[s] = append(a.SASS[s], in)
			case s.File == "":
				a.Remainder.Unlocated++
			case filepath.Base(s.File) == unit.GeneratedName():
				a.Remainder.Generated++
			default:
				a.Remainder.Headers++
			}
		}
	}
	return a
}

// Unattributed reports whether the join failed outright: the disassembly
// holds instructions and not one of them names a line of the program.
//
// It is not the same as a program with no hot lines. Every report is empty
// in this state, and the cause is upstream of the measurement -- a unit
// that knows only a kernel's name, or file names that do not match the ones
// the compiler recorded -- so a report says so rather than printing an
// empty table under a heading.
func (a *Attribution) Unattributed() bool {
	return a.Attributed == 0 && a.Instructions > 0
}

// Sites returns every site any tier attributes to, in file and line order.
func (a *Attribution) Sites() []Site {
	seen := map[Site]bool{}
	var sites []Site
	add := func(s Site) {
		if !seen[s] {
			seen[s] = true
			sites = append(sites, s)
		}
	}
	for s := range a.CUDA {
		add(s)
	}
	for s := range a.PTXLines {
		add(s)
	}
	for s := range a.SASS {
		add(s)
	}
	sortSites(sites)
	return sites
}

// A Row is one site with what each tier attributes to it.
type Row struct {
	Site Site
	CUDA []int
	PTX  int
	SASS []cudasass.Instruction
}

// Rows returns one row per site, in file and line order.
func (a *Attribution) Rows() []Row {
	var rows []Row
	for _, s := range a.Sites() {
		rows = append(rows, Row{Site: s, CUDA: a.CUDA[s], PTX: len(a.PTXLines[s]), SASS: a.SASS[s]})
	}
	return rows
}

// Instruction returns the instruction at a function-relative offset in the
// named function, which is how PC sampling and SASS metrics report a PC.
func (a *Attribution) Instruction(function string, offset uint64) (cudasass.Instruction, bool) {
	if a.Disassembly == nil {
		return cudasass.Instruction{}, false
	}
	fn := a.Disassembly.Function(function)
	if fn == nil {
		return cudasass.Instruction{}, false
	}
	for _, in := range fn.Instructions {
		if in.Address == offset {
			return in, true
		}
	}
	return cudasass.Instruction{}, false
}

// SiteAt returns the program site of the instruction at a function-relative
// offset. It reports false when there is no such instruction or it names no
// site of the program.
func (a *Attribution) SiteAt(function string, offset uint64) (Site, bool) {
	in, ok := a.Instruction(function, offset)
	if !ok || !a.Unit.ProgramFile(in.SourceFile) {
		return Site{}, false
	}
	return Site{File: in.SourceFile, Line: in.SourceLine}, true
}

// ptxSites lists, by the site they name, the one-based lines of ptx holding
// a .loc directive. PTX refers to files by index and emits the .file table
// after the directives that use it, so the names are collected in a pass of
// their own. A .loc with line 0 names no line and is skipped, and a directive
// inside a comment is not a directive.
func ptxSites(ptx string) map[Site][]int {
	if ptx == "" {
		return nil
	}
	lines := strings.Split(ptx, "\n")
	inComment := false
	for i, text := range lines {
		lines[i], inComment = stripPTXComments(text, inComment)
	}
	files := map[int]string{}
	for _, text := range lines {
		rest, ok := directive(text, ".file")
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) < 2 {
			continue
		}
		idx, err := strconv.Atoi(f[0])
		if err != nil {
			continue
		}
		// The name is quoted and may hold spaces: take it from the text,
		// not from the fields.
		_, name, ok := strings.Cut(rest, `"`)
		if !ok {
			continue
		}
		if name, _, ok = strings.Cut(name, `"`); ok {
			files[idx] = name
		}
	}
	sites := map[Site][]int{}
	for i, text := range lines {
		rest, ok := directive(text, ".loc")
		if !ok {
			continue
		}
		// ".loc 1 5 3, inlined_at 1 9 2": the first triple is the location.
		triple, _, _ := strings.Cut(rest, ",")
		f := strings.Fields(triple)
		if len(f) < 2 {
			continue
		}
		idx, err := strconv.Atoi(f[0])
		if err != nil {
			continue
		}
		name, ok := files[idx]
		if !ok {
			continue
		}
		line, err := strconv.Atoi(f[1])
		if err != nil || line <= 0 {
			continue
		}
		s := Site{File: name, Line: line}
		sites[s] = append(sites[s], i+1)
	}
	return sites
}

// directive returns the text after name when text is that directive.
func directive(text, name string) (string, bool) {
	rest, ok := strings.CutPrefix(strings.TrimSpace(text), name)
	if !ok || rest == "" || !strings.ContainsRune(" \t", rune(rest[0])) {
		return "", false
	}
	return rest, true
}

// stripPTXComments removes the // and /* */ comments of one line of PTX,
// where inComment reports a block comment left open by the line before, and
// reports whether one is open at the end of this line. A comment marker
// inside a quoted file name is part of the name.
func stripPTXComments(text string, inComment bool) (string, bool) {
	var b strings.Builder
	quoted := false
	for i := 0; i < len(text); i++ {
		switch {
		case inComment:
			if strings.HasPrefix(text[i:], "*/") {
				inComment = false
				i++
				b.WriteByte(' ')
			}
		case quoted:
			quoted = text[i] != '"'
			b.WriteByte(text[i])
		case strings.HasPrefix(text[i:], "//"):
			return b.String(), false
		case strings.HasPrefix(text[i:], "/*"):
			inComment = true
			i++
		default:
			quoted = text[i] == '"'
			b.WriteByte(text[i])
		}
	}
	return b.String(), inComment
}

func (r Remainder) String() string {
	parts := []string{
		fmt.Sprintf("%d generated", r.Generated),
		fmt.Sprintf("%d unlocated", r.Unlocated),
	}
	if r.Headers > 0 {
		parts = append(parts, fmt.Sprintf("%d header", r.Headers))
	}
	for _, name := range sortedKeys(r.Subroutines) {
		parts = append(parts, fmt.Sprintf("%d in %s", r.Subroutines[name], name))
	}
	return strings.Join(parts, ", ")
}
