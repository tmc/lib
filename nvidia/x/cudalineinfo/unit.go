package cudalineinfo

import (
	"fmt"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A Site is one place in a source file. It is the key that joins the tiers:
// a generated CUDA line, a PTX .loc directive, and a SASS instruction each
// name a file and a line, and nothing else survives all three.
type Site struct {
	File string
	Line int
}

// SiteOf returns the site of a source location.
func SiteOf(loc cudair.SourceLocation) Site {
	return Site{File: loc.File, Line: loc.Line}
}

// String formats the site as base name and line, the way a compiler
// diagnostic would.
func (s Site) String() string {
	return fmt.Sprintf("%s:%d", filepath.Base(s.File), s.Line)
}

// less orders sites by file, then line.
func (s Site) less(t Site) bool {
	if s.File != t.File {
		return s.File < t.File
	}
	return s.Line < t.Line
}

func sortSites(sites []Site) {
	sort.Slice(sites, func(i, j int) bool { return sites[i].less(sites[j]) })
}

// A Unit is one generated CUDA C++ source with its provenance: for each
// generated line that has one, the source location it was lowered from.
//
// Name is the kernel's name and is also used, with a .cu suffix, as the file
// name the unit's own lines are reported under. Source carries no #line
// directives; [Unit.SourceWithLineDirectives] adds them.
type Unit struct {
	Name    string
	Source  string
	LineMap map[int]cudair.SourceLocation
}

// GeneratedName is the file name the unit's own lines are reported under.
func (u Unit) GeneratedName() string {
	return u.Name + ".cu"
}

// SourceWithLineDirectives returns the unit's source with C preprocessor
// #line directives attributing generated lines to the code they were lowered
// from.
//
// Compiling the result with line information (nvcc -lineinfo, or NVRTC's
// equivalent) makes the device binary record those coordinates, so a
// disassembler reports the program's own file and line instead of generated
// text that was never written to disk. After each annotated run a directive
// restores the generated name, so a codegen bug still points into the
// generated source.
func (u Unit) SourceWithLineDirectives() string {
	if len(u.LineMap) == 0 {
		return u.Source
	}
	generated := u.GeneratedName()
	lines := strings.Split(u.Source, "\n")
	var b strings.Builder
	redirected := false
	for i, line := range lines {
		n := i + 1
		loc, ok := u.LineMap[n]
		switch {
		case ok && loc.File != "" && loc.Line > 0:
			fmt.Fprintf(&b, "#line %d %q\n", loc.Line, loc.File)
			redirected = true
		case redirected:
			fmt.Fprintf(&b, "#line %d %q\n", n, generated)
			redirected = false
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return strings.TrimSuffix(b.String(), "\n")
}

// Sites returns the sites the unit's lines were lowered from, in file and
// line order, each once.
func (u Unit) Sites() []Site {
	seen := map[Site]bool{}
	var sites []Site
	for _, loc := range u.LineMap {
		s := SiteOf(loc)
		if s.File == "" || s.Line <= 0 || seen[s] {
			continue
		}
		seen[s] = true
		sites = append(sites, s)
	}
	sortSites(sites)
	return sites
}

// Location returns the recorded origin of a site: the location of the
// lowest generated line mapped to it. Locations carry what a site does not,
// such as the function name and the callers above it.
func (u Unit) Location(s Site) (cudair.SourceLocation, bool) {
	best, found := 0, false
	var loc cudair.SourceLocation
	for n, l := range u.LineMap {
		if SiteOf(l) != s {
			continue
		}
		if !found || n < best {
			best, found, loc = n, true, l
		}
	}
	return loc, found
}

// Lines returns the generated lines lowered from a site, in order.
func (u Unit) Lines(s Site) []int {
	var lines []int
	for n, l := range u.LineMap {
		if SiteOf(l) == s {
			lines = append(lines, n)
		}
	}
	sort.Ints(lines)
	return lines
}

// isUserFile reports whether a file named by the toolchain belongs to the
// program rather than to the unit itself, a header, or nvdisasm's embedded
// PTX text. Those are the three other kinds of name a line directive can
// carry, and an instruction naming one of them is not attributed to the
// program.
func (u Unit) isUserFile(file string) bool {
	if file == "" || file == cudasass.PTXTextFile {
		return false
	}
	if file == u.GeneratedName() || filepath.Base(file) == u.GeneratedName() {
		return false
	}
	switch strings.ToLower(path.Ext(file)) {
	case ".h", ".hpp", ".hxx", ".hh", ".cuh", ".inl":
		return false
	}
	return true
}
