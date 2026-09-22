package cudalineinfo

import (
	"fmt"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/ianlancetaylor/demangle"
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
// diagnostic would. Two files with the same base name print alike, so a
// report that must tell every site of a set apart names them with
// [SiteNames] instead.
func (s Site) String() string {
	return fmt.Sprintf("%s:%d", filepath.Base(s.File), s.Line)
}

// SiteNames returns a display name for each site: the shortest suffix of
// path elements that tells its file apart from the others, with the line.
// A report of one file prints base names, and one spanning two files that
// share a base name prints enough of the directory to tell them apart, so
// a row or a graph node names exactly one site.
func SiteNames(sites []Site) map[Site]string {
	files := map[string]bool{}
	for _, s := range sites {
		files[s.File] = true
	}
	suffix := fileSuffixes(files)
	names := make(map[Site]string, len(sites))
	for _, s := range sites {
		names[s] = fmt.Sprintf("%s:%d", suffix[s.File], s.Line)
	}
	return names
}

// fileSuffixes names each file by the shortest tail of its path elements
// that no other file shares. Paths are split on both separators, because a
// file name reaches here from a compiler's line table and may have been
// written on another system.
func fileSuffixes(files map[string]bool) map[string]string {
	elems := make(map[string][]string, len(files))
	longest := 0
	for f := range files {
		e := strings.FieldsFunc(f, func(r rune) bool { return r == '/' || r == '\\' })
		elems[f] = e
		longest = max(longest, len(e))
	}
	names := make(map[string]string, len(files))
	for n := 1; n <= longest; n++ {
		count := map[string]int{}
		for f := range files {
			count[tail(elems[f], n)]++
		}
		for f := range files {
			if _, done := names[f]; done {
				continue
			}
			if t := tail(elems[f], n); count[t] == 1 {
				names[f] = t
			}
		}
	}
	// Two files with identical paths are one file, so anything still
	// unnamed has no distinguishing suffix and is named by its whole path.
	for f := range files {
		if _, done := names[f]; !done {
			names[f] = f
		}
	}
	return names
}

// tail joins the last n elements of a path.
func tail(elems []string, n int) string {
	if n > len(elems) {
		n = len(elems)
	}
	return path.Join(elems[len(elems)-n:]...)
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

// A Unit is one CUDA C++ source with its provenance: for each generated
// line that has one, the source location it was lowered from. A unit with
// no line map is hand-written, and its own file is the program.
//
// Name is used, with a .cu suffix, as the file name the unit's own lines are
// reported under, and, when Kernels is empty, as the name of its one kernel.
// Source carries no #line directives; [Unit.SourceWithLineDirectives] adds
// them.
type Unit struct {
	Name    string
	Source  string
	LineMap map[int]cudair.SourceLocation

	// Kernels names the functions of the compiled unit that are its
	// kernels, for a source holding more than one. Empty means the unit
	// has the one kernel Name names, or, when Name is empty too, that
	// every function of the disassembly is a kernel of the unit.
	Kernels []string
}

// Kernel reports whether a function of the unit's disassembly is one of its
// kernels. Instructions of any other function are the compiler's, such as a
// division slowpath, and are not attributed to the program.
//
// A __global__ function declared without extern "C" reaches the disassembly
// under its C++ mangled symbol, so a name is matched against three spellings
// of the function: the symbol itself, which is how an extern "C" kernel
// arrives; the demangled name without parameters or template arguments, such
// as ns::geglu; and that name's last :: component. A caller therefore names a
// kernel the way it is written in the source, whether or not the compiler
// mangled it. The full demangled signature is not a spelling: its parameter
// text is the demangler's, not the source's.
func (u Unit) Kernel(name string) bool {
	if len(u.Kernels) == 0 {
		return u.Name == "" || name == u.Name
	}
	return slices.ContainsFunc(u.Kernels, func(k string) bool {
		return slices.Contains(kernelSpellings(name), k)
	})
}

// kernelSpellings returns the names a disassembly function answers to.
func kernelSpellings(name string) []string {
	spellings := []string{name}
	d, err := demangle.ToString(name, demangle.NoParams, demangle.NoTemplateParams)
	if err != nil || d == "" || d == name {
		return spellings
	}
	spellings = append(spellings, d)
	if i := strings.LastIndex(d, "::"); i >= 0 {
		spellings = append(spellings, d[i+len("::"):])
	}
	return spellings
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

// ProgramFile reports whether a file named by the toolchain belongs to the
// program rather than to the unit itself, a header, or nvdisasm's embedded
// PTX text. Those are the three other kinds of name a line directive can
// carry, and an instruction naming one of them is not attributed to the
// program. The unit's own file is matched by base name, since the compiler
// and a profile may record it under different paths.
//
// The unit's own file is the program when the unit has source and no line
// map, which is how hand-written CUDA reaches here: nothing was lowered
// into it, so there is no generated tier to keep out.
func (u Unit) ProgramFile(file string) bool {
	if file == "" || file == cudasass.PTXTextFile {
		return false
	}
	if file == u.GeneratedName() || filepath.Base(file) == u.GeneratedName() {
		// A unit carrying source but no line map is hand-written: there is
		// no generated tier below it, so its own file is the program. A
		// bare unit, which holds neither, knows only a name and is not
		// entitled to say what its file is.
		return len(u.LineMap) == 0 && u.Source != ""
	}
	switch strings.ToLower(path.Ext(file)) {
	case ".h", ".hpp", ".hxx", ".hh", ".cuh", ".inl":
		return false
	}
	return true
}
