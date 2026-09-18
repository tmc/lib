package cudasass

import (
	"strconv"
	"strings"
)

// A SourceFileKind classifies the file named in an nvdisasm line directive.
//
// nvdisasm spells a real source file and its own PTX cross-reference tier with
// the same syntax, so the name alone does not say which one a directive means.
// Every consumer that joins on (file, line) has to tell them apart, and doing
// it by comparing against a literal at each call site is how one call site
// drifts from the rest.
type SourceFileKind int

const (
	// CUDASource is a CUDA source or header file that took part in the
	// compilation. Its line numbers index that file.
	CUDASource SourceFileKind = iota

	// PTXText is nvdisasm's pseudo-file ".nv_debug_ptx_txt", which holds the
	// PTX text embedded in the CUBIN rather than any file on disk. Its line
	// numbers index that embedded text and run far past the end of any real
	// source: a module of a few thousand source lines routinely names PTX
	// lines past 140000. Fusing those numbers into a source file's namespace
	// attributes instructions to lines that file does not have.
	PTXText
)

func (k SourceFileKind) String() string {
	switch k {
	case CUDASource:
		return "CUDASource"
	case PTXText:
		return "PTXText"
	}
	return "SourceFileKind(" + strconv.Itoa(int(k)) + ")"
}

// PTXTextFile is the name nvdisasm gives the embedded PTX text. It is exported
// so a caller can recognise the name in data it did not parse here; a caller
// that has a [SourceDirective] should read its Kind instead.
const PTXTextFile = ".nv_debug_ptx_txt"

// A SourceDirective is one parsed nvdisasm line directive:
//
//	//## File "kernels.cu", line 42
//	//## File "util.cuh", line 7 inlined at "kernels.cu", line 42
//
// The directive establishes a location for every instruction that follows it,
// up to the next directive or the next function, which is nvdisasm's own
// convention.
type SourceDirective struct {
	File string         // file named by the directive, as nvdisasm spelled it
	Line int            // line within File
	Kind SourceFileKind // what File is: a CUDA source, or the PTX text

	// InlinedFile and InlinedLine are the call site the directive attributes
	// File:Line to, when it carries an "inlined at" annotation. nvdisasm emits
	// one annotation per inlining level; only the first, the immediate call
	// site, is reported here. InlinedFile is empty exactly when InlinedLine is
	// zero.
	InlinedFile string
	InlinedLine int
}

// ParseSourceDirective parses an nvdisasm line directive. It reports false for
// any other line, including a line that merely contains one: a directive is
// recognised only at the start of the line, after leading space.
func ParseSourceDirective(line string) (SourceDirective, bool) {
	m := sourceLocationPattern.FindStringSubmatch(strings.TrimSpace(line))
	if m == nil {
		return SourceDirective{}, false
	}
	d := SourceDirective{File: m[1], Kind: classifySourceFile(m[1])}
	d.Line, _ = strconv.Atoi(m[2])
	if mi := inlineLocationPattern.FindStringSubmatch(line); mi != nil {
		d.InlinedFile = mi[1]
		d.InlinedLine, _ = strconv.Atoi(mi[2])
		if d.InlinedLine == 0 {
			d.InlinedFile = ""
		}
	}
	return d, true
}

// classifySourceFile reports what kind of file name is. The name is compared
// whole: a real source may legitimately be named ".nv_debug_ptx_txt.cu", and a
// prefix or substring test would misclassify it.
func classifySourceFile(name string) SourceFileKind {
	if name == PTXTextFile {
		return PTXText
	}
	return CUDASource
}
