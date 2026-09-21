package cudafuse

import (
	"fmt"
	"strings"
)

// SourceWithLineDirectives returns the kernel source with C preprocessor #line
// directives attributing generated lines to the Go code they were lowered from.
//
// Compiling the result with line information (nvcc -lineinfo, or NVRTC's
// equivalent) makes the device binary record those Go coordinates, so a
// disassembler reports the program's own file and line instead of generated
// text that was never written to disk.
//
// generatedName is the file name reported for lines with no Go origin; the
// directives restore it after each annotated run so a codegen bug still points
// into the generated source.
func (fk FusedKernel) SourceWithLineDirectives(generatedName string) string {
	if len(fk.LineMap) == 0 {
		return fk.Source
	}
	lines := strings.Split(fk.Source, "\n")
	var b strings.Builder
	redirected := false
	for i, line := range lines {
		n := i + 1
		loc, ok := fk.LineMap[n]
		switch {
		case ok && loc.File != "" && loc.Line > 0:
			fmt.Fprintf(&b, "#line %d %q\n", loc.Line, loc.File)
			redirected = true
		case redirected:
			fmt.Fprintf(&b, "#line %d %q\n", n, generatedName)
			redirected = false
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return strings.TrimSuffix(b.String(), "\n")
}
