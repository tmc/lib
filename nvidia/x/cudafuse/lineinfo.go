package cudafuse

import "github.com/tmc/lib/nvidia/x/cudalineinfo"

// Unit returns the kernel as a cudalineinfo unit: the same name, source, and
// line map, for attributing the compiled kernel back to its Go statements.
func (fk FusedKernel) Unit() cudalineinfo.Unit {
	return cudalineinfo.Unit{Name: fk.Name, Source: fk.Source, LineMap: fk.LineMap}
}

// SourceWithLineDirectives returns the kernel source with C preprocessor #line
// directives attributing generated lines to the Go code they were lowered
// from, as [cudalineinfo.Unit.SourceWithLineDirectives] describes. Lines with
// no Go origin are reported under the kernel's name with a .cu suffix.
func (fk FusedKernel) SourceWithLineDirectives() string {
	return fk.Unit().SourceWithLineDirectives()
}
