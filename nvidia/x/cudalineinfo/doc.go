// Package cudalineinfo attributes device code to the source it was generated
// from.
//
// A program that lowers Go to CUDA C++ knows, for every generated line, the Go
// statement it came from. Emitting that as #line directives and compiling with
// line information makes the compiler record the Go coordinates instead of
// the generated ones, so PTX .loc directives and nvdisasm's line comments name
// the Go file and line directly. This package emits the directives, compiles
// and disassembles the result, and joins each tier back to its [Site]: how
// many generated lines, PTX locations, and SASS instructions each Go statement
// produced, and which instructions they are.
//
// [Attribute] does the join from artifacts already in hand; [Compile] produces
// them with NVRTC and nvdisasm first. [Diff] compares two attributions of the
// same unit, for example with and without fast math. [WriteTable],
// [WriteListing], and [WriteDiff] print the results.
//
//	unit := fk.Unit() // cudafuse.FusedKernel
//	a, err := cudalineinfo.Compile(ctx, unit, cudalineinfo.Options{Arch: "sm_121"})
//	if err != nil {
//		return err
//	}
//	cudalineinfo.WriteTable(os.Stdout, a)
//
// Attribution is only as exact as the compiler keeps it. An optimizer that
// fuses one statement's operation into its consumer attributes the result to
// one line and the other gets zero instructions; a statement is not
// guaranteed to survive as instructions of its own.
package cudalineinfo
