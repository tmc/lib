// Cudainspect reports static facts about one kernel in a CUBIN.
//
// Usage:
//
//	cudainspect report -symbol NAME [-require a,b] [-optional c] [-sass FILE] [-max-bytes N] FILE.cubin
//
// The report subcommand writes a [cudainspect.Report] as JSON to standard
// output. -require and -optional name features, comma separated, ignoring
// spaces around each name; -sass supplies cuobjdump -sass or nvdisasm output
// for the same image, from which the instruction features come. Cudainspect runs no tools and loads nothing.
//
// The exit status is 0 when every required feature has a value, 1 when a
// required feature has none (the report is still written), and 2 when the
// input is refused or the command line is wrong: a missing or unknown symbol,
// an image with no code sections, a non-CUDA input, a CUDA image whose ELF
// version is not 1, an input over its bound, or SASS that does not list the
// symbol.
//
// [cudainspect.Report]: https://pkg.go.dev/github.com/tmc/lib/nvidia/x/cudainspect#Report
package main
