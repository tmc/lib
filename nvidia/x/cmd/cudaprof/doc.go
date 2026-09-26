// Cudaprof records a CUDA program's device timeline with CUPTI and reads the
// resulting capture: kernel totals and source through upstream pprof, PTX and
// NVIDIA disassembly, idle gaps, a per-op comparison of two captures, the
// verdict of an ABBA timing campaign, and a Perfetto trace.
//
// Capture requires Linux and the CUPTI agent built by go generate. Every
// other command reads a capture file and requires neither a GPU nor CUDA.
//
// Run "cudaprof help" for the commands, and see README.md for the workflow.
//
// # Ops files
//
// The -ops flag of compare, qualify and advise names a file that maps
// kernels to ops. Blank lines and lines starting with # are ignored. Each
// other line is a rule,
//
//	regex op
//
// and the first rule whose regular expression matches a kernel's match key
// (its demangled name without parameters, and without template arguments
// unless -templates is given) names the kernel's op. A regex ends at the
// first space or tab, so it cannot contain one; use \s. Kernels no rule
// matches are op (unmatched), a name no rule may use.
//
// Compare, and ab through it, also reads combine lines, whose second field
// is "=":
//
//	name = op[:glob,...] ...
//
// A combine line reports one more row that adds up the kernels of op name,
// if a rule names it, and those each term takes: every kernel of the term's
// op, or with globs, the kernels of that op whose functor or match key a
// glob matches, ignoring case. A functor is what a PyTorch elementwise kernel
// applies, such as MulFunctor or GeluCUDAKernelImpl, read from its template
// arguments. Globs are those of [path.Match]. Every term must name an op
// that a rule names, and no kernel may be taken by two combine lines. The
// rows sit outside compare's identity: they read fused work on one side
// against the kernels it replaced on the other.
//
//	GeGLU geglu
//	elementwise elementwise
//	.  other
//	geglu = elementwise:gelu*,mulfunctor
//
// compare -ops-check reports the kernels each rule won and the kernels it
// matched that an earlier rule won, and fails when a rule wins no kernel in
// either capture or a combine term takes none. Without it, such lines are
// listed only in the JSON output's unused_rules.
package main
