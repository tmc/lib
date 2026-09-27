// Package trainstat writes training runs in the Go benchmark format, for
// comparison with the trainstat command or benchstat.
//
// A run is file configuration (the model, data, options and seed) followed
// by result lines, each an observation with an iteration count of 1:
//
//	seed: 3
//	lr-encoder: 5e-4
//	Unit accuracy better=higher assume=normal delta=abs
//	BenchmarkTrain/step=1 1 1.412 loss 3.91 grad-norm 64 seqs 2.412 sec/step
//	BenchmarkEval/split=val/group=all/cal=raw/step=458 1 0.5501 loss 0.7968 accuracy
//	BenchmarkFinal/split=val/group=all/cal=temp 1 0.4203 loss 0.8119 accuracy
//	BenchmarkRun 1 1830 steps 117120 seqs 27054720 tokens 3978.1 train-sec 1 done 0 nonfinite
//
// The reserved benchmark names are Train (a per-step curve, keyed by
// /step), Eval (an evaluation curve, keyed by /split, /group, /cal and
// /step), Final (end-of-run values), Run (status and totals) and
// Throughput (a steady-state aggregate whose iteration count is the
// number of steps it covers). Other names pass through unchanged.
//
// The configuration key seed is the replicate: the trainstat command
// merges bit-identical results with the same seed and compares columns
// across seeds. The key run and keys beginning with out- identify one run
// and are never compared as configuration.
//
// Three unit metadata keys are added to the predefined better and assume:
// delta=abs prints differences as absolute values rather than percentages,
// kind=time marks a nondeterministic unit, and assume=normal requests
// t intervals. [Units] lists the metadata of the standard units.
//
// In tests, [Report] records values as test attributes and [Artifact]
// writes curves to the test's artifact directory. The attribute keys are
//
//	trainstat:<unit>              a value for the test's own row
//	trainstat:<result>:<unit>     a value for a named row
//	trainstat.unit:<unit>         unit metadata, such as "better=higher delta=abs"
//	trainstat.config:<key>        run configuration
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package trainstat
