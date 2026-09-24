// Trainstat compares training runs, as benchstat compares benchmarks.
//
// Usage:
//
//	trainstat [flags] inputs...
//	trainstat jsonl [-config file]... [-x step] [-time ms] [-group-key group] events.jsonl
//	trainstat repro [-pin keys] a b
//	trainstat gate -rules file [-lenient] [-min-n n] [-json] [flags] base new
//
// Every input becomes Go benchmark format results, in the shape the
// [github.com/tmc/lib/nvidia/x/trainstat] package writes. An input is
// recognized by its content: benchmark format text, including go test
// -bench output; a go test -json stream or go test -v text, whose
// trainstat attributes and artifact directories become runs; or a JSONL
// event log. A directory is a run directory: DIR/trainstat.txt, or else
// DIR/progress.jsonl converted with DIR/train/state.json as configuration.
// A test's artifact directory is read as a run directory.
// An input "-" is standard input, and label=path names an input's label.
//
// The summary compares each run's Final, Best, Run and Throughput results
// across columns. Columns default to the configuration keys whose values
// differ between inputs, other than the replicate key (seed), run and
// out-* keys. A result without such a key takes its input's only value
// of it, so benchmark lines join the tests beside them. Within a column,
// results of a unit that is not a time are merged by replicate: results
// with the same seed must be bit-identical and count once. Units with
// assume=normal show a mean with a Student t interval, and are compared
// by a t test paired by seed when both columns
// have the same seeds and validation sets, and by Welch's t test
// otherwise. Other units are summarized and compared as benchstat does.
// Differences are absolute for delta=abs units and percentages otherwise.
// A cell with one sample shows its value, labeled n=1, with no interval.
// A block of a relative unit with several rows and columns ends with a
// geometric mean row when every value is positive.
//
// Best/keys is derived from each run's Eval/keys/step=N results: the best
// value by the unit's better= direction. Throughput is derived from Train
// results: the mean sec/step, and tokens/s, over steps after -skip,
// excluding each step that follows an evaluation.
//
// The -col, -row, -table, -filter and -ignore flags take benchstat's
// projection and filter syntax. -format benchfmt writes the converted and
// derived results for benchstat.
//
// Jsonl converts a JSONL event log to benchmark format: "step" events to
// Train/step=N results, "eval" events to Eval/split=S/group=G/cal=raw|temp/step=N
// results, the last evaluation also to Final results, and "done" and
// "stop" to the Run result, with nonfinite 1 if the stop event has
// "nonfinite": true. An object with no event field, such as a single evaluation,
// becomes Final results. A "start" event's scalar fields, and those of
// each -config file, become configuration.
//
// Repro checks that two runs are bit-identical: every value except those
// of time units, and every out-* configuration key. It has three outcomes.
// If any pinning key differs (-pin, by default commit, lib-nvidia, nvrtc,
// cublas, driver, device, exact, tf32, mlx-c and libmlx), the runs need
// not agree, and repro prints "refused: pinning keys differ: " naming each
// key and its two values, and exits with status 1. A pinning key whose
// value is "unknown" differs even from itself. If the values or out-*
// keys differ, it prints the first difference with its distance in float32
// ulps and the number of differing values, and exits with status 1; when
// the whole configuration is equal apart from run and host, it first
// prints "nondeterministic: same configuration, different results". The
// first difference is where the runs diverge, not necessarily why.
// Otherwise it prints "ok". Configuration differences other than pinning
// keys are reported but do not by themselves fail.
//
// Gate judges rules on the difference between two columns. Each line of
// the rules file is
//
//	row unit op value
//
// where op is >= or <=, bounding the difference (value may end in %, for
// a bound on new/base-1), or ==, requiring every value in both columns to
// equal value exactly. A bound passes when the difference's 1-alpha
// confidence interval lies within it, fails when the interval lies
// outside, and is otherwise INCONCLUSIVE. Each of the k rules is judged at
// -alpha on its own; there is no correction for k. A rule is INSUFFICIENT
// when a column has fewer samples than -min-n, or, for a bound, one seed,
// or fewer than 6 samples of a time unit or of any unit without
// assume=normal or assume=exact. A rule that matches no row fails, and a
// line that is not a rule is INCONCLUSIVE. Gate exits with status 1 if
// any rule fails, or is INCONCLUSIVE or INSUFFICIENT without -lenient,
// and 2 on a usage error.
package main
