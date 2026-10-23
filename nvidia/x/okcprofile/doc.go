// Package okcprofile reads Open Kernel Challenge judge profiles: the JSON a
// judge returns after timing a submission against its baseline.
//
// [Decode] reads the file. [File.Series] turns its paired samples into
// [pairedtiming.Series], so the judge's score can be recomputed and
// bootstrapped offline; consecutive baseline and candidate samples form one
// ABBA block and the judge's replay overhead is the intercept.
//
// [ParseReport] reads the human-readable kernel table the judge attaches to
// each profiled case, and [Kernels] sums those tables into one weighted
// per-kernel table per load. The tables are lossy in ways the parser reports
// rather than hides:
//
//   - times are rounded to 0.1 µs;
//   - kernel names longer than the column end in "..." and two different
//     kernels can truncate to the same name, so a row may be several kernels;
//   - "... N more" hides N kernel groups, so visible launch counts are a lower
//     bound;
//   - when kernels overlap (PDL or streams) each row includes time spent
//     waiting for its predecessor, so the rows sum to more than the GPU was
//     busy, and the judge prints the busy time separately.
//
// A line the parser does not recognize is an error, not a skipped row: a
// format change must not shrink the table silently.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package okcprofile
