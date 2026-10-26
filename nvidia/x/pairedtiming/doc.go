// Package pairedtiming estimates the speedup of a candidate over a baseline
// from paired GPU timings run in ABBA blocks.
//
// A [Block] holds one baseline A and one candidate B timed in the order
// A1, B1, B2, A2, so a linear drift within the block cancels. [Paired]
// reduces blocks to a median per-block ratio after removing a fixed replay
// intercept, the per-call cost (graph launch, event recording) that a
// smaller kernel cannot remove. [Intercept] estimates that intercept from
// control blocks timing one and two copies of a trivial operation.
//
// [Score] combines shapes the way a weighted benchmark suite does: the
// weighted sums of baseline and candidate times within each load give one
// ratio per load, and the score is the geometric mean of those ratios.
// [Bootstrap] resamples whole blocks within each shape to give a percentile
// interval for that score. The interval describes one campaign's noise only;
// it says nothing about another host, process or day.
//
// [SignTest] is the exact two-sided binomial sign test for the number of
// blocks in which the candidate won.
//
// All times are in one caller-chosen unit; the package never converts units.
package pairedtiming
