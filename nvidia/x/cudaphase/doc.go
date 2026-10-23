// Package cudaphase times the phases of a kernel from inside it, one row
// of timestamps per block, and summarizes them where no profiler runs,
// such as a remote judge that returns only the tail of a program's output.
//
// The device side is the header [Header] (cudaphase.h): a kernel takes a
// trace buffer and marks phase boundaries with CUDAPHASE_MARK and
// cudaphase_now, which reads %globaltimer. [Python] (cudaphase.py) defines
// cudaphase_summary, which a submission calls after a traced launch to
// print one line of per-phase median, minimum and maximum across blocks,
// the launch's span, and the number of SMs used. [Parse] reads those lines
// back from a log, and [Summarize] computes the same summary in Go from a
// raw buffer.
//
// globaltimer's resolution is coarse (256 ns steps were observed on an
// H200), so each summary reports the smallest step between its stamps;
// phases near that step are not resolved. For finer phases a kernel
// stamps cudaphase_clock, the SM's %clock64 cycle counter, and anchors
// each block with globaltimer and the SM number at its start and end.
// [SummarizeClock] measures the SM clock from the blocks spanning at
// least 2 µs, converts cycles at that rate, places each block on
// globaltimer's timeline, and leaves out blocks whose SM changed.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaphase
