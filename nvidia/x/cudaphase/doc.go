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
// phases near that step are not resolved.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaphase
