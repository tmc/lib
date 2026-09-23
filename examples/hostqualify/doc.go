// Hostqualify refuses to record a throughput number from a degraded host.
//
// Every throughput comparison in this port assumes the two runs saw the
// same machine. One set of Metal measurements was taken at load average
// 23 on a 16-core host, next to another session's test suite, and the
// oracle's own benchmark reproduced its published rate at a third of
// the published value. Nothing in the tooling noticed; the numbers were
// written down as if they meant something.
//
// Run it before a measurement to check the host is quiet:
//
//	hostqualify
//
// Run it after, with the rates the run reported and the reference the
// oracle publishes for this machine, to check the host was quiet enough
// for the numbers to stand:
//
//	hostqualify -reference speed-bench/m4_max.csv -ctx 2048 \
//	    -prefill 343.8 -gen 26.8
//
// A single reading before a run and another after cannot say whether
// the limit was ever within reach. Sample the idle host to find out,
// before blaming a failed after-check on the run:
//
//	hostqualify -baseline -samples 16 -interval 15s
//
// It reports the load distribution and qualifies only if every sample
// is under the limit. On the M4 Max this port measures, sixteen samples
// taken with no work running gave min 3.45, max 4.84, mean 4.09 against
// a registered limit of 4.00, ten of them over the line. The quiet
// windows there are unstable rather than absent: six samples qualified,
// so a run can still start on a host that passes, and whether it holds
// for a full interval is what this does not establish. The answer to a
// failed after-check is a quieter host, never a wider limit.
//
// It exits non-zero when the host does not qualify, so a measurement
// script can gate on it. The reference file is a ds4-bench CSV: its
// header names the columns, so both the Metal and the CUDA reference
// files parse.
package main
