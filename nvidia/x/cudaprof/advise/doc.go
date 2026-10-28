// Package advise ranks the performance findings in a cudaprof capture.
//
// [Analyze] reads a capture's profile, journals and code tiers (compiler
// remarks, PTX, SASS, CUBIN and register pressure) and returns a [Report]:
// capture integrity, occupancy, launch geometry, uncoalesced memory access,
// kernel sequences, host dispatch and API cost, device duty cycle,
// communication that compute does not overlap, and precise float math that
// costs a kernel local memory. Each
// [Item] carries a [Severity] saying in what order to read it, an
// [EvidenceLevel] saying how far to trust it, and the device or host time it
// accounts for.
//
// A capture of code alone -- SASS, a cubin, PTX or a ptxas -v report
// compiled for a target with no device, such as sm_103a -- yields the
// findings the code supports: registers, shared memory, spills, Tensor
// Memory and cluster occupancy limits, and which tensor-core instruction
// family (mma.sync, wgmma, tcgen05) a GEMM-like kernel uses against what its
// target offers. [Report.CompileOnly] marks such a report.
//
// [AnalyzeRanks] analyzes the captures of a job's ranks together, and
// [Report.ProjectTo] models a capture's step on another device.
//
// [Report.RankAgainst] reorders a report by how much of the gap to a base
// capture each finding could explain. The comparison itself is a [Baseline],
// supplied by the caller.
//
// [Report.Markdown] and [Report.MarkdownWithLimit] render the report as text,
// and [Report.JSON] as indented JSON.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package advise
