// Package advise ranks the performance findings in a cudaprof capture.
//
// [Analyze] reads a capture's profile, journals and code tiers (compiler
// remarks, PTX, SASS, CUBIN and register pressure) and returns a [Report]:
// capture integrity, occupancy, launch geometry, uncoalesced memory access,
// kernel sequences, host dispatch and API cost, device duty cycle, and
// precise float math that costs a kernel local memory. Each
// [Item] carries a [Severity] saying in what order to read it, an
// [EvidenceLevel] saying how far to trust it, and the device or host time it
// accounts for.
//
// Wide address instructions are static context observations. Their presence
// alone proves neither compiler fallback nor inefficiency, even alongside
// vector or TMA operations. These findings cannot acquire runtime share or
// comparison-gap credit; inspect memory width, stride and layout first.
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
