// Package cuptiactivity collects bounded CUPTI activity records and exports
// complete records to Perfetto.
//
// Start requires exclusive ownership of process-wide CUPTI activity APIs.
// Synchronize application work before Session.Close when complete device
// timestamps are required. Close drains buffers, not application CUDA work;
// failed cleanup retains ownership for retry. Inspect Session.Result for losses,
// pending buffers, incomplete records and errors before using a trace.
//
// Record decoding requires CUPTI 13.x and a callable cuptiActivityGetStructSize.
// A missing API returns an error wrapping cupti.SymbolError before collection
// starts. The package does not infer layouts from the version number.
//
// Each decodable kind is checked against the loaded library's current emitted
// record size. Known appended layouts are admitted as follows, in bytes:
//
//	Kind              CUPTI 13.x       Additional sizes for CUPTI >= 13.4
//	ConcurrentKernel  216, 224         232, 248
//	Memcpy            96               112
//	Memset            88               104
//
// Other kinds require the exact Go layout size. Unknown sizes, newer major
// versions, and newer layouts reported by earlier versions return LayoutError
// or a version error. A 13.3 runtime reporting a 232-byte kernel record remains
// refused. The known 13.4 tails preserve the decoded prefix; size checks and
// CPU decoder tests qualify that layout contract, not collection from every
// toolkit or GPU. Native acceptance requires records from the actual loaded
// CUPTI, driver and device. The CUPTI loader path is independent of the NVRTC
// and ptxas versions used to compile a workload.
//
// WriteRecord produces a standalone CUPTI-relative timeline; it does not align
// CUPTI timestamps with host clocks. This package does not collect hardware
// metrics; see github.com/tmc/lib/nvidia/x/cuptirange for Range Profiling.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cuptiactivity
