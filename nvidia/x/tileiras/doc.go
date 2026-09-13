// Package tileiras provides programmatic invocation and diagnostics parsing
// for NVIDIA's tileiras optimizing Tile IR assembler.
//
// tileiras translates CUDA Tile IR bytecode into ELF CUBIN machine code and
// can package outputs into fatbinaries (.fatbin). It parses compiler
// optimization remarks to detect TMA instruction-selection bottlenecks,
// Tensor Core MMA mapping, and optimization hint diagnostics.
//
// # Remarks & TMA Diagnostics
//
// When remarks are enabled with WithRemarks, tileiras produces structured
// diagnostic remarks explaining compiler decisions, such as whether a load/store
// was promoted to Tensor Memory Accelerator (TMA) hardware instructions or
// fell back due to stride, alignment, or shape constraints:
//
//	res, err := tileiras.Assemble(ctx, bytecode,
//		tileiras.WithGPUName("sm_90"),
//		tileiras.WithRemarks("all"),
//	)
//	for _, fail := range res.TMAFailures() {
//		log.Printf("TMA bottleneck at %s:%d: %s", fail.File, fail.Line, fail.Remark)
//	}
package tileiras
