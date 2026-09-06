package main

import (
	"fmt"
	"strings"
)

// WeightReport describes the weight storage this engine is actually running on.
//
// DType comes from the bound module and Kernels are the entry point names that
// module resolved, so a run can be shown to have executed in the mode it
// claims and an f32 execution can never be recorded as an f16 success.
//
// Bytes is not measured. It is GPUModelWeights.WeightBytes, which sums
// elems*dtype.Bytes() over the tensors the loader recorded; it never consults
// the allocator, so it reports what the weights should occupy rather than what
// was allocated. The device-side figure is what a cudamem.Tracker on the
// weight arena reports, and that is what the native tests assert against.
type WeightReport struct {
	DType   string   // dtype the kernels were bound for
	Storage string   // dtype the weights are stored in on the device
	Bytes   int      // device bytes occupied by weights alone
	PerElem int      // bytes per weight
	Kernels []string // entry point names actually bound
}

// String renders the report as one line.
func (r WeightReport) String() string {
	return fmt.Sprintf("weights=%s storage=%s bytes=%d per-weight=%d kernels=[%s]",
		r.DType, r.Storage, r.Bytes, r.PerElem, strings.Join(r.Kernels, " "))
}

