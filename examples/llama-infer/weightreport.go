package main

import (
	"fmt"
	"strings"
)

// WeightReport describes the weight storage this engine is actually running on.
//
// DType identifies the execution dtype; Storage identifies parameter storage.
// PJRT can execute half-precision graphs using parameters stored in another
// dtype. Storage is "mixed" and PerElem is zero for heterogeneous parameters.
// Kernels lists resolved entry points when the backend exposes them.
//
// Bytes is not measured. It is GPUModelWeights.WeightBytes, which sums
// elems*dtype.Bytes() over the tensors the loader recorded; it never consults
// the allocator, so it reports what the weights should occupy rather than what
// was allocated. The device-side figure is what a cudamem.Tracker on the
// weight arena reports, and that is what the native tests assert against.
// PJRT reports uploaded parameter payload bytes, excluding allocator overhead.
type WeightReport struct {
	DType   string   // execution dtype
	Storage string   // dtype the weights are stored in on the device
	Bytes   int      // calculated weight payload bytes
	PerElem int      // bytes per weight
	Kernels []string // entry point names actually bound
}

// String renders the report as one line.
func (r WeightReport) String() string {
	return fmt.Sprintf("weights=%s storage=%s bytes=%d per-weight=%d kernels=[%s]",
		r.DType, r.Storage, r.Bytes, r.PerElem, strings.Join(r.Kernels, " "))
}

